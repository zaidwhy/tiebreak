"""HTTP surface: one rerank handler mounted on the paths Cohere, Jina and Ollama-style clients call."""
from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tiebreak import __version__
from tiebreak.engine import DeadlineExceeded, LimitError, to_relevance
from tiebreak.settings import Settings

log = logging.getLogger("tiebreak.server")

RERANK_PATHS = ("/v1/rerank", "/v2/rerank", "/rerank", "/api/rerank")


class BadRequest(Exception):
    def __init__(self, field: str, message: str, status: int = 400):
        super().__init__(message)
        self.field, self.status = field, status


def _error(status: int, message: str, field: str | None = None, limit: int | None = None) -> JSONResponse:
    body: dict = {"message": message}
    if field:
        body["field"] = field
    if limit is not None:
        body["limit"] = limit
    return JSONResponse(body, status_code=status)


def parse_documents(raw, cfg: Settings) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise BadRequest("documents", "documents must be a non-empty list of strings or {\"text\": ...} objects")
    if len(raw) > cfg.max_documents:
        raise LimitError("documents", cfg.max_documents,
                         f"documents has {len(raw)} items, limit TIEBREAK_MAX_DOCUMENTS={cfg.max_documents}")
    texts: list[str] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            item = item.get("text")
        if not isinstance(item, str):
            raise BadRequest(f"documents[{i}]", f"documents[{i}] must be a string or an object with a string 'text'")
        if len(item) > cfg.max_chars_per_document:
            raise LimitError(f"documents[{i}]", cfg.max_chars_per_document,
                             f"documents[{i}] has {len(item)} characters, "
                             f"limit TIEBREAK_MAX_CHARS_PER_DOCUMENT={cfg.max_chars_per_document}")
        texts.append(item)
    return texts


def create_app(cfg: Settings, engine) -> FastAPI:
    app = FastAPI(title="tiebreak", version=__version__, docs_url=None, redoc_url=None)
    gate = threading.Lock()  # one scoring pass at a time: ORT already uses every core
    body_cap = cfg.max_documents * cfg.max_chars_per_document

    def unauthorized(request: Request) -> bool:
        if not cfg.api_key:
            return False
        sent = request.headers.get("authorization", "")
        return not hmac.compare_digest(sent, f"Bearer {cfg.api_key}")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_id, "version": __version__}

    @app.get("/v1/models")
    def models(request: Request):
        if unauthorized(request):
            return _error(401, "missing or wrong bearer token")
        return {"object": "list", "data": [{"id": engine.model_id, "object": "model", "owned_by": "tiebreak"}]}

    def rerank(request: Request, payload: dict):
        started = time.monotonic()
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise BadRequest("query", "query must be a non-empty string")
        if len(query) > cfg.max_chars_per_document:
            raise LimitError("query", cfg.max_chars_per_document,
                             f"query has {len(query)} characters, "
                             f"limit TIEBREAK_MAX_CHARS_PER_DOCUMENT={cfg.max_chars_per_document}")
        docs = parse_documents(payload.get("documents"), cfg)
        top_n = payload.get("top_n")
        if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
            raise BadRequest("top_n", "top_n must be a positive integer")
        top_n = min(top_n or len(docs), len(docs))
        want_docs = payload.get("return_documents", False)
        if not isinstance(want_docs, bool):
            raise BadRequest("return_documents", "return_documents must be true or false")
        scale = payload.get("score_scale", cfg.scores)
        if scale not in ("sigmoid", "raw"):
            raise BadRequest("score_scale", "score_scale must be 'sigmoid' or 'raw'")

        with gate:
            scored = engine.score(query, docs, max_total_tokens=cfg.max_total_tokens,
                                  deadline=started + cfg.request_timeout_s)
        relevance = to_relevance(scored.scores, scale)
        order = sorted(range(len(docs)), key=lambda i: (-relevance[i], i))[:top_n]
        results = []
        for i in order:
            row: dict = {"index": i, "relevance_score": relevance[i]}
            if want_docs:
                row["document"] = {"text": docs[i]}
            results.append(row)
        warnings = [
            f"documents[{i}] exceeds {engine.max_length} tokens and was truncated; "
            "its score reflects only the first part of the text"
            for i in scored.truncated
        ]
        meta = {"api_version": {"version": "2"}, "score_scale": scale, "truncated_indices": scored.truncated}
        if warnings:
            meta["warnings"] = warnings
        log.info("rerank n=%d top_n=%d truncated=%d ms=%d", len(docs), top_n, len(scored.truncated),
                 (time.monotonic() - started) * 1000)
        return {
            "id": str(uuid.uuid4()),
            "model": engine.model_id,
            "results": results,
            "usage": {"total_tokens": scored.tokens, "prompt_tokens": scored.tokens},
            "meta": meta,
            "warnings": warnings,
        }

    def handler(request: Request, payload: dict):
        if unauthorized(request):
            return _error(401, "missing or wrong bearer token")
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > body_cap:
            return _error(413, f"request body is {length} bytes, limit {body_cap} "
                               "(TIEBREAK_MAX_DOCUMENTS * TIEBREAK_MAX_CHARS_PER_DOCUMENT)", "body", body_cap)
        try:
            return rerank(request, payload)
        except BadRequest as exc:
            return _error(exc.status, str(exc), exc.field)
        except LimitError as exc:
            return _error(exc.status, str(exc), exc.field, exc.limit)
        except DeadlineExceeded:
            return _error(504, f"scoring exceeded TIEBREAK_TIMEOUT_S={cfg.request_timeout_s}", "timeout",
                          cfg.request_timeout_s)

    for path in RERANK_PATHS:
        app.add_api_route(path, handler, methods=["POST"])

    @app.exception_handler(422)
    async def _invalid(request: Request, exc):
        return _error(422, "request body must be a JSON object")

    return app
