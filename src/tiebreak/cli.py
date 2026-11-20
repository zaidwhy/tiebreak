from __future__ import annotations

import argparse
import json
import logging
import sys

from tiebreak.settings import Settings

OLLAMA_DEFAULT = "http://127.0.0.1:11434"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tiebreak", description="Rerank server to run beside Ollama.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the HTTP server")
    s.add_argument("--model", help="alias, Hugging Face repo id, or local directory (default: $TIEBREAK_MODEL or minilm)")
    s.add_argument("--onnx-file", help="path of the .onnx file inside the repo")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--ollama", nargs="?", const=OLLAMA_DEFAULT, metavar="URL",
                   help=f"proxy every other request to Ollama (default {OLLAMA_DEFAULT}) and rerank with "
                        "GGUF rerankers Ollama has pulled")

    r = sub.add_parser("rerank", help="rerank documents in-process and print the order")
    r.add_argument("query")
    r.add_argument("documents", nargs="+")
    r.add_argument("--model")
    r.add_argument("--onnx-file")
    r.add_argument("--top-n", type=int)
    r.add_argument("--json", action="store_true", help="print the full response body")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per proxied or llama-server call is noise
    try:
        cfg = Settings.from_env().with_overrides(model=args.model, onnx_file=args.onnx_file,
                                                 host=getattr(args, "host", None), port=getattr(args, "port", None))
        if args.cmd == "serve":
            cfg.check_exposure()
            if args.ollama:
                if not args.ollama.startswith(("http://", "https://")):
                    raise ValueError(f"--ollama must be an http:// or https:// URL, got {args.ollama!r}")
                cfg = cfg.with_overrides(ollama_url=args.ollama.rstrip("/"))
    except ValueError as exc:
        print(f"tiebreak: {exc}", file=sys.stderr)
        return 2

    from tiebreak.engine import Reranker

    if args.cmd == "serve" and cfg.ollama_url:
        return _serve_ollama(cfg)

    engine = Reranker(cfg.model, cfg.onnx_file, cfg.max_length, cfg.batch_size)
    if args.cmd == "serve":
        import uvicorn

        from tiebreak.server import create_app

        logging.getLogger("tiebreak").info("serving %s (%s) on http://%s:%d, auth %s", engine.model_id,
                                           engine.onnx_file, cfg.host, cfg.port, "bearer" if cfg.api_key else "none")
        uvicorn.run(create_app(cfg, engine), host=cfg.host, port=cfg.port, log_level="warning")
        return 0

    from tiebreak.engine import to_relevance

    scored = engine.score(args.query, args.documents, cfg.max_total_tokens)
    rel = to_relevance(scored.scores, cfg.scores)
    order = sorted(range(len(rel)), key=lambda i: (-rel[i], i))[: args.top_n or len(rel)]
    if args.json:
        print(json.dumps({"model": engine.model_id, "truncated_indices": scored.truncated,
                          "results": [{"index": i, "relevance_score": rel[i]} for i in order]}, indent=2))
    else:
        for i in order:
            print(f"{rel[i]:8.4f}  [{i}] {args.documents[i][:90]}")
        for i in scored.truncated:
            print(f"warning: documents[{i}] exceeds {engine.max_length} tokens and was truncated", file=sys.stderr)
    return 0


def _serve_ollama(cfg: Settings) -> int:
    import uvicorn

    from tiebreak.engine import Reranker
    from tiebreak.gguf import GgufError, NotAReranker
    from tiebreak.llama_server import LlamaServerError, LlamaServerReranker
    from tiebreak.router import EngineRouter, UnknownModel
    from tiebreak.server import create_app

    router = EngineRouter(
        cfg.model,
        onnx_factory=lambda name: Reranker(name, cfg.onnx_file if name == cfg.model else None, cfg.max_length,
                                           cfg.batch_size),
        gguf_factory=lambda name, blob: LlamaServerReranker(name, blob, cfg.max_length, cfg.llama_start_timeout_s),
    )
    try:
        default = router.get(cfg.model)  # fail at startup, not on the first request
    except (UnknownModel, NotAReranker, GgufError, LlamaServerError) as exc:
        print(f"tiebreak: {exc}", file=sys.stderr)
        return 2
    logging.getLogger("tiebreak").info("serving rerank with %s on http://%s:%d, everything else proxied to %s, "
                                       "auth %s", default.model_id, cfg.host, cfg.port, cfg.ollama_url,
                                       "bearer" if cfg.api_key else "none")
    try:
        uvicorn.run(create_app(cfg, router=router), host=cfg.host, port=cfg.port, log_level="warning")
    finally:
        router.close_gguf()
    return 0
