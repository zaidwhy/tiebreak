from __future__ import annotations

import argparse
import json
import logging
import sys

from tiebreak.settings import Settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tiebreak", description="Rerank server to run beside Ollama.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the HTTP server")
    s.add_argument("--model", help="alias, Hugging Face repo id, or local directory (default: $TIEBREAK_MODEL or bge-reranker-base-int8)")
    s.add_argument("--onnx-file", help="path of the .onnx file inside the repo")
    s.add_argument("--host")
    s.add_argument("--port", type=int)

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
    try:
        cfg = Settings.from_env().with_overrides(model=args.model, onnx_file=args.onnx_file,
                                                 host=getattr(args, "host", None), port=getattr(args, "port", None))
        if args.cmd == "serve":
            cfg.check_exposure()
    except ValueError as exc:
        print(f"tiebreak: {exc}", file=sys.stderr)
        return 2

    from tiebreak.engine import Reranker

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
