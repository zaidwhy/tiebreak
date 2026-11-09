"""PyTorch CPU vs ONNX Runtime (fp32 and int8) on the same SciFact pairs.

Needs torch + transformers, which tiebreak itself does not depend on:
    python bench/runtime.py --hf cross-encoder/ms-marco-MiniLM-L-6-v2 --onnx minilm --onnx minilm-int8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ndcg import load_pool
from tiebreak.engine import Reranker

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--onnx", action="append", required=True)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    docs, _, queries, pool = load_pool(100)
    qid, qtext = queries[0]
    cands = [docs[d] for d in pool[qid]]
    texts = [(qtext, cands[i % len(cands)]) for i in range(args.pairs)]
    rows = []

    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf).eval()
    torch.set_num_threads(torch.get_num_threads())

    def run_torch() -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), args.batch):
                b = texts[i:i + args.batch]
                enc = tok([q for q, _ in b], [d for _, d in b], truncation="only_second", max_length=512,
                          padding=True, return_tensors="pt")
                out.append(model(**enc).logits[:, 0].numpy())
        return np.concatenate(out)

    run_torch()  # warmup
    best = 1e9
    for _ in range(3):
        started = time.perf_counter()
        ref = run_torch()
        best = min(best, time.perf_counter() - started)
    rows.append({"runtime": "torch-fp32", "ms_per_pair": round(1000 * best / len(texts), 2)})

    for name in args.onnx:
        eng = Reranker(name, max_length=512, batch_size=args.batch)
        queries_only = qtext
        docs_only = [d for _, d in texts]
        eng.score(queries_only, docs_only[:args.batch])  # warmup
        elapsed = 1e9
        for _ in range(3):
            started = time.perf_counter()
            got = np.array(eng.score(queries_only, docs_only).scores)
            elapsed = min(elapsed, time.perf_counter() - started)
        agree = float((np.argsort(-ref)[:10] == np.argsort(-got)[:10]).mean())
        rows.append({"runtime": f"ort-{name}", "ms_per_pair": round(1000 * elapsed / len(texts), 2),
                     "max_abs_logit_diff_vs_torch": round(float(np.abs(ref - got).max()), 4),
                     "top10_same_position": agree})
    for r in rows:
        print(json.dumps(r))
    with open(RESULTS / "runtime.jsonl", "a") as fh:
        fh.write(json.dumps({"hf": args.hf, "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
