"""nDCG@10 on a BEIR test set: BM25 top-100, then rerank with a tiebreak engine.

    python bench/ndcg.py --model minilm --queries 100 --dataset nfcorpus

Rows without a "dataset" field predate the flag and are SciFact.

The BM25 candidate lists are cached so every model reranks the identical pool.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import ir_datasets
from rank_bm25 import BM25Okapi

from tiebreak.engine import Reranker

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TOKEN = re.compile(r"[a-z0-9]+")


def ndcg_at_10(ranked: list[str], rels: dict[str, int]) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


DATASETS = {"scifact": "beir/scifact/test", "nfcorpus": "beir/nfcorpus/test"}


def load_pool(dataset: str, n_queries: int, depth: int = 100):
    cache = RESULTS / f"{dataset}_bm25_q{n_queries}_k{depth}.json"
    ds = ir_datasets.load(DATASETS[dataset])
    docs = {d.doc_id: f"{d.title} {d.text}".strip() for d in ds.docs_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for q in ds.qrels_iter():
        qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance
    queries = [(q.query_id, q.text) for q in ds.queries_iter() if q.query_id in qrels][:n_queries]
    if cache.exists():
        return docs, qrels, queries, json.loads(cache.read_text())
    ids = list(docs)
    bm25 = BM25Okapi([TOKEN.findall(docs[i].lower()) for i in ids])
    pool = {}
    for qid, text in queries:
        scores = bm25.get_scores(TOKEN.findall(text.lower()))
        top = sorted(range(len(ids)), key=lambda i: -scores[i])[:depth]
        pool[qid] = [ids[i] for i in top]
    RESULTS.mkdir(exist_ok=True)
    cache.write_text(json.dumps(pool))
    return docs, qrels, queries, pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minilm")
    ap.add_argument("--onnx-file")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="scifact")
    args = ap.parse_args()

    docs, qrels, queries, pool = load_pool(args.dataset, args.queries)
    bm25_scores = [ndcg_at_10(pool[qid], qrels[qid]) for qid, _ in queries]
    engine = Reranker(args.model, args.onnx_file, max_length=args.max_length)
    reranked, pairs, cut, spent = [], 0, 0, 0.0
    for qid, text in queries:
        cands = pool[qid]
        started = time.perf_counter()
        scored = engine.score(text, [docs[d] for d in cands])
        spent += time.perf_counter() - started
        pairs += len(cands)
        cut += len(scored.truncated)
        order = sorted(range(len(cands)), key=lambda i: -scored.scores[i])
        reranked.append(ndcg_at_10([cands[i] for i in order], qrels[qid]))
    row = {
        "dataset": args.dataset, "model": args.model, "onnx_file": engine.onnx_file, "queries": len(queries), "max_length": args.max_length,
        "ndcg10_bm25": round(sum(bm25_scores) / len(queries), 4),
        "ndcg10_reranked": round(sum(reranked) / len(queries), 4),
        "pairs": pairs, "ms_per_pair": round(1000 * spent / pairs, 2), "truncated_pairs": cut,
    }
    print(json.dumps(row))
    with open(RESULTS / "ndcg.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
