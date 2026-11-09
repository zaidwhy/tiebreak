# tiebreak handoff (2026-11-10)

State: 2 commits on main, local only. 18 offline + 4 model tests pass. Live curl verified on /v1/rerank and /api/rerank.

Next: Zaid creates the private remote (the auto-mode classifier blocks Claude from `gh repo create`):
`cd "C:/Users/Asus/projects/tiebreak" && gh repo create zaidwhy/tiebreak --private --source=. --remote=origin --push`

Done 2026-11-10: bge-reranker-base-int8 on 100 queries scored 0.7501 against BM25 0.7303 (+0.020). Open work: a second dataset (NFCorpus or FiQA) would show whether MiniLM's loss is SciFact-specific.

## Overnight decisions (review)
- Default model changed from `minilm` to `bge-reranker-base-int8`. Reason: MiniLM lowered nDCG@10 on SciFact (0.7409 to 0.7211 on 30 queries, 0.7303 to 0.7218 on 100); bge-base int8 raised it to 0.7776 at about 3.8x the cost per pair. bge-reranker-v2-m3 (0.8264) is 10x slower than MiniLM, too slow for a CPU default. Revert with `TIEBREAK_MODEL=minilm` or by changing `settings.py`.
- sameness.py flags 67% entry overlap with bhashantar; the shared entries are `src`, `tests`, `bench`, `.pytest_cache` (standard Python layout), headings 0%. Left as is.
