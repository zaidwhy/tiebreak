# tiebreak handoff (2026-11-10)

State: main pushed to zaidwhy/tiebreak (public since 2026-11-10), tag `v0.1.0` at 666f4b9. 18 offline + 4 model tests pass. Live curl verified on /v1/rerank and /api/rerank.

Remote created by Zaid on 2026-11-10.

Done 2026-11-10: bge-reranker-base-int8 on 100 queries scored 0.7501 against BM25 0.7303 (+0.020). NFCorpus and FiQA done (MiniLM mean +0.049, bge-base int8 +0.041 over three sets); bge-base fp32 equals int8 on NFCorpus (0.3430 vs 0.3431). Open work: none planned; a GPU provider only if someone asks.

## Overnight decisions (review)
- Default stays `minilm` (briefly switched to `bge-reranker-base-int8`, then reverted the same night). Reason: on SciFact MiniLM lost to BM25 (-0.009) and bge-base won (+0.020), but on NFCorpus MiniLM gained +0.048 and bge-base +0.004. MiniLM has the larger mean gain and is 3 to 4 times faster. The README states where the default loses. Revisit with a third dataset or bge-reranker-base fp32 (int8 is a community export).
- sameness.py flags 67% entry overlap with bhashantar; the shared entries are `src`, `tests`, `bench`, `.pytest_cache` (standard Python layout), headings 0%. Left as is.

Published to PyPI 2026-11-10 as tiebreak-rerank 0.1.0 (command and import stay tiebreak).

Ready to upload: 0.1.1 (adds `[project.urls]`, nothing else). Built with `python -m build`, `twine check` passed, tag `v0.1.1` is on the bump commit. Zaid runs `uvx twine upload dist/tiebreak_rerank-0.1.1*` in his own terminal; PyPI serves 0.1.0 until then.
