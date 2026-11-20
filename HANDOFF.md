# tiebreak handoff (2026-11-20)

State: branch `ollama-mode` (local only, not pushed, not published). main is unchanged at a168dbd and matches origin; PyPI is still tiebreak-rerank 0.1.1.

## What ollama-mode adds
- `tiebreak serve --ollama [URL]`: owns POST /api/rerank, /v1/rerank (also /v2/rerank, /rerank) and GET /health; every other method and path is forwarded to Ollama byte for byte with streaming (`src/tiebreak/proxy.py`).
- GGUF backend (`ollama.py` manifest -> blob, `gguf.py` header checks, `llama_server.py`): spawns the llama-server Ollama ships in `lib/ollama/` with `--reranking --pooling rank`, calls its /v1/rerank. Windows job object kills the child if tiebreak is hard-killed (checked by hand: Stop-Process on the parent, child gone).
- `router.py`: request `model` picks ONNX (tiebreak alias or no model) or GGUF (pulled in Ollama); anything else 404 naming the model.
- Refusals: no `cls.*` head, or a decoder arch without `tokenizer.chat_template.rerank`. `dengcao/Qwen3-Reranker-0.6B:Q8_0` has neither (header read via a 24 MB range request) and is refused.
- `bench/ndcg.py --model <ollama name>` runs the GGUF path.
- New dependency: httpx (runtime).

## Verified
- `pytest -q`: 54 passed, 6 deselected (Python 3.14 system and a fresh Python 3.12 `.venv`).
- `pytest -q -m model tests/test_gguf_model.py`: 2 passed (needs `ollama pull qllama/bge-reranker-v2-m3`).
- Live: `tiebreak serve --ollama --port 11436 --model qllama/bge-reranker-v2-m3` against Ollama 0.32.15: `/`, `/api/version`, `/api/tags` proxied; `/api/generate` with qwen2.5:7b streamed through (first byte 9.08 s, last 9.31 s, token lines 70 to 90 ms apart); `/api/rerank` scored. The first generate attempt was a 500 from Ollama itself (its CUDA llama-server crashed while loading, see Ollama server.log 19:01:26), passed through unchanged.

## Measured (real numbers, row appended to bench/results/ndcg.jsonl)
- `.venv/Scripts/python bench/ndcg.py --model qllama/bge-reranker-v2-m3 --queries 30 --dataset scifact` (45 min wall): BM25 0.7409, GGUF reranked 0.8235, 900.17 ms per pair, 711 of 3000 pairs truncated. ONNX int8 baseline row (2026-11-10): 0.8264, 492.16 ms per pair, 659 truncated.
- Only download this run: `ollama pull qllama/bge-reranker-v2-m3` (635 MB) plus a 24 MB range read of the dengcao Qwen3 header. Existing Ollama models untouched.
- `.venv` (Python 3.12, gitignored) now exists with tiebreak editable, ir_datasets, rank_bm25, pytest.

## Not done / open
- Not pushed, not on PyPI, no version bump (Zaid decides).
- The ONNX `pytest -m model` set (4 tests) was not rerun: the ONNX weights are no longer in the HF cache and re-downloading them was outside this run's budget.
- Qwen3-Reranker through a correctly converted GGUF: only the offline P(yes) to logit test; no real file run.
- llama-server discovery on Linux/macOS Ollama installs is coded (PATH, `<ollama>/../lib/ollama`) but untested; the job-object cleanup is Windows only.
- Proxy does not handle WebSocket upgrades (Ollama does not use them).

## Before this branch (from the 2026-11-10 handoff)
- Public at zaidwhy/tiebreak; tags v0.1.0 and v0.1.1 (no GitHub release objects); tiebreak-rerank 0.1.0 and 0.1.1 on PyPI.
- CI (`.github/workflows/tests.yml`) runs `pytest -q` on Ubuntu and Windows, Python 3.10 and 3.12.
- Default stays `minilm`: larger mean nDCG gain over SciFact, NFCorpus and FiQA than bge-base int8 and 3 to 4 times faster; README states where it loses (SciFact).
