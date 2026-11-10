# tiebreak

Ollama serves chat and embeddings but has no rerank endpoint (ollama#3368, open, 290 thumbs-up on 2026-11-10). tiebreak is the missing piece for a local RAG stack: a CPU-only cross-encoder server, ONNX Runtime with no torch, that answers the Cohere and Jina `/v1/rerank` request shapes so existing clients point at it with a base URL change.

The opinion: no reranker is "better" in general, and a shim that ships one without showing where it loses is selling a guess. tiebreak ships the fast model as the default, publishes the dataset where that default makes ranking worse, and ships the bench script so the choice is made on your data. Every response also says which documents were cut to fit the model.

## Measured

Three BEIR test sets, BM25 top-100 candidates, the identical cached pool for every model, nDCG@10. One laptop CPU (Windows, onnxruntime CPU provider). Raw rows: `bench/results/ndcg.jsonl`; rerun with `python bench/ndcg.py --model <alias> --queries 100 --dataset scifact|nfcorpus|fiqa`.

| Model (alias) | Dataset | Queries | BM25 | Reranked | Change | ms per pair |
|---|---|---|---|---|---|---|
| `minilm` (default, ms-marco-MiniLM-L-6-v2 fp32) | SciFact | 100 | 0.7303 | 0.7218 | -0.009 | 55.4 |
| `minilm-int8` | SciFact | 100 | 0.7303 | 0.7178 | -0.013 | 40.6 |
| `bge-reranker-base-int8` | SciFact | 100 | 0.7303 | 0.7501 | +0.020 | 174.8 |
| `minilm` (default) | NFCorpus | 100 | 0.3395 | 0.3871 | +0.048 | 45.2 |
| `bge-reranker-base-int8` | NFCorpus | 100 | 0.3395 | 0.3431 | +0.004 | 171.9 |
| `bge-reranker-base` fp32 (Xenova `onnx/model.onnx`) | NFCorpus | 100 | 0.3395 | 0.3430 | +0.004 | 210.0 |
| `minilm` (default) | FiQA | 100 | 0.2074 | 0.3164 | +0.109 | 33.4 |
| `bge-reranker-base-int8` | FiQA | 100 | 0.2074 | 0.3053 | +0.098 | 110.0 |
| `minilm` (default) | SciFact | 30 | 0.7409 | 0.7211 | -0.020 | 46.6 |
| `bge-reranker-base-int8` | SciFact | 30 | 0.7409 | 0.7776 | +0.037 | 177.3 |
| `bge-reranker-v2-m3` (int8) | SciFact | 30 | 0.7409 | 0.8264 | +0.085 | 492.2 |

What the table says:

- MiniLM lowered nDCG@10 on SciFact (scientific claims, where BM25 is already strong; 13% of its pairs were cut at 512 tokens) and raised it most of any model run on NFCorpus (medical questions) and FiQA (finance questions). It is the default because it is 3 to 4 times faster than bge-base and its average gain over the three sets is larger (+0.049 against +0.041).
- bge-reranker-base never went below BM25 here, but it gained almost nothing on NFCorpus, and that is the model, not quantization: the fp32 file scores 0.3430 against int8's 0.3431.
- bge-reranker-v2-m3 had the largest gain measured, on 30 queries of one dataset only, at about 10 times MiniLM's cost.
- 30-query rows move a lot: bge-base went from +0.037 on 30 SciFact queries to +0.020 on 100.

Speed in practice: reranking 20 candidates costs about 1 s with `minilm`, about 3.5 s with `bge-reranker-base-int8`, about 10 s with `bge-reranker-v2-m3` on this machine. Before trusting any of them, run `bench/ndcg.py` or your own labelled queries and keep the model only if it beats your first-stage retriever.

`bench/runtime.py` checks the ONNX path against the torch reference for MiniLM: fp32 logits match exactly (max diff 0.0); the int8 file drifts up to 0.74 logits and keeps only 1 in 10 top-10 positions identical, which is why int8 MiniLM is opt-in.

## Run

```
pip install tiebreak-rerank        # the PyPI name; the command and import are `tiebreak`
tiebreak serve                      # 127.0.0.1:11435, downloads the default model once
tiebreak serve --model bge-reranker-v2-m3
tiebreak rerank "what lowers blood pressure" "The Eiffel Tower is in Paris." "ACE inhibitors relax blood vessels."
```

A live call against `tiebreak serve --model bge-reranker-base-int8`:

```
$ curl -s 127.0.0.1:11435/v1/rerank -H 'content-type: application/json' -d '{"query":"what lowers blood pressure","documents":["The Eiffel Tower is in Paris.","ACE inhibitors reduce blood pressure by relaxing blood vessels.","Salt intake raises blood pressure in sensitive people."],"top_n":2,"return_documents":true}'
{"id":"...","model":"Xenova/bge-reranker-base","results":[{"index":1,"relevance_score":0.983...,"document":{"text":"ACE inhibitors reduce blood pressure by relaxing blood vessels."}},{"index":2,"relevance_score":0.582...,"document":{"text":"Salt intake raises blood pressure in sensitive people."}}],"usage":{"total_tokens":59,"prompt_tokens":59},"meta":{"api_version":{"version":"2"},"score_scale":"sigmoid","truncated_indices":[]},"warnings":[]}
```

The same handler is mounted on `/v1/rerank`, `/v2/rerank`, `/rerank` and `/api/rerank`. Documents may be strings or `{"text": ...}` objects. `GET /health` and `GET /v1/models` exist for probes and clients that list models. The Cohere Python SDK works against it unchanged (`cohere.ClientV2(base_url="http://127.0.0.1:11435", api_key="unused")`; covered by a model test).

Model aliases: `minilm`, `minilm-int8`, `bge-reranker-base`, `bge-reranker-base-int8`, `bge-reranker-v2-m3`, `jina-reranker-v1-tiny-en`. Anything else is read as a Hugging Face repo id or a local directory holding `tokenizer.json` and an ONNX file (`--onnx-file` picks which). Only the chosen weights file and the tokenizer are downloaded, never the whole repo.

## Limits and safety

Every cap is an environment variable and every refusal names the field and the limit:

| Variable | Default | Refusal |
|---|---|---|
| `TIEBREAK_MAX_DOCUMENTS` | 1000 | 413, `documents` |
| `TIEBREAK_MAX_CHARS_PER_DOCUMENT` | 50000 | 413 |
| `TIEBREAK_MAX_TOTAL_TOKENS` | 400000 | 413 |
| `TIEBREAK_TIMEOUT_S` | 30 | 504 |
| `TIEBREAK_MAX_LENGTH` | 512 | truncates, listed in `meta.truncated_indices` and `warnings` |

The server binds 127.0.0.1. It refuses to start on any other address unless `TIEBREAK_API_KEY` is set, and then every request needs `Authorization: Bearer <key>`. Scores are sigmoid of the logit by default (0 to 1, like Cohere); `TIEBREAK_SCORES=raw` returns logits.

## Tests

```
pytest -q            # 18 passed: request shapes, caps, auth, refusal text (no model download)
pytest -q -m model   # 4 passed: real scoring, truncation reporting, Cohere SDK round trip, CLI
```

## Not done

No GPU provider, no streaming, no batching across concurrent requests. Quality numbers cover three small BEIR sets at 100 queries (bge-reranker-v2-m3 at 30), not the full test sets.

MIT licence.
