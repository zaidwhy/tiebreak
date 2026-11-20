# tiebreak

Ollama has no rerank endpoint (ollama#3368 "Reranking models", open, 387 reactions on 2026-11-20). People already pull reranker GGUFs into Ollama and nothing serves them. tiebreak is that endpoint, using the reranker you already pulled: `tiebreak serve --ollama` sits in front of Ollama, answers `/api/rerank` and `/v1/rerank` with the GGUF from Ollama's own model store, and forwards every other request to Ollama unchanged, streamed chat included. Clients change one port, 11434 to 11435.

The opinion: a rerank endpoint that returns a number is easy; one that returns the right number is the work. tiebreak refuses a GGUF it cannot score correctly instead of answering with numbers that mean nothing, reports every document it cut to fit the model, and publishes where each model loses, with the bench script to check it on your data.

## Ollama mode

```
ollama pull qllama/bge-reranker-v2-m3          # 635 MB, a GGUF with its classifier head
tiebreak serve --ollama                          # 127.0.0.1:11435 in front of http://127.0.0.1:11434
tiebreak serve --ollama http://10.0.0.5:11434 --model qllama/bge-reranker-v2-m3
```

A live call on this machine (Windows, Ollama 0.32.15):

```
$ curl -s 127.0.0.1:11435/api/rerank -H 'content-type: application/json' -d '{"model":"qllama/bge-reranker-v2-m3","query":"what lowers blood pressure","documents":["The Eiffel Tower is in Paris.","ACE inhibitors reduce blood pressure by relaxing blood vessels.","Salt intake raises blood pressure in sensitive people."],"top_n":2}'
{"id":"...","model":"qllama/bge-reranker-v2-m3","results":[{"index":1,"relevance_score":0.906...},{"index":2,"relevance_score":0.074...}],"usage":{"total_tokens":59,"prompt_tokens":59},"meta":{"api_version":{"version":"2"},"score_scale":"sigmoid","truncated_indices":[]},"warnings":[]}
```

How a request is routed:

- `POST /api/rerank`, `/v1/rerank` (and `/v2/rerank`, `/rerank`) and `GET /health` are answered by tiebreak. Every other method and path, `/v1/models` included, goes to Ollama byte for byte; responses stream through chunk by chunk, so `stream: true` chat and generate behave as they do against Ollama directly. An unreachable Ollama is a 502 that names the URL.
- The rerank `model` field picks the backend. A tiebreak alias (`minilm`, `bge-reranker-base-int8`, ...) or no model runs the ONNX engine below. Any other name must be a model Ollama has pulled: tiebreak reads its manifest under `$OLLAMA_MODELS` or `~/.ollama/models` (`manifests/<host>/<namespace>/<model>/<tag>`), takes the `application/vnd.ollama.image.model` layer, and serves that blob. A name that is neither is a 404 naming the model and the `ollama pull` that would fix it, never a silent fallback to another model.
- One GGUF is loaded at a time; asking for a different one stops the previous llama-server.

Backend: tiebreak starts the `llama-server` that Ollama itself ships (`lib/ollama/llama-server`, built with `--reranking` and `--pooling rank`) as an argument-list subprocess and calls its `/v1/rerank`, because on Windows that binary is already installed and working wherever Ollama is, while llama-cpp-python has to be compiled with MSVC against a matching Python. `TIEBREAK_LLAMA_SERVER` or a `llama-server` on PATH takes precedence. On Windows the child is placed in a job object, so even a hard kill of tiebreak takes llama-server with it.

Before starting anything, tiebreak reads the GGUF header and refuses, by model name, a file that rank pooling would score wrongly:

- no classifier head (`cls.weight` or `cls.output.weight`): not a reranker, or converted without its head;
- a decoder reranker (for example Qwen3-Reranker) without the `tokenizer.chat_template.rerank` prompt that current llama.cpp converters write: llama.cpp would feed it a bare query and document instead of the yes/no judging prompt it was trained on.

`dengcao/Qwen3-Reranker-0.6B:Q8_0`, the Qwen3 reranker most often suggested in ollama#3368, fails the first check: its header (read with a 24 MB range request, not a full pull) has 310 tensors, none of them `cls.*`, and no rerank prompt. It is refused with that reason. A Qwen3-Reranker GGUF converted by a current `convert_hf_to_gguf.py` passes both checks; its server returns P(yes), which tiebreak turns back into the yes-minus-no logit so `score_scale` means the same thing for every backend. That path is covered by offline tests only; no such GGUF was run here.

Long documents are cut with the model's own tokenizer (llama-server's `/tokenize` and `/detokenize`) to fit `TIEBREAK_MAX_LENGTH` and listed in `meta.truncated_indices`, as with ONNX.

### Parity: the same model through both backends

bge-reranker-v2-m3 on the identical cached BM25 top-100 pool, SciFact, 30 queries, nDCG@10, `TIEBREAK_MAX_LENGTH=512`, same laptop CPU. The GGUF row ran on 2026-11-20 with `python bench/ndcg.py --model qllama/bge-reranker-v2-m3 --queries 30`; the ONNX row is the earlier one from the table below.

| Backend | File | BM25 | Reranked | Change | ms per pair | Pairs cut to 512 |
|---|---|---|---|---|---|---|
| ONNX Runtime | `onnx-community/bge-reranker-v2-m3-ONNX` int8 | 0.7409 | 0.8264 | +0.085 | 492.2 | 659 of 3000 |
| llama-server (Ollama's build), rank pooling | `qllama/bge-reranker-v2-m3` Q8_0 GGUF | 0.7409 | 0.8235 | +0.083 | 900.2 | 711 of 3000 |

The GGUF path keeps the ranking gain within 0.003 of the ONNX path on this set and costs about 1.8 times as much per pair here. It cuts more documents because it reserves a few tokens of margin around the detokenize round trip, so part of the 0.003 may be that rather than the quantization. 30 queries of one dataset; the 100-query and other-dataset runs have not been done for the GGUF path.

## Measured (ONNX backend)

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
| request body | `MAX_DOCUMENTS * MAX_CHARS_PER_DOCUMENT` bytes | 413, `body` |
| `TIEBREAK_PROXY_TIMEOUT_S` | 300 | longest silence between two chunks from Ollama; 504 if no answer starts in time |
| `TIEBREAK_LLAMA_START_TIMEOUT_S` | 120 | a GGUF that does not load in time is a 502 naming the variable |

The server binds 127.0.0.1. It refuses to start on any other address unless `TIEBREAK_API_KEY` is set, and then every request needs `Authorization: Bearer <key>`. In `--ollama` mode that includes proxied requests, and the key is stripped before forwarding, so binding a public address never exposes Ollama without it. Scores are sigmoid of the logit by default (0 to 1, like Cohere); `TIEBREAK_SCORES=raw` returns logits.

## Tests

```
pytest -q            # 54 passed: request shapes, caps, auth, refusal text, proxy pass-through against a
                     # fake Ollama (a streamed chat included), manifest resolution, GGUF header checks
pytest -q -m model   # ONNX: real scoring, truncation, Cohere SDK round trip, CLI (4, last run 2026-11-10);
                     # GGUF: 2 tests, need `ollama pull qllama/bge-reranker-v2-m3`
```

## Not done

No GPU provider for ONNX, no batching across concurrent requests, one GGUF loaded at a time. The GGUF path is verified on Windows with Ollama 0.32.15 only; the llama-server lookup for Linux and macOS installs is untested, and a hard kill of tiebreak there can leave llama-server running. Quality numbers cover three small BEIR sets at 100 queries (bge-reranker-v2-m3 at 30), not the full test sets.

MIT licence.
