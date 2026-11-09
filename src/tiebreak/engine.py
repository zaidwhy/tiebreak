"""ONNX cross-encoder scoring with explicit truncation accounting."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("tiebreak.engine")

# alias -> (hf repo, onnx file). Any other value is read as a repo id or a local directory.
ALIASES: dict[str, tuple[str, str]] = {
    "minilm": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx"),
    "minilm-int8": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "onnx/model_quint8_avx2.onnx"),
    "bge-reranker-base": ("BAAI/bge-reranker-base", "onnx/model.onnx"),
    "bge-reranker-base-int8": ("Xenova/bge-reranker-base", "onnx/model_quantized.onnx"),
    "bge-reranker-v2-m3": ("onnx-community/bge-reranker-v2-m3-ONNX", "onnx/model_int8.onnx"),
    "jina-reranker-v1-tiny-en": ("jinaai/jina-reranker-v1-tiny-en", "onnx/model_quantized.onnx"),
}


class LimitError(Exception):
    """A request exceeded a named cap."""

    def __init__(self, field: str, limit: int, message: str, status: int = 413):
        super().__init__(message)
        self.field, self.limit, self.status = field, limit, status


class DeadlineExceeded(Exception):
    pass


@dataclass
class Scored:
    scores: list[float]   # raw model logits, input order
    truncated: list[int]  # document indices cut to fit max_length
    tokens: int           # tokens actually fed to the model


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def resolve_model(model: str, onnx_file: str | None) -> tuple[Path, str, str]:
    """Return (directory with tokenizer.json and the onnx file, onnx relative path, display id)."""
    local = Path(model)
    if local.is_dir():
        return local, onnx_file or "onnx/model.onnx", local.name
    repo, rel = ALIASES.get(model, (model, "onnx/model.onnx"))
    rel = onnx_file or rel
    from huggingface_hub import snapshot_download

    # Fetch only the chosen weights file and tokenizer, never the whole repo.
    path = snapshot_download(repo, allow_patterns=[rel, rel + "_data", "tokenizer.json", "config.json"])
    return Path(path), rel, repo


class Reranker:
    def __init__(self, model: str, onnx_file: str | None = None, max_length: int = 512,
                 batch_size: int = 16, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root, rel, self.model_id = resolve_model(model, onnx_file)
        self.onnx_file = rel
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(root / rel), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        tok_path = str(root / "tokenizer.json")
        self.tok_full = Tokenizer.from_file(tok_path)
        self.tok_full.no_truncation()
        self.tok_full.no_padding()
        self.max_length = max_length
        self.batch_size = batch_size
        # only_second keeps the whole query and cuts the document; longest_first is the
        # fallback for a query that alone exceeds the window.
        self.tok_cut = Tokenizer.from_file(tok_path)
        self.tok_cut.no_padding()
        self.tok_cut.enable_truncation(max_length, strategy="only_second")
        self.tok_fallback = Tokenizer.from_file(tok_path)
        self.tok_fallback.no_padding()
        self.tok_fallback.enable_truncation(max_length, strategy="longest_first")
        pad = self.tok_full.token_to_id("<pad>")
        if pad is None:
            pad = self.tok_full.token_to_id("[PAD]")
        self.pad_id = pad or 0

    def count_tokens(self, query: str, documents: list[str]) -> list[int]:
        return [len(e.ids) for e in self.tok_full.encode_batch([(query, d) for d in documents])]

    def _encode_cut(self, query: str, doc: str):
        try:
            return self.tok_cut.encode(query, doc)
        except Exception:
            return self.tok_fallback.encode(query, doc)

    def score(self, query: str, documents: list[str], max_total_tokens: int | None = None,
              deadline: float | None = None) -> Scored:
        lengths = self.count_tokens(query, documents)
        if max_total_tokens is not None and sum(lengths) > max_total_tokens:
            raise LimitError("documents", max_total_tokens,
                             f"request needs {sum(lengths)} tokens (query + documents), "
                             f"limit TIEBREAK_MAX_TOTAL_TOKENS={max_total_tokens}")
        truncated = [i for i, n in enumerate(lengths) if n > self.max_length]
        if truncated:
            log.warning("truncated %d document(s) to %d tokens", len(truncated), self.max_length)
        # similar lengths per batch means less padding
        order = sorted(range(len(documents)), key=lambda i: lengths[i])
        logits = np.zeros(len(documents), dtype=np.float64)
        fed = 0
        for start in range(0, len(order), self.batch_size):
            if deadline is not None and time.monotonic() > deadline:
                raise DeadlineExceeded("request timeout reached while scoring")
            idx = order[start:start + self.batch_size]
            encs = [self._encode_cut(query, documents[i]) for i in idx]
            width = max(len(e.ids) for e in encs)
            ids = np.full((len(encs), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(encs), width), dtype=np.int64)
            types = np.zeros((len(encs), width), dtype=np.int64)
            for r, e in enumerate(encs):
                n = len(e.ids)
                ids[r, :n], mask[r, :n], types[r, :n] = e.ids, e.attention_mask, e.type_ids
                fed += n
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            out = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
            out = np.asarray(out, dtype=np.float64)
            if out.ndim == 2 and out.shape[1] == 2:
                out = out[:, 1] - out[:, 0]
            logits[idx] = out.reshape(len(idx), -1)[:, 0]
        return Scored(scores=[float(x) for x in logits], truncated=truncated, tokens=fed)


def to_relevance(raw: list[float], scale: str) -> list[float]:
    if scale == "raw":
        return [float(x) for x in raw]
    return [float(x) for x in sigmoid(np.asarray(raw))]
