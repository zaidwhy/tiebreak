"""Read the GGUF header (metadata and tensor names) to decide whether a file can rerank.

Only the header is parsed; tensor data is never read. Rank pooling in llama.cpp needs a
classifier head (`cls.weight` or `cls.output.weight`), and a decoder model such as
Qwen3-Reranker also needs the `tokenizer.chat_template.rerank` prompt that llama.cpp's
converter writes. A file missing either would load and return scores that mean nothing,
so it is refused before any server is started.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"
# Sanity caps so a corrupt or hostile file cannot make the reader allocate gigabytes.
MAX_KV = 100_000
MAX_TENSORS = 1_000_000
MAX_STRING = 64 * 1024 * 1024

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

# Architectures that read [CLS] query [SEP] document [SEP] and need no prompt template.
ENCODER_ARCHS = {"bert", "nomic-bert", "jina-bert-v2", "jina-bert-v3", "roberta", "xlm-roberta", "modern-bert",
                 "neo-bert", "t5encoder"}
# Decoder rerankers whose rank pooling ends in a softmax over ("yes", "no"): the score is P(yes).
SOFTMAX_ARCHS = {"qwen3", "qwen3vl"}


class GgufError(ValueError):
    pass


class NotAReranker(ValueError):
    """The file loads, but rank pooling over it would not produce a relevance score."""


@dataclass
class GgufInfo:
    version: int
    arch: str
    metadata: dict = field(repr=False)
    tensors: list[str] = field(repr=False)

    @property
    def has_classifier(self) -> bool:
        names = set(self.tensors)
        return "cls.weight" in names or "cls.output.weight" in names

    @property
    def rerank_template(self) -> str | None:
        value = self.metadata.get("tokenizer.chat_template.rerank")
        return value if isinstance(value, str) else None

    @property
    def context_length(self) -> int | None:
        value = self.metadata.get(f"{self.arch}.context_length")
        return int(value) if isinstance(value, int) else None


def _read(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise GgufError("file ends inside the GGUF header")
    return data


def _u32(fh: BinaryIO) -> int:
    return struct.unpack("<I", _read(fh, 4))[0]


def _u64(fh: BinaryIO) -> int:
    return struct.unpack("<Q", _read(fh, 8))[0]


def _string(fh: BinaryIO) -> str:
    n = _u64(fh)
    if n > MAX_STRING:
        raise GgufError(f"string of {n} bytes in the header exceeds {MAX_STRING}")
    return _read(fh, n).decode("utf-8", errors="replace")


def _value(fh: BinaryIO, kind: int):
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, _read(fh, struct.calcsize(fmt)))[0]
    if kind == _STRING:
        return _string(fh)
    if kind == _ARRAY:
        item, n = _u32(fh), _u64(fh)
        if item in _SCALARS:
            size = struct.calcsize(_SCALARS[item])
            if n * size > MAX_STRING * 16:
                raise GgufError(f"array of {n} items in the header is implausibly large")
            fh.seek(n * size, 1)  # arrays (vocab scores, token types) are never needed here
            return None
        for _ in range(n):
            _value(fh, item)
        return None
    raise GgufError(f"unknown GGUF value type {kind}")


def read_header(path: str | Path) -> GgufInfo:
    with open(path, "rb") as fh:
        if fh.read(4) != MAGIC:
            raise GgufError(f"{path} is not a GGUF file (bad magic)")
        version = _u32(fh)
        if version not in (2, 3):
            raise GgufError(f"{path} is GGUF version {version}; versions 2 and 3 are supported")
        n_tensors, n_kv = _u64(fh), _u64(fh)
        if n_kv > MAX_KV or n_tensors > MAX_TENSORS:
            raise GgufError(f"{path} declares {n_kv} metadata keys and {n_tensors} tensors; refusing to parse")
        meta: dict = {}
        for _ in range(n_kv):
            key = _string(fh)
            meta[key] = _value(fh, _u32(fh))
        tensors: list[str] = []
        for _ in range(n_tensors):
            tensors.append(_string(fh))
            n_dims = _u32(fh)
            if n_dims > 8:
                raise GgufError(f"tensor {tensors[-1]!r} has {n_dims} dimensions")
            fh.seek(8 * n_dims + 4 + 8, 1)  # dims, ggml type, data offset
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        raise GgufError(f"{path} has no general.architecture key")
    return GgufInfo(version=version, arch=arch, metadata=meta, tensors=tensors)


def check_reranker(info: GgufInfo, name: str) -> None:
    """Raise NotAReranker, naming the model, unless rank pooling gives a real relevance score."""
    if not info.has_classifier:
        raise NotAReranker(
            f"model {name!r} ({info.arch}) has no classifier head (cls.weight or cls.output.weight), so llama.cpp "
            "rank pooling cannot score it; it is either not a reranker or was converted to GGUF without its head "
            "(early Qwen3-Reranker GGUFs were converted as plain language models; a current llama.cpp "
            "convert_hf_to_gguf.py writes the head)")
    if info.arch not in ENCODER_ARCHS and info.rerank_template is None:
        raise NotAReranker(
            f"model {name!r} ({info.arch}) is a decoder reranker without the tokenizer.chat_template.rerank prompt "
            "in its GGUF; without that prompt llama.cpp feeds it query and document in the wrong format and the "
            "scores are not relevance scores. Re-convert it with a current llama.cpp convert_hf_to_gguf.py")
