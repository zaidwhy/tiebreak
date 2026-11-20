"""Pick the scoring backend for a request's `model` field in --ollama mode.

tiebreak's own aliases (and local ONNX directories) go to the ONNX engine. Any other name
must be a model Ollama has pulled; its GGUF blob is served by a llama-server subprocess.
A name that is neither is refused rather than silently scored with a different model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from tiebreak import ollama
from tiebreak.engine import ALIASES

log = logging.getLogger("tiebreak.router")


class UnknownModel(LookupError):
    pass


def is_onnx_name(name: str) -> bool:
    return name in ALIASES or Path(name).is_dir()


class EngineRouter:
    def __init__(self, default_model: str, onnx_factory: Callable[[str], object],
                 gguf_factory: Callable[[str, Path], object], models_root: Path | None = None,
                 preloaded: object | None = None):
        self.default_model = default_model
        self.onnx_factory, self.gguf_factory = onnx_factory, gguf_factory
        self.models_root = models_root
        self.onnx: dict[str, object] = {}
        if preloaded is not None:
            self.onnx[default_model] = preloaded
        self.gguf_name: str | None = None
        self.gguf = None  # one llama-server at a time: each holds a model in RAM

    def loaded(self) -> list[str]:
        return list(self.onnx) + ([self.gguf_name] if self.gguf_name else [])

    def get(self, requested: str | None):
        name = requested or self.default_model
        if is_onnx_name(name):
            if name not in self.onnx:
                log.info("loading ONNX model %s", name)
                self.onnx[name] = self.onnx_factory(name)
            return self.onnx[name]
        if name == self.gguf_name:
            return self.gguf
        try:
            blob = ollama.resolve_blob(name, self.models_root)
        except ollama.OllamaModelError as exc:
            raise UnknownModel(f"model {name!r} is not a tiebreak alias ({', '.join(ALIASES)}) and Ollama "
                               f"cannot provide it: {exc}") from None
        if self.gguf is not None:
            log.info("stopping llama-server for %s to load %s", self.gguf_name, name)
            self.close_gguf()
        self.gguf = self.gguf_factory(name, blob)
        self.gguf_name = name
        return self.gguf

    def close_gguf(self) -> None:
        if self.gguf is not None and hasattr(self.gguf, "close"):
            self.gguf.close()
        self.gguf, self.gguf_name = None, None
