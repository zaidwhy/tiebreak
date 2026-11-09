"""Every knob is an environment variable; CLI flags override a few of them."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model: str = "minilm"
    onnx_file: str | None = None
    host: str = "127.0.0.1"
    port: int = 11435
    api_key: str | None = None
    max_documents: int = 1000
    max_chars_per_document: int = 50_000
    max_total_tokens: int = 400_000
    max_length: int = 512
    batch_size: int = 16
    request_timeout_s: int = 30
    scores: str = "sigmoid"  # "sigmoid" (0..1, Cohere-like) or "raw" (model logits)

    @classmethod
    def from_env(cls) -> "Settings":
        scores = os.environ.get("TIEBREAK_SCORES", "sigmoid")
        if scores not in ("sigmoid", "raw"):
            raise ValueError(f"TIEBREAK_SCORES must be 'sigmoid' or 'raw', got {scores!r}")
        return cls(
            model=os.environ.get("TIEBREAK_MODEL", cls.model),
            onnx_file=os.environ.get("TIEBREAK_ONNX_FILE") or None,
            host=os.environ.get("TIEBREAK_HOST", cls.host),
            port=_int("TIEBREAK_PORT", cls.port),
            api_key=os.environ.get("TIEBREAK_API_KEY") or None,
            max_documents=_int("TIEBREAK_MAX_DOCUMENTS", cls.max_documents),
            max_chars_per_document=_int("TIEBREAK_MAX_CHARS_PER_DOCUMENT", cls.max_chars_per_document),
            max_total_tokens=_int("TIEBREAK_MAX_TOTAL_TOKENS", cls.max_total_tokens),
            max_length=_int("TIEBREAK_MAX_LENGTH", cls.max_length),
            batch_size=_int("TIEBREAK_BATCH_SIZE", cls.batch_size),
            request_timeout_s=_int("TIEBREAK_TIMEOUT_S", cls.request_timeout_s),
            scores=scores,
        )

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})

    def check_exposure(self) -> None:
        if self.host not in LOOPBACK and not self.api_key:
            raise ValueError(
                f"refusing to bind {self.host} without TIEBREAK_API_KEY: "
                "without a key the server is only safe on loopback"
            )
