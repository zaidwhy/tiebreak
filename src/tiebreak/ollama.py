"""Find the GGUF file behind a model name Ollama has already pulled.

Ollama stores a manifest per model at
<models>/manifests/<host>/<namespace>/<model>/<tag>; its layer with media type
application/vnd.ollama.image.model names the GGUF blob at <models>/blobs/sha256-<hex>.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_HOST = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
_DIGEST = re.compile(r"^sha256[:-]([0-9a-f]{64})$")
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class OllamaModelError(LookupError):
    pass


def models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env).expanduser() if env else Path.home() / ".ollama" / "models"


def split_name(name: str) -> tuple[str, str, str, str]:
    """'qllama/bge-reranker-v2-m3' -> ('registry.ollama.ai', 'qllama', 'bge-reranker-v2-m3', 'latest')."""
    path, tag = name, DEFAULT_TAG
    if ":" in name.rsplit("/", 1)[-1]:
        path, tag = name.rsplit(":", 1)
    parts = path.split("/")
    if len(parts) == 1:
        parts = [DEFAULT_HOST, DEFAULT_NAMESPACE, parts[0]]
    elif len(parts) == 2:
        parts = [DEFAULT_HOST, *parts]
    if len(parts) != 3 or not all(_PART.match(p) for p in (*parts, tag)):
        raise OllamaModelError(f"{name!r} is not an Ollama model name ([host/][namespace/]model[:tag])")
    host, namespace, model = parts
    return host, namespace, model, tag


def _find_casefold(root: Path, parts: tuple[str, ...]) -> Path | None:
    """Ollama keeps the case a name was pulled with; match it case-insensitively like the CLI does."""
    cur = root
    for part in parts:
        if (cur / part).exists():
            cur = cur / part
            continue
        if not cur.is_dir():
            return None
        match = next((c for c in cur.iterdir() if c.name.lower() == part.lower()), None)
        if match is None:
            return None
        cur = match
    return cur


def manifest_path(name: str, root: Path | None = None) -> Path:
    root = root or models_dir()
    found = _find_casefold(root / "manifests", split_name(name))
    if found is None or not found.is_file():
        raise OllamaModelError(f"model {name!r} is not pulled in Ollama (no manifest under {root / 'manifests'}); "
                               f"run: ollama pull {name}")
    return found


def resolve_blob(name: str, root: Path | None = None) -> Path:
    """Return the path of the GGUF blob for an Ollama model name."""
    root = root or models_dir()
    manifest = manifest_path(name, root)
    try:
        layers = json.loads(manifest.read_text(encoding="utf-8")).get("layers", [])
    except (OSError, ValueError) as exc:
        raise OllamaModelError(f"manifest for {name!r} at {manifest} is unreadable: {exc}") from None
    model_layers = [layer for layer in layers if isinstance(layer, dict) and layer.get("mediaType") == MODEL_MEDIA_TYPE]
    if len(model_layers) != 1:
        raise OllamaModelError(f"manifest for {name!r} has {len(model_layers)} model layers, expected exactly 1")
    match = _DIGEST.match(str(model_layers[0].get("digest", "")))
    if not match:
        raise OllamaModelError(f"manifest for {name!r} has a model layer without a sha256 digest")
    blob = root / "blobs" / f"sha256-{match.group(1)}"
    if not blob.is_file():
        raise OllamaModelError(f"manifest for {name!r} points at {blob}, which does not exist")
    return blob


def is_pulled(name: str, root: Path | None = None) -> bool:
    try:
        manifest_path(name, root)
    except OllamaModelError:
        return False
    return True
