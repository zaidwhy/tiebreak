"""Write minimal GGUF headers (no tensor data) for offline tests."""
import struct
from pathlib import Path


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, arch: str, tensors: list[str], extra: dict | None = None) -> Path:
    kv = [(b"general.architecture", 8, _s(arch)),
          (f"{arch}.context_length".encode(), 4, struct.pack("<I", 8192)),
          # a string array and a float array, like a real vocab, to exercise the skipping code
          (b"tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _s("yes") + _s("no")),
          (b"tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 2) + struct.pack("<ff", 0.0, -1.0))]
    for key, value in (extra or {}).items():
        kv.append((key.encode(), 8, _s(value)))
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, kind, payload in kv:
        out += struct.pack("<Q", len(key)) + key + struct.pack("<I", kind) + payload
    for name in tensors:
        out += _s(name) + struct.pack("<I", 2) + struct.pack("<QQ", 4, 4) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    path.write_bytes(out)
    return path
