import socket
import threading
import time
from dataclasses import dataclass

import pytest
import uvicorn
from fastapi.testclient import TestClient

from tiebreak.engine import DeadlineExceeded, LimitError, Scored
from tiebreak.server import create_app
from tiebreak.settings import Settings


class WordOverlapEngine:
    """Deterministic stand-in: score is the share of query words found in the document.
    A document longer than max_length words is cut, like the real tokenizer would."""

    model_id = "word-overlap"
    max_length = 20

    def score(self, query, documents, max_total_tokens=None, deadline=None):
        words = [d.split() for d in documents]
        total = sum(len(w) for w in words)
        if max_total_tokens is not None and total > max_total_tokens:
            raise LimitError("documents", max_total_tokens,
                             f"request needs {total} tokens, limit TIEBREAK_MAX_TOTAL_TOKENS={max_total_tokens}")
        q = set(query.lower().split())
        truncated = [i for i, w in enumerate(words) if len(w) > self.max_length]
        scores = [len(q & {x.lower() for x in w[: self.max_length]}) / len(q) * 10 - 5 for w in words]
        return Scored(scores=scores, truncated=truncated, tokens=total)


class SlowEngine(WordOverlapEngine):
    """Reports a missed deadline the way the real engine does between batches."""

    def score(self, query, documents, max_total_tokens=None, deadline=None):
        raise DeadlineExceeded("request timeout reached while scoring")


@pytest.fixture
def make_client():
    def build(engine=None, **overrides):
        cfg = Settings().with_overrides(**overrides)
        return TestClient(create_app(cfg, engine or WordOverlapEngine()))
    return build


@pytest.fixture
def client(make_client):
    return make_client()


@dataclass
class LiveServer:
    url: str


@pytest.fixture
def live_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(Settings(), WordOverlapEngine())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        thread.join(0.05)
    assert server.started, "uvicorn did not start"
    yield LiveServer(f"http://127.0.0.1:{port}")
    server.should_exit = True
    thread.join(timeout=5)
