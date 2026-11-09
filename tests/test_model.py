"""Real ONNX model, real client library. Run with: pytest -m model"""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from tiebreak.engine import Reranker
from tiebreak.server import create_app
from tiebreak.settings import Settings

pytestmark = pytest.mark.model

DOCS = [
    "Corona is a Mexican brand of beer produced by Grupo Modelo.",
    "Bears eat salmon and berries.",
    "COVID-19 is a contagious illness caused by the virus SARS-CoV-2.",
]
QUERY = "What is COVID-19?"


@pytest.fixture(scope="module")
def engine():
    return Reranker("minilm", max_length=64)


def test_real_model_ranks_the_disease_passage_first(engine):
    s = engine.score(QUERY, DOCS)
    assert max(range(3), key=lambda i: s.scores[i]) == 2


def test_long_document_is_flagged_and_its_tail_is_ignored(engine):
    tail_answer = "filler words about nothing " * 60 + "COVID-19 is a contagious illness caused by SARS-CoV-2."
    s = engine.score(QUERY, [tail_answer, DOCS[2]])
    assert s.truncated == [0]
    assert s.scores[1] > s.scores[0]


def test_cohere_sdk_call_against_a_live_server(engine):
    import cohere
    import threading
    import socket
    import time
    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(Settings(), engine), host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        thread.join(0.05)
    try:
        co = cohere.ClientV2(api_key="unused", base_url=f"http://127.0.0.1:{port}")
        out = co.rerank(model="anything", query=QUERY, documents=DOCS, top_n=2)
        assert [r.index for r in out.results][0] == 2
        assert len(out.results) == 2
        assert out.results[0].relevance_score >= out.results[1].relevance_score
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_cli_rerank_prints_best_document_first():
    out = subprocess.run([sys.executable, "-m", "tiebreak", "rerank", QUERY, *DOCS, "--top-n", "1"],
                         capture_output=True, text=True, check=True).stdout
    assert "[2]" in out.splitlines()[0]
