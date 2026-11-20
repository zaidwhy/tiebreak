"""Real GGUF through the llama-server Ollama bundles. Run with: pytest -m model

Needs `ollama pull qllama/bge-reranker-v2-m3` (635 MB) first; the fixture fails with that
instruction otherwise, the way the ONNX model tests need network for their first download.
"""
import pytest

from tiebreak import ollama
from tiebreak.llama_server import LlamaServerReranker

pytestmark = pytest.mark.model

NAME = "qllama/bge-reranker-v2-m3"
QUERY = "what lowers blood pressure"
DOCS = ["The Eiffel Tower is in Paris.",
        "ACE inhibitors reduce blood pressure by relaxing blood vessels.",
        "Salt intake raises blood pressure in sensitive people."]


@pytest.fixture(scope="module")
def backend():
    assert ollama.is_pulled(NAME), f"run `ollama pull {NAME}` before pytest -m model"
    b = LlamaServerReranker(NAME, ollama.resolve_blob(NAME), max_length=512)
    yield b
    b.close()


def test_gguf_ranks_the_answer_first_and_the_off_topic_passage_last(backend):
    s = backend.score(QUERY, DOCS).scores
    assert max(range(3), key=lambda i: s[i]) == 1
    assert min(range(3), key=lambda i: s[i]) == 0


def test_gguf_long_document_is_cut_and_reported(backend):
    out = backend.score(QUERY, [DOCS[1], "filler words " * 1000])
    assert out.truncated == [1]
    assert out.scores[0] > out.scores[1]
