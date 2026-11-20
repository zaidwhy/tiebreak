"""Ollama manifest resolution, GGUF reranker checks, backend routing and llama-server scoring, all offline."""
import json
import math
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import WordOverlapEngine
from gguf_fixture import write_gguf
from tiebreak import ollama
from tiebreak.gguf import GgufError, NotAReranker, check_reranker, read_header
from tiebreak.llama_server import LlamaServerReranker, prob_to_logit
from tiebreak.router import EngineRouter
from tiebreak.server import create_app
from tiebreak.settings import Settings

HEX_A = "a" * 64
HEX_B = "b" * 64
QWEN_TEMPLATE = "<|im_start|>user\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"


def add_model(root: Path, parts: tuple[str, str, str, str], digest_hex: str, blob: bool = True) -> None:
    host, ns, model, tag = parts
    manifest = root / "manifests" / host / ns / model / tag
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": [
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "c" * 64, "size": 1},
        {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest_hex}", "size": 10},
    ]}))
    (root / "blobs").mkdir(exist_ok=True)
    if blob:
        (root / "blobs" / f"sha256-{digest_hex}").write_bytes(b"GGUF")


@pytest.fixture
def models(tmp_path):
    add_model(tmp_path, ("registry.ollama.ai", "qllama", "bge-reranker-v2-m3", "latest"), HEX_A)
    add_model(tmp_path, ("registry.ollama.ai", "dengcao", "Qwen3-Reranker-0.6B", "Q8_0"), HEX_B)
    add_model(tmp_path, ("registry.ollama.ai", "library", "gone", "latest"), "d" * 64, blob=False)
    return tmp_path


# --- manifest resolution ---------------------------------------------------------------

def test_name_defaults_to_registry_library_and_latest():
    assert ollama.split_name("qwen2.5") == ("registry.ollama.ai", "library", "qwen2.5", "latest")
    assert ollama.split_name("qllama/bge-reranker-v2-m3") == ("registry.ollama.ai", "qllama", "bge-reranker-v2-m3",
                                                             "latest")
    assert ollama.split_name("hf.co/user/repo:Q4_K_M") == ("hf.co", "user", "repo", "Q4_K_M")


def test_resolves_the_model_layer_blob(models):
    assert ollama.resolve_blob("qllama/bge-reranker-v2-m3", models) == models / "blobs" / f"sha256-{HEX_A}"
    assert ollama.resolve_blob("dengcao/Qwen3-Reranker-0.6B:Q8_0", models) == models / "blobs" / f"sha256-{HEX_B}"


def test_name_case_does_not_have_to_match_the_pull(models):
    assert ollama.resolve_blob("dengcao/qwen3-reranker-0.6b:q8_0", models).name == f"sha256-{HEX_B}"


def test_ollama_models_env_sets_the_root(models, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    assert ollama.models_dir() == models
    assert ollama.resolve_blob("qllama/bge-reranker-v2-m3").name == f"sha256-{HEX_A}"


def test_missing_model_says_how_to_pull_it(models):
    with pytest.raises(ollama.OllamaModelError, match="ollama pull qllama/nope"):
        ollama.resolve_blob("qllama/nope", models)


def test_manifest_pointing_at_a_missing_blob_is_reported(models):
    with pytest.raises(ollama.OllamaModelError, match="does not exist"):
        ollama.resolve_blob("gone", models)


@pytest.mark.parametrize("name", ["../../etc/passwd", "a/b/c/d", "qllama/..", "", "x:"])
def test_names_that_could_escape_the_models_dir_are_refused(name, models):
    with pytest.raises(ollama.OllamaModelError):
        ollama.resolve_blob(name, models)


# --- GGUF header checks --------------------------------------------------------------------

def test_encoder_reranker_with_classifier_head_is_accepted(tmp_path):
    info = read_header(write_gguf(tmp_path / "m.gguf", "bert", ["token_embd.weight", "cls.weight", "cls.output.weight"]))
    assert info.arch == "bert" and info.has_classifier and info.context_length == 8192
    check_reranker(info, "qllama/bge-reranker-v2-m3")


def test_model_without_classifier_head_is_refused_by_name(tmp_path):
    info = read_header(write_gguf(tmp_path / "m.gguf", "bert", ["token_embd.weight"]))
    with pytest.raises(NotAReranker, match="'nomic-embed-text'.*no classifier head"):
        check_reranker(info, "nomic-embed-text")


def test_qwen3_without_rerank_prompt_is_refused_by_name(tmp_path):
    info = read_header(write_gguf(tmp_path / "m.gguf", "qwen3", ["token_embd.weight", "cls.output.weight"]))
    with pytest.raises(NotAReranker, match="'dengcao/Qwen3-Reranker-0.6B:Q8_0'.*chat_template.rerank"):
        check_reranker(info, "dengcao/Qwen3-Reranker-0.6B:Q8_0")


def test_qwen3_with_rerank_prompt_and_head_is_accepted(tmp_path):
    info = read_header(write_gguf(tmp_path / "m.gguf", "qwen3", ["cls.output.weight"],
                                  {"tokenizer.chat_template.rerank": QWEN_TEMPLATE}))
    assert info.rerank_template == QWEN_TEMPLATE
    check_reranker(info, "qwen3-reranker")


def test_non_gguf_and_truncated_files_are_refused(tmp_path):
    (tmp_path / "x").write_bytes(b"NOPE")
    with pytest.raises(GgufError, match="bad magic"):
        read_header(tmp_path / "x")
    full = write_gguf(tmp_path / "m.gguf", "bert", ["cls.weight"]).read_bytes()
    (tmp_path / "cut").write_bytes(full[:40])
    with pytest.raises(GgufError, match="ends inside"):
        read_header(tmp_path / "cut")


# --- routing in --ollama mode --------------------------------------------------------------

class FakeGguf(WordOverlapEngine):
    def __init__(self, name, blob):
        self.model_id, self.blob, self.closed = name, blob, False

    def close(self):
        self.closed = True


def make_router(models):
    built = []

    def gguf(name, blob):
        built.append(FakeGguf(name, blob))
        return built[-1]

    router = EngineRouter("minilm", onnx_factory=lambda name: WordOverlapEngine(), gguf_factory=gguf,
                          models_root=models)
    return router, built


def ollama_client(models, **overrides):
    router, built = make_router(models)
    cfg = Settings(ollama_url="http://127.0.0.1:9").with_overrides(**overrides)
    return TestClient(create_app(cfg, router=router)), router, built


BODY = {"query": "capital of France", "documents": ["Paris is the capital of France", "bananas"]}


def test_tiebreak_alias_or_no_model_uses_onnx(models):
    client, router, built = ollama_client(models)
    assert client.post("/api/rerank", json=BODY).json()["model"] == "word-overlap"
    assert client.post("/api/rerank", json={**BODY, "model": "minilm"}).status_code == 200
    assert built == []


def test_pulled_ollama_name_uses_its_gguf_blob(models):
    client, router, built = ollama_client(models)
    r = client.post("/v1/rerank", json={**BODY, "model": "qllama/bge-reranker-v2-m3"})
    assert r.status_code == 200
    assert r.json()["model"] == "qllama/bge-reranker-v2-m3"
    assert built[0].blob == models / "blobs" / f"sha256-{HEX_A}"
    assert r.json()["results"][0]["index"] == 0


def test_switching_gguf_models_stops_the_previous_server(models):
    client, router, built = ollama_client(models)
    client.post("/api/rerank", json={**BODY, "model": "qllama/bge-reranker-v2-m3"})
    client.post("/api/rerank", json={**BODY, "model": "dengcao/Qwen3-Reranker-0.6B:Q8_0"})
    assert built[0].closed and not built[1].closed
    assert router.loaded() == ["dengcao/Qwen3-Reranker-0.6B:Q8_0"]


def test_unknown_model_is_refused_not_scored_with_another_model(models):
    client, _, _ = ollama_client(models)
    r = client.post("/api/rerank", json={**BODY, "model": "rerank-v3.5"})
    assert r.status_code == 404
    assert r.json()["field"] == "model"
    assert "'rerank-v3.5'" in r.json()["message"] and "ollama pull rerank-v3.5" in r.json()["message"]
    assert client.post("/api/rerank", json={**BODY, "model": 7}).json()["field"] == "model"


def test_bounds_apply_in_ollama_mode(models):
    client, _, _ = ollama_client(models, max_documents=1)
    r = client.post("/api/rerank", json={**BODY, "model": "qllama/bge-reranker-v2-m3"})
    assert r.status_code == 413 and r.json()["field"] == "documents" and r.json()["limit"] == 1
    client, _, _ = ollama_client(models, max_chars_per_document=5)
    r = client.post("/api/rerank", json=BODY)
    assert r.status_code == 413 and r.json()["field"] == "query"


def test_request_body_over_the_cap_is_refused_before_parsing(models):
    client, _, _ = ollama_client(models, max_documents=2, max_chars_per_document=10)
    r = client.post("/api/rerank", content=json.dumps({**BODY, "pad": "x" * 100}),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json()["field"] == "body" and r.json()["limit"] == 20


def test_health_reports_mode_and_loaded_models(models):
    client, _, _ = ollama_client(models)
    client.post("/api/rerank", json={**BODY, "model": "qllama/bge-reranker-v2-m3"})
    body = client.get("/health").json()
    assert body["mode"] == "ollama" and body["upstream"] == "http://127.0.0.1:9"
    assert "qllama/bge-reranker-v2-m3" in body["loaded"]


# --- llama-server scoring logic (HTTP faked, no binary) -----------------------------------

def fake_llama(scores_by_doc, seen):
    """Word-level tokenizer and a fixed score per document text, like llama-server's endpoints."""
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(len(body["content"].split())))})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": " ".join(["w"] * len(body["tokens"]))})
        seen.append(body)
        rows = [{"index": i, "relevance_score": scores_by_doc(d)} for i, d in enumerate(body["documents"])]
        return httpx.Response(200, json={"results": sorted(rows, key=lambda r: -r["relevance_score"])})
    return handle


def bare_backend(handler, max_length=64, softmax=False, overhead=4):
    backend = LlamaServerReranker.__new__(LlamaServerReranker)
    backend.model_id, backend.max_length, backend.softmax_scores = "fake", max_length, softmax
    backend._overhead = overhead
    backend.http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://llama")
    return backend


def test_scores_come_back_in_input_order():
    seen = []
    backend = bare_backend(fake_llama(lambda d: {"a b": 0.1, "c": 2.0}[d], seen))
    out = backend.score("q", ["a b", "c"])
    assert out.scores == [0.1, 2.0] and out.truncated == []


def test_long_document_is_cut_to_the_window_and_reported():
    seen = []
    backend = bare_backend(fake_llama(lambda d: 1.0, seen), max_length=32)
    out = backend.score("one two", ["short doc", "word " * 100])
    assert out.truncated == [1]
    budget = 32 - 4 - 2 - LlamaServerReranker.ROUNDTRIP_MARGIN
    assert len(seen[0]["documents"][1].split()) == budget


def test_query_that_fills_the_window_is_refused_naming_the_field():
    backend = bare_backend(fake_llama(lambda d: 1.0, []), max_length=16)
    from tiebreak.engine import LimitError
    with pytest.raises(LimitError) as err:
        backend.score("q " * 20, ["doc"])
    assert err.value.field == "query"


def test_qwen3_probabilities_become_logits_so_sigmoid_round_trips():
    backend = bare_backend(fake_llama(lambda d: {"yes": 0.9, "no": 0.2}[d], []), softmax=True)
    out = backend.score("q", ["yes", "no"])
    assert out.scores[0] == pytest.approx(math.log(9))
    assert 1 / (1 + math.exp(-out.scores[1])) == pytest.approx(0.2)
    assert prob_to_logit(1.0) < 40 and prob_to_logit(0.0) > -40
