from conftest import SlowEngine

DOCS = [
    "bananas are yellow fruit",
    "the capital of France is Paris",
    "Paris is the capital city of France and sits on the Seine",
]
QUERY = "capital of France Paris"


def post(client, path="/v1/rerank", **body):
    return client.post(path, json=body)


def test_cohere_shape_returns_sorted_results_with_original_indices(client):
    r = post(client, model="rerank-v3.5", query=QUERY, documents=DOCS, top_n=3)
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["index"] in (1, 2)
    assert results[-1]["index"] == 0
    scores = [x["relevance_score"] for x in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all("document" not in x for x in results)


def test_jina_shape_with_text_objects_and_return_documents(client):
    r = post(client, model="jina-reranker-v2-base-multilingual", query=QUERY,
             documents=[{"text": d} for d in DOCS], top_n=2, return_documents=True)
    body = r.json()
    assert len(body["results"]) == 2
    top = body["results"][0]
    assert top["document"]["text"] == DOCS[top["index"]]
    assert body["usage"]["total_tokens"] > 0


def test_top_n_is_honoured_and_larger_than_documents_returns_all(client):
    assert len(post(client, query=QUERY, documents=DOCS, top_n=1).json()["results"]) == 1
    assert len(post(client, query=QUERY, documents=DOCS, top_n=50).json()["results"]) == 3


def test_same_handler_on_every_advertised_path(client):
    for path in ("/v1/rerank", "/v2/rerank", "/rerank", "/api/rerank"):
        assert post(client, path, query=QUERY, documents=DOCS).status_code == 200


def test_raw_scores_are_unbounded_logits(client):
    r = post(client, query=QUERY, documents=DOCS, score_scale="raw")
    assert r.json()["meta"]["score_scale"] == "raw"
    assert min(x["relevance_score"] for x in r.json()["results"]) < 0


def test_long_document_raises_truncation_warning_and_names_index(client):
    long_doc = "filler " * 100 + "capital of France Paris"
    body = post(client, query=QUERY, documents=[DOCS[0], long_doc]).json()
    assert body["meta"]["truncated_indices"] == [1]
    assert "documents[1]" in body["warnings"][0]
    assert "truncated" in body["warnings"][0]


def test_no_warning_when_nothing_is_cut(client):
    body = post(client, query=QUERY, documents=DOCS).json()
    assert body["warnings"] == []
    assert body["meta"]["truncated_indices"] == []


def test_too_many_documents_is_refused_with_the_limit_named(make_client):
    r = post(make_client(max_documents=2), query=QUERY, documents=DOCS)
    assert r.status_code == 413
    assert r.json()["limit"] == 2
    assert "TIEBREAK_MAX_DOCUMENTS" in r.json()["message"]


def test_oversized_document_is_refused_with_index_and_limit(make_client):
    r = post(make_client(max_chars_per_document=10), query="a", documents=["short", "x" * 11])
    assert r.status_code == 413
    assert r.json()["field"] == "documents[1]"
    assert "TIEBREAK_MAX_CHARS_PER_DOCUMENT=10" in r.json()["message"]


def test_total_token_cap_is_refused_with_the_limit_named(make_client):
    r = post(make_client(max_total_tokens=5), query=QUERY, documents=DOCS)
    assert r.status_code == 413
    assert "TIEBREAK_MAX_TOTAL_TOKENS=5" in r.json()["message"]


def test_bad_inputs_name_the_field(client):
    assert post(client, documents=DOCS).json()["field"] == "query"
    assert post(client, query=QUERY, documents=[]).json()["field"] == "documents"
    assert post(client, query=QUERY, documents=[3]).json()["field"] == "documents[0]"
    assert post(client, query=QUERY, documents=DOCS, top_n=0).json()["field"] == "top_n"
    assert client.post("/v1/rerank", json=[1, 2]).status_code == 422


def test_missed_deadline_returns_504_naming_the_timeout(make_client):
    r = post(make_client(SlowEngine(), request_timeout_s=7), query=QUERY, documents=DOCS)
    assert r.status_code == 504
    assert "TIEBREAK_TIMEOUT_S=7" in r.json()["message"]


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["data"][0]["id"] == "word-overlap"


def test_api_key_protects_rerank_and_models_but_not_health(make_client):
    c = make_client(api_key="s3cret")
    assert c.get("/health").status_code == 200
    assert post(c, query=QUERY, documents=DOCS).status_code == 401
    assert c.get("/v1/models").status_code == 401
    ok = c.post("/v1/rerank", json={"query": QUERY, "documents": DOCS}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
