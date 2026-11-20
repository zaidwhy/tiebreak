"""--ollama mode forwards what it does not own, unchanged and streamed, against a fake Ollama."""
import asyncio
import json
import threading
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from conftest import WordOverlapEngine, run_server
from tiebreak.router import EngineRouter
from tiebreak.server import create_app
from tiebreak.settings import Settings


class FakeOllama:
    def __init__(self):
        self.seen: list[dict] = []
        self.release = threading.Event()  # the stream's second line waits for this

    async def record(self, request: Request) -> dict:
        body = await request.body()
        row = {"method": request.method, "path": request.url.path, "query": request.url.query,
               "headers": dict(request.headers), "body": body}
        self.seen.append(row)
        return row

    async def tags(self, request):
        await self.record(request)
        return JSONResponse({"models": [{"name": "qwen2.5:7b"}]}, headers={"x-upstream": "yes"})

    async def chat(self, request):
        await self.record(request)

        async def lines():
            yield b'{"message":{"content":"Hel"},"done":false}\n'
            while not self.release.is_set():
                await asyncio.sleep(0.01)
            yield b'{"message":{"content":"lo"},"done":true}\n'
        return StreamingResponse(lines(), media_type="application/x-ndjson")

    async def echo(self, request):
        row = await self.record(request)
        return Response(row["body"], status_code=201, media_type="application/octet-stream")

    async def missing(self, request):
        await self.record(request)
        return JSONResponse({"error": "model 'nope' not found"}, status_code=404)

    def app(self):
        return Starlette(routes=[
            Route("/api/tags", self.tags),
            Route("/api/chat", self.chat, methods=["POST"]),
            Route("/v1/chat/completions", self.echo, methods=["POST"]),
            Route("/v1/models", self.tags),
            Route("/api/show", self.missing, methods=["POST"]),
            Route("/api/rerank", self.echo, methods=["POST"]),  # must never be reached
            Route("/v1/rerank", self.echo, methods=["POST"]),
        ])


@pytest.fixture
def stack():
    fake = FakeOllama()
    with run_server(fake.app()) as upstream:
        router = EngineRouter("minilm", onnx_factory=lambda name: WordOverlapEngine(),
                              gguf_factory=lambda name, blob: None)
        cfg = Settings(ollama_url=upstream, proxy_timeout_s=5)
        with run_server(create_app(cfg, router=router)) as front:
            yield fake, front
        fake.release.set()


def test_get_is_forwarded_with_status_body_and_headers(stack):
    fake, front = stack
    r = httpx.get(f"{front}/api/tags?verbose=1")
    assert r.status_code == 200
    assert r.json() == {"models": [{"name": "qwen2.5:7b"}]}
    assert r.headers["x-upstream"] == "yes"
    assert fake.seen[-1]["path"] == "/api/tags" and fake.seen[-1]["query"] == "verbose=1"


def test_post_body_is_forwarded_byte_for_byte(stack):
    fake, front = stack
    payload = b'{"model":"qwen2.5:7b","messages":[{"role":"user","content":"caf\xc3\xa9"}],"stream":false}'
    r = httpx.post(f"{front}/v1/chat/completions", content=payload,
                   headers={"content-type": "application/json", "x-client": "test"})
    assert r.status_code == 201
    assert r.content == payload
    assert fake.seen[-1]["body"] == payload
    assert fake.seen[-1]["headers"]["x-client"] == "test"


def test_upstream_error_status_is_passed_through(stack):
    _, front = stack
    r = httpx.post(f"{front}/api/show", json={"model": "nope"})
    assert r.status_code == 404
    assert r.json() == {"error": "model 'nope' not found"}


def test_v1_models_belongs_to_ollama_in_this_mode(stack):
    fake, front = stack
    assert httpx.get(f"{front}/v1/models").json()["models"][0]["name"] == "qwen2.5:7b"
    assert fake.seen[-1]["path"] == "/v1/models"


def test_chat_stream_reaches_the_client_before_the_upstream_finishes(stack):
    fake, front = stack
    with httpx.stream("POST", f"{front}/api/chat", json={"model": "qwen2.5:7b", "stream": True}, timeout=10) as r:
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = r.iter_lines()
        started = time.monotonic()
        first = next(lines)
        # the upstream is still holding its second line, so a buffering proxy would block here
        assert not fake.release.is_set()
        assert json.loads(first)["message"]["content"] == "Hel"
        assert time.monotonic() - started < 5
        fake.release.set()
        rest = [json.loads(line) for line in lines if line]
    assert rest == [{"message": {"content": "lo"}, "done": True}]


def test_rerank_paths_are_answered_here_not_forwarded(stack):
    fake, front = stack
    for path in ("/api/rerank", "/v1/rerank"):
        r = httpx.post(f"{front}{path}", json={"query": "capital of France",
                                               "documents": ["bananas", "Paris is the capital of France"]})
        assert r.status_code == 200
        assert r.json()["results"][0]["index"] == 1
    assert not [row for row in fake.seen if row["path"].endswith("rerank")]


def test_health_is_answered_here(stack):
    fake, front = stack
    assert httpx.get(f"{front}/health").json()["mode"] == "ollama"
    assert fake.seen == []


def test_unreachable_ollama_is_a_502_naming_the_url():
    with run_server(create_app(Settings(ollama_url="http://127.0.0.1:9"),
                               router=EngineRouter("minilm", lambda n: WordOverlapEngine(), lambda n, b: None))) as front:
        r = httpx.get(f"{front}/api/tags")
    assert r.status_code == 502
    assert "http://127.0.0.1:9" in r.json()["error"]


def test_api_key_guards_the_proxy_and_is_not_forwarded():
    fake = FakeOllama()
    with run_server(fake.app()) as upstream:
        cfg = Settings(ollama_url=upstream, api_key="s3cret")
        app = create_app(cfg, router=EngineRouter("minilm", lambda n: WordOverlapEngine(), lambda n, b: None))
        with run_server(app) as front:
            assert httpx.get(f"{front}/api/tags").status_code == 401
            ok = httpx.get(f"{front}/api/tags", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert len(fake.seen) == 1 and "authorization" not in fake.seen[0]["headers"]
