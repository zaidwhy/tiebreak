"""Forward every request tiebreak does not own to Ollama, byte for byte, streaming both ways."""
from __future__ import annotations

import logging

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("tiebreak.proxy")

# Hop-by-hop headers (RFC 9110 section 7.6.1) describe one connection, not the message.
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
              "trailer", "transfer-encoding", "upgrade"}


def make_client(proxy_timeout_s: int) -> httpx.AsyncClient:
    # read is the gap allowed between two chunks, not the whole response: a long chat stream is fine
    # as long as tokens keep coming. Loading a large model before the first token can take a while.
    return httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=float(proxy_timeout_s), write=60.0, pool=10.0),
                             follow_redirects=False)


async def forward(request: Request, client: httpx.AsyncClient, upstream: str, strip_auth: bool) -> Response:
    url = upstream.rstrip("/") + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    drop = HOP_BY_HOP | {"host"} | ({"authorization"} if strip_auth else set())
    headers = [(k, v) for k, v in request.headers.raw if k.decode("latin-1").lower() not in drop]
    upstream_req = client.build_request(request.method, url, headers=headers, content=request.stream())
    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        log.warning("upstream timeout %s %s", request.method, url)
        return JSONResponse({"error": f"Ollama at {upstream} did not answer in time "
                                      "(TIEBREAK_PROXY_TIMEOUT_S)"}, status_code=504)
    except httpx.TransportError as exc:
        log.warning("upstream unreachable %s %s: %s", request.method, url, exc)
        return JSONResponse({"error": f"Ollama at {upstream} is unreachable: {type(exc).__name__}"},
                            status_code=502)

    async def body():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            # Headers are already sent, so the status cannot change; end the stream and say why in the log.
            log.warning("upstream stream broke %s %s: %s", request.method, url, exc)

    response = StreamingResponse(body(), status_code=upstream_resp.status_code,
                                 background=BackgroundTask(upstream_resp.aclose))
    # raw bytes are relayed undecoded, so content-encoding and content-length stay valid
    response.raw_headers = [(k, v) for k, v in upstream_resp.headers.raw
                            if k.decode("latin-1").lower() not in HOP_BY_HOP]
    return response
