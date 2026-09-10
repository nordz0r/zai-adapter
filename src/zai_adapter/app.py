"""FastAPI application: OpenAI-compatible facade over the signed ZCode protocol."""
from __future__ import annotations

import json
import os
import secrets
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .signing import Signer, parse_quota
from .translate import (TranslationError, anthropic_to_openai_chunks,
                        anthropic_to_openai_response, models_payload, openai_to_anthropic)

UPSTREAM = os.environ.get("ZAI_UPSTREAM_URL", "https://zcode.z.ai/api/v1/ultra-zai/anthropic")
HANDSHAKE_URL = os.environ.get("ZAI_HANDSHAKE_URL", "https://api.z.ai/api/paas/c1f3a7e2/v2/client")
MONITOR_URL = os.environ.get("ZAI_MONITOR_URL", "https://api.z.ai/api/monitor/usage/quota/limit")
UPSTREAM_TIMEOUT = float(os.environ.get("ZAI_UPSTREAM_TIMEOUT_S", "120"))

_signer: Signer | None = None
_signer_key: str | None = None


def _zai_api_key() -> str:
    return os.environ.get("ZAI_API_KEY", "")


def _adapter_api_key() -> str:
    return os.environ.get("ADAPTER_API_KEY", "")


def _default_model() -> str:
    return os.environ.get("ZAI_DEFAULT_MODEL", "glm-5.3-flash")


def _require_signer() -> Signer:
    global _signer, _signer_key
    api_key = _zai_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="ZAI_API_KEY is not configured")
    if _signer is None or _signer_key != api_key:
        _signer = Signer(api_key, HANDSHAKE_URL)
        _signer_key = api_key
    return _signer


def _check_auth(request: Request) -> None:
    expected = _adapter_api_key()
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else request.headers.get("x-api-key", "")
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid adapter API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    created = False
    if not hasattr(app.state, "http"):
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=15.0))
        created = True
    yield
    if created:
        await app.state.http.aclose()


app = FastAPI(title="zai-adapter", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "upstream": UPSTREAM, "signing": bool(_zai_api_key())}


@app.get("/v1/models")
@app.get("/models")
async def models(request: Request):
    _check_auth(request)
    return models_payload()


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    s = _require_signer()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    try:
        anthropic_body = openai_to_anthropic(body, _default_model())
    except TranslationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session_id = f"adapter-{secrets.token_hex(6)}"
    headers = s.signed_headers(session_id)
    headers["Content-Type"] = "application/json"
    headers["anthropic-version"] = "2023-06-01"
    headers["x-api-key"] = _zai_api_key()
    model = anthropic_body["model"]

    want_stream = bool(body.get("stream"))
    try:
        resp = await request.app.state.http.post(
            f"{UPSTREAM}/v1/messages", json=anthropic_body, headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="upstream timeout")
    if resp.status_code == 401:
        s.invalidate()
        headers = s.signed_headers(f"adapter-{secrets.token_hex(6)}")
        headers["Content-Type"] = "application/json"
        headers["anthropic-version"] = "2023-06-01"
        headers["x-api-key"] = _zai_api_key()
        resp = await request.app.state.http.post(
            f"{UPSTREAM}/v1/messages", json=anthropic_body, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"upstream {resp.status_code}: {resp.text[:300]}")
    data = resp.json()

    if want_stream:
        chunks = anthropic_to_openai_chunks(data, model)

        def sse():
            for chunk in chunks:
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")
    return JSONResponse(anthropic_to_openai_response(data, model))


@app.get("/quota")
async def quota():
    """Plan quota straight from the Z.AI monitor endpoint (window percentages, resets)."""
    api_key = _zai_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="ZAI_API_KEY is not configured")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(MONITOR_URL, headers={
                "Authorization": f"Bearer {api_key}", "Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"monitor unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"monitor returned {resp.status_code}")
    payload = resp.json()
    if payload.get("code") != 200:
        raise HTTPException(status_code=502, detail="monitor rejected the key")
    return {
        "plan": payload["data"].get("level"),
        "windows": parse_quota(payload["data"]),
        "generated_at": int(time.time()),
    }


@app.get("/quota/text")
async def quota_text():
    """Human-readable quota line, used by cron jobs."""
    data = await quota()
    parts = []
    for w in data["windows"]:
        reset_str = f" (reset {time.strftime('%d.%m %H:%M UTC', time.gmtime(w['reset_at'] / 1000))})" if w.get("reset_at") else ""
        parts.append(f"{w['remaining_pct']}% free{reset_str}")
    return {"text": f"zai plan {data.get('plan')}: " + "; ".join(parts)}

