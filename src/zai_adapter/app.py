"""FastAPI application: OpenAI-compatible facade over the signed ZCode protocol."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("zai_adapter")

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .signing import KeyPool, Signer, parse_quota
from .translate import (StreamTranslator, TranslationError, anthropic_to_openai_chunks,
                        anthropic_to_openai_response, models_payload, openai_to_anthropic)

UPSTREAM = os.environ.get("ZAI_UPSTREAM_URL", "https://zcode.z.ai/api/v1/ultra-zai/anthropic")
HANDSHAKE_URL = os.environ.get("ZAI_HANDSHAKE_URL", "https://api.z.ai/api/paas/c1f3a7e2/v2/client")
MONITOR_URL = os.environ.get("ZAI_MONITOR_URL", "https://api.z.ai/api/monitor/usage/quota/limit")
UPSTREAM_TIMEOUT = float(os.environ.get("ZAI_UPSTREAM_TIMEOUT_S", "120"))
STREAM_PACING = os.environ.get("ZAI_STREAM_PACING", "true").lower() in ("1", "true", "yes")

_signer: Signer | None = None
_signer_key: str | None = None
_pool: KeyPool | None = None
_pool_keys_str: str | None = None


def _get_api_keys() -> list[str]:
    raw = os.environ.get("ZAI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _adapter_api_key() -> str:
    return os.environ.get("ADAPTER_API_KEY", "")


def _default_model() -> str:
    return os.environ.get("ZAI_DEFAULT_MODEL", "glm-5.3-flash")


def is_free_window() -> bool:
    """Free campaign window: 23:00 - 09:00 Singapore Time (UTC+8) -> 15:00 - 01:00 UTC (18:00 - 04:00 MSK)."""
    utc_hour = time.gmtime().tm_hour
    return utc_hour >= 15 or utc_hour < 1


def check_free_window_guard() -> None:
    if os.environ.get("ZAI_ONLY_FREE_HOURS", "false").lower() in ("true", "1", "yes"):
        if not is_free_window():
            raise HTTPException(
                status_code=403,
                detail=(
                    "Free campaign window is currently inactive (active 15:00 - 01:00 UTC / 18:00 - 04:00 MSK). "
                    "Request blocked by ZAI_ONLY_FREE_HOURS guard."
                ),
            )


def _require_pool() -> KeyPool:
    global _pool, _pool_keys_str, _signer
    raw_keys = os.environ.get("ZAI_API_KEY", "")
    keys = _get_api_keys()
    if not keys:
        raise HTTPException(status_code=503, detail="ZAI_API_KEY is not configured")
    if _signer is not None:
        pool = KeyPool(keys, HANDSHAKE_URL)
        pool._signers[keys[0]] = _signer
        return pool
    if _pool is None or _pool_keys_str != raw_keys:
        _pool = KeyPool(keys, HANDSHAKE_URL)
        _pool_keys_str = raw_keys
    return _pool


def _check_auth(request: Request) -> None:
    expected = _adapter_api_key()
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else request.headers.get("x-api-key", "")
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid adapter API key")


def _get_http_client(request: Request) -> httpx.AsyncClient:
    if hasattr(request.app.state, "http") and request.app.state.http is not None:
        return request.app.state.http
    client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=15.0))
    request.app.state.http = client
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    created = False
    if not hasattr(app.state, "http"):
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=15.0))
        created = True
    yield
    if created:
        await app.state.http.aclose()


app = FastAPI(title="zai-adapter", version="1.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    keys = _get_api_keys()
    return {
        "status": "ok",
        "upstream": UPSTREAM,
        "signing": bool(keys),
        "keys_count": len(keys),
        "free_window_active": is_free_window(),
        "free_hours_guard": os.environ.get("ZAI_ONLY_FREE_HOURS", "false").lower() in ("true", "1", "yes")
    }


@app.get("/v1/models")
@app.get("/models")
async def models(request: Request):
    _check_auth(request)
    return models_payload()


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    check_free_window_guard()
    pool = _require_pool()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    try:
        anthropic_body = openai_to_anthropic(body, _default_model())
    except TranslationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model = anthropic_body["model"]
    want_stream = bool(body.get("stream"))
    logger.info(
        "Chat request: model=%s stream=%s messages=%d tools=%d max_tokens=%s thinking=%s",
        model, want_stream, len(body.get("messages", [])), len(body.get("tools", [])),
        anthropic_body.get("max_tokens"), bool(anthropic_body.get("thinking"))
    )

    all_keys = pool.all_keys()
    last_error_detail = ""

    client = _get_http_client(request)

    # Multi-key retry loop on 429 / 1313 / quota limits
    for attempt in range(len(all_keys)):
        api_key, signer = pool.get_signer()
        session_id = f"adapter-{secrets.token_hex(6)}"
        headers = signer.signed_headers(session_id)
        headers["Content-Type"] = "application/json"
        headers["anthropic-version"] = "2023-06-01"
        headers["x-api-key"] = api_key

        if want_stream:
            stream_body = dict(anthropic_body)
            stream_body["stream"] = True
            req = client.build_request("POST", f"{UPSTREAM}/v1/messages", json=stream_body, headers=headers)
            resp = await client.send(req, stream=True)

            if resp.status_code == 401:
                signer.invalidate()
                headers = signer.signed_headers(f"adapter-{secrets.token_hex(6)}")
                headers["Content-Type"] = "application/json"
                headers["anthropic-version"] = "2023-06-01"
                headers["x-api-key"] = api_key
                await resp.aclose()
                req = client.build_request("POST", f"{UPSTREAM}/v1/messages", json=stream_body, headers=headers)
                resp = await client.send(req, stream=True)

            if resp.status_code in (429, 503) or resp.status_code >= 400:
                err_bytes = await resp.aread()
                err_text = err_bytes.decode("utf-8", errors="replace")
                await resp.aclose()
                last_error_detail = f"upstream {resp.status_code}: {err_text[:300]}"
                logger.warning("Upstream error on key %s: %s", api_key[:10], last_error_detail)
                if resp.status_code == 429 or "1313" in err_text or "1113" in err_text:
                    pool.mark_cooldown(api_key, duration_s=60.0)
                    if attempt < len(all_keys) - 1:
                        continue
                raise HTTPException(status_code=502, detail=last_error_detail)

            # Check if upstream returned SSE stream or plain JSON
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                translator = StreamTranslator(model)

                async def sse_realtime():
                    if not STREAM_PACING:
                        current_event = "message"
                        try:
                            async for line in resp.aiter_lines():
                                if line.startswith("event: "):
                                    current_event = line[7:].strip()
                                elif line.startswith("data: "):
                                    raw_data = line[6:].strip()
                                    if not raw_data:
                                        continue
                                    try:
                                        ev_data = json.loads(raw_data)
                                    except Exception:
                                        continue
                                    ev_type = ev_data.get("type") or current_event
                                    chunks = translator.feed_event(ev_type, ev_data)
                                    for chunk in chunks:
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        finally:
                            await resp.aclose()
                        yield "data: [DONE]\n\n"
                        return

                    # River-Flow Adaptive Stream Pacer:
                    # Gathers upstream token bursts into an async queue and emits chunks with adaptive
                    # micro-delays so thinking thoughts and answer text flow continuously like a river.
                    queue: asyncio.Queue[dict | None] = asyncio.Queue()
                    fetch_errors: list[Exception] = []

                    async def fetcher():
                        current_event = "message"
                        try:
                            async for line in resp.aiter_lines():
                                if line.startswith("event: "):
                                    current_event = line[7:].strip()
                                elif line.startswith("data: "):
                                    raw_data = line[6:].strip()
                                    if not raw_data:
                                        continue
                                    try:
                                        ev_data = json.loads(raw_data)
                                    except Exception:
                                        continue
                                    ev_type = ev_data.get("type") or current_event
                                    chunks = translator.feed_event(ev_type, ev_data)
                                    for chunk in chunks:
                                        await queue.put(chunk)
                        except Exception as exc:
                            logger.error("Error in upstream stream fetcher: %s", exc)
                            fetch_errors.append(exc)
                        finally:
                            await queue.put(None)

                    fetch_task = asyncio.create_task(fetcher())

                    try:
                        while True:
                            chunk = await queue.get()
                            if chunk is None:
                                break

                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                            choices = chunk.get("choices", [])
                            delta = choices[0].get("delta", {}) if choices else {}
                            finish_reason = choices[0].get("finish_reason") if choices else None

                            # Only pace reasoning thoughts and text content.
                            # Never delay tool calls, stop signals, or empty deltas.
                            is_text = bool(delta.get("reasoning_content") or delta.get("content"))
                            has_tool = bool(delta.get("tool_calls"))

                            if is_text and not has_tool and not finish_reason:
                                if fetch_task.done():
                                    delay = 0.002
                                else:
                                    qsize = queue.qsize()
                                    if qsize > 35:
                                        delay = 0.004
                                    elif qsize > 20:
                                        delay = 0.010
                                    elif qsize > 8:
                                        delay = 0.020
                                    else:
                                        delay = 0.030
                                await asyncio.sleep(delay)

                        if fetch_errors:
                            raise fetch_errors[0]
                    except asyncio.CancelledError:
                        fetch_task.cancel()
                        raise
                    except Exception as exc:
                        logger.error("Error in sse_realtime generator: %s", exc)
                        raise
                    finally:
                        if not fetch_task.done():
                            fetch_task.cancel()
                        await resp.aclose()

                    yield "data: [DONE]\n\n"

                stream_headers = {
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
                return StreamingResponse(sse_realtime(), media_type="text/event-stream", headers=stream_headers)
            else:
                # JSON fallback for upstream/mock
                raw_bytes = await resp.aread()
                await resp.aclose()
                data = json.loads(raw_bytes.decode("utf-8"))
                chunks = anthropic_to_openai_chunks(data, model)

                async def sse_fallback():
                    for chunk in chunks:
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        if STREAM_PACING:
                            await asyncio.sleep(0.015)
                    yield "data: [DONE]\n\n"

                return StreamingResponse(sse_fallback(), media_type="text/event-stream")

        else:
            # Non-streaming request
            try:
                resp = await client.post(
                    f"{UPSTREAM}/v1/messages", json=anthropic_body, headers=headers)
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="upstream timeout")

            if resp.status_code == 401:
                signer.invalidate()
                headers = signer.signed_headers(f"adapter-{secrets.token_hex(6)}")
                headers["Content-Type"] = "application/json"
                headers["anthropic-version"] = "2023-06-01"
                headers["x-api-key"] = api_key
                resp = await client.post(
                    f"{UPSTREAM}/v1/messages", json=anthropic_body, headers=headers)

            if resp.status_code in (429, 503) or resp.status_code >= 400:
                last_error_detail = f"upstream {resp.status_code}: {resp.text[:300]}"
                if resp.status_code == 429 or "1313" in resp.text or "1113" in resp.text:
                    pool.mark_cooldown(api_key, duration_s=60.0)
                    if attempt < len(all_keys) - 1:
                        continue
                raise HTTPException(status_code=502, detail=last_error_detail)

            data = resp.json()
            return JSONResponse(anthropic_to_openai_response(data, model))

    raise HTTPException(status_code=502, detail=last_error_detail or "all keys in pool failed")


@app.get("/quota")
async def quota():
    """Plan quota from Z.AI monitor endpoint across configured keys."""
    pool = _require_pool()
    all_keys = pool.all_keys()
    results = []

    async with httpx.AsyncClient(timeout=20) as client:
        for k in all_keys:
            try:
                resp = await client.get(MONITOR_URL, headers={
                    "Authorization": f"Bearer {k}", "Accept": "application/json"})
                if resp.status_code == 200:
                    payload = resp.json()
                    if payload.get("code") == 200:
                        results.append({
                            "key_id": k.split(".")[0],
                            "plan": payload["data"].get("level"),
                            "windows": parse_quota(payload["data"]),
                        })
            except Exception:
                continue

    if not results:
        raise HTTPException(status_code=502, detail="monitor rejected or unreachable for all keys")

    primary = results[0]
    return {
        "plan": primary["plan"],
        "windows": primary["windows"],
        "all_keys": results,
        "keys_count": len(all_keys),
        "free_window_active": is_free_window(),
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

    status_suffix = " [FREE WINDOW ACTIVE]" if data.get("free_window_active") else " [PAID HOURS]"
    return {"text": f"zai plan {data.get('plan')}: " + "; ".join(parts) + status_suffix}
