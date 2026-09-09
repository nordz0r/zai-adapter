"""OpenAI Chat Completions <-> Z.AI Anthropic-messages translation."""
from __future__ import annotations

import time
import uuid

MODELS = ["glm-5.3-flash", "glm-5.3", "glm-5.2", "glm-5.1", "glm-5", "glm-4.6"]


class TranslationError(ValueError):
    pass


def openai_to_anthropic(body: dict, default_model: str = "glm-5.3-flash") -> dict:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TranslationError("messages[] is required")
    system_parts: list[str] = []
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            continue
        if role not in ("user", "assistant"):
            raise TranslationError(f"unsupported role: {role!r}")
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = [{"type": "text", "text": p.get("text", "")}
                      for p in content if isinstance(p, dict) and p.get("type") == "text"]
        else:
            raise TranslationError("message content must be a string or parts list")
        out.append({"role": role, "content": blocks})
    result: dict = {
        "model": body.get("model") or default_model,
        "messages": out,
        "max_tokens": body.get("max_tokens") or 4096,
    }
    if system_parts:
        result["system"] = "\n\n".join(system_parts)
    if isinstance(body.get("temperature"), (int, float)):
        result["temperature"] = body["temperature"]
    if isinstance(body.get("stop"), (str, list)):
        result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    return result


def anthropic_to_openai_response(resp: dict, request_model: str) -> dict:
    text_parts: list[str] = []
    reasoning: list[str] = []
    for block in resp.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning.append(block.get("thinking", ""))
    usage = resp.get("usage", {})
    message: dict = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model or resp.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _map_finish(resp.get("stop_reason")),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def anthropic_to_openai_chunks(resp: dict, request_model: str) -> list[dict]:
    """Single-chunk streaming fallback: the upstream answer arrives as one SSE delta."""
    full = anthropic_to_openai_response(resp, request_model)
    created = full["created"]
    choice = full["choices"][0]
    chunks = [{
        "id": full["id"], "object": "chat.completion.chunk", "created": created,
        "model": full["model"],
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }]
    message = choice["message"]
    delta: dict = {}
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("content"):
        delta["content"] = message["content"]
    chunks.append({
        "id": full["id"], "object": "chat.completion.chunk", "created": created,
        "model": full["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}],
    })
    chunks.append({
        "id": full["id"], "object": "chat.completion.chunk", "created": created,
        "model": full["model"],
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "usage": full["usage"],
    })
    return chunks


def _map_finish(stop_reason: str | None) -> str | None:
    return {"stop": "stop", "length": "length", "max_tokens": "length"}.get(stop_reason or "", "stop")


def models_payload() -> dict:
    now = int(time.time())
    return {"object": "list", "data": [
        {"id": m, "object": "model", "created": now, "owned_by": "zai-adapter"} for m in MODELS]}
