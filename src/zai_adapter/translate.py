"""OpenAI Chat Completions <-> Z.AI Anthropic-messages translation."""
from __future__ import annotations

import json
import secrets
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
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            continue

        if role == "tool":
            # Anthropic tool_result must be inside a user message
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                raise TranslationError("tool message requires tool_call_id")
            tool_res_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            tool_block = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_res_content
            }
            if out and out[-1]["role"] == "user":
                out[-1]["content"].append(tool_block)
            else:
                out.append({"role": "user", "content": [tool_block]})
            continue

        if role not in ("user", "assistant"):
            raise TranslationError(f"unsupported role: {role!r}")

        blocks: list[dict] = []
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        blocks.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "image_url":
                        # Support OpenAI vision format
                        url_obj = p.get("image_url", {})
                        url_str = url_obj.get("url", "") if isinstance(url_obj, dict) else url_obj
                        if url_str.startswith("data:"):
                            header, data_b64 = url_str.split(",", 1)
                            media_type = header.split(";")[0].removeprefix("data:")
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data_b64
                                }
                            })
        elif content is None and role == "assistant":
            pass
        elif not isinstance(content, (str, list)):
            raise TranslationError("message content must be a string or parts list")

        # Handle tool_calls on assistant message
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                args = tc.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"call_{secrets.token_hex(8)}",
                    "name": tc.get("function", {}).get("name"),
                    "input": args
                })

        if blocks:
            out.append({"role": role, "content": blocks})

    # Normalize messages for Anthropic: alternating roles, first message must be user
    normalized_messages: list[dict] = []
    for msg in out:
        r = msg["role"]
        c = msg["content"]
        if not normalized_messages:
            if r == "assistant":
                normalized_messages.append({"role": "user", "content": [{"type": "text", "text": "Hello"}]})
            normalized_messages.append({"role": r, "content": list(c)})
        else:
            if normalized_messages[-1]["role"] == r:
                normalized_messages[-1]["content"].extend(c)
            else:
                normalized_messages.append({"role": r, "content": list(c)})

    req_max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    max_tokens = int(req_max_tokens) if req_max_tokens else 32768

    result: dict = {
        "model": body.get("model") or default_model,
        "messages": normalized_messages,
    }

    if system_parts:
        result["system"] = "\n\n".join(system_parts)
    if isinstance(body.get("temperature"), (int, float)):
        result["temperature"] = body["temperature"]
    if isinstance(body.get("stop"), (str, list)):
        result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    # Reasoning effort / thinking budget
    effort = body.get("reasoning_effort") or body.get("reasoningEffort")
    budget: int | None = None
    if effort:
        budgets = {"low": 2048, "medium": 4096, "high": 8192, "max": 16384}
        budget = budgets.get(str(effort).lower(), 8192)
    elif "thinking" in body and isinstance(body["thinking"], dict):
        budget = body["thinking"].get("budget_tokens", 8192)

    if budget is not None:
        result["thinking"] = {"type": "enabled", "budget_tokens": budget}
        max_tokens = max(max_tokens, budget + 8192)

    result["max_tokens"] = min(max_tokens, 65536)

    # Convert tools
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                params = fn.get("parameters") or {"type": "object", "properties": {}}
                if isinstance(params, dict) and params.get("type") == "object" and "properties" not in params:
                    params = dict(params)
                    params["properties"] = {}
                anthropic_tools.append({
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": params
                })
            elif "name" in t:
                anthropic_tools.append(t)
        if anthropic_tools:
            result["tools"] = anthropic_tools

    # Convert tool_choice
    tc = body.get("tool_choice")
    if tc:
        if tc == "auto":
            result["tool_choice"] = {"type": "auto"}
        elif tc == "required":
            result["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            fn_name = tc.get("function", {}).get("name")
            if fn_name:
                result["tool_choice"] = {"type": "tool", "name": fn_name}

    return result


def anthropic_to_openai_response(resp: dict, request_model: str) -> dict:
    text_parts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []

    for block in resp.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                }
            })

    usage = resp.get("usage", {})
    content_val = "".join(text_parts) if (text_parts or not tool_calls) else None

    message: dict = {
        "role": "assistant",
        "content": content_val,
    }
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = resp.get("stop_reason")
    finish_reason = "tool_calls" if tool_calls else _map_finish(stop_reason)

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model or resp.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


class StreamTranslator:
    """Translates an upstream Anthropic SSE event stream into OpenAI chat.completion.chunk objects."""

    def __init__(self, request_model: str):
        self.request_model = request_model
        self.completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.started = False
        self.tool_calls_map: dict[int, dict] = {}

    def feed_event(self, ev_type: str, data: dict) -> list[dict]:
        chunks: list[dict] = []

        if ev_type == "message_start":
            if not self.started:
                self.started = True
                chunks.append({
                    "id": self.completion_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                })
            return chunks

        if ev_type == "content_block_start":
            block = data.get("content_block", {})
            btype = block.get("type")
            idx = data.get("index", 0)
            if btype == "tool_use":
                t_idx = len(self.tool_calls_map)
                self.tool_calls_map[idx] = {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "index": t_idx
                }
                chunks.append({
                    "id": self.completion_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": t_idx,
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": ""
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                })
            return chunks

        if ev_type == "content_block_delta":
            delta = data.get("delta", {})
            dtype = delta.get("type")
            idx = data.get("index", 0)

            if dtype == "thinking_delta":
                thought = delta.get("thinking", "")
                chunks.append({
                    "id": self.completion_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "reasoning_content": thought,
                            "reasoning": thought,
                        },
                        "finish_reason": None
                    }]
                })
            elif dtype == "text_delta":
                chunks.append({
                    "id": self.completion_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None
                    }]
                })
            elif dtype == "input_json_delta":
                t_info = self.tool_calls_map.get(idx, {"index": 0})
                chunks.append({
                    "id": self.completion_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.request_model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": t_info["index"],
                                "function": {"arguments": delta.get("partial_json", "")}
                            }]
                        },
                        "finish_reason": None
                    }]
                })
            return chunks

        if ev_type == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason")
            finish_reason = "tool_calls" if stop_reason == "tool_use" else _map_finish(stop_reason)
            usage = data.get("usage")

            chunk = {
                "id": self.completion_id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.request_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            }
            if usage:
                chunk["usage"] = {"completion_tokens": usage.get("output_tokens", 0)}
            chunks.append(chunk)
            return chunks

        return chunks


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
    if message.get("tool_calls"):
        delta["tool_calls"] = message["tool_calls"]

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
    return {
        "stop": "stop",
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "length": "length",
        "max_tokens": "length"
    }.get(stop_reason or "", "stop")


def models_payload() -> dict:
    now = int(time.time())
    return {"object": "list", "data": [
        {"id": m, "object": "model", "created": now, "owned_by": "zai-adapter"} for m in MODELS]}
