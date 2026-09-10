import pytest

from zai_adapter.translate import (TranslationError, anthropic_to_openai_chunks,
                                   anthropic_to_openai_response, models_payload,
                                   openai_to_anthropic)


def test_request_translation_system_and_roles():
    body = {
        "model": "glm-5.3-flash",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "ок"},
            {"role": "user", "content": [{"type": "text", "text": "ещё"}]},
        ],
        "max_tokens": 128,
        "temperature": 0.5,
        "stop": "END",
    }
    out = openai_to_anthropic(body)
    assert out["model"] == "glm-5.3-flash"
    assert out["system"] == "be brief"
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user"]
    assert out["max_tokens"] == 128
    assert out["temperature"] == 0.5
    assert out["stop_sequences"] == ["END"]


def test_request_translation_rejects_bad():
    with pytest.raises(TranslationError):
        openai_to_anthropic({"messages": []})
    with pytest.raises(TranslationError):
        openai_to_anthropic({"messages": [{"role": "tool", "content": "x"}]})


def test_response_translation_text_and_thinking():
    resp = {
        "model": "glm-5.3-flash",
        "content": [
            {"type": "thinking", "thinking": "думаю"},
            {"type": "text", "text": "ответ"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    out = anthropic_to_openai_response(resp, "glm-5.3-flash")
    choice = out["choices"][0]
    assert choice["message"]["content"] == "ответ"
    assert choice["message"]["reasoning_content"] == "думаю"
    assert choice["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 15


def test_stream_chunks_shape():
    resp = {"model": "m", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
    chunks = anthropic_to_openai_chunks(resp, "m")
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "ok"
    assert chunks[-1]["choices"][0]["finish_reason"] is None


def test_models_payload():
    data = models_payload()["data"]
    assert "glm-5.3-flash" in [m["id"] for m in data]


def test_tools_and_reasoning_effort_translation():
    from zai_adapter.translate import StreamTranslator

    body = {
        "model": "glm-5.3-flash",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_fn", "arguments": "{\"x\": 1}"}
            }]},
            {"role": "tool", "tool_call_id": "call_123", "content": "res"}
        ],
        "tools": [{
            "type": "function",
            "function": {"name": "test_fn", "description": "desc", "parameters": {"type": "object"}}
        }],
        "tool_choice": "auto",
        "reasoning_effort": "high"
    }
    out = openai_to_anthropic(body)
    assert out["tools"][0]["name"] == "test_fn"
    assert out["tool_choice"] == {"type": "auto"}
    assert out["thinking"]["type"] == "enabled"
    assert out["thinking"]["budget_tokens"] == 8192
    assert out["messages"][1]["content"][0]["type"] == "tool_use"
    assert out["messages"][2]["content"][0]["type"] == "tool_result"

    # Test StreamTranslator
    st = StreamTranslator("glm-5.3-flash")
    ch1 = st.feed_event("message_start", {})
    assert len(ch1) == 1
    assert ch1[0]["choices"][0]["delta"]["role"] == "assistant"

    ch2 = st.feed_event("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}})
    assert ch2[0]["choices"][0]["delta"]["reasoning_content"] == "plan"

    ch3 = st.feed_event("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "hi"}})
    assert ch3[0]["choices"][0]["delta"]["content"] == "hi"

    ch4 = st.feed_event("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10}})
    assert ch4[0]["choices"][0]["finish_reason"] == "stop"
    assert ch4[0]["usage"]["completion_tokens"] == 10

