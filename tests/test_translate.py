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
