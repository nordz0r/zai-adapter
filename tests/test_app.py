"""End-to-end app tests with a mocked upstream and mocked handshake."""
import json

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

import zai_adapter.app as app_module

API_KEY = "testkey.testsecret"


def _make_client(monkeypatch, upstream_handler, adapter_key=None):
    """TestClient with mocked upstream HTTP and mocked handshake (returns fixed key)."""
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    if adapter_key is None:
        monkeypatch.delenv("ADAPTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ADAPTER_API_KEY", adapter_key)

    priv = Ed25519PrivateKey.generate()
    counter = {"handshakes": 0}

    def fake_handshake(self):
        counter["handshakes"] += 1
        return priv

    monkeypatch.setattr(app_module.Signer, "_perform_handshake", fake_handshake)
    monkeypatch.setattr(app_module, "_signer", None)
    monkeypatch.setattr(app_module, "_signer_key", None)

    app_module.app.state.http = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream_handler), timeout=httpx.Timeout(10))
    return TestClient(app_module.app), counter


def _upstream_ok(request):
    assert request.headers["x-session-id"].startswith("adapter-")
    assert request.headers["x-app-id"] == "zcode"
    assert request.headers["x-client-pow"]
    assert request.headers["x-client-sig"]
    return httpx.Response(200, json={
        "model": "glm-5.3-flash",
        "content": [{"type": "text", "text": "привет"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    })


def test_chat_completions_signed_roundtrip(monkeypatch):
    def upstream(request):
        seen["url"] = str(request.url)
        return _upstream_ok(request)

    seen = {}
    client, counter = _make_client(monkeypatch, upstream)
    with client as c:
        resp = c.post("/v1/chat/completions", json={
            "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "привет"
        assert body["usage"]["prompt_tokens"] == 12
        assert seen["url"].endswith("/v1/messages")
        # Key is cached: second call must not re-handshake.
        resp2 = c.post("/v1/chat/completions", json={
            "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "again"}]})
        assert resp2.status_code == 200
    assert counter["handshakes"] == 1


def test_chat_completions_streaming(monkeypatch):
    client, _ = _make_client(monkeypatch, _upstream_ok)
    resp = client.post("/v1/chat/completions", json={
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    payload = resp.text
    assert "data: [DONE]" in payload
    assert '"content":"привет"' in payload.replace(" ", "")


def test_auth_required_when_configured(monkeypatch):
    client, _ = _make_client(monkeypatch, _upstream_ok, adapter_key="topsecret")
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer topsecret"})
    assert ok.status_code == 200
    blocked = client.post("/v1/chat/completions", json={
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert blocked.status_code == 401


def test_bad_model_body_rejected(monkeypatch):
    client, counter = _make_client(monkeypatch, _upstream_ok)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "nope", "content": "x"}]})
    assert resp.status_code == 400
    assert counter["handshakes"] == 0


def test_healthz_reports_signing(monkeypatch):
    client, _ = _make_client(monkeypatch, _upstream_ok)
    data = client.get("/healthz").json()
    assert data["status"] == "ok" and data["signing"] is True


def test_free_window_guard(monkeypatch):
    client, _ = _make_client(monkeypatch, _upstream_ok)
    monkeypatch.setenv("ZAI_ONLY_FREE_HOURS", "true")

    # Mock is_free_window to False
    monkeypatch.setattr(app_module, "is_free_window", lambda: False)
    resp = client.post("/v1/chat/completions", json={
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 403
    assert "18:00 - 04:00 MSK" in resp.json()["detail"]

    # Mock is_free_window to True
    monkeypatch.setattr(app_module, "is_free_window", lambda: True)
    resp2 = client.post("/v1/chat/completions", json={
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert resp2.status_code == 200

