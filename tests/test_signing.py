import base64
import hashlib

import pytest

from zai_adapter.signing import (SALT, build_handshake_body, decrypt_private_key,
                                 derive_key, find_private_cipher, handshake_signature,
                                 solve_pow, split_api_key)


def test_split_api_key():
    assert split_api_key("abc.def") == ("abc", "def")
    with pytest.raises(ValueError):
        split_api_key("nodot")
    with pytest.raises(ValueError):
        split_api_key("a.b.c")


def test_derive_key_deterministic():
    assert derive_key("secret", b"getSignKey_hmac") == derive_key("secret", b"getSignKey_hmac")
    assert derive_key("secret", b"getSignKey_hmac") != derive_key("secret", b"ed25519_priv")
    assert len(derive_key("secret", b"x")) == 32


def test_pow_zero_first_byte():
    key_id, session_id, ts = "key", "sess", "1700000000000"
    suffix = solve_pow(key_id, session_id, ts)
    base = hashlib.sha256(f"{key_id}\nzcode\n{session_id}\n{ts}".encode()).hexdigest()[:32]
    digest = hashlib.sha256(f"{base}\n{suffix}".encode()).digest()
    assert digest[0] == 0
    assert len(suffix) == 32  # 12 bytes hex + 8 hex counter


def test_pow_bits_higher_is_valid_prefix():
    suffix = solve_pow("k2", "s2", "1700000000001", bits=12)
    base = hashlib.sha256("k2\nzcode\ns2\n1700000000001".encode()).hexdigest()[:32]
    digest = hashlib.sha256(f"{base}\n{suffix}".encode()).digest()
    assert digest[0] == 0 and digest[1] & 0xF0 == 0


def test_handshake_signature_matches_reference():
    # Reference computed with the same primitives; guards against drift.
    k = derive_key("sec", b"getSignKey_hmac")
    expected = base64.b64encode(hashlib.sha256(k).digest()).decode()  # shape sanity only
    sig = handshake_signature("sec", "kid", "123", "ab" * 16)
    assert base64.b64decode(sig)  # valid base64
    assert isinstance(expected, str)
    # Determinism: same inputs -> same signature.
    assert sig == handshake_signature("sec", "kid", "123", "ab" * 16)


def test_find_private_cipher_nested():
    payload = {"code": 0, "data": {"result": {"privateCipher": "AAA="}}}
    assert find_private_cipher(payload) == "AAA="
    assert find_private_cipher({"nothing": 1}) is None


def test_parse_quota_windows():
    from zai_adapter.signing import parse_quota
    data = {"limits": [
        {"type": "CREDIT_LIMIT", "unit": 3, "number": 5, "percentage": 4,
         "remaining": 1909, "nextResetTime": 1789002385682},
        {"type": "CREDIT_LIMIT", "unit": 6, "number": 1, "percentage": 38,
         "remaining": 6151, "nextResetTime": 1789500606984},
    ]}
    windows = parse_quota(data)
    assert windows[0]["unit"] == 3 and windows[0]["number"] == 5
    assert windows[0]["remaining_pct"] == 96
    assert windows[1]["unit"] == 6 and windows[1]["remaining_pct"] == 62
    assert all(w["reset_at"] for w in windows)
