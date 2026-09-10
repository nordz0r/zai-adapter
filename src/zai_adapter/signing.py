"""ZCode client signing: the exact scheme the ZCode desktop app uses to talk
to the Z.AI GLM Coding Plan ("via ZCode" tier).

Handshake: POST {handshake_url} {apiKey, nonce, sig, ts}, Authorization: <apiKey>
  sig = base64(HMAC-SHA256(k, "get_sign_key\n{apiKeyId}\n{ts}\n{nonce}"))
  k   = HKDF-SHA256(ikm=secret, salt="WD_CLIENT_SIGN_KDF_SALT", info="getSignKey_hmac")
Response: data.privateCipher (base64(iv12 + AES-GCM(ct))) -> base64 text -> PKCS8 Ed25519.
  AES key = HKDF-SHA256(ikm=secret, salt=same, info="ed25519_priv"), AAD = apiKeyId.

Per request:
  X-Client-Ts, X-Client-Nonce (16B hex), X-App-Id: zcode, X-Client-Version,
  X-Session-Id, X-Client-Sig = base64(Ed25519("{id}\n{ts}\n{ver}\n{sid}\n{nonce}")),
  X-Client-Pow = suffix s where sha256("{powBase}\n{s}") has 8 leading zero bits,
  powBase = sha256("{apiKeyId}\nzcode\n{sessionId}\n{ts}").hex()[:32].
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

SALT = b"WD_CLIENT_SIGN_KDF_SALT"
POW_BITS = 8
POW_APP_ID = "zcode"
NONCE_BYTES = 16
HANDSHAKE_MESSAGE = "get_sign_key"


def derive_key(secret: str, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=SALT, info=info).derive(secret.encode())


def split_api_key(api_key: str) -> tuple[str, str]:
    # Exactly one separator, mirroring the client-side credential parser.
    if api_key.count(".") != 1:
        raise ValueError("coding plan apiKey must contain exactly one separator")
    key_id, secret = api_key.split(".")
    if not key_id.strip() or not secret.strip():
        raise ValueError("coding plan apiKey halves must be non-empty")
    return key_id, secret


def handshake_signature(secret: str, key_id: str, ts: str, nonce: str) -> str:
    k = derive_key(secret, b"getSignKey_hmac")
    msg = f"{HANDSHAKE_MESSAGE}\n{key_id}\n{ts}\n{nonce}".encode()
    return base64.b64encode(hmac.new(k, msg, hashlib.sha256).digest()).decode()


def decrypt_private_key(cipher_b64: str, key_id: str, secret: str) -> Ed25519PrivateKey:
    aes = derive_key(secret, b"ed25519_priv")
    raw = base64.b64decode(cipher_b64)
    pt = AESGCM(aes).decrypt(raw[:12], raw[12:], key_id.encode())
    pkcs8 = base64.b64decode(pt.decode())
    key = serialization.load_der_private_key(pkcs8, None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("decrypted material is not an Ed25519 private key")
    return key


def solve_pow(key_id: str, session_id: str, ts: str, bits: int = POW_BITS) -> str:
    base = hashlib.sha256(f"{key_id}\n{POW_APP_ID}\n{session_id}\n{ts}".encode()).hexdigest()[:32]
    prefix = secrets.token_bytes(12).hex()
    nbytes, rem = divmod(bits, 8)
    mask = (0xFF << (8 - rem)) & 0xFF if rem else 0
    for i in range(1 << 32):
        suffix = f"{prefix}{i:08x}"
        digest = hashlib.sha256(f"{base}\n{suffix}".encode()).digest()
        if digest[:nbytes] == b"\x00" * nbytes and (rem == 0 or digest[nbytes] & mask == 0):
            return suffix
    raise RuntimeError("proof of work unsolvable")


def business_signature(priv: Ed25519PrivateKey, key_id: str, ts: str, version: str,
                       session_id: str, nonce: str) -> str:
    msg = f"{key_id}\n{ts}\n{version}\n{session_id}\n{nonce}".encode()
    return base64.b64encode(priv.sign(msg)).decode()


def find_private_cipher(resp: object) -> str | None:
    if isinstance(resp, dict):
        for k, v in resp.items():
            if k == "privateCipher" and isinstance(v, str):
                return v
            found = find_private_cipher(v)
            if found:
                return found
    elif isinstance(resp, list):
        for item in resp:
            found = find_private_cipher(item)
            if found:
                return found
    return None


class Signer:
    """Performs the handshake once, caches the Ed25519 key, re-handshakes on expiry."""

    def __init__(self, api_key: str, handshake_url: str, client_version: str = "0.16.5",
                 transport: httpx.Client | None = None):
        self.api_key = api_key
        self.handshake_url = handshake_url
        self.client_version = client_version
        self._transport = transport
        self._priv: Ed25519PrivateKey | None = None
        self._lock = threading.Lock()

    def _http(self) -> httpx.Client:
        return self._transport or httpx.Client(timeout=30)

    def _perform_handshake(self) -> Ed25519PrivateKey:
        key_id, secret = split_api_key(self.api_key)
        ts = str(int(time.time() * 1000))
        nonce = secrets.token_bytes(NONCE_BYTES).hex()
        sig = handshake_signature(secret, key_id, ts, nonce)
        headers = {"Authorization": self.api_key, "Content-Type": "application/json",
                   "User-Agent": f"ZCode/{self.client_version}"}
        payload = {"apiKey": self.api_key, "nonce": nonce, "sig": sig, "ts": ts}
        with self._http() as client:
            resp = client.post(self.handshake_url, json=payload, headers=headers)
        resp.raise_for_status()
        cipher = find_private_cipher(resp.json())
        if not cipher:
            raise RuntimeError("handshake response has no privateCipher")
        return decrypt_private_key(cipher, key_id, secret)

    def private_key(self) -> Ed25519PrivateKey:
        with self._lock:
            if self._priv is None:
                self._priv = self._perform_handshake()
            return self._priv

    def invalidate(self) -> None:
        with self._lock:
            self._priv = None

    def signed_headers(self, session_id: str) -> dict[str, str]:
        key_id, _ = split_api_key(self.api_key)
        ts = str(int(time.time() * 1000))
        nonce = secrets.token_bytes(NONCE_BYTES).hex()
        pow_value = solve_pow(key_id, session_id, ts)
        sig = business_signature(self.private_key(), key_id, ts, self.client_version, session_id, nonce)
        return {
            "X-Session-Id": session_id,
            "X-Client-Ts": ts,
            "X-Client-Version": self.client_version,
            "X-Client-Sig": sig,
            "X-Client-Nonce": nonce,
            "X-App-Id": POW_APP_ID,
            "X-Client-Pow": pow_value,
            "User-Agent": f"ZCode/{self.client_version}",
        }


class KeyPool:
    """Manages a pool of Signer instances, rotating on 429/1313 or rate limits."""

    def __init__(self, api_keys: list[str], handshake_url: str, client_version: str = "0.16.5",
                 transport: httpx.Client | None = None):
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.handshake_url = handshake_url
        self.client_version = client_version
        self._transport = transport
        self._signers = {k: Signer(k, handshake_url, client_version, transport) for k in self.api_keys}
        self._cooldowns: dict[str, float] = {}
        self._current_idx = 0
        self._lock = threading.Lock()

    def get_signer(self) -> tuple[str, Signer]:
        with self._lock:
            if not self.api_keys:
                raise ValueError("no API keys configured in pool")
            now = time.time()
            for i in range(len(self.api_keys)):
                idx = (self._current_idx + i) % len(self.api_keys)
                k = self.api_keys[idx]
                if self._cooldowns.get(k, 0) <= now:
                    self._current_idx = idx
                    return k, self._signers[k]
            earliest_key = min(self.api_keys, key=lambda k: self._cooldowns.get(k, 0))
            return earliest_key, self._signers[earliest_key]

    def mark_cooldown(self, key: str, duration_s: float = 60.0) -> None:
        with self._lock:
            self._cooldowns[key] = time.time() + duration_s
            if self.api_keys:
                self._current_idx = (self._current_idx + 1) % len(self.api_keys)

    def all_keys(self) -> list[str]:
        return list(self.api_keys)


def build_handshake_body(api_key: str) -> tuple[dict, str, str, str]:

    """Pure helper (used by tests): returns (body, key_id, secret, ts)."""
    key_id, secret = split_api_key(api_key)
    ts = str(int(time.time() * 1000))
    nonce = secrets.token_bytes(NONCE_BYTES).hex()
    sig = handshake_signature(secret, key_id, ts, nonce)
    return {"apiKey": api_key, "nonce": nonce, "sig": sig, "ts": ts}, key_id, secret, ts


def parse_quota(data: dict) -> list[dict]:
    """Normalize the monitor endpoint payload into compact windows."""
    windows = []
    for lim in data.get("limits", []):
        windows.append({
            "type": lim.get("type"),
            "unit": lim.get("unit"),
            "number": lim.get("number"),
            "used_pct": lim.get("percentage"),
            "remaining_pct": 100 - lim.get("percentage", 0),
            "remaining": lim.get("remaining"),
            "reset_at": lim.get("nextResetTime"),
        })
    return windows


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
