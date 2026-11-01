"""Query/document encoders.

Two backends share one interface:
  * SentenceTransformerEncoder - the real model, CPU-only, thread-limited.
  * HashEncoder - deterministic feature-hashing encoder. No model download,
    fast, lexical only. Used by unit tests and the "fast" tier, never to
    judge semantic quality.

The encoder fingerprint is the compatibility contract between an index and
the serving process: the builder stores the vectors of a few probe sentences
in the manifest, and the server refuses to become ready if its own encoder
does not reproduce them. That catches a wrong model, a wrong revision of the
right model, or a changed pre-processing step, even when the names match.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import re
import threading
import time
import urllib.parse
from functools import lru_cache
from typing import Protocol

import numpy as np

PROBE_TEXTS = [
    "cordless drill with two batteries",
    "outdoor led flood light with motion sensor",
    "kitchen faucet with pull-down sprayer",
]


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32, copy=False)


_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


@lru_cache(maxsize=200_000)
def _slot(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashEncoder:
    """Bag of hashed unigrams and bigrams, L2-normalised."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-bow-{dim}"

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            for tok in tokens:
                idx, sign = _slot(tok, self.dim)
                out[row, idx] += sign
            for a, b in zip(tokens, tokens[1:]):
                idx, sign = _slot(a + " " + b, self.dim)
                out[row, idx] += 0.5 * sign
        return _normalize(out)


class SentenceTransformerEncoder:
    """sentence-transformers on CPU with a fixed thread budget.

    torch defaults to one thread per *node* core, not per container CPU
    limit. On a 2-CPU pod on a 16-core node that means 16 threads fighting
    for 2 CPUs of CFS quota: throttling and tail-latency spikes. Pin it.
    """

    def __init__(self, model_path: str, model_name: str | None = None, threads: int | None = None, device: str = "cpu"):
        import torch

        if threads and threads > 0:
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # can only be set once per process
        from sentence_transformers import SentenceTransformer

        self._torch = torch
        self.model = SentenceTransformer(model_path, device=device)
        self.name = model_name or model_path
        self.dim = int(self.encode(["probe"]).shape[1])

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        with self._torch.inference_mode():
            vecs = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vecs, dtype=np.float32)


class LockedEncoder:
    """Lets only one thread at a time into a shared encoder.

    Hugging Face fast tokenizers keep mutable state (padding/truncation
    settings) and can fail with "RuntimeError: Already borrowed" when one
    tokenizer is used from several threads at once. Stage 6A lets you turn
    this lock off (ENCODER_LOCK=0) and see whether your versions hit it.
    """

    def __init__(self, inner: Encoder):
        self.inner, self.name, self.dim = inner, inner.name, inner.dim
        self._lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        with self._lock:
            return self.inner.encode(texts, batch_size)


class EncoderUnavailable(RuntimeError):
    pass


class RemoteEncoder:
    """Client for the separate embedding service (retrieval/encoder_server.py).

    One keep-alive HTTP connection per calling thread. /embed is idempotent,
    so a request that fails on a stale connection is retried once.
    Vectors travel as base64 float32, about 3x smaller than JSON floats.
    """

    def __init__(self, url: str, timeout: float = 10.0, wait_s: float = 120.0):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.prefix = u.hostname, u.port or 80, u.path.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()
        deadline = time.monotonic() + wait_s
        while True:  # the encoder service may still be starting
            try:
                info = self._request("GET", "/info")
                break
            except EncoderUnavailable:
                if time.monotonic() > deadline:
                    raise
                time.sleep(2)
        self.name, self.dim = info["model"], int(info["dim"])

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn = None

    def _request(self, method: str, path: str, body: bytes | None = None) -> dict:
        last = None
        for _ in (1, 2):
            conn = self._conn()
            try:
                conn.request(method, self.prefix + path, body=body, headers={"Content-Type": "application/json"} if body else {})
                resp = conn.getresponse()
                data = resp.read()
                if (resp.getheader("connection") or "").lower() == "close":
                    self._drop()
                if resp.status != 200:
                    raise EncoderUnavailable(f"encoder service {path} -> HTTP {resp.status}: {data[:200]!r}")
                return json.loads(data)
            except (http.client.HTTPException, OSError) as exc:
                self._drop()
                last = exc
        raise EncoderUnavailable(f"encoder service unreachable: {last}")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        doc = self._request("POST", "/embed", json.dumps({"texts": list(texts)}).encode())
        vecs = np.frombuffer(base64.b64decode(doc["vectors_b64"]), dtype="<f4")
        return vecs.reshape(len(texts), int(doc["dim"])).astype(np.float32, copy=True)


def load_encoder(backend: str, model_path: str = "", model_name: str = "", threads: int = 0, url: str = "", device: str = "cpu") -> Encoder:
    if backend == "hash":
        return HashEncoder()
    if backend == "st":
        return SentenceTransformerEncoder(model_path, model_name or None, threads, device)
    if backend == "remote":
        if not url:
            raise ValueError("ENCODER_BACKEND=remote needs ENCODER_URL")
        return RemoteEncoder(url)
    raise ValueError(f"unknown encoder backend {backend!r} (use 'st', 'hash' or 'remote')")


def fingerprint(encoder: Encoder) -> dict:
    vecs = encoder.encode(PROBE_TEXTS)
    return {
        "probe_texts": PROBE_TEXTS,
        "vectors": [[round(float(x), 6) for x in row] for row in vecs],
    }


def fingerprint_cosines(stored: dict, encoder: Encoder) -> list[float]:
    """Cosine between stored probe vectors and the live encoder's vectors."""
    ref = np.asarray(stored["vectors"], dtype=np.float32)
    live = encoder.encode(list(stored["probe_texts"]))
    if ref.shape != live.shape:
        return [-1.0] * len(ref)
    ref = _normalize(ref)
    return [float(x) for x in np.sum(ref * live, axis=1)]


def fingerprint_id(fp: dict) -> str:
    """Short stable id of an encoder, used to key the embedding cache."""
    blob = np.asarray(fp["vectors"], dtype=np.float32).round(4).tobytes()
    return hashlib.sha256(blob).hexdigest()[:12]
