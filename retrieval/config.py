"""Runtime settings, read once from environment variables.

Every knob that matters for latency, throughput or safety is here so an
experiment changes one env var in the Deployment, not the code.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Where the fetched, verified index version lives on local disk.
    index_dir: str = "artifacts/indexes/v1"
    verify_checksums: bool = True

    # Encoder
    encoder_backend: str = "st"  # "st" = sentence-transformers, "hash" = lexical test encoder
    model_path: str = "models/all-MiniLM-L6-v2"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    torch_threads: int = 2

    # Micro-batching and backpressure
    max_batch: int = 32
    max_wait_ms: float = 2.0
    max_queue: int = 256
    request_timeout_ms: float = 1000.0
    query_cache_size: int = 0  # LRU of query text -> embedding; 0 disables

    # ANN
    ef: int = 0  # 0 = use the manifest default
    filter_overfetch: int = 5

    # Startup validation
    golden_min_overlap: float = 0.9
    fingerprint_min_cosine: float = 0.999
    warmup_requests: int = 64

    # Identity and drills
    pod_name: str = ""
    build_id: str = "dev"
    allow_crash: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            index_dir=_env("INDEX_DIR", cls.index_dir),
            verify_checksums=_bool("VERIFY_CHECKSUMS", True),
            encoder_backend=_env("ENCODER_BACKEND", cls.encoder_backend),
            model_path=_env("MODEL_PATH", cls.model_path),
            model_name=_env("MODEL_NAME", cls.model_name),
            torch_threads=int(_env("TORCH_THREADS", str(cls.torch_threads))),
            max_batch=int(_env("MAX_BATCH", str(cls.max_batch))),
            max_wait_ms=float(_env("MAX_WAIT_MS", str(cls.max_wait_ms))),
            max_queue=int(_env("MAX_QUEUE", str(cls.max_queue))),
            request_timeout_ms=float(_env("REQUEST_TIMEOUT_MS", str(cls.request_timeout_ms))),
            query_cache_size=int(_env("QUERY_CACHE_SIZE", str(cls.query_cache_size))),
            ef=int(_env("EF", str(cls.ef))),
            filter_overfetch=int(_env("FILTER_OVERFETCH", str(cls.filter_overfetch))),
            golden_min_overlap=float(_env("GOLDEN_MIN_OVERLAP", str(cls.golden_min_overlap))),
            fingerprint_min_cosine=float(_env("FINGERPRINT_MIN_COSINE", str(cls.fingerprint_min_cosine))),
            warmup_requests=int(_env("WARMUP_REQUESTS", str(cls.warmup_requests))),
            pod_name=_env("POD_NAME", socket.gethostname()),
            build_id=_env("BUILD_ID", cls.build_id),
            allow_crash=_bool("ALLOW_CRASH", True),
        )
