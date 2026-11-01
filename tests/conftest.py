"""Shared fixtures: a tiny lab (2,000 products) built with the hash encoder.

Runs without the transformer model or network, in seconds. It still exercises
the real code paths: change log, embedding cache, hnswlib build, checksums,
manifest, startup validation, batching, HTTP server, load generator.
"""
from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(module: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", module, *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture(scope="session")
def lab(tmp_path_factory) -> Path:
    """A working directory with data/, artifacts/indexes/v1 and v2 built."""
    d = tmp_path_factory.mktemp("lab")
    for pkg in ("retrieval", "data", "loadgen"):
        shutil.copytree(ROOT / pkg, d / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.jsonl"))
    run("data.generate_catalog", "--n", "2000", cwd=d)
    run("data.generate_queries", "--n", "200", cwd=d)
    (d / "data" / "changes.jsonl").write_text("")
    run("retrieval.build_index", "--version", "v1", "--encoder", "hash", "--eval-queries", "100", cwd=d)
    run("data.generate_changes", "--n", "300", "--seed", "3", cwd=d)
    run("retrieval.build_index", "--version", "v2", "--encoder", "hash", "--eval-queries", "100", cwd=d)
    return d


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """uvicorn in a background thread, serving a real app on a free port."""

    def __init__(self, app):
        import uvicorn

        self.port = free_port()
        self.config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", timeout_graceful_shutdown=5)
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=15)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def make_server(lab):
    from retrieval.config import Settings
    from retrieval.server import create_app

    def factory(version: str = "v1", **overrides):
        s = Settings(index_dir=str(lab / "artifacts/indexes" / version), encoder_backend="hash", pod_name="test-pod", warmup_requests=4, **overrides)
        return LiveServer(create_app(settings=s))

    return factory


def arun(coro):
    return asyncio.run(coro)
