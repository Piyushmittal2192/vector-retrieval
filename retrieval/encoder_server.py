"""Separate embedding service (Stage 9A).

Moves the model out of the retrieval pods so the two halves of a search can
scale on their own bottlenecks: encoder pods on CPU (or GPU) for inference,
retrieval pods on RAM for the index. Retrieval pods call it with
ENCODER_BACKEND=remote ENCODER_URL=http://encoder.

Endpoints
  POST /embed   {"texts": [str, ...]} -> {"model", "dim", "vectors_b64"}  (float32, little endian)
  GET  /info    {"model", "dim"}
  GET  /health, /ready, /metrics, POST /admin/drain   (same contract as the retrieval service)

Requests from many retrieval pods are micro-batched together here, so the
model sees bigger batches than any single retrieval pod could form.

Run:
  ENCODER_BACKEND=st uvicorn retrieval.encoder_server:app --host 0.0.0.0 --port 8001 --no-access-log
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from retrieval.batcher import MicroBatcher, Overloaded
from retrieval.config import Settings
from retrieval.encoders import load_encoder
from retrieval.metrics import METRICS

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("retrieval.encoder_server")
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


def create_app(settings: Settings | None = None, encoder=None) -> Starlette:
    st: dict = {"settings": settings, "encoder": encoder, "batcher": None, "ready": False, "draining": False, "identity": ""}

    def process(texts: list[str]) -> list[np.ndarray]:
        t0 = time.perf_counter()
        vecs = st["encoder"].encode(texts, batch_size=max(1, len(texts)))
        METRICS.encode_seconds.observe(time.perf_counter() - t0)
        METRICS.batch_size.observe(len(texts))
        return list(vecs)

    @asynccontextmanager
    async def lifespan(app):
        s = st["settings"] = st["settings"] or Settings.from_env()
        if s.encoder_backend == "remote":
            raise RuntimeError("the encoder service needs a local model: ENCODER_BACKEND=st or hash")
        st["identity"] = f"{s.pod_name}/{os.getpid()}"
        loop = asyncio.get_running_loop()
        if st["encoder"] is None:
            st["encoder"] = await loop.run_in_executor(
                None, load_encoder, s.encoder_backend, s.model_path, s.model_name, s.torch_threads, "", s.device
            )
            if s.encoder_backend == "st":
                st["encoder"].name = s.model_name
        st["batcher"] = MicroBatcher(process, s.max_batch, s.max_wait_ms / 1000.0, s.max_queue)
        await st["batcher"].start()
        await asyncio.gather(*(st["batcher"].submit("warm up the encoder", 30.0) for _ in range(s.warmup_requests)))
        st["ready"] = True
        METRICS.ready.set(1)
        log.info("READY encoder=%s dim=%s device=%s max_batch=%s worker=%s", st["encoder"].name, st["encoder"].dim, s.device, s.max_batch, st["identity"])
        yield
        st["ready"] = False
        await st["batcher"].stop()

    def headers() -> dict:
        h = {"X-Served-By": st["identity"]}
        if st["draining"]:
            h["Connection"] = "close"
        return h

    async def embed(request: Request) -> Response:
        if not st["ready"]:
            return JSONResponse({"error": "not_ready"}, status_code=503)
        try:
            req = EmbedRequest.model_validate_json(await request.body())
        except ValidationError:
            METRICS.requests.inc(code="422")
            return JSONResponse({"error": "invalid_request"}, status_code=422)
        timeout = st["settings"].request_timeout_ms / 1000.0
        t0 = time.perf_counter()
        try:
            vecs = await asyncio.gather(*(st["batcher"].submit(t, timeout) for t in req.texts))
        except Overloaded:
            METRICS.requests.inc(code="503")
            return JSONResponse({"error": "overloaded"}, status_code=503, headers={**headers(), "Retry-After": "1"})
        except asyncio.TimeoutError:
            METRICS.requests.inc(code="504")
            return JSONResponse({"error": "timeout"}, status_code=504, headers=headers())
        METRICS.requests.inc(code="200")
        METRICS.request_seconds.observe(time.perf_counter() - t0)
        blob = np.ascontiguousarray(np.vstack(vecs), dtype="<f4").tobytes()
        return JSONResponse(
            {"model": st["encoder"].name, "dim": st["encoder"].dim, "vectors_b64": base64.b64encode(blob).decode()},
            headers=headers(),
        )

    async def info(request: Request) -> Response:
        if not st["ready"]:
            return JSONResponse({"error": "not_ready"}, status_code=503)
        return JSONResponse({"model": st["encoder"].name, "dim": st["encoder"].dim, "worker": st["identity"]})

    async def health(request: Request) -> Response:
        b = st["batcher"]
        if b is not None and not b.alive():
            return JSONResponse({"status": "worker_dead"}, status_code=500)
        return JSONResponse({"status": "alive"})

    async def ready(request: Request) -> Response:
        if st["draining"] or not st["ready"]:
            return JSONResponse({"status": "draining" if st["draining"] else "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    async def metrics(request: Request) -> Response:
        if st["batcher"]:
            METRICS.queue_depth.set(st["batcher"].depth)
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")

    async def drain(request: Request) -> Response:
        if request.client is None or request.client.host not in LOOPBACK:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        st["draining"] = True
        METRICS.ready.set(0)
        return JSONResponse({"status": "draining"})

    async def crash(request: Request) -> Response:
        if request.client is None or request.client.host not in LOOPBACK or not st["settings"].allow_crash:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        log.error("CRASH drill requested")
        asyncio.get_running_loop().call_later(0.05, os._exit, 1)
        return JSONResponse({"status": "crashing"})

    routes = [
        Route("/admin/crash", crash, methods=["POST"]),
        Route("/embed", embed, methods=["POST"]),
        Route("/info", info, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/ready", ready, methods=["GET"]),
        Route("/metrics", metrics, methods=["GET"]),
        Route("/admin/drain", drain, methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


app = create_app()
