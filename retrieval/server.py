"""HTTP API (Starlette on uvicorn).

Endpoints
  POST /search        {"query": str, "k": 1..100, "category": str|null}
  GET  /health        liveness: process + worker alive
  GET  /ready         readiness: loaded, validated, warmed, not draining
  GET  /version       manifest summary of the loaded index
  GET  /metrics       Prometheus text format
  POST /admin/drain   loopback only; used by the preStop hook
  POST /admin/crash   loopback only; drill: exit the process without warning

Starlette is the ASGI core FastAPI is built on; on this hot path we only need
routing and pydantic validation, so the thinner layer keeps per-request
overhead and dependencies down.

Run:
  uvicorn retrieval.server:app --host 0.0.0.0 --port 8000 --no-access-log \
      --timeout-graceful-shutdown 20 --timeout-keep-alive 5
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from retrieval.batcher import MicroBatcher, Overloaded
from retrieval.config import Settings
from retrieval.engine import SearchEngine, SearchItem
from retrieval.metrics import METRICS

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("retrieval.server")

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=10, ge=1, le=100)
    category: str | None = Field(default=None, max_length=64)


class ServiceState:
    def __init__(self, settings: Settings | None, engine: SearchEngine | None):
        self.settings = settings
        self.engine = engine
        self.batcher: MicroBatcher | None = None
        self.ready = False
        self.draining = False
        self.started = time.time()
        self.load_report: dict = {}

    def headers(self) -> dict:
        h = {
            "X-Index-Version": self.engine.manifest.get("index_version", ""),
            "X-Served-By": self.settings.pod_name,
        }
        if self.draining:
            # Tell keep-alive clients to reconnect; their new connection lands
            # on a pod that is still in the Service endpoints.
            h["Connection"] = "close"
        return h


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None) -> Starlette:
    state = ServiceState(settings, engine)

    @asynccontextmanager
    async def lifespan(app):
        if state.settings is None:
            state.settings = Settings.from_env()
        s = state.settings
        if state.engine is None:
            state.engine = SearchEngine(s)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        log.info("loading index from %s", s.index_dir)
        state.load_report = await loop.run_in_executor(None, state.engine.load)
        state.batcher = MicroBatcher(state.engine.process_batch, s.max_batch, s.max_wait_ms / 1000.0, s.max_queue)
        await state.batcher.start()

        t1 = time.perf_counter()
        golden = [g["query"] for g in state.engine.manifest.get("golden", [])] or ["drill", "light", "faucet"]
        warm = [state.batcher.submit(SearchItem(golden[i % len(golden)], 10), 30.0) for i in range(s.warmup_requests)]
        await asyncio.gather(*warm)
        state.load_report["warmup"] = round(time.perf_counter() - t1, 3)
        METRICS.startup_seconds.set(state.load_report["warmup"], stage="warmup")
        METRICS.startup_seconds.set(round(time.perf_counter() - t0, 3), stage="total")

        state.ready = True
        METRICS.ready.set(1)
        log.info(
            "READY index=%s vectors=%s pod=%s startup=%s",
            state.engine.manifest["index_version"],
            state.engine.manifest["index"]["count"],
            s.pod_name,
            state.load_report,
        )
        yield
        state.ready = False
        METRICS.ready.set(0)
        await state.batcher.stop()
        log.info("stopped")

    async def health(request: Request) -> Response:
        if state.batcher is not None and not state.batcher.alive():
            return JSONResponse({"status": "worker_dead"}, status_code=500)
        return JSONResponse({"status": "alive"})

    async def ready(request: Request) -> Response:
        if state.draining:
            return JSONResponse({"status": "draining"}, status_code=503)
        if not state.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse(
            {"status": "ready", "index_version": state.engine.manifest["index_version"], "pod": state.settings.pod_name}
        )

    async def version(request: Request) -> Response:
        if not state.engine or not state.engine.manifest:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        m = state.engine.manifest
        return JSONResponse(
            {
                "index_version": m["index_version"],
                "created_at": m.get("created_at"),
                "encoder": {k: v for k, v in m["encoder"].items() if k != "fingerprint"},
                "index": m["index"],
                "source": m.get("source"),
                "quality": m.get("quality"),
                "build": state.settings.build_id,
                "pod": state.settings.pod_name,
                "startup": state.load_report,
                "draining": state.draining,
            }
        )

    async def search(request: Request) -> Response:
        t0 = time.perf_counter()
        if not state.ready:
            METRICS.requests.inc(code="503")
            return JSONResponse({"error": "not_ready"}, status_code=503)
        try:
            req = SearchRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            METRICS.requests.inc(code="422")
            return JSONResponse({"error": "invalid_request", "detail": json.loads(exc.json())}, status_code=422)

        METRICS.inflight.inc()
        try:
            result = await state.batcher.submit(
                SearchItem(req.query, req.k, req.category), state.settings.request_timeout_ms / 1000.0
            )
        except Overloaded:
            METRICS.requests.inc(code="503")
            return JSONResponse(
                {"error": "overloaded"}, status_code=503, headers={**state.headers(), "Retry-After": "1"}
            )
        except asyncio.TimeoutError:
            METRICS.requests.inc(code="504")
            return JSONResponse({"error": "timeout"}, status_code=504, headers=state.headers())
        except Exception:
            log.exception("search failed")
            METRICS.requests.inc(code="500")
            return JSONResponse({"error": "internal"}, status_code=500, headers=state.headers())
        finally:
            METRICS.inflight.dec()

        took = time.perf_counter() - t0
        METRICS.requests.inc(code="200")
        METRICS.request_seconds.observe(took)
        m = state.engine.manifest
        body = {
            "index_version": m["index_version"],
            "watermark": m.get("source", {}).get("watermark", 0),
            "served_by": state.settings.pod_name,
            "build": state.settings.build_id,
            "took_ms": round(took * 1000, 3),
            "results": result["results"],
        }
        return JSONResponse(body, headers=state.headers())

    async def metrics(request: Request) -> Response:
        if state.batcher:
            METRICS.queue_depth.set(state.batcher.depth)
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")

    def _loopback(request: Request) -> bool:
        return request.client is not None and request.client.host in LOOPBACK

    async def drain(request: Request) -> Response:
        if not _loopback(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        state.draining = True
        METRICS.ready.set(0)
        log.warning("DRAINING: readiness now fails, keep-alive connections will be closed")
        return JSONResponse({"status": "draining"})

    async def crash(request: Request) -> Response:
        if not _loopback(request) or not state.settings.allow_crash:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        log.error("CRASH drill requested: exiting without draining")
        asyncio.get_running_loop().call_later(0.05, os._exit, 1)
        return JSONResponse({"status": "crashing"})

    routes = [
        Route("/search", search, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/ready", ready, methods=["GET"]),
        Route("/version", version, methods=["GET"]),
        Route("/metrics", metrics, methods=["GET"]),
        Route("/admin/drain", drain, methods=["POST"]),
        Route("/admin/crash", crash, methods=["POST"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.service = state
    return app


app = create_app()
