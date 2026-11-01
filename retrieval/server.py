"""HTTP API (Starlette on uvicorn).

Endpoints
  POST /search        {"query": str, "k": 1..100, "category": str|null, "simulate_compute_ms": lab only}
                      header X-Priority: low  -> shed first under load (batcher mode)
  GET  /health        liveness: process + worker alive
  GET  /ready         readiness: loaded, validated, warmed, not draining
  GET  /version       manifest summary, serving settings, startup timings
  GET  /metrics       Prometheus text format
  POST /admin/drain   loopback only; used by the preStop hook
  POST /admin/crash   loopback only; drill: exit the process without warning

Starlette is the ASGI core FastAPI is built on; on this hot path we only need
routing and pydantic validation, so the thinner layer keeps per-request
overhead and dependencies down.

One request = three kinds of work, and they need different tools:
  receive/parse/respond   network I/O        -> the event loop (async)
  encode + HNSW search    CPU                -> a thread, chosen by EXEC_MODE
  metadata lookup         network I/O (sim.) -> awaited, never blocking the loop

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
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
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
EXEC_MODES = {"batcher", "threadpool", "semaphore", "inline"}
METADATA_MODES = {"none", "async", "thread", "blocking"}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    k: int = Field(default=10, ge=1, le=100)
    category: str | None = Field(default=None, max_length=64)
    simulate_compute_ms: float = Field(default=0.0, ge=0.0, le=5000.0)


class ServiceState:
    def __init__(self, settings: Settings | None, engine: SearchEngine | None):
        self.settings = settings
        self.engine = engine
        self.batcher: MicroBatcher | None = None
        self.slots: asyncio.Semaphore | None = None
        self.pool: ThreadPoolExecutor | None = None
        self.ready = False
        self.draining = False
        self.load_report: dict = {}
        self.identity = ""

    def headers(self) -> dict:
        h = {"X-Index-Version": self.engine.manifest.get("index_version", ""), "X-Served-By": self.identity}
        if self.draining:
            # Tell keep-alive clients to reconnect; their new connection lands
            # on a pod that is still in the Service endpoints.
            h["Connection"] = "close"
        return h


async def _loop_lag_monitor() -> None:
    loop = asyncio.get_running_loop()
    while True:
        t0 = loop.time()
        await asyncio.sleep(0.1)
        METRICS.loop_lag.observe(max(0.0, loop.time() - t0 - 0.1))


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None) -> Starlette:
    state = ServiceState(settings, engine)

    # ------------------------------------------------------------ compute strategies

    def _process_one(item: SearchItem) -> dict:
        METRICS.compute_busy.inc()
        try:
            return state.engine.process_batch([item])[0]
        finally:
            METRICS.compute_busy.dec()

    async def compute(item: SearchItem, low_priority: bool = False) -> dict:
        """Run encode + search with the strategy chosen by EXEC_MODE."""
        s = state.settings
        timeout = s.request_timeout_ms / 1000.0
        mode = s.exec_mode
        if mode == "batcher":
            # one worker thread, micro-batches, bounded queue -> Overloaded when full
            return await state.batcher.submit(item, timeout, low_priority)
        if mode == "inline":
            # ANTI-PATTERN: CPU work on the event loop. While it runs, this
            # process cannot accept connections, parse requests, answer /ready,
            # or send responses for anyone. Timeouts cannot interrupt it.
            return _process_one(item)
        if mode == "threadpool":
            # What a plain `def` endpoint gets: Starlette's thread pool (anyio,
            # 40 threads by default). No admission control: under overload,
            # work piles up in threads that all compete for the same CPUs.
            task = asyncio.ensure_future(run_in_threadpool(_process_one, item))
            # shield: on timeout we answer 504 now; the thread keeps running
            # until it finishes, because Python cannot kill a running thread.
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        if mode == "semaphore":
            # Bounded concurrency + admission control: at most COMPUTE_SLOTS
            # computations; a request that cannot get a slot within
            # ADMISSION_TIMEOUT_MS is rejected quickly instead of queueing.
            try:
                await asyncio.wait_for(state.slots.acquire(), s.admission_timeout_ms / 1000.0)
            except asyncio.TimeoutError:
                METRICS.shed.inc(reason="admission_timeout")
                raise Overloaded() from None
            fut = asyncio.get_running_loop().run_in_executor(state.pool, _process_one, item)
            # Release the slot when the THREAD finishes, not when the caller
            # stops waiting. Releasing on timeout would let more computations
            # run than there are slots: the classic way this pattern leaks.
            fut.add_done_callback(lambda _f: state.slots.release())
            return await asyncio.wait_for(asyncio.shield(fut), timeout)
        raise RuntimeError(f"unknown EXEC_MODE {mode!r}")

    async def lookup_metadata(n: int) -> None:
        """Simulated call to a product-metadata store after search."""
        s = state.settings
        delay = s.metadata_latency_ms / 1000.0
        if s.metadata_mode == "none" or delay <= 0:
            return
        if s.metadata_mode == "async":
            await asyncio.sleep(delay)  # an async client: the loop serves others meanwhile
        elif s.metadata_mode == "thread":
            await asyncio.to_thread(time.sleep, delay)  # a sync client moved off the loop (default executor size!)
        elif s.metadata_mode == "blocking":
            time.sleep(delay)  # ANTI-PATTERN: a sync client called from async code blocks everyone

    # ------------------------------------------------------------ lifecycle

    @asynccontextmanager
    async def lifespan(app):
        if state.settings is None:
            state.settings = Settings.from_env()
        s = state.settings
        if s.exec_mode not in EXEC_MODES:
            raise RuntimeError(f"EXEC_MODE must be one of {sorted(EXEC_MODES)}")
        if s.metadata_mode not in METADATA_MODES:
            raise RuntimeError(f"METADATA_MODE must be one of {sorted(METADATA_MODES)}")
        state.identity = f"{s.pod_name}/{os.getpid()}"
        if state.engine is None:
            state.engine = SearchEngine(s)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        log.info("loading index from %s (exec_mode=%s)", s.index_dir, s.exec_mode)
        state.load_report = await loop.run_in_executor(None, state.engine.load)
        state.batcher = MicroBatcher(
            state.engine.process_batch, s.max_batch, s.max_wait_ms / 1000.0, s.max_queue, s.low_priority_shed_at
        )
        await state.batcher.start()
        state.slots = asyncio.Semaphore(s.compute_slots)
        state.pool = ThreadPoolExecutor(max_workers=s.compute_slots, thread_name_prefix="compute")
        lag_task = asyncio.create_task(_loop_lag_monitor())

        t1 = time.perf_counter()
        golden = [g["query"] for g in state.engine.manifest.get("golden", [])] or ["drill", "light", "faucet"]
        warm = [compute(SearchItem(golden[i % len(golden)], 10)) for i in range(s.warmup_requests)]
        await asyncio.gather(*warm, return_exceptions=True)
        state.load_report["warmup"] = round(time.perf_counter() - t1, 3)
        METRICS.startup_seconds.set(state.load_report["warmup"], stage="warmup")
        METRICS.startup_seconds.set(round(time.perf_counter() - t0, 3), stage="total")

        state.ready = True
        METRICS.ready.set(1)
        log.info(
            "READY index=%s vectors=%s worker=%s exec_mode=%s startup=%s",
            state.engine.manifest["index_version"],
            state.engine.manifest["index"]["count"],
            state.identity,
            s.exec_mode,
            state.load_report,
        )
        yield
        state.ready = False
        METRICS.ready.set(0)
        lag_task.cancel()
        await state.batcher.stop()
        state.pool.shutdown(wait=False)
        log.info("stopped")

    # ------------------------------------------------------------ endpoints

    async def health(request: Request) -> Response:
        if state.batcher is not None and not state.batcher.alive():
            return JSONResponse({"status": "worker_dead"}, status_code=500)
        return JSONResponse({"status": "alive"})

    async def ready(request: Request) -> Response:
        if state.draining:
            return JSONResponse({"status": "draining"}, status_code=503)
        if not state.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready", "index_version": state.engine.manifest["index_version"], "pod": state.identity})

    async def version(request: Request) -> Response:
        if not state.engine or not state.engine.manifest:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        m, s = state.engine.manifest, state.settings
        return JSONResponse(
            {
                "index_version": m["index_version"],
                "created_at": m.get("created_at"),
                "encoder": {k: v for k, v in m["encoder"].items() if k != "fingerprint"},
                "index": m["index"],
                "source": m.get("source"),
                "quality": m.get("quality"),
                "build": s.build_id,
                "pod": state.identity,
                "serving": {
                    "exec_mode": s.exec_mode,
                    "compute_slots": s.compute_slots,
                    "max_batch": s.max_batch,
                    "max_wait_ms": s.max_wait_ms,
                    "max_queue": s.max_queue,
                    "encoder_backend": s.encoder_backend,
                    "metadata_mode": s.metadata_mode,
                    "metadata_latency_ms": s.metadata_latency_ms,
                },
                "startup": state.load_report,
                "draining": state.draining,
            }
        )

    async def search(request: Request) -> Response:
        t0 = time.perf_counter()
        s = state.settings
        if not state.ready:
            METRICS.requests.inc(code="503")
            return JSONResponse({"error": "not_ready"}, status_code=503)
        try:
            req = SearchRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            METRICS.requests.inc(code="422")
            return JSONResponse({"error": "invalid_request", "detail": json.loads(exc.json())}, status_code=422)
        if len(req.query) > s.max_query_chars or (req.simulate_compute_ms and not s.allow_debug):
            METRICS.requests.inc(code="422")
            return JSONResponse({"error": "invalid_request", "detail": "query too long or debug fields disabled"}, status_code=422)

        METRICS.inflight.inc()
        try:
            low = request.headers.get("x-priority", "").lower() == "low"
            result = await compute(SearchItem(req.query, req.k, req.category, req.simulate_compute_ms), low)
            await lookup_metadata(len(result["results"]))
        except Overloaded:
            METRICS.requests.inc(code="503")
            return JSONResponse({"error": "overloaded"}, status_code=503, headers={**state.headers(), "Retry-After": "1"})
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
            "served_by": state.identity,
            "build": s.build_id,
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
