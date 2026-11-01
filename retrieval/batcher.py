"""Async micro-batcher with bounded queue (backpressure) and deadlines.

Why: a transformer encoder is far cheaper per query in a batch of 16 than 16
times alone, and one worker thread avoids torch threads from concurrent
requests fighting over the pod's CPU quota.

Behaviour:
  * submit() puts a request on a bounded queue. Full queue -> Overloaded
    immediately (the API turns it into 503 + Retry-After). Shedding fast is
    what keeps p99 bounded when offered load exceeds capacity; an unbounded
    queue just converts overload into timeouts for everyone.
  * The worker takes whatever is queued (up to max_batch). If the batch is
    small it waits at most max_wait for more, then runs encode+search for
    the whole batch in its single executor thread.
  * Requests whose caller already gave up (deadline passed) are dropped
    before doing work for them.
  * Priority-aware shedding: low-priority requests (for example a background
    recommendation widget) are refused once the queue is `low_shed_at` full,
    so the remaining room is kept for interactive search.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from retrieval.metrics import METRICS

log = logging.getLogger("retrieval.batcher")


class Overloaded(Exception):
    pass


class MicroBatcher:
    def __init__(
        self,
        process: Callable[[list], list],
        max_batch: int = 32,
        max_wait_s: float = 0.002,
        max_queue: int = 256,
        low_shed_at: float = 0.5,
    ):
        self.process = process
        self.low_limit = max(1, int(max_queue * low_shed_at))
        self.max_batch = max_batch
        self.max_wait_s = max_wait_s
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-worker")
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="micro-batcher")

    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def submit(self, item: Any, timeout_s: float, low_priority: bool = False) -> Any:
        if low_priority and self.queue.qsize() >= self.low_limit:
            METRICS.shed.inc(reason="low_priority")
            raise Overloaded()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        enqueued = loop.time()
        try:
            self.queue.put_nowait((item, fut, enqueued, enqueued + timeout_s))
        except asyncio.QueueFull:
            METRICS.shed.inc(reason="queue_full")
            raise Overloaded() from None
        METRICS.queue_depth.set(self.queue.qsize())
        try:
            return await asyncio.wait_for(fut, timeout_s)
        except asyncio.TimeoutError:
            METRICS.shed.inc(reason="timeout")
            raise

    async def _collect(self) -> list:
        loop = asyncio.get_running_loop()
        batch = [await self.queue.get()]
        deadline = loop.time() + self.max_wait_s
        while len(batch) < self.max_batch:
            try:
                batch.append(self.queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.0005))
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            METRICS.queue_depth.set(self.queue.qsize())
            now = loop.time()
            live = []
            for item, fut, enqueued, deadline in batch:
                if fut.done() or now > deadline:
                    if not fut.done():
                        fut.cancel()
                    METRICS.shed.inc(reason="expired_in_queue")
                    continue
                METRICS.queue_wait.observe(now - enqueued)
                live.append((item, fut))
            if not live:
                continue
            try:
                results = await loop.run_in_executor(self.executor, self.process, [i for i, _ in live])
            except Exception as exc:  # one bad batch must not kill the worker
                log.exception("batch of %d failed", len(live))
                for _, fut in live:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), res in zip(live, results):
                if not fut.done():
                    fut.set_result(res)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        while not self.queue.empty():
            _, fut, _, _ = self.queue.get_nowait()
            if not fut.done():
                fut.set_exception(RuntimeError("shutting down"))
        self.executor.shutdown(wait=False)

