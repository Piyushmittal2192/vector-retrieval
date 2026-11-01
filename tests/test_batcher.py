import asyncio
import threading
import time

import pytest

from retrieval.batcher import MicroBatcher, Overloaded


def test_requests_are_batched():
    sizes = []

    def process(items):
        sizes.append(len(items))
        time.sleep(0.01)
        return [i * 2 for i in items]

    async def main():
        b = MicroBatcher(process, max_batch=16, max_wait_s=0.005, max_queue=100)
        await b.start()
        results = await asyncio.gather(*(b.submit(i, 2.0) for i in range(50)))
        await b.stop()
        return results

    assert asyncio.run(main()) == [i * 2 for i in range(50)]
    assert max(sizes) > 1 and max(sizes) <= 16


def test_full_queue_sheds_immediately():
    gate = threading.Event()

    def process(items):
        gate.wait(2)
        return items

    async def main():
        b = MicroBatcher(process, max_batch=1, max_wait_s=0, max_queue=2)
        await b.start()
        first = asyncio.create_task(b.submit(0, 5))
        await asyncio.sleep(0.05)  # worker now blocked on item 0
        queued = [asyncio.create_task(b.submit(i, 5)) for i in (1, 2)]
        await asyncio.sleep(0.01)
        with pytest.raises(Overloaded):
            await b.submit(3, 5)
        gate.set()
        await asyncio.gather(first, *queued)
        await b.stop()

    asyncio.run(main())


def test_deadline_and_errors():
    def slow(items):
        time.sleep(0.2)
        return items

    def boom(items):
        raise ValueError("bad batch")

    async def main():
        b = MicroBatcher(slow, max_batch=4, max_wait_s=0, max_queue=10)
        await b.start()
        with pytest.raises(asyncio.TimeoutError):
            await b.submit(1, 0.05)
        await b.stop()
        b2 = MicroBatcher(boom, max_batch=4, max_wait_s=0, max_queue=10)
        await b2.start()
        with pytest.raises(ValueError):
            await b2.submit(1, 1)
        assert b2.alive(), "a failing batch must not kill the worker"
        await b2.stop()

    asyncio.run(main())
