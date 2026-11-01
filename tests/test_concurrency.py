"""Execution modes, admission control, priority shedding, the encoder
service, closed-loop load, and the background index worker."""
import asyncio
import json
import shutil
import subprocess
import sys
import threading

import pytest

from loadgen.loadgen import main as loadgen_main
from retrieval.batcher import MicroBatcher, Overloaded
from retrieval.config import Settings
from retrieval.encoder_server import create_app as create_encoder_app
from retrieval.encoders import RemoteEncoder
from retrieval.engine import SearchEngine, SearchItem
from tests.conftest import LiveServer
from tests.test_server_and_loadgen import call


@pytest.mark.parametrize("mode", ["threadpool", "semaphore", "inline", "batcher"])
def test_every_exec_mode_serves(make_server, mode):
    with make_server("v1", exec_mode=mode) as srv:
        status, _, body = call(srv.url, "POST", "/search", {"query": "led shop light 4 ft", "k": 3})
        assert status == 200 and len(json.loads(body)["results"]) == 3
        assert json.loads(call(srv.url, "GET", "/version")[2])["serving"]["exec_mode"] == mode


def test_semaphore_admission_control_sheds_and_releases_slots(make_server):
    with make_server("v1", exec_mode="semaphore", compute_slots=1, admission_timeout_ms=20) as srv:
        results = []

        def one():
            results.append(call(srv.url, "POST", "/search", {"query": "hammer", "simulate_compute_ms": 300})[0])

        threads = [threading.Thread(target=one) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(200) >= 1 and results.count(503) >= 1, results
        metrics = call(srv.url, "GET", "/metrics")[2].decode()
        assert 'retrieval_shed_total{reason="admission_timeout"}' in metrics
        assert "retrieval_compute_busy 0.0" in metrics, "slots must be released when threads finish"


def test_low_priority_is_shed_first():
    async def main():
        gate = threading.Event()
        b = MicroBatcher(lambda items: (gate.wait(2), items)[1], max_batch=1, max_wait_s=0, max_queue=4, low_shed_at=0.5)
        await b.start()
        first = asyncio.create_task(b.submit("busy", 5))
        await asyncio.sleep(0.05)
        queued = [asyncio.create_task(b.submit(i, 5)) for i in range(2)]  # queue now half full
        await asyncio.sleep(0.01)
        with pytest.raises(Overloaded):
            await b.submit("low", 5, low_priority=True)
        normal = asyncio.create_task(b.submit("normal", 5))  # still admitted
        gate.set()
        await asyncio.gather(first, normal, *queued)
        await b.stop()

    asyncio.run(main())


def test_metadata_modes_and_debug_guard(make_server):
    for mode in ("async", "thread", "blocking"):
        with make_server("v1", metadata_mode=mode, metadata_latency_ms=5) as srv:
            assert call(srv.url, "POST", "/search", {"query": "garden hose"})[0] == 200
    with make_server("v1", allow_debug=False) as srv:
        assert call(srv.url, "POST", "/search", {"query": "garden hose", "simulate_compute_ms": 50})[0] == 422


def test_remote_encoder_service(lab):
    enc_settings = Settings(encoder_backend="hash", pod_name="enc", warmup_requests=2)
    with LiveServer(create_encoder_app(settings=enc_settings)) as enc:
        remote = RemoteEncoder(enc.url, wait_s=5)
        assert remote.name == "hash-bow-384" and remote.dim == 384
        s = Settings(index_dir=str(lab / "artifacts/indexes/v1"), encoder_backend="remote", encoder_url=enc.url)
        eng = SearchEngine(s)
        report = eng.load()  # fingerprint and golden checks run through the service
        assert report["golden_overlap"] >= 0.9
        assert len(eng.process_batch([SearchItem("tape measure 25 ft", 4)])[0]["results"]) == 4


def test_closed_loop_and_slow_requests(make_server, tmp_path, lab):
    with make_server("v1") as srv:
        out = tmp_path / "closed"
        assert loadgen_main([
            "--target", srv.url, "--queries", str(lab / "data/queries.jsonl"), "--mode", "closed",
            "--concurrency", "1,4", "--step-duration", "2", "--warmup", "0.5", "--pause", "0.2",
            "--report-every", "10", "--out", str(out), "--slow-frac", "0.2", "--slow-ms", "30",
        ]) == 0
        rows = (out / "closed.csv").read_text().splitlines()
        assert len(rows) == 3
        s = json.loads((out / "summary.json").read_text())
        for row in s["closed_loop"]:
            assert abs(row["littles_law_users"] - row["concurrency"]) <= 0.5 * row["concurrency"] + 0.5
        assert s["latency_by_class"]["slow"]["count"] > 0


def test_index_worker_is_idempotent(lab, tmp_path):
    d = tmp_path / "lab"
    shutil.copytree(lab, d)
    base = [sys.executable, "-m", "retrieval.index_worker", "--once", "--encoder", "hash", "--baseline", "v2"]
    out = subprocess.run(base, cwd=d, capture_output=True, text=True, check=True).stdout
    assert "up to date at watermark 300" in out
    subprocess.run([sys.executable, "-m", "data.generate_changes", "--n", "40"], cwd=d, check=True, capture_output=True)
    out = subprocess.run(base + ["--min-events", "10"], cwd=d, capture_output=True, text=True, check=True).stdout
    assert "auto-w340 PASSED" in out
    latest = json.loads((d / "artifacts/channels/latest.json").read_text())
    assert latest["version"] == "auto-w340" and latest["baseline"] == "v2"
    again = subprocess.run(base + ["--min-events", "10"], cwd=d, capture_output=True, text=True, check=True).stdout
    assert "up to date at watermark 340" in again
