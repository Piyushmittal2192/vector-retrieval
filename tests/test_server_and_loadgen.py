import http.client
import json
import urllib.parse

from loadgen import stats
from loadgen.loadgen import main as loadgen_main


def call(url: str, method: str, path: str, body: dict | None = None):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.request(method, path, body=json.dumps(body) if body is not None else None, headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    headers = {k.lower(): v for k, v in r.getheaders()}
    conn.close()
    return r.status, headers, data


def test_endpoints(make_server):
    with make_server("v1") as srv:
        assert call(srv.url, "GET", "/health")[0] == 200
        status, _, body = call(srv.url, "GET", "/ready")
        assert status == 200 and json.loads(body)["index_version"] == "v1"
        status, headers, body = call(srv.url, "POST", "/search", {"query": "cordless drill 20v", "k": 5})
        doc = json.loads(body)
        assert status == 200 and len(doc["results"]) == 5
        assert headers["x-index-version"] == "v1" and headers["x-served-by"].startswith("test-pod/")
        assert call(srv.url, "POST", "/search", {"query": "", "k": 5})[0] == 422
        assert call(srv.url, "POST", "/search", {"query": "x", "k": 1000})[0] == 422
        metrics = call(srv.url, "GET", "/metrics")[2].decode()
        for name in ("retrieval_requests_total", "retrieval_batch_size_bucket", "retrieval_index_info", "process_resident_memory_bytes"):
            assert name in metrics
        version = json.loads(call(srv.url, "GET", "/version")[2])
        assert version["source"]["watermark"] == 0 and "fingerprint" not in version["encoder"]


def test_drain_fails_readiness_and_closes_keepalive(make_server):
    with make_server("v1") as srv:
        assert call(srv.url, "POST", "/admin/drain")[0] == 200  # from loopback
        assert call(srv.url, "GET", "/ready")[0] == 503
        status, headers, _ = call(srv.url, "POST", "/search", {"query": "hammer"})
        assert status == 200, "draining pods still serve requests that reach them"
        assert headers.get("connection", "").lower() == "close"


def test_loadgen_end_to_end(make_server, tmp_path, lab):
    with make_server("v2") as srv:
        out = tmp_path / "run"
        rc = loadgen_main([
            "--target", srv.url, "--queries", str(lab / "data/queries.jsonl"), "--rate", "60", "--duration", "3",
            "--connections", "8", "--out", str(out), "--report-every", "1", "--category-frac", "0.2",
        ])
        assert rc == 0
        s = json.loads((out / "summary.json").read_text())
        assert s["error_rate"] == 0 and s["ok"] > 100
        assert s["versions"] == {"v2": s["ok"]}
        assert s["consistency"]["inconsistent"] == 0
        assert (out / "timeline.csv").exists() and (tmp_path / "runs.csv").exists()


def test_stats_detect_mixed_versions_and_inconsistency():
    recs = []
    for t in range(10):
        ver = "v1" if t < 5 else "v2"
        lagging = "v1" if t == 5 else ver  # pod-b switches one second later
        recs.append((t + 0.1, 0.01, 0.01, 200, None, ver, "pod-a", "1.0", ("q1", 10, None), 111 if ver == "v1" else 222))
        recs.append((t + 0.2, 0.02, 0.02, 200, None, lagging, "pod-b", "1.0", ("q1", 10, None), 111 if lagging == "v1" else 222))
    recs.append((9.5, 0.0, 0.0, 0, "connect_refused", "", "", "", None, 0))
    s = stats.summarize(recs, 10, slo_p99_ms=100, slo_error_rate=0.1)
    assert s["mixed_version_seconds"] == 1
    assert s["errors"] == {"connect_refused": 1}
    assert s["consistency"]["inconsistent"] == 0
    recs.append((9.9, 0.02, 0.02, 200, None, "v2", "pod-c", "1.0", ("q1", 10, None), 999))
    assert stats.summarize(recs, 10, 100, 0.1)["consistency"]["inconsistent"] == 1
    assert stats.percentile([1, 2, 3, 4], 50) == 2 and stats.percentile([1, 2, 3, 4], 99) == 4
