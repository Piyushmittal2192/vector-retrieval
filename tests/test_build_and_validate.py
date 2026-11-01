import json
import shutil
import subprocess
import sys

import pytest

from retrieval import artifacts
from retrieval.config import Settings
from retrieval.encoders import HashEncoder
from retrieval.engine import CompatibilityError, SearchEngine, SearchItem


def test_manifest_lists_verified_files(lab):
    vdir = lab / "artifacts/indexes/v1"
    m = artifacts.read_manifest(vdir)
    assert set(m["files"]) == {"index.bin", "docs.json"}
    artifacts.verify_files(vdir, m)
    assert m["quality"]["recall_at_10"] > 0.9
    assert len(m["golden"]) == 20


def test_versions_are_immutable(lab):
    r = subprocess.run([sys.executable, "-m", "retrieval.build_index", "--version", "v1", "--encoder", "hash"], cwd=lab, capture_output=True, text=True)
    assert r.returncode != 0 and "immutable" in (r.stdout + r.stderr)


def test_v2_reuses_embeddings_and_tracks_watermark(lab):
    m = artifacts.read_manifest(lab / "artifacts/indexes/v2")
    assert m["source"]["watermark"] == 300
    assert m["build"]["embeddings_reused"] > 1500


def test_validation_gate_passes_against_baseline(lab):
    r = subprocess.run(
        [sys.executable, "-m", "retrieval.validate_index", "--version", "v2", "--baseline", "v1", "--encoder", "hash", "--max-ndcg-drop", "0.2"],
        cwd=lab, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((lab / "artifacts/validation/v2.json").read_text())
    assert report["passed"] and report["summary"]["baseline"] == "v1"


def test_corrupt_artifact_is_rejected(lab, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(lab / "artifacts/indexes/v1", bad)
    with open(bad / "index.bin", "r+b") as f:
        f.seek(100)
        b = f.read(1)
        f.seek(100)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(artifacts.ArtifactError, match="sha256"):
        artifacts.verify_files(bad, artifacts.read_manifest(bad))


def test_wrong_encoder_is_rejected(lab):
    s = Settings(index_dir=str(lab / "artifacts/indexes/v1"), encoder_backend="hash")
    other = HashEncoder()
    other.encode = lambda texts, batch_size=64: HashEncoder().encode([t[::-1] for t in texts])  # same name, different "weights"
    with pytest.raises(CompatibilityError, match="fingerprint"):
        SearchEngine(s, encoder=other).load()


def test_engine_search_and_category_filter(lab):
    eng = SearchEngine(Settings(index_dir=str(lab / "artifacts/indexes/v1"), encoder_backend="hash"))
    report = eng.load()
    assert report["golden_overlap"] >= 0.9
    out = eng.process_batch([SearchItem("cordless drill", 5), SearchItem("string lights", 5, "lighting"), SearchItem("cordless drill", 3)])
    assert len(out[0]["results"]) == 5 and len(out[2]["results"]) == 3
    assert all(r["category"] == "lighting" for r in out[1]["results"])


def test_fetch_from_file_and_http(lab, tmp_path):
    import functools
    import http.server
    import threading

    dest = tmp_path / "dest"
    p = artifacts.fetch_version(str(lab / "artifacts/indexes"), "v1", dest)
    assert (p / "manifest.json").exists()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(lab / "artifacts/indexes"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        p2 = artifacts.fetch_version(f"http://127.0.0.1:{httpd.server_port}", "v2", tmp_path / "http")
        artifacts.verify_files(p2, artifacts.read_manifest(p2))
    finally:
        httpd.shutdown()
