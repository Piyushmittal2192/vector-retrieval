"""Immutable, checksummed index artifacts.

Layout of one version (never modified after publish):

    <root>/<version>/
        index.bin       hnswlib graph + vectors
        docs.json       label -> id, title, category
        manifest.json   written LAST; lists every other file with sha256 + size

Publishing = build in a temp dir next to the target, write the manifest last,
then one atomic rename. A reader that sees manifest.json therefore sees a
complete version. A version that already exists is never overwritten.

Fetching supports file://, plain paths, http(s):// and gs:// (GCS JSON API
with the GKE metadata-server token, so Workload Identity works without extra
libraries). Every file is verified against the manifest before the version
directory appears at the destination.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST = "manifest.json"
CHUNK = 8 * 1024 * 1024


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_json_atomic(path: Path, obj, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o644)  # mkstemp creates 0600; the artifact server must be able to read it
    os.replace(tmp, path)


def read_manifest(version_dir: Path) -> dict:
    path = Path(version_dir) / MANIFEST
    if not path.exists():
        raise ArtifactError(f"no {MANIFEST} in {version_dir} (incomplete or missing version)")
    with open(path) as f:
        manifest = json.load(f)
    for key in ("index_version", "encoder", "index", "files"):
        if key not in manifest:
            raise ArtifactError(f"manifest missing required key {key!r}")
    return manifest


def verify_files(version_dir: Path, manifest: dict) -> None:
    version_dir = Path(version_dir)
    for name, meta in manifest["files"].items():
        path = version_dir / name
        if not path.exists():
            raise ArtifactError(f"{name}: missing")
        size = path.stat().st_size
        if size != meta["bytes"]:
            raise ArtifactError(f"{name}: size {size} != manifest {meta['bytes']}")
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            raise ArtifactError(f"{name}: sha256 mismatch (corrupt or tampered artifact)")


def publish(tmp_dir: Path, final_dir: Path) -> None:
    final_dir = Path(final_dir)
    if final_dir.exists():
        raise ArtifactError(f"{final_dir} already exists: versions are immutable, pick a new version name")
    with open(Path(tmp_dir) / MANIFEST) as f:
        json.load(f)  # must exist and parse before we publish
    os.chmod(tmp_dir, 0o755)  # mkdtemp creates 0700; nginx (another uid) must traverse it
    for child in Path(tmp_dir).iterdir():
        os.chmod(child, 0o644)
    os.rename(tmp_dir, final_dir)


# ---------------------------------------------------------------- fetching


def _gcs_token() -> str:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)["access_token"]


def _open(uri: str, timeout: float = 60):
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in ("", "file"):
        return open(parsed.path if parsed.scheme else uri, "rb")
    if parsed.scheme in ("http", "https"):
        return urllib.request.urlopen(uri, timeout=timeout)
    if parsed.scheme == "gs":
        obj = urllib.parse.quote(parsed.path.lstrip("/"), safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{parsed.netloc}/o/{obj}?alt=media"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_gcs_token()}"})
        return urllib.request.urlopen(req, timeout=timeout)
    raise ArtifactError(f"unsupported URI scheme in {uri!r}")


def _download(uri: str, dest: Path, expected: dict | None, attempts: int = 4) -> None:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            h = hashlib.sha256()
            with _open(uri) as src, open(dest, "wb") as out:
                while block := src.read(CHUNK):
                    h.update(block)
                    out.write(block)
            if expected:
                if dest.stat().st_size != expected["bytes"]:
                    raise ArtifactError(f"{uri}: size {dest.stat().st_size} != {expected['bytes']}")
                if h.hexdigest() != expected["sha256"]:
                    raise ArtifactError(f"{uri}: sha256 mismatch")
            return
        except ArtifactError:
            raise  # corruption is not transient; do not retry it
        except Exception as exc:  # network errors: retry with backoff
            last = exc
            time.sleep(min(2**attempt, 10))
    raise ArtifactError(f"{uri}: download failed after {attempts} attempts: {last}")


def fetch_version(source: str, version: str, dest_root: Path) -> Path:
    """Download and verify <source>/<version>/ into <dest_root>/<version>/."""
    dest_root = Path(dest_root)
    final = dest_root / version
    if final.exists():
        manifest = read_manifest(final)
        verify_files(final, manifest)
        return final
    dest_root.mkdir(parents=True, exist_ok=True)
    base = source.rstrip("/") + "/" + version + "/"
    tmp = Path(tempfile.mkdtemp(dir=dest_root, prefix=f".{version}."))
    try:
        _download(base + MANIFEST, tmp / MANIFEST, None)
        manifest = read_manifest(tmp)
        if manifest["index_version"] != version:
            raise ArtifactError(f"manifest says {manifest['index_version']!r}, expected {version!r}")
        for name, meta in manifest["files"].items():
            _download(base + name, tmp / name, meta)
        os.rename(tmp, final)
        return final
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
