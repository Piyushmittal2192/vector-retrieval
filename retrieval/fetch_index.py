"""Init-container entrypoint: download one index version and verify it.

  python -m retrieval.fetch_index --source http://artifact-server/indexes --version v2 --dest /index

Exit code 0 only if every file matched the manifest checksums. A non-zero
exit leaves the pod in Init:Error, so a corrupt artifact can never reach a
serving container, and the old pods keep taking traffic.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from retrieval.artifacts import ArtifactError, fetch_version, read_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="base URI: file:///path, http(s)://host/path or gs://bucket/path")
    ap.add_argument("--version", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    try:
        path = fetch_version(args.source, args.version, Path(args.dest))
    except ArtifactError as exc:
        print(f"FETCH FAILED for {args.version}: {exc}", file=sys.stderr, flush=True)
        return 1
    m = read_manifest(path)
    total = sum(f["bytes"] for f in m["files"].values())
    print(
        f"fetched {args.version} ({total / 1e6:.1f} MB, {m['index']['count']:,} vectors) "
        f"to {path} in {time.perf_counter() - t0:.1f}s; checksums verified",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
