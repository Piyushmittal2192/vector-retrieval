"""Background indexing worker: change log -> new, validated index versions.

  python -m retrieval.index_worker --min-events 500 --max-delay 60        # loop until Ctrl+C
  python -m retrieval.index_worker --once                                  # one decision, then exit

This is the *other* concurrency problem in a search system. Interactive
search wants low latency and sheds load it cannot serve. Index updates want
the opposite: never lose an event, never corrupt or interrupt the active
index, and it is fine if they take minutes.

Durable-queue semantics without a broker. data/changes.jsonl is an
append-only log with sequence numbers, like a single-partition topic (in
GCP: Pub/Sub with ordering, or a Kafka partition). The worker's committed
offset is not stored anywhere separately: it is the watermark of the newest
published version. That makes the worker idempotent:

  * version name = <prefix>-w<watermark>, so the same input always maps to
    the same version; a build that already exists is never redone;
  * a crash mid-build leaves only a hidden temp dir, never a half version;
    on restart the worker recomputes the same decision and rebuilds;
  * a crash after publish but before validation re-runs only validation.

Trigger policy (micro-batching again, on a different time scale): build when
at least --min-events are pending, or when the oldest pending event is older
than --max-delay seconds. The trade-off is freshness against build cost.

Output: artifacts/channels/latest.json names the newest version that passed
validation. Promotion (blue/green) stays a separate, deliberate step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from retrieval import artifacts
from retrieval.catalog import iter_jsonl


def published(root: Path, prefix: str) -> list[tuple[int, str]]:
    out = []
    for m in root.glob(f"{prefix}-w*/manifest.json"):
        try:
            out.append((int(json.loads(m.read_text())["source"]["watermark"]), m.parent.name))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(out)


def pending_events(changes: Path, after: int) -> list[dict]:
    if not changes.exists():
        return []
    return [ev for ev in iter_jsonl(changes) if int(ev["seq"]) > after]


def clean_stale_tmp(root: Path, prefix: str, older_than_s: float = 3600) -> None:
    for d in root.glob(f".{prefix}-w*.*"):
        if d.is_dir() and time.time() - d.stat().st_mtime > older_than_s:
            shutil.rmtree(d, ignore_errors=True)
            print(f"removed stale temp dir {d} (left by a crashed build)", flush=True)


def run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def step(args) -> str:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    done = published(root, args.prefix)
    if done:
        committed, latest = done[-1]
    elif args.baseline:  # first auto build: continue from what the baseline already covers
        committed, latest = int(artifacts.read_manifest(root / args.baseline)["source"]["watermark"]), args.baseline
    else:
        committed, latest = 0, ""
    events = pending_events(Path(args.changes), committed)
    if not events:
        return f"up to date at watermark {committed}"
    now = time.time()
    oldest_age = now - float(events[0].get("ts", 0) or 0)
    if len(events) < args.min_events and oldest_age < args.max_delay:
        return f"waiting: {len(events)} pending (< {args.min_events}), oldest {oldest_age:.0f}s (< {args.max_delay}s)"

    watermark = int(events[-1]["seq"])  # everything complete in the log right now
    version = f"{args.prefix}-w{watermark}"
    print(f"{dt.datetime.now():%H:%M:%S} building {version}: {len(events)} events, oldest {oldest_age:.0f}s old", flush=True)
    common = ["--encoder", args.encoder, "--model-path", args.model_path, "--model-name", args.model_name]
    if not (root / version).exists():
        rc = run([sys.executable, "-m", "retrieval.build_index", "--version", version, "--watermark", str(watermark),
                  "--catalog", args.catalog, "--changes", args.changes, "--queries", args.queries, "--out", str(root),
                  "--M", str(args.M), "--ef-construction", str(args.ef_construction), *common])
        if rc != 0:
            return f"build of {version} failed (exit {rc}); will retry next poll"
    cmd = [sys.executable, "-m", "retrieval.validate_index", "--version", version, "--root", str(root),
           "--catalog", args.catalog, "--changes", args.changes, "--queries", args.queries, "--out", args.validation_dir, *common]
    if latest:
        cmd += ["--baseline", latest]
    passed = run(cmd) == 0
    channel = Path(args.channels) / ("latest.json" if passed else "rejected.json")
    artifacts.write_json_atomic(
        channel,
        {"version": version, "watermark": watermark, "events": len(events), "baseline": latest,
         "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
    )
    return f"{version} {'PASSED -> ' if passed else 'FAILED validation -> '}{channel}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--changes", default="data/changes.jsonl")
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--root", default="artifacts/indexes")
    ap.add_argument("--channels", default="artifacts/channels")
    ap.add_argument("--validation-dir", default="artifacts/validation")
    ap.add_argument("--prefix", default="auto")
    ap.add_argument("--baseline", default="v1", help="version to validate the first auto build against")
    ap.add_argument("--min-events", type=int, default=500)
    ap.add_argument("--max-delay", type=float, default=60.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--encoder", default="st", choices=["st", "hash"])
    ap.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    args = ap.parse_args()
    if args.baseline and not (Path(args.root) / args.baseline).exists():
        args.baseline = ""
    clean_stale_tmp(Path(args.root), args.prefix)
    try:
        while True:
            print(f"{dt.datetime.now():%H:%M:%S} {step(args)}", flush=True)
            if args.once:
                return 0
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("worker stopped; restart it any time, it resumes from the newest published watermark")
        return 0


if __name__ == "__main__":
    sys.exit(main())
