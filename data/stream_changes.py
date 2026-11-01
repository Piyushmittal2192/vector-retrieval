"""Simulated live producer: keeps appending product changes to the log.

  python -m data.stream_changes --rate 20        # about 20 events/s until Ctrl+C

Run it next to `make index-worker` to watch freshness lag build up and drain.
Only one producer at a time: sequence numbers come from the log itself.
"""
from __future__ import annotations

import argparse
import time

from data.generate_changes import append_changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--changes", default="data/changes.jsonl")
    ap.add_argument("--rate", type=float, default=20.0, help="events per second")
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()
    interval = args.batch / args.rate
    try:
        while True:
            t0 = time.monotonic()
            first, last, counts = append_changes(args.catalog, args.changes, args.batch)
            print(f"seq {first}..{last} {counts}", flush=True)
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("producer stopped")


if __name__ == "__main__":
    main()
