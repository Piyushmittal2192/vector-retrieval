"""Encoder benchmark: latency and throughput by batch size and thread count.

This is usually the dominant cost per request (milliseconds) next to HNSW
search (tens to hundreds of microseconds), and it decides the micro-batching
settings and how many CPUs a pod needs.

  python -m bench.bench_encoder --threads 1,2,4 --batch-sizes 1,2,4,8,16,32,64
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from retrieval.encoders import load_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="st", choices=["st", "hash"])
    ap.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default="reports/encoder_bench.csv")
    args = ap.parse_args()

    with open(args.queries) as f:
        texts = [json.loads(l)["query"] for l in f if l.strip()]
    enc = load_encoder(args.encoder, args.model_path, "", 0)
    rows = []
    for threads in [int(x) for x in args.threads.split(",")]:
        if args.encoder == "st":
            import torch

            torch.set_num_threads(threads)
        for bs in [int(x) for x in args.batch_sizes.split(",")]:
            enc.encode(texts[:bs])  # warm up this shape
            lat = []
            for r in range(args.repeats):
                batch = [texts[(r * bs + i) % len(texts)] for i in range(bs)]
                t0 = time.perf_counter()
                enc.encode(batch, batch_size=bs)
                lat.append(time.perf_counter() - t0)
            lat.sort()
            p50 = lat[len(lat) // 2]
            row = {
                "encoder": enc.name, "threads": threads, "batch_size": bs,
                "p50_ms": round(p50 * 1000, 2), "p99_ms": round(lat[max(0, int(len(lat) * 0.99) - 1)] * 1000, 2),
                "queries_per_s": round(bs / p50, 1), "ms_per_query": round(p50 * 1000 / bs, 3),
            }
            rows.append(row)
            print(row, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
