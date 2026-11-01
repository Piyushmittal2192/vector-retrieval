"""Offline ANN benchmark: recall@10 vs latency/throughput/memory across HNSW settings.

Measures the index alone (no encoder, no HTTP), so the numbers isolate what
M, ef_construction and ef cost. Ground truth is exact brute-force search.

  # real catalog embeddings from the builder's cache, real query embeddings
  python -m bench.bench_ann --vectors auto --encoder st --n 200000
  # synthetic clustered vectors (no model needed), e.g. to see 1M behaviour
  python -m bench.bench_ann --synthetic 1000000

Change one parameter at a time when you read the CSV: fix M and
ef_construction, vary ef; then fix ef and vary M.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import hnswlib
import numpy as np

from retrieval.ann import brute_force_topk, recall_at_k


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def synthetic(n: int, dim: int, nq: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((max(10, n // 500), dim), dtype=np.float32)
    assign = rng.integers(0, len(centers), n)
    base = centers[assign] + 0.35 * rng.standard_normal((n, dim), dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    q = base[rng.integers(0, n, nq)] + 0.2 * rng.standard_normal((nq, dim), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return base, q.astype(np.float32)


def from_cache(path: str, n: int, queries: str, encoder: str, model_path: str, nq: int):
    if path == "auto":
        found = sorted(Path("cache/embeddings").glob("*/vecs.npy"))
        if len(found) != 1:
            raise SystemExit(f"--vectors auto needs exactly one cache, found {[str(p) for p in found]}")
        path = str(found[0])
    vecs = np.load(path, mmap_mode="r")
    n = min(n, vecs.shape[0]) if n else vecs.shape[0]
    base = np.ascontiguousarray(vecs[:n], dtype=np.float32)
    from retrieval.encoders import load_encoder

    enc = load_encoder(encoder, model_path, "", 0)
    with open(queries) as f:
        texts = [json.loads(l)["query"] for l in f if l.strip()][:nq]
    return base, enc.encode(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=None, help="path to vecs.npy from cache/embeddings, or 'auto'")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n", type=int, default=0, help="limit base vectors")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--encoder", default="st", choices=["st", "hash"])
    ap.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--M", default="8,16,32")
    ap.add_argument("--efc", default="100,200")
    ap.add_argument("--ef", default="10,20,40,64,100,200")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="reports/ann_bench.csv")
    args = ap.parse_args()

    if args.synthetic:
        base, q = synthetic(args.synthetic, args.dim, args.nq)
        source = f"synthetic-{args.synthetic}"
    elif args.vectors:
        base, q = from_cache(args.vectors, args.n, args.queries, args.encoder, args.model_path, args.nq)
        source = f"catalog-{base.shape[0]}"
    else:
        raise SystemExit("pass --vectors auto|PATH or --synthetic N")
    print(f"{source}: {base.shape[0]:,} x {base.shape[1]} base, {q.shape[0]} queries; computing exact top-10 ...")
    exact = brute_force_topk(base, q, 10)

    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Exact search baseline: "when is HNSW the wrong choice?" starts here.
    # Brute force needs no build, no graph memory, has perfect recall, and
    # with BLAS it is batch-friendly; HNSW only wins once N is large enough.
    lat = []
    for i in range(min(200, q.shape[0])):
        t0 = time.perf_counter()
        brute_force_topk(base, q[i : i + 1], 10)
        lat.append(time.perf_counter() - t0)
    lat.sort()
    t0 = time.perf_counter()
    for s in range(0, min(q.shape[0], 320), 32):
        brute_force_topk(base, q[s : s + 32], 10)
    exact_qps = min(q.shape[0], 320) / (time.perf_counter() - t0)
    exact_row = {
        "source": source, "n": base.shape[0], "M": "exact", "ef_construction": "", "ef": "",
        "recall_at_10": 1.0,
        "p50_us": round(lat[len(lat) // 2] * 1e6, 1), "p99_us": round(lat[int(len(lat) * 0.99) - 1] * 1e6, 1),
        "qps_1thread_batch32": round(exact_qps, 1),
        "build_s": 0.0, "index_mb": round(base.nbytes / 2**20, 1), "load_s": "", "rss_delta_mb": "",
    }
    rows.append(exact_row)
    print(exact_row, flush=True)
    print("(exact search uses numpy/BLAS threads; set OMP_NUM_THREADS=1 for a like-for-like single-thread comparison)")
    for M in [int(x) for x in args.M.split(",")]:
        for efc in [int(x) for x in args.efc.split(",")]:
            idx = hnswlib.Index(space="cosine", dim=base.shape[1])
            idx.init_index(max_elements=base.shape[0], ef_construction=efc, M=M)
            t0 = time.perf_counter()
            idx.add_items(base, np.arange(base.shape[0]), num_threads=args.threads or -1)
            build_s = time.perf_counter() - t0
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "i.bin")
                idx.save_index(p)
                size_mb = os.path.getsize(p) / 2**20
                del idx
                before = rss_mb()
                t0 = time.perf_counter()
                idx = hnswlib.Index(space="cosine", dim=base.shape[1])
                idx.load_index(p)
                load_s = time.perf_counter() - t0
                rss_delta = rss_mb() - before
            idx.set_num_threads(1)
            for ef in [int(x) for x in args.ef.split(",")]:
                idx.set_ef(max(ef, 10))
                labels, _ = idx.knn_query(q, k=10, num_threads=1)
                recall = recall_at_k(labels.astype(np.int64), exact, 10)
                lat = []
                for i in range(min(500, q.shape[0])):
                    t0 = time.perf_counter()
                    idx.knn_query(q[i : i + 1], k=10, num_threads=1)
                    lat.append(time.perf_counter() - t0)
                lat.sort()
                t0 = time.perf_counter()
                for s in range(0, q.shape[0], 32):
                    idx.knn_query(q[s : s + 32], k=10, num_threads=1)
                qps = q.shape[0] / (time.perf_counter() - t0)
                row = {
                    "source": source, "n": base.shape[0], "M": M, "ef_construction": efc, "ef": ef,
                    "recall_at_10": round(recall, 4),
                    "p50_us": round(lat[len(lat) // 2] * 1e6, 1), "p99_us": round(lat[int(len(lat) * 0.99) - 1] * 1e6, 1),
                    "qps_1thread_batch32": round(qps, 1),
                    "build_s": round(build_s, 2), "index_mb": round(size_mb, 1), "load_s": round(load_s, 3), "rss_delta_mb": round(rss_delta, 1),
                }
                rows.append(row)
                print(row, flush=True)
            del idx
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
