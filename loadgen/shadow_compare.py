"""Shadow comparison: replay the same queries against two live deployments.

  python -m loadgen.shadow_compare --a http://retrieval-blue.search --b http://retrieval-green.search \
      --queries /artifacts/queries/queries.jsonl --out /reports/shadow-v1-v2

Unlike validate_index (offline, one process), this goes through the real
pods: image, config, init-container fetch, network path. It reports result
overlap, top-1 agreement, latency of each side, and errors, and exits 1 if
the candidate (b) breaches the gates.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loadgen.httpclient import HTTPError, HTTPPool
from loadgen.stats import latency_block


async def query_all(base: str, queries: list[dict], k: int, concurrency: int, timeout: float):
    pool = HTTPPool(base, max_conns=concurrency)
    sem = asyncio.Semaphore(concurrency)
    results: list = [None] * len(queries)
    lats: list[float] = []
    errors = 0
    versions = set()

    async def one(i: int, q: dict):
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                status, headers, body = await asyncio.wait_for(
                    pool.request("POST", "/search", json.dumps({"query": q["query"], "k": k}).encode()), timeout
                )
                if status != 200:
                    errors += 1
                    return
                doc = json.loads(body)
                versions.add(doc["index_version"])
                results[i] = [r["id"] for r in doc["results"]]
                lats.append(time.perf_counter() - t0)
            except (HTTPError, asyncio.TimeoutError):
                errors += 1

    await asyncio.gather(*(one(i, q) for i, q in enumerate(queries)))
    await pool.close()
    return results, lats, errors, sorted(versions)


async def main_async(args) -> int:
    with open(args.queries) as f:
        queries = [json.loads(l) for l in f if l.strip()][: args.limit]
    (ra, la, ea, va), (rb, lb, eb, vb) = await asyncio.gather(
        query_all(args.a, queries, args.k, args.concurrency, args.timeout),
        query_all(args.b, queries, args.k, args.concurrency, args.timeout),
    )
    pairs = [(x, y) for x, y in zip(ra, rb) if x is not None and y is not None]
    overlap = sum(len(set(x) & set(y)) / max(1, len(x)) for x, y in pairs) / max(1, len(pairs))
    top1 = sum(1 for x, y in pairs if x and y and x[0] == y[0]) / max(1, len(pairs))
    changed = sum(1 for x, y in pairs if x != y)
    lat_a, lat_b = latency_block(la), latency_block(lb)
    p99_ratio = (lat_b["p99_ms"] or 0) / max(1e-9, lat_a["p99_ms"] or 1e-9)
    report = {
        "a": {"target": args.a, "versions": va, "errors": ea, "latency": lat_a},
        "b": {"target": args.b, "versions": vb, "errors": eb, "latency": lat_b},
        "queries": len(queries),
        "compared": len(pairs),
        "overlap_at_k": round(overlap, 4),
        "top1_agreement": round(top1, 4),
        "queries_with_changed_results": changed,
        "p99_ratio_b_over_a": round(p99_ratio, 3),
    }
    gates = {
        "b_errors": eb <= args.max_errors,
        "overlap": overlap >= args.min_overlap,
        "latency": p99_ratio <= args.max_p99_ratio,
    }
    report["gates"] = gates
    report["pass"] = all(gates.values())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shadow.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"SHADOW {'PASS' if report['pass'] else 'FAIL'}: overlap@{args.k} {overlap:.3f}, top-1 {top1:.3f}, "
          f"p99 a={lat_a['p99_ms']}ms b={lat_b['p99_ms']}ms, errors a={ea} b={eb}")
    return 0 if report["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="current (control) base URL")
    ap.add_argument("--b", required=True, help="candidate base URL")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--max-p99-ratio", type=float, default=1.5)
    ap.add_argument("--max-errors", type=int, default=0)
    ap.add_argument("--out", default="reports/shadow")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
