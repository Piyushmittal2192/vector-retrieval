"""Open-loop load generator for the retrieval service.

Open loop: requests are scheduled on a fixed arrival process (Poisson by
default) regardless of how fast responses come back, like real users. A
closed-loop tool (N workers, each waiting for its last response) slows down
when the server slows down and hides the queueing it should be measuring
("coordinated omission"). Latency here is measured from the SCHEDULED send
time, so time spent waiting for a free connection counts.

Examples
  # 60 s at 200 req/s, results in reports/steady-200
  python -m loadgen.loadgen --target http://localhost:8080 --rate 200 --duration 60 --out reports/steady-200

  # capacity sweep
  python -m loadgen.loadgen --target http://retrieval.search --mode sweep --rates 50,100,200,400 --step-duration 30

Outputs in --out: summary.json, timeline.csv (per second), sweep.csv (sweep
mode), and one line appended to <out>/../runs.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import gzip
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

from loadgen.httpclient import HTTPError, HTTPPool
from loadgen.stats import LAT, STATUS, VER, latency_block, summarize, timeline


def load_queries(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Picker:
    """Zipf-skewed query popularity: a few head queries, a long tail."""

    def __init__(self, queries: list[dict], zipf: float, seed: int):
        self.rng = random.Random(seed)
        self.queries = list(queries)
        self.rng.shuffle(self.queries)
        weights = [1.0 / (r**zipf) for r in range(1, len(self.queries) + 1)]
        total, acc, self.cum = sum(weights), 0.0, []
        for w in weights:
            acc += w / total
            self.cum.append(acc)

    def pick(self) -> dict:
        return self.rng.choices(self.queries, cum_weights=self.cum, k=1)[0]


class Runner:
    def __init__(self, args):
        self.args = args
        self.pool = HTTPPool(args.target, max_conns=args.connections, idle_timeout=args.idle_timeout)
        self.picker = Picker(load_queries(args.queries), args.zipf, args.seed)
        self.rng = random.Random(args.seed + 1)
        self.recs: list[tuple] = []
        self.inflight = 0
        self.max_lag = 0.0
        self.dropped_client = 0

    async def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status, _, _ = await asyncio.wait_for(self.pool.request("GET", "/ready"), 2.0)
                if status == 200:
                    return
            except (HTTPError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1)
        raise SystemExit(f"target {self.args.target} not ready after {timeout}s")

    async def fire(self, t0_run: float, t_sched: float, q: dict) -> None:
        a = self.args
        loop = asyncio.get_running_loop()
        category = q.get("category") if a.category_frac and self.rng.random() < a.category_frac else None
        payload = {"query": q["query"], "k": a.k}
        if category:
            payload["category"] = category
        body = json.dumps(payload).encode()
        key = (q["qid"], a.k, category)
        status, err, ver, pod, build, rhash = 0, None, "", "", "", 0
        t_send = loop.time()
        self.inflight += 1
        try:
            status, headers, data = await asyncio.wait_for(self.pool.request("POST", "/search", body), a.timeout)
            ver, pod = headers.get("x-index-version", ""), headers.get("x-served-by", "")
            if status == 200:
                doc = json.loads(data)
                build = doc.get("build", "")
                rhash = zlib.crc32("|".join(r["id"] for r in doc["results"]).encode())
            else:
                err = f"http_{status}"
        except asyncio.TimeoutError:
            err = "client_timeout"
        except HTTPError as exc:
            err = exc.kind
        except Exception as exc:  # malformed body etc.
            err = f"client_{type(exc).__name__}"
        finally:
            self.inflight -= 1
        end = loop.time()
        self.recs.append((t_sched - t0_run, end - t_sched, end - t_send, status, err, ver, pod, build, key, rhash))

    async def phase(self, rate: float, duration: float, t0_run: float, start_offset: float) -> None:
        """Schedule `rate` req/s for `duration` seconds starting at t0_run + start_offset."""
        a = self.args
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task] = set()
        t_sched = t0_run + start_offset
        end = t_sched + duration
        while True:
            t_sched += self.rng.expovariate(rate) if a.arrival == "poisson" else 1.0 / rate
            if t_sched >= end:
                break
            delay = t_sched - loop.time()
            if delay > 0.0005:
                await asyncio.sleep(delay)
            self.max_lag = max(self.max_lag, loop.time() - t_sched)
            if self.inflight >= a.max_inflight:
                self.dropped_client += 1
                self.recs.append((t_sched - t0_run, 0.0, 0.0, 0, "client_overflow", "", "", "", None, 0))
                continue
            task = asyncio.create_task(self.fire(t0_run, t_sched, self.picker.pick()))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks)

    async def reporter(self, t0_run: float) -> None:
        every = self.args.report_every
        seen = 0
        while True:
            await asyncio.sleep(every)
            window = self.recs[seen:]
            seen = len(self.recs)
            ok = [r for r in window if r[STATUS] == 200]
            lat = latency_block([r[LAT] for r in ok])
            vers: dict[str, int] = {}
            for r in ok:
                vers[r[VER]] = vers.get(r[VER], 0) + 1
            vtxt = ",".join(f"{v}:{c * 100 // max(1, len(ok))}%" for v, c in sorted(vers.items())) or "-"
            pods = len({r[6] for r in ok})
            print(
                f"t={time.monotonic() - t0_run:6.0f}s done={len(window) / every:7.1f}/s ok={len(ok) / every:7.1f}/s "
                f"err={len(window) - len(ok):4d} p50={lat['p50_ms']}ms p99={lat['p99_ms']}ms inflight={self.inflight} "
                f"pods={pods} versions={vtxt} conns={self.pool._open}",
                flush=True,
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()})


async def main_async(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    if args.wait_ready:
        await runner.wait_ready(args.wait_ready)
    loop = asyncio.get_running_loop()
    t0 = loop.time() + 0.2
    rep = asyncio.create_task(runner.reporter(loop.time()))
    sweep_rows = []
    if args.mode == "constant":
        await runner.phase(args.rate, args.duration, t0, 0.0)
        duration = args.duration
        recs = runner.recs
    else:
        offset = 0.0
        recs = []
        for rate in [float(x) for x in args.rates.split(",")]:
            print(f"--- step {rate:g} req/s for {args.step_duration}s (first {args.warmup}s excluded) ---", flush=True)
            before = len(runner.recs)
            await runner.phase(rate, args.step_duration, t0, offset)
            step = [r for r in runner.recs[before:] if r[0] >= offset + args.warmup]
            measured = args.step_duration - args.warmup
            s = summarize(step, measured, args.slo_p99_ms, args.slo_error_rate)
            sweep_rows.append(
                {
                    "offered_rps": rate,
                    "achieved_rps": s["achieved_rps"],
                    "error_rate": s["error_rate"],
                    **{k: s["latency"][k] for k in ("p50_ms", "p95_ms", "p99_ms", "max_ms")},
                    "slo_pass": s["slo"]["pass"],
                }
            )
            print(f"    -> achieved {s['achieved_rps']}/s p99 {s['latency']['p99_ms']}ms errors {s['error_rate']:.4f} SLO {'PASS' if s['slo']['pass'] else 'FAIL'}", flush=True)
            recs.extend(step)
            offset = loop.time() - t0 + args.pause
            await asyncio.sleep(args.pause)
        duration = sum(args.step_duration - args.warmup for _ in sweep_rows)
    rep.cancel()
    await runner.pool.close()

    summary = summarize(recs, duration, args.slo_p99_ms, args.slo_error_rate)
    summary.update(
        {
            "name": args.name or out.name,
            "mode": args.mode,
            "target": args.target,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config": {k: v for k, v in vars(args).items() if k not in ("queries",)},
            "client": {**runner.pool.stats, "max_schedule_lag_ms": round(runner.max_lag * 1000, 2), "client_overflow": runner.dropped_client},
        }
    )
    if sweep_rows:
        passing = [r["offered_rps"] for r in sweep_rows if r["slo_pass"]]
        summary["sweep"] = sweep_rows
        summary["capacity_rps_at_slo"] = max(passing) if passing else 0
        write_csv(out / "sweep.csv", sweep_rows)
    write_csv(out / "timeline.csv", timeline(runner.recs))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_requests:
        with gzip.open(out / "requests.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_sched", "latency", "service_time", "status", "error", "version", "pod", "build"])
            for r in runner.recs:
                w.writerow([round(r[0], 4), round(r[1], 5), round(r[2], 5), r[3], r[4] or "", r[5], r[6], r[7]])

    runs = out.parent / "runs.csv"
    new = not runs.exists()
    with open(runs, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["started_at", "name", "mode", "offered_rps", "achieved_rps", "p50_ms", "p99_ms", "error_rate", "versions", "mixed_version_s", "pods", "capacity_rps", "slo_pass"])
        w.writerow(
            [
                summary["started_at"], summary["name"], args.mode, summary["offered_rps"], summary["achieved_rps"],
                summary["latency"]["p50_ms"], summary["latency"]["p99_ms"], summary["error_rate"],
                json.dumps(summary["versions"]), summary["mixed_version_seconds"], summary["pod_count"],
                summary.get("capacity_rps_at_slo", ""), summary["slo"]["pass"],
            ]
        )

    lat = summary["latency"]
    print("\n==== summary ====")
    print(f"requests {summary['requests']:,}  ok {summary['ok']:,}  achieved {summary['achieved_rps']}/s  error rate {summary['error_rate']:.5f}")
    print(f"latency  p50 {lat['p50_ms']}  p95 {lat['p95_ms']}  p99 {lat['p99_ms']}  p99.9 {lat['p999_ms']}  max {lat['max_ms']} ms")
    print(f"errors   {summary['errors'] or 'none'}")
    print(f"versions {summary['versions']}  mixed-version seconds {summary['mixed_version_seconds']}")
    print(f"pods     {summary['pod_count']} (max/min share {summary['pod_imbalance_max_over_min']})  consistency {summary['consistency']}")
    print(f"client   {summary['client']}")
    if sweep_rows:
        print(f"capacity at SLO: {summary['capacity_rps_at_slo']} req/s")
    print(f"SLO (p99 <= {args.slo_p99_ms} ms, errors <= {args.slo_error_rate}): {'PASS' if summary['slo']['pass'] else 'FAIL'}")
    print(f"results in {out}")
    if summary["client"]["max_schedule_lag_ms"] > 50:
        print("WARNING: the generator fell behind its schedule; give it more CPU or lower the rate", file=sys.stderr)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=os.getenv("TARGET", "http://localhost:8080"))
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--mode", choices=["constant", "sweep"], default="constant")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rates", default="25,50,100,200,400")
    ap.add_argument("--step-duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--pause", type=float, default=3.0)
    ap.add_argument("--arrival", choices=["poisson", "uniform"], default="poisson")
    ap.add_argument("--connections", type=int, default=32)
    ap.add_argument("--idle-timeout", type=float, default=2.0)
    ap.add_argument("--max-inflight", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--zipf", type=float, default=1.0, help="query popularity skew; 0 = uniform")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--category-frac", type=float, default=0.0, help="fraction of requests with a category filter")
    ap.add_argument("--slo-p99-ms", type=float, default=100.0)
    ap.add_argument("--slo-error-rate", type=float, default=0.001)
    ap.add_argument("--wait-ready", type=float, default=60.0, help="seconds to wait for /ready first (0 = skip)")
    ap.add_argument("--report-every", type=float, default=5.0)
    ap.add_argument("--save-requests", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=f"reports/run-{dt.datetime.now():%Y%m%d-%H%M%S}")
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
