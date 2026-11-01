#!/usr/bin/env python3
"""Cost per million requests, per architecture, from your measured capacity.

  python3 ops/cost_model.py \\
      --price-vcpu-hour <$> --price-gib-hour <$> [--price-gpu-hour <$>] \\
      --arch "name=combined-batched,run=reports/sweep-3pods,vcpu=3,gib=4.5" \\
      --arch "name=separated-batched,run=reports/sweep-sep,vcpu=5,gib=4" \\
      --arch "name=gpu-encoder,rps=1800,vcpu=4,gib=12,gpu=1"

Each --arch needs the capacity at SLO (from a sweep's summary.json via
run=..., or rps=... directly) and the TOTAL resources that run used: every
pod in the request path (retrieval + encoder), counted the way you are billed
(on GKE Autopilot: pod requests; on Standard: the share of node capacity).

Look up current prices for your region; this script ships with none, on
purpose. It answers two Staff-level questions with your own numbers: what one
million searches cost, and which architectural change moves that number most.
"""
import argparse
import json
from pathlib import Path


def parse_arch(text: str) -> dict:
    d = dict(part.split("=", 1) for part in text.split(","))
    if "run" in d:
        summary = json.loads((Path(d["run"]) / "summary.json").read_text())
        d.setdefault("rps", summary.get("capacity_rps_at_slo") or summary.get("achieved_rps"))
    for key in ("rps", "vcpu", "gib", "gpu"):
        d[key] = float(d.get(key, 0) or 0)
    if d["rps"] <= 0:
        raise SystemExit(f"{d.get('name')}: no capacity (rps); pass run=<sweep dir> or rps=<n>")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", action="append", required=True)
    ap.add_argument("--price-vcpu-hour", type=float, required=True)
    ap.add_argument("--price-gib-hour", type=float, required=True)
    ap.add_argument("--price-gpu-hour", type=float, default=0.0)
    ap.add_argument("--utilization", type=float, default=0.7, help="share of capacity you plan to run at (headroom)")
    args = ap.parse_args()

    rows = []
    for a in map(parse_arch, args.arch):
        per_hour = a["vcpu"] * args.price_vcpu_hour + a["gib"] * args.price_gib_hour + a["gpu"] * args.price_gpu_hour
        sustained = a["rps"] * args.utilization
        per_million = per_hour / (sustained * 3600) * 1e6
        rows.append((per_million, a["name"], a["rps"], sustained, per_hour, a["vcpu"], a["gib"], a["gpu"]))
    rows.sort()
    print(f"{'architecture':<24}{'capacity':>10}{'planned':>10}{'$/hour':>10}{'vCPU':>6}{'GiB':>6}{'GPU':>5}{'$ per 1M req':>14}")
    for per_million, name, rps, sustained, per_hour, vcpu, gib, gpu in rows:
        print(f"{name:<24}{rps:>10.0f}{sustained:>10.0f}{per_hour:>10.3f}{vcpu:>6.1f}{gib:>6.1f}{gpu:>5.0f}{per_million:>14.4f}")
    best = rows[0]
    print(f"\ncheapest per request: {best[1]}. Planned load = capacity x {args.utilization} headroom.")


if __name__ == "__main__":
    main()
