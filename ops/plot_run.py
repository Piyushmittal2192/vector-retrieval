#!/usr/bin/env python3
"""Plot a load-test run: throughput/errors, latency and index-version share per second.

  python3 ops/plot_run.py reports/steady-200     -> reports/steady-200/timeline.png (+ sweep.png)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

run = Path(sys.argv[1])
rows = list(csv.DictReader(open(run / "timeline.csv")))
t = [int(r["second"]) for r in rows]
num = lambda v: float(v) if v not in ("", "None") else float("nan")  # noqa: E731
versions = sorted({v for r in rows for v in json.loads(r["versions"] or "{}")})

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(t, [int(r["ok"]) for r in rows], label="ok / s", color="#2a7ab8")
ax[0].bar(t, [int(r["errors"]) for r in rows], label="errors / s", color="#d1495b", width=1.0)
ax[0].set_ylabel("requests / s")
ax[0].legend(loc="upper right")
ax[1].plot(t, [num(r["p50_ms"]) for r in rows], label="p50", color="#2a7ab8")
ax[1].plot(t, [num(r["p99_ms"]) for r in rows], label="p99", color="#e8a33d")
ax[1].set_ylabel("latency (ms)")
ax[1].legend(loc="upper right")
bottom = [0.0] * len(rows)
for i, v in enumerate(versions):
    share = []
    for r in rows:
        d = json.loads(r["versions"] or "{}")
        share.append(d.get(v, 0) / max(1, sum(d.values())))
    ax[2].bar(t, share, bottom=bottom, width=1.0, label=v, color=["#2a7ab8", "#3fa34d", "#e8a33d", "#8e6bbf"][i % 4])
    bottom = [b + s for b, s in zip(bottom, share)]
ax[2].set_ylabel("index version share")
ax[2].set_xlabel("seconds since start")
if versions:
    ax[2].legend(loc="upper right")
summary = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else {}
fig.suptitle(f"{run.name}: p99 {summary.get('latency', {}).get('p99_ms')} ms, error rate {summary.get('error_rate')}")
fig.tight_layout()
fig.savefig(run / "timeline.png", dpi=110)
print(f"wrote {run / 'timeline.png'}")

if (run / "sweep.csv").exists():
    sw = list(csv.DictReader(open(run / "sweep.csv")))
    x = [float(r["offered_rps"]) for r in sw]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color in (("p50_ms", "#2a7ab8"), ("p95_ms", "#3fa34d"), ("p99_ms", "#e8a33d")):
        ax.plot(x, [num(r[key]) for r in sw], marker="o", label=key.replace("_ms", ""), color=color)
    slo = summary.get("slo", {}).get("p99_ms")
    if slo:
        ax.axhline(slo, color="#d1495b", linestyle="--", label=f"SLO p99 {slo} ms")
    ax.set_xlabel("offered load (req/s)")
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"{run.name}: capacity at SLO = {summary.get('capacity_rps_at_slo')} req/s")
    fig.tight_layout()
    fig.savefig(run / "sweep.png", dpi=110)
    print(f"wrote {run / 'sweep.png'}")
