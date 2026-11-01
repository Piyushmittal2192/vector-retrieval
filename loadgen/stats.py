"""Aggregation for load-generator records (standard library only).

A record is a tuple:
  (t_sched, latency, service_time, status, error, version, pod, build, key, result_hash)
    t_sched       seconds since run start when the request was SCHEDULED
    latency       completion - scheduled time (includes any client-side
                  queueing: corrected for coordinated omission)
    service_time  completion - actual send time
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

T, LAT, SVC, STATUS, ERR, VER, POD, BUILD, KEY, RHASH = range(10)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    rank = max(1, math.ceil(p / 100.0 * len(sorted_vals)))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def latency_block(values: list[float]) -> dict:
    v = sorted(values)
    ms = lambda x: round(x * 1000, 2) if not math.isnan(x) else None  # noqa: E731
    return {
        "p50_ms": ms(percentile(v, 50)),
        "p90_ms": ms(percentile(v, 90)),
        "p95_ms": ms(percentile(v, 95)),
        "p99_ms": ms(percentile(v, 99)),
        "p999_ms": ms(percentile(v, 99.9)),
        "max_ms": ms(v[-1]) if v else None,
    }


def summarize(recs: list[tuple], duration: float, slo_p99_ms: float, slo_error_rate: float) -> dict:
    ok = [r for r in recs if r[STATUS] == 200]
    errors = Counter(r[ERR] for r in recs if r[STATUS] != 200)
    total = len(recs)
    error_rate = (total - len(ok)) / total if total else 0.0
    lat = latency_block([r[LAT] for r in ok])
    pods = Counter(r[POD] for r in ok)
    versions = Counter(r[VER] for r in ok)

    # consistency: same query + same index version must give the same ids on every pod
    seen: dict[tuple, set] = defaultdict(set)
    pods_for: dict[tuple, set] = defaultdict(set)
    for r in ok:
        seen[(r[KEY], r[VER])].add(r[RHASH])
        pods_for[(r[KEY], r[VER])].add(r[POD])
    multi_pod = [k for k, p in pods_for.items() if len(p) > 1]
    inconsistent = [k for k in multi_pod if len(seen[k]) > 1]

    tl = timeline(recs)
    mixed = sum(1 for row in tl if len([v for v in row["versions"] if v]) > 1)
    shares = {p: round(c / len(ok), 4) for p, c in pods.most_common()} if ok else {}
    imbalance = (max(pods.values()) / min(pods.values())) if len(pods) > 1 else 1.0

    slow_ok = [r for r in ok if r[KEY] is not None and len(r[KEY]) > 3 and r[KEY][3]]
    extra = {}
    if slow_ok:
        fast_ok = [r for r in ok if not (r[KEY] is not None and len(r[KEY]) > 3 and r[KEY][3])]
        extra["latency_by_class"] = {
            "fast": {**latency_block([r[LAT] for r in fast_ok]), "count": len(fast_ok)},
            "slow": {**latency_block([r[LAT] for r in slow_ok]), "count": len(slow_ok)},
        }

    keyed = [r for r in recs if r[KEY] is not None and len(r[KEY]) > 4]
    if any(r[KEY][4] for r in keyed):
        extra["by_priority"] = {}
        for name, flag in (("normal", False), ("low", True)):
            group = [r for r in keyed if bool(r[KEY][4]) == flag]
            good = [r for r in group if r[STATUS] == 200]
            extra["by_priority"][name] = {
                "requests": len(group),
                "error_rate": round((len(group) - len(good)) / len(group), 6) if group else 0.0,
                "latency": latency_block([r[LAT] for r in good]),
            }

    verdict = bool(ok) and lat["p99_ms"] is not None and lat["p99_ms"] <= slo_p99_ms and error_rate <= slo_error_rate
    return {
        "requests": total,
        "ok": len(ok),
        "offered_rps": round(total / duration, 2) if duration else None,
        "achieved_rps": round(len(ok) / duration, 2) if duration else None,
        "error_rate": round(error_rate, 6),
        "errors": dict(errors),
        "latency": lat,
        "service_time": latency_block([r[SVC] for r in ok]),
        "versions": dict(versions),
        "builds": dict(Counter(r[BUILD] for r in ok)),
        "mixed_version_seconds": mixed,
        "pods": shares,
        "pod_count": len(pods),
        "pod_imbalance_max_over_min": round(imbalance, 3),
        "consistency": {"query_versions_seen_on_multiple_pods": len(multi_pod), "inconsistent": len(inconsistent)},
        "slo": {"p99_ms": slo_p99_ms, "error_rate": slo_error_rate, "pass": verdict},
        **extra,
    }


def timeline(recs: list[tuple]) -> list[dict]:
    buckets: dict[int, list] = defaultdict(list)
    for r in recs:
        buckets[int(r[T])].append(r)
    rows = []
    for sec in sorted(buckets):
        rs = buckets[sec]
        ok = sorted(r[LAT] for r in rs if r[STATUS] == 200)
        errs = Counter(r[ERR] for r in rs if r[STATUS] != 200)
        rows.append(
            {
                "second": sec,
                "sent": len(rs),
                "ok": len(ok),
                "errors": sum(errs.values()),
                "error_kinds": dict(errs),
                "p50_ms": round(percentile(ok, 50) * 1000, 2) if ok else None,
                "p99_ms": round(percentile(ok, 99) * 1000, 2) if ok else None,
                "versions": dict(Counter(r[VER] for r in rs if r[STATUS] == 200)),
                "pods": len({r[POD] for r in rs if r[STATUS] == 200}),
            }
        )
    return rows
