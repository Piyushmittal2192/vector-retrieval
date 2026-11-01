"""Small, dependency-free Prometheus metrics (text exposition format 0.0.4).

Counters, gauges and histograms with labels, safe to update from the event
loop and the worker thread. Enough for this service; in a larger codebase you
would use prometheus_client, which exposes the same format.
"""
from __future__ import annotations

import os
import threading
import time


def _fmt_labels(labels: tuple) -> str:
    if not labels:
        return ""
    parts = []
    for k, v in labels:
        v = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{k}="{v}"')
    return "{" + ",".join(parts) + "}"


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str):
        self.name, self.help = name, help_text
        self._lock = threading.Lock()
        self._values: dict[tuple, float] = {}

    @staticmethod
    def _key(labels: dict) -> tuple:
        return tuple(sorted(labels.items()))

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]

    def render(self) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        return self.header() + [f"{self.name}{_fmt_labels(k)} {v}" for k, v in items]


class Counter(_Metric):
    kind = "counter"

    def inc(self, amount: float = 1.0, **labels) -> None:
        if amount == 0:
            return
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, **labels) -> float:
        return self._values.get(self._key(labels), 0.0)


class Gauge(_Metric):
    kind = "gauge"

    def set(self, value: float, **labels) -> None:
        with self._lock:
            self._values[self._key(labels)] = float(value)

    def inc(self, amount: float = 1.0, **labels) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels) -> None:
        self.inc(-amount, **labels)

    def get(self, **labels) -> float:
        return self._values.get(self._key(labels), 0.0)


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name: str, help_text: str, buckets: list[float]):
        super().__init__(name, help_text)
        self.buckets = sorted(buckets)
        self._series: dict[tuple, list] = {}

    def observe(self, value: float, **labels) -> None:
        key = self._key(labels)
        with self._lock:
            s = self._series.get(key)
            if s is None:
                s = self._series[key] = [[0] * len(self.buckets), 0.0, 0]
            for i, b in enumerate(self.buckets):
                if value <= b:
                    s[0][i] += 1
                    break
            s[1] += value
            s[2] += 1

    def render(self) -> list[str]:
        lines = self.header()
        with self._lock:
            series = [(k, list(v[0]), v[1], v[2]) for k, v in self._series.items()]
        for key, counts, total, n in series:
            cum = 0
            for b, c in zip(self.buckets, counts):
                cum += c
                lines.append(f"{self.name}_bucket{_fmt_labels(key + (('le', repr(float(b))),))} {cum}")
            lines.append(f"{self.name}_bucket{_fmt_labels(key + (('le', '+Inf'),))} {n}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {total}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {n}")
        return lines


LATENCY = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5]


class Metrics:
    def __init__(self):
        self.requests = Counter("retrieval_requests_total", "Search requests by HTTP status code.")
        self.request_seconds = Histogram("retrieval_request_seconds", "Server-side latency of successful searches.", LATENCY)
        self.queue_wait = Histogram("retrieval_queue_wait_seconds", "Time a request waited before its batch started.", LATENCY)
        self.encode_seconds = Histogram("retrieval_encode_seconds", "Encoder time per batch.", LATENCY)
        self.ann_seconds = Histogram("retrieval_ann_seconds", "hnswlib query time per batch.", LATENCY)
        self.batch_size = Histogram("retrieval_batch_size", "Requests per micro-batch.", [1, 2, 4, 8, 16, 32, 64, 128])
        self.queue_depth = Gauge("retrieval_queue_depth", "Requests waiting for the worker.")
        self.inflight = Gauge("retrieval_inflight_requests", "Search requests being handled.")
        self.shed = Counter("retrieval_shed_total", "Requests rejected or abandoned, by reason.")
        self.query_cache = Counter("retrieval_query_cache_total", "Query embedding cache lookups.")
        self.index_info = Gauge("retrieval_index_info", "Loaded index (value is always 1).")
        self.index_vectors = Gauge("retrieval_index_vectors", "Vectors in the loaded index.")
        self.startup_seconds = Gauge("retrieval_startup_stage_seconds", "Startup time by stage.")
        self.ready = Gauge("retrieval_ready", "1 when serving traffic, 0 when loading or draining.")
        self._all = [v for v in vars(self).values() if isinstance(v, _Metric)]

    def render(self) -> str:
        lines: list[str] = []
        for m in self._all:
            lines.extend(m.render())
        lines.extend(_process_metrics())
        return "\n".join(lines) + "\n"


def _process_metrics() -> list[str]:
    out = []
    t = os.times()
    out += ["# TYPE process_cpu_seconds_total counter", f"process_cpu_seconds_total {t.user + t.system}"]
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        out += ["# TYPE process_resident_memory_bytes gauge", f"process_resident_memory_bytes {rss_pages * os.sysconf('SC_PAGE_SIZE')}"]
    except (OSError, ValueError):
        pass
    out += ["# TYPE process_time_seconds gauge", f"process_time_seconds {time.time()}"]
    return out


METRICS = Metrics()
