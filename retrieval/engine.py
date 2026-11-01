"""Search engine: loads one verified index version and answers batches.

load() is the readiness contract. The pod reports ready only after ALL of:
  1. manifest present and well formed
  2. every file matches its sha256 (catches truncated or corrupt downloads)
  3. encoder name and dimension match the manifest
  4. encoder fingerprint reproduces the builder's probe vectors
  5. index vector count == manifest == docs count
  6. golden queries reproduce the builder's top-10 (overlap >= threshold)
Then the server warms the full request path before flipping /ready.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retrieval import artifacts
from retrieval.ann import AnnIndex
from retrieval.config import Settings
from retrieval.encoders import Encoder, LockedEncoder, fingerprint_cosines, load_encoder
from retrieval.metrics import METRICS

log = logging.getLogger("retrieval.engine")


class CompatibilityError(RuntimeError):
    pass


@dataclass
class SearchItem:
    query: str
    k: int = 10
    category: str | None = None
    simulate_ms: float = 0.0  # lab only: extra compute time, to create slow requests on demand


class SearchEngine:
    def __init__(self, settings: Settings, encoder: Encoder | None = None):
        self.settings = settings
        self.encoder = encoder
        self.manifest: dict = {}
        self.ann: AnnIndex | None = None
        self.ids: list[str] = []
        self.titles: list[str] = []
        self.categories: list[str | None] = []
        self.real_count = 0
        self.timings: dict[str, float] = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------ loading

    def _stage(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.timings[name] = round(now - t0, 3)
        METRICS.startup_seconds.set(self.timings[name], stage=name)
        log.info("startup stage %-12s %.3fs", name, self.timings[name])
        return now

    def load(self) -> dict:
        s = self.settings
        vdir = Path(s.index_dir)
        t = time.perf_counter()

        self.manifest = artifacts.read_manifest(vdir)
        if s.verify_checksums:
            artifacts.verify_files(vdir, self.manifest)
        t = self._stage("verify", t)

        m_enc = self.manifest["encoder"]
        if self.encoder is None:
            self.encoder = load_encoder(s.encoder_backend, s.model_path, s.model_name, s.torch_threads, s.encoder_url, s.device)
            # In the concurrent execution modes several threads share one model.
            if s.exec_mode != "batcher" and s.encoder_lock and s.encoder_backend == "st":
                self.encoder = LockedEncoder(self.encoder)
        t = self._stage("encoder", t)
        if self.encoder.name != m_enc["name"]:
            raise CompatibilityError(f"encoder {self.encoder.name!r} != index encoder {m_enc['name']!r}")
        if self.encoder.dim != m_enc["dimension"]:
            raise CompatibilityError(f"encoder dim {self.encoder.dim} != index dim {m_enc['dimension']}")
        cosines = fingerprint_cosines(m_enc["fingerprint"], self.encoder)
        if min(cosines) < s.fingerprint_min_cosine:
            raise CompatibilityError(
                f"encoder fingerprint mismatch (cosines {cosines}); same name, different weights or preprocessing?"
            )
        t = self._stage("fingerprint", t)

        idx = self.manifest["index"]
        ef = s.ef or idx["default_ef"]
        self.ann = AnnIndex(str(vdir / "index.bin"), m_enc["dimension"], idx["count"], ef, idx.get("metric", "cosine"))
        t = self._stage("ann_load", t)

        with open(vdir / "docs.json") as f:
            docs = json.load(f)
        self.ids, self.titles, self.categories = docs["ids"], docs["titles"], docs["categories"]
        self.real_count = len(self.ids)
        if self.real_count + idx.get("pad_count", 0) != idx["count"]:
            raise CompatibilityError("docs.json and index disagree on the number of items")
        t = self._stage("docs", t)

        golden = self.manifest.get("golden", [])
        if golden:
            results = self.process_batch([SearchItem(g["query"], 10) for g in golden])
            overlaps = []
            for g, res in zip(golden, results):
                got = [r["id"] for r in res["results"]]
                overlaps.append(len(set(got) & set(g["top_ids"])) / max(1, len(g["top_ids"])))
            mean = float(np.mean(overlaps))
            if mean < s.golden_min_overlap:
                raise CompatibilityError(f"golden query overlap {mean:.3f} < {s.golden_min_overlap}")
            self.timings["golden_overlap"] = round(mean, 4)
        t = self._stage("golden", t)

        METRICS.index_vectors.set(idx["count"])
        METRICS.index_info.set(
            1,
            index_version=self.manifest["index_version"],
            encoder=m_enc["name"],
            watermark=str(self.manifest.get("source", {}).get("watermark", 0)),
        )
        return dict(self.timings)

    # ------------------------------------------------------------ serving

    def _embed(self, texts: list[str]) -> np.ndarray:
        size = self.settings.query_cache_size
        if size <= 0:
            return self.encoder.encode(texts)
        out = np.empty((len(texts), self.encoder.dim), dtype=np.float32)
        missing = []
        with self._cache_lock:
            for i, t in enumerate(texts):
                v = self._cache.get(t)
                if v is None:
                    missing.append(i)
                else:
                    self._cache.move_to_end(t)
                    out[i] = v
        METRICS.query_cache.inc(len(texts) - len(missing), result="hit")
        METRICS.query_cache.inc(len(missing), result="miss")
        if missing:
            vecs = self.encoder.encode([texts[i] for i in missing])  # outside the lock: the slow part
            with self._cache_lock:
                for i, v in zip(missing, vecs):
                    out[i] = v
                    self._cache[texts[i]] = v
                while len(self._cache) > size:
                    self._cache.popitem(last=False)
        return out

    def process_batch(self, items: list[SearchItem]) -> list[dict]:
        """Encode + search a batch. Runs in a worker thread, never on the event loop
        (except in the deliberately bad EXEC_MODE=inline)."""
        unique: dict[str, int] = {}
        for it in items:
            unique.setdefault(it.query, len(unique))
        texts = list(unique)

        t0 = time.perf_counter()
        vecs = self._embed(texts)
        t1 = time.perf_counter()
        overfetch = max(1, self.settings.filter_overfetch)
        k_max = max(it.k * (overfetch if it.category else 1) for it in items)
        labels, dists = self.ann.query(vecs, k_max)
        t2 = time.perf_counter()
        slow = max((it.simulate_ms for it in items), default=0.0)
        if slow > 0:
            # A batch finishes when its slowest member does: everyone batched with
            # a slow request waits for it (head-of-line blocking).
            time.sleep(slow / 1000.0)
        METRICS.encode_seconds.observe(t1 - t0)
        METRICS.ann_seconds.observe(t2 - t1)
        METRICS.batch_size.observe(len(items))

        out = []
        for it in items:
            row = unique[it.query]
            hits = []
            for label, dist in zip(labels[row], dists[row]):
                label = int(label)
                if label >= self.real_count:  # filler vector from a padded scale-test index
                    if it.category:
                        continue
                    hits.append({"id": f"pad-{label}", "score": round(1.0 - float(dist), 5), "title": "", "category": None})
                else:
                    cat = self.categories[label]
                    if it.category and cat != it.category:
                        continue
                    hits.append(
                        {"id": self.ids[label], "score": round(1.0 - float(dist), 5), "title": self.titles[label], "category": cat}
                    )
                if len(hits) == it.k:
                    break
            out.append({"results": hits})
        return out
