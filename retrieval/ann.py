"""hnswlib build and query helpers.

Thread-safety: hnswlib's knn_query may run from several threads at once
(it releases the GIL while searching), but ef is one shared setting on the
index. When a request needs k > ef, ef is raised under a lock and never
lowered again: concurrent searches may then run with a larger ef (slower,
never less accurate). Restarting the pod resets it.
"""
from __future__ import annotations

import threading
import time

import hnswlib
import numpy as np


def hnswlib_version() -> str:
    try:
        from importlib.metadata import version

        return version("hnswlib")
    except Exception:
        return "unknown"


def build_hnsw(
    vectors: np.ndarray,
    M: int = 16,
    ef_construction: int = 200,
    threads: int = -1,
    seed: int = 100,
    chunk: int = 100_000,
    log=print,
) -> hnswlib.Index:
    n, dim = vectors.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M, random_seed=seed)
    index.set_num_threads(threads if threads and threads > 0 else -1)
    t0 = time.perf_counter()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        index.add_items(vectors[start:end], np.arange(start, end), num_threads=threads if threads > 0 else -1)
        if n > chunk:
            rate = end / max(time.perf_counter() - t0, 1e-9)
            log(f"  hnsw: {end:,}/{n:,} vectors ({rate:,.0f}/s)")
    return index


class AnnIndex:
    def __init__(self, path: str, dim: int, count: int, default_ef: int, metric: str = "cosine"):
        self.index = hnswlib.Index(space=metric, dim=dim)
        self.index.load_index(path)
        actual = self.index.get_current_count()
        if actual != count:
            raise RuntimeError(f"index holds {actual} vectors, manifest says {count}")
        self.count = count
        self.dim = dim
        self.default_ef = default_ef
        self._ef = default_ef
        self._ef_lock = threading.Lock()
        self.index.set_ef(default_ef)
        self.index.set_num_threads(1)

    def query(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Batch query; safe to call from several threads."""
        k = max(1, min(k, self.count))
        if k > self._ef:  # hnswlib needs ef >= k
            with self._ef_lock:
                if k > self._ef:
                    self.index.set_ef(k)
                    self._ef = k
        return self.index.knn_query(vectors, k=k, num_threads=1)


def brute_force_topk(base: np.ndarray, queries: np.ndarray, k: int, chunk: int = 50_000) -> np.ndarray:
    """Exact top-k by inner product (vectors are normalised, so = cosine)."""
    nq = queries.shape[0]
    best_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    best_ids = np.full((nq, k), -1, dtype=np.int64)
    for start in range(0, base.shape[0], chunk):
        block = base[start : start + chunk]
        scores = queries @ block.T
        kk = min(k, scores.shape[1])
        idx = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        cand_scores = np.take_along_axis(scores, idx, axis=1)
        merged_scores = np.concatenate([best_scores, cand_scores], axis=1)
        merged_ids = np.concatenate([best_ids, idx + start], axis=1)
        order = np.argsort(-merged_scores, axis=1)[:, :k]
        best_scores = np.take_along_axis(merged_scores, order, axis=1)
        best_ids = np.take_along_axis(merged_ids, order, axis=1)
    return best_ids


def recall_at_k(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    hits = 0
    for a, e in zip(approx[:, :k], exact[:, :k]):
        hits += len(set(a.tolist()) & set(e.tolist()))
    return hits / (exact.shape[0] * k)
