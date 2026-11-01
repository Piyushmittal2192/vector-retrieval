"""Offline index builder: catalog @ watermark -> immutable index version.

  python -m retrieval.build_index --version v1
  python -m retrieval.build_index --version v2 --watermark 5000 --baseline-cache
  python -m retrieval.build_index --version v1-2m --pad 1700000 --ef-construction 100

Steps
  1. Apply change events up to the watermark; keep active products, sorted by id.
  2. Embed product text, reusing cached vectors for unchanged text (keyed by
     encoder fingerprint + sha1 of text). A v2 build re-embeds only what changed.
  3. Optionally append random unit "filler" vectors to reach a scale tier
     without hours of encoding (they almost never rank for real queries).
  4. Build HNSW, write index.bin + docs.json into a temp dir.
  5. Measure quality: recall@10 vs brute force on labeled queries, nDCG@10.
  6. Record golden-query results the server must reproduce at startup.
  7. Write manifest.json last (checksums of everything), atomically publish.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from retrieval import artifacts
from retrieval.ann import brute_force_topk, build_hnsw, hnswlib_version, recall_at_k
from retrieval.catalog import Judge, active_sorted, iter_jsonl, load_products, product_text
from retrieval.encoders import fingerprint, fingerprint_id, load_encoder


class EmbeddingCache:
    """sha1(text) -> vector, one cache per encoder fingerprint."""

    def __init__(self, root: Path, encoder_key: str):
        self.dir = Path(root) / encoder_key
        self.keys_path, self.vecs_path = self.dir / "keys.npy", self.dir / "vecs.npy"
        self.index: dict[bytes, int] = {}
        self.vecs = None
        if self.keys_path.exists() and self.vecs_path.exists():
            keys = np.load(self.keys_path)  # (n, 20) uint8; NOT an 'S20' array, which strips trailing NUL bytes
            self.vecs = np.load(self.vecs_path, mmap_mode="r")
            self.index = {k.tobytes(): i for i, k in enumerate(keys)}
        self.new_keys: list[bytes] = []
        self.new_vecs: list[np.ndarray] = []

    @staticmethod
    def key(text: str) -> bytes:
        return hashlib.sha1(text.encode()).digest()

    def save(self) -> None:
        if not self.new_keys:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        old_keys = np.load(self.keys_path) if self.keys_path.exists() else np.zeros((0, 20), dtype=np.uint8)
        new_keys = np.frombuffer(b"".join(self.new_keys), dtype=np.uint8).reshape(-1, 20)
        keys = np.concatenate([old_keys, new_keys])
        parts = ([np.asarray(self.vecs)] if self.vecs is not None else []) + [np.vstack(self.new_vecs)]
        vecs = np.vstack(parts).astype(np.float32)
        for path, arr in ((self.keys_path, keys), (self.vecs_path, vecs)):
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, path)


def embed_with_cache(encoder, texts: list[str], cache: EmbeddingCache | None, batch: int = 256, log=print) -> tuple[np.ndarray, int]:
    out = np.empty((len(texts), encoder.dim), dtype=np.float32)
    todo = []
    for i, t in enumerate(texts):
        row = cache.index.get(EmbeddingCache.key(t)) if cache else None
        if row is None:
            todo.append(i)
        else:
            out[i] = cache.vecs[row]
    hits = len(texts) - len(todo)
    log(f"  embeddings: {hits:,} reused from cache, {len(todo):,} to encode")
    t0 = time.perf_counter()
    step = 4096
    for start in range(0, len(todo), step):
        idx = todo[start : start + step]
        vecs = encoder.encode([texts[i] for i in idx], batch_size=batch)
        out[idx] = vecs
        if cache:
            cache.new_keys.extend(EmbeddingCache.key(texts[i]) for i in idx)
            cache.new_vecs.append(vecs)
        done = start + len(idx)
        rate = done / max(time.perf_counter() - t0, 1e-9)
        eta = (len(todo) - done) / max(rate, 1e-9)
        log(f"  encoded {done:,}/{len(todo):,} ({rate:,.0f}/s, eta {eta:,.0f}s)")
    return out, hits


def fill_random_unit(out: np.ndarray, seed: int) -> None:
    """Fill `out` in place with random unit vectors (no second full-size copy)."""
    rng = np.random.default_rng(seed)
    for start in range(0, out.shape[0], 200_000):
        block = rng.standard_normal((min(200_000, out.shape[0] - start), out.shape[1]), dtype=np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        out[start : start + len(block)] = block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="new, never-used version name, e.g. v1, v2, v1-2m")
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--changes", default="data/changes.jsonl")
    ap.add_argument("--watermark", type=int, default=None, help="apply change events with seq <= this (default: all)")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--out", default="artifacts/indexes")
    ap.add_argument("--cache-dir", default="cache/embeddings")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--encoder", default=os.getenv("ENCODER_BACKEND", "st"), choices=["st", "hash"])
    ap.add_argument("--model-path", default=os.getenv("MODEL_PATH", "models/all-MiniLM-L6-v2"))
    ap.add_argument("--model-name", default=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--threads", type=int, default=0, help="build/encode threads (0 = all cores)")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef", type=int, default=64, help="default query-time ef stored in the manifest")
    ap.add_argument("--pad", type=int, default=0, help="add N random filler vectors (scale tests)")
    ap.add_argument("--eval-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    out_root = Path(args.out)
    final = out_root / args.version
    if final.exists():
        raise SystemExit(f"{final} exists. Versions are immutable: choose a new --version.")
    t_all = time.perf_counter()
    stats: dict = {}

    print(f"[1/6] loading catalog (watermark={args.watermark if args.watermark is not None else 'latest'})")
    products, watermark, change_stats = load_products(Path(args.catalog), Path(args.changes), args.watermark)
    items = active_sorted(products)
    texts = [product_text(p) for p in items]
    print(f"  {len(items):,} active products, applied watermark {watermark}, {change_stats}")

    print(f"[2/6] encoder backend={args.encoder}")
    if args.threads > 0 and args.encoder == "st":
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    encoder = load_encoder(args.encoder, args.model_path, args.model_name, args.threads)
    if args.encoder == "st":
        encoder.name = args.model_name
    fp = fingerprint(encoder)
    cache = None if args.no_cache else EmbeddingCache(Path(args.cache_dir), f"{encoder.name.replace('/', '__')}-{fingerprint_id(fp)}")
    t0 = time.perf_counter()
    vectors, reused = embed_with_cache(encoder, texts, cache)
    stats["embed_seconds"] = round(time.perf_counter() - t0, 2)
    stats["embeddings_reused"] = reused
    if cache:
        cache.save()

    if args.pad:
        print(f"  padding with {args.pad:,} filler vectors")
        full = np.empty((len(vectors) + args.pad, encoder.dim), dtype=np.float32)
        full[: len(vectors)] = vectors
        del vectors
        fill_random_unit(full[len(items):], args.seed)
        vectors = full

    print(f"[3/6] building HNSW (M={args.M}, ef_construction={args.ef_construction}) over {len(vectors):,} vectors")
    t0 = time.perf_counter()
    index = build_hnsw(vectors, args.M, args.ef_construction, args.threads or -1, args.seed)
    index.set_ef(args.ef)
    stats["hnsw_build_seconds"] = round(time.perf_counter() - t0, 2)

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out_root, prefix=f".{args.version}."))
    try:
        index.save_index(str(tmp / "index.bin"))
        with open(tmp / "docs.json", "w") as f:
            json.dump({"ids": [p["id"] for p in items], "titles": [p["title"] for p in items], "categories": [p["category"] for p in items]}, f)

        print("[4/6] evaluating quality")
        queries = list(iter_jsonl(Path(args.queries))) if Path(args.queries).exists() else []
        quality = {}
        golden_rows = []
        if queries:
            eval_q = queries[: args.eval_queries]
            qv = encoder.encode([q["query"] for q in eval_q])
            labels, _ = index.knn_query(qv, k=10)
            exact = brute_force_topk(vectors, qv, 10)
            judge = Judge(products)
            ids = [p["id"] for p in items]

            def to_ids(row):
                return [ids[int(l)] if int(l) < len(ids) else f"pad-{int(l)}" for l in row]

            ndcgs = [judge.ndcg(to_ids(row), q["target"]) for row, q in zip(labels, eval_q)]
            quality = {
                "recall_at_10": round(recall_at_k(labels.astype(np.int64), exact, 10), 4),
                "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
                "eval_queries": len(eval_q),
                "ef": args.ef,
            }
            golden = [q for q in queries if q.get("golden")]
            if golden:
                gl, _ = index.knn_query(encoder.encode([g["query"] for g in golden]), k=10)
                golden_rows = [{"query": g["query"], "top_ids": to_ids(row)} for g, row in zip(golden, gl)]
            print(f"  recall@10={quality['recall_at_10']} ndcg@10={quality['ndcg_at_10']} on {len(eval_q)} queries")

        print("[5/6] writing manifest")
        index_bytes = (tmp / "index.bin").stat().st_size
        docs_bytes = (tmp / "docs.json").stat().st_size
        # index + Python objects for docs + torch runtime and model (~1 GiB measured headroom)
        est_mib = int((index_bytes * 1.1 + docs_bytes * 6) / 2**20 + 1024)
        manifest = {
            "index_version": args.version,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "encoder": {"name": encoder.name, "backend": args.encoder, "dimension": encoder.dim, "normalized": True, "fingerprint": fp},
            "index": {
                "library": "hnswlib",
                "library_version": hnswlib_version(),
                "metric": "cosine",
                "M": args.M,
                "ef_construction": args.ef_construction,
                "default_ef": args.ef,
                "count": int(len(vectors)),
                "real_count": len(items),
                "pad_count": args.pad,
                "estimated_pod_memory_mib": est_mib,
            },
            "source": {"catalog": str(args.catalog), "changes": str(args.changes), "watermark": watermark, **change_stats},
            "quality": quality,
            "golden": golden_rows,
            "build": {**stats, "total_seconds": round(time.perf_counter() - t_all, 2), "host": platform.node(), "python": platform.python_version()},
            "files": {name: artifacts.file_entry(tmp / name) for name in ("index.bin", "docs.json")},
        }
        artifacts.write_json_atomic(tmp / artifacts.MANIFEST, manifest)

        print("[6/6] publishing")
        artifacts.publish(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"published {final}: {len(vectors):,} vectors, index {index_bytes / 1e6:.1f} MB, "
          f"suggested pod memory {est_mib} MiB, total {manifest['build']['total_seconds']}s")


if __name__ == "__main__":
    main()
