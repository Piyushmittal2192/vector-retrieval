"""Promotion gate for a built index version. Exit code 0 = safe to deploy.

  python -m retrieval.validate_index --version v2 --baseline v1

Checks
  integrity   manifest present, every file matches sha256 and size
  load        index loads, vector count == manifest == docs
  encoder     current encoder reproduces the manifest fingerprint
  golden      golden queries reproduce the build-time top-10
  recall      build-time recall@10 vs brute force >= --min-recall
  ndcg        live nDCG@10 on labeled queries >= --min-ndcg
  freshness   no product deleted at or before the watermark is in the index
  baseline    (with --baseline) nDCG drop <= --max-ndcg-drop, overlap@10 reported

Writes artifacts/validation/<version>.json; it is NOT written into the
version directory, which stays immutable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from retrieval import artifacts
from retrieval.catalog import Judge, iter_jsonl, load_products
from retrieval.config import Settings
from retrieval.engine import SearchEngine, SearchItem


def load_engine(root: Path, version: str, args, encoder=None) -> SearchEngine:
    s = Settings(
        index_dir=str(root / version),
        encoder_backend=args.encoder,
        model_path=args.model_path,
        model_name=args.model_name,
        torch_threads=args.threads,
        golden_min_overlap=0.0,  # checked explicitly below so it is reported, not raised
    )
    eng = SearchEngine(s, encoder)
    eng.load()
    return eng


def search_all(eng: SearchEngine, queries: list[dict], batch: int = 64) -> list[list[str]]:
    out = []
    for start in range(0, len(queries), batch):
        res = eng.process_batch([SearchItem(q["query"], 10) for q in queries[start : start + batch]])
        out.extend([[r["id"] for r in x["results"]] for x in res])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--root", default="artifacts/indexes")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--changes", default="data/changes.jsonl")
    ap.add_argument("--encoder", default="st", choices=["st", "hash"])
    ap.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--min-ndcg", type=float, default=0.0)
    ap.add_argument("--max-ndcg-drop", type=float, default=0.02)
    ap.add_argument("--min-golden", type=float, default=0.9)
    ap.add_argument("--out", default="artifacts/validation")
    args = ap.parse_args()

    root = Path(args.root)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<10} {detail}")

    print(f"validating {args.version}" + (f" against baseline {args.baseline}" if args.baseline else ""))
    try:
        manifest = artifacts.read_manifest(root / args.version)
        artifacts.verify_files(root / args.version, manifest)
        check("integrity", True, f"{len(manifest['files'])} files match sha256")
    except artifacts.ArtifactError as exc:
        check("integrity", False, str(exc))
        return finish(args, checks)

    try:
        eng = load_engine(root, args.version, args)
        check("load", True, f"{manifest['index']['count']:,} vectors, startup {eng.timings}")
        check("encoder", True, f"fingerprint matches {manifest['encoder']['name']}")
    except Exception as exc:
        check("load", False, f"{type(exc).__name__}: {exc}")
        return finish(args, checks)

    golden = eng.timings.get("golden_overlap", 1.0)
    check("golden", golden >= args.min_golden, f"mean overlap {golden:.3f} (min {args.min_golden})")

    recall = manifest.get("quality", {}).get("recall_at_10")
    check("recall", recall is not None and recall >= args.min_recall, f"recall@10 {recall} (min {args.min_recall})")

    wm = manifest.get("source", {}).get("watermark")
    products, _, _ = load_products(Path(args.catalog), Path(args.changes), wm)
    queries = list(iter_jsonl(Path(args.queries)))
    judge = Judge(products)
    results = search_all(eng, queries)
    ndcg = float(np.mean([judge.ndcg(r, q["target"]) for r, q in zip(results, queries)]))
    check("ndcg", ndcg >= args.min_ndcg, f"nDCG@10 {ndcg:.4f} on {len(queries)} queries (min {args.min_ndcg})")

    inactive = {pid for pid, p in products.items() if not p.get("active", True)}
    stale = [i for i in eng.ids if i in inactive]
    check("freshness", not stale, f"{len(stale)} deleted products present (watermark {wm})")

    summary = {"ndcg_at_10": round(ndcg, 4), "golden_overlap": golden, "watermark": wm}
    if args.baseline:
        base = load_engine(root, args.baseline, args, encoder=eng.encoder)
        base_results = search_all(base, queries)
        base_ndcg = float(np.mean([judge.ndcg(r, q["target"]) for r, q in zip(base_results, queries)]))
        overlap = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(results, base_results)]))
        top1 = float(np.mean([bool(a and b and a[0] == b[0]) for a, b in zip(results, base_results)]))
        drop = base_ndcg - ndcg
        check(
            "baseline",
            drop <= args.max_ndcg_drop,
            f"nDCG {base_ndcg:.4f} -> {ndcg:.4f} (drop {drop:+.4f}, max {args.max_ndcg_drop}); "
            f"overlap@10 {overlap:.3f}, top-1 agreement {top1:.3f}",
        )
        summary.update({"baseline": args.baseline, "baseline_ndcg_at_10": round(base_ndcg, 4), "overlap_at_10": round(overlap, 4), "top1_agreement": round(top1, 4)})
    return finish(args, checks, summary)


def finish(args, checks: list[dict], summary: dict | None = None) -> int:
    ok = all(c["pass"] for c in checks)
    report = {
        "version": args.version,
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "passed": ok,
        "checks": checks,
        "summary": summary or {},
    }
    artifacts.write_json_atomic(Path(args.out) / f"{args.version}.json", report)
    print(f"{'PASSED' if ok else 'FAILED'}: report in {Path(args.out) / (args.version + '.json')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
