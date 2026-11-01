"""Generate labeled search queries with graded relevance targets.

Each line: {"qid", "query", "category", "target": {"type", "brand"?, "attrs"?}, "golden": bool}
A product's grade for a query = 0 if the type differs, else 1 + matched brand
+ matched attributes (see retrieval.catalog.grade). About a third of queries
use a synonym instead of the type name, so lexical overlap alone is not
enough to score well.

  python -m data.generate_queries --n 2000 --out data/queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.taxonomy import BRANDS, all_types

N_GOLDEN = 20


def make_query(rng: random.Random, category: str, type_name: str, spec: dict) -> tuple[str, dict]:
    noun = rng.choice(spec["synonyms"]) if rng.random() < 0.35 else type_name
    attr_key = rng.choice(list(spec["attrs"]))
    attr_val = rng.choice(spec["attrs"][attr_key])
    brand = rng.choice(BRANDS[category])
    r = rng.random()
    if r < 0.25:
        return f"{brand} {noun}", {"type": type_name, "brand": brand}
    if r < 0.50:
        return f"{attr_val} {noun}", {"type": type_name, "attrs": {attr_key: attr_val}}
    if r < 0.70:
        return f"{brand} {attr_val} {noun}", {"type": type_name, "brand": brand, "attrs": {attr_key: attr_val}}
    if r < 0.85:
        return f"{noun} for {rng.choice(spec['uses'])}", {"type": type_name}
    feature = rng.choice(spec["features"])
    return f"{noun} with {feature}", {"type": type_name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/queries.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    types = list(all_types())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    while len(rows) < args.n:
        category, type_name, spec = types[len(rows) % len(types)] if len(rows) < N_GOLDEN else rng.choice(types)
        text, target = make_query(rng, category, type_name, spec)
        text = text.lower()
        if text in seen:
            continue
        seen.add(text)
        rows.append({"qid": f"q{len(rows):05d}", "query": text, "category": category, "target": target, "golden": len(rows) < N_GOLDEN})
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows):,} queries ({N_GOLDEN} golden) to {out}")


if __name__ == "__main__":
    main()
