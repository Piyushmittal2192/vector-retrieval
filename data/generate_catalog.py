"""Generate a synthetic home-improvement catalog with realistic variety.

  python -m data.generate_catalog --n 100000 --out data/catalog.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.taxonomy import ADJECTIVES, BRANDS, all_types

BASE_PRICE = 40.0
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"


def make_product(rng: random.Random, num: int, category: str, type_name: str, spec: dict) -> dict:
    brand = rng.choice(BRANDS[category])
    attrs = {k: rng.choice(v) for k, v in spec["attrs"].items()}
    feats = rng.sample(spec["features"], k=min(len(spec["features"]), rng.randint(2, 3)))
    use = rng.choice(spec["uses"])
    adj = rng.choice(ADJECTIVES)
    model = f"{brand[:2].upper()}{rng.randint(100, 9999)}{rng.choice(LETTERS)}"
    title = f"{brand} {' '.join(attrs.values())} {type_name} with {feats[0]} {model}"
    specs = "; ".join(f"{k}: {v}" for k, v in attrs.items())
    description = (
        f"{adj.capitalize()} {type_name} from {brand}, built for {use}. "
        f"Features {', '.join(feats)}. Specs: {specs}."
    )
    return {
        "id": f"sku-{num:08d}",
        "title": title,
        "description": description,
        "category": category,
        "type": type_name,
        "brand": brand,
        "attrs": attrs,
        "price": round(BASE_PRICE * rng.lognormvariate(0.8, 0.6), 2),
        "active": True,
        "seq": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--out", default="data/catalog.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    types = list(all_types())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    titles = set()
    with open(out, "w") as f:
        for num in range(1, args.n + 1):
            category, type_name, spec = rng.choice(types)
            p = make_product(rng, num, category, type_name, spec)
            titles.add(p["title"])
            f.write(json.dumps(p) + "\n")
    print(f"wrote {args.n:,} products to {out} ({out.stat().st_size / 1e6:.1f} MB); "
          f"{len(titles):,} distinct titles across {len(types)} product types")


if __name__ == "__main__":
    main()
