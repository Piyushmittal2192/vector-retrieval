"""Append product change events to the change log.

  python -m data.generate_changes --catalog data/catalog.jsonl --changes data/changes.jsonl --n 5000

Mix: 55% updates (new description/price), 25% new products, 20% deletes.
Each event gets the next sequence number; the last one printed is the
watermark you pass to the index builder.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.generate_catalog import make_product
from data.taxonomy import T
from retrieval.catalog import load_products


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--changes", default="data/changes.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    products, last_seq, _ = load_products(Path(args.catalog), Path(args.changes))
    rng = random.Random(args.seed if args.seed is not None else last_seq + 1)
    active = [pid for pid, p in products.items() if p.get("active", True)]
    next_num = max(int(pid.split("-")[1]) for pid in products) + 1
    counts = {"update": 0, "add": 0, "delete": 0}
    seq = last_seq
    with open(args.changes, "a") as f:
        for _ in range(args.n):
            seq += 1
            r = rng.random()
            if r < 0.55 and active:
                pid = rng.choice(active)
                old = products[pid]
                spec = T[old["category"]][old["type"]]
                fresh = make_product(rng, int(pid.split("-")[1]), old["category"], old["type"], spec)
                new = {**old, "description": fresh["description"], "price": fresh["price"]}
                products[pid] = new
                f.write(json.dumps({"seq": seq, "op": "upsert", "product": new}) + "\n")
                counts["update"] += 1
            elif r < 0.80:
                category = rng.choice(list(T))
                type_name = rng.choice(list(T[category]))
                p = make_product(rng, next_num, category, type_name, T[category][type_name])
                next_num += 1
                products[p["id"]] = p
                active.append(p["id"])
                f.write(json.dumps({"seq": seq, "op": "upsert", "product": p}) + "\n")
                counts["add"] += 1
            else:
                pid = active.pop(rng.randrange(len(active)))
                products[pid] = {**products[pid], "active": False}
                f.write(json.dumps({"seq": seq, "op": "delete", "id": pid}) + "\n")
                counts["delete"] += 1
    print(f"appended {args.n:,} events seq {last_seq + 1}..{seq} to {args.changes}: {counts}")
    print(f"WATERMARK={seq}")


if __name__ == "__main__":
    main()
