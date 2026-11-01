"""Catalog snapshot + change log -> the exact product set an index version covers.

catalog.jsonl : one product per line, the base snapshot (seq 0)
changes.jsonl : ordered events {"seq": N, "op": "upsert"|"delete", ...}

An index built "at watermark W" contains the snapshot with every event whose
seq <= W applied, and only products with active == true. The watermark is
stored in the manifest and returned on every response, so a client can tell
how fresh the results are.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_products(catalog: Path, changes: Path | None = None, watermark: int | None = None):
    """Return ({id: product}, applied_watermark, stats)."""
    products: dict[str, dict] = {p["id"]: p for p in iter_jsonl(catalog)}
    applied = 0
    stats = {"snapshot": len(products), "upserts": 0, "deletes": 0}
    if changes and Path(changes).exists():
        last_seq = 0
        for ev in iter_jsonl(changes):
            seq = int(ev["seq"])
            if seq <= last_seq:
                raise ValueError(f"change log out of order at seq {seq}")
            last_seq = seq
            if watermark is not None and seq > watermark:
                break
            if ev["op"] == "upsert":
                p = dict(ev["product"])
                p["seq"] = seq
                products[p["id"]] = p
                stats["upserts"] += 1
            elif ev["op"] == "delete":
                if ev["id"] in products:
                    products[ev["id"]] = {**products[ev["id"]], "active": False, "seq": seq}
                stats["deletes"] += 1
            else:
                raise ValueError(f"unknown op {ev['op']!r} at seq {seq}")
            applied = seq
    return products, applied, stats


def active_sorted(products: dict[str, dict]) -> list[dict]:
    """Active products in id order: label i of the index == position i here."""
    return [products[k] for k in sorted(products) if products[k].get("active", True)]


def product_text(p: dict) -> str:
    return f"{p['title']}. {p['description']} Category: {p['category']}."


# ---------------------------------------------------------------- relevance


def grade(product: dict, target: dict) -> int:
    """0 if the product type is wrong, else 1 + matched brand + matched attributes."""
    if product.get("type") != target["type"] or not product.get("active", True):
        return 0
    g = 1
    if target.get("brand") and product.get("brand") == target["brand"]:
        g += 1
    for key, value in (target.get("attrs") or {}).items():
        if (product.get("attrs") or {}).get(key) == value:
            g += 1
    return g


class Judge:
    """Graded relevance for labeled queries over one product set."""

    def __init__(self, products: dict[str, dict]):
        self.products = products
        self.by_type: dict[str, list[dict]] = {}
        for p in products.values():
            if p.get("active", True):
                self.by_type.setdefault(p["type"], []).append(p)
        self._ideal: dict[str, list[int]] = {}

    def ideal_grades(self, target: dict, k: int) -> list[int]:
        key = json.dumps(target, sort_keys=True)
        if key not in self._ideal:
            grades = sorted((grade(p, target) for p in self.by_type.get(target["type"], [])), reverse=True)
            self._ideal[key] = grades[:k]
        return self._ideal[key]

    def ndcg(self, result_ids: list[str], target: dict, k: int = 10) -> float:
        def dcg(grades):
            return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

        got = [grade(self.products[i], target) if i in self.products else 0 for i in result_ids[:k]]
        ideal = dcg(self.ideal_grades(target, k))
        return dcg(got) / ideal if ideal > 0 else 0.0
