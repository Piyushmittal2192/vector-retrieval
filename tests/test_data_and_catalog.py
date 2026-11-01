import json

from retrieval.catalog import Judge, active_sorted, grade, load_products


def test_catalog_is_varied(lab):
    rows = [json.loads(l) for l in open(lab / "data/catalog.jsonl")]
    assert len(rows) == 2000
    assert len({r["title"] for r in rows}) == len(rows), "every SKU text must be distinct"
    assert len({r["type"] for r in rows}) >= 30


def test_change_log_applies_up_to_watermark(lab):
    events = [json.loads(l) for l in open(lab / "data/changes.jsonl")]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    deleted = next(e["id"] for e in events if e["op"] == "delete")
    seq = next(e["seq"] for e in events if e["op"] == "delete")
    before, wm_before, _ = load_products(lab / "data/catalog.jsonl", lab / "data/changes.jsonl", seq - 1)
    after, wm_after, _ = load_products(lab / "data/catalog.jsonl", lab / "data/changes.jsonl", seq)
    assert wm_before == seq - 1 and wm_after == seq
    assert deleted not in {p["id"] for p in active_sorted(after)}


def test_grading_and_ndcg():
    p = {"id": "a", "type": "cordless drill", "brand": "Voltara", "attrs": {"voltage": "20V"}, "active": True}
    assert grade(p, {"type": "cordless drill", "brand": "Voltara", "attrs": {"voltage": "20V"}}) == 3
    assert grade(p, {"type": "impact driver"}) == 0
    judge = Judge({"a": p})
    assert judge.ndcg(["a"], {"type": "cordless drill"}) == 1.0
