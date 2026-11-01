"""Download the encoder once into ./models so the builder and the image use
byte-identical weights (and the container never talks to Hugging Face).

  python -m retrieval.download_model --model sentence-transformers/all-MiniLM-L6-v2 --out models/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--revision", default=None, help="pin a commit hash for full reproducibility")
    ap.add_argument("--out", default="models/all-MiniLM-L6-v2")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device="cpu", revision=args.revision)
    out = Path(args.out)
    model.save(str(out))
    (out / "LAB_SOURCE.json").write_text(json.dumps({"model": args.model, "revision": args.revision}, indent=2))
    print(f"saved {args.model} to {out}")


if __name__ == "__main__":
    main()
