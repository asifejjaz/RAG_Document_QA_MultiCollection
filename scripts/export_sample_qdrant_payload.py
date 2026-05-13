#!/usr/bin/env python3
"""
Export one scroll point from a Qdrant collection as JSON (canonical payload sample for clients).

Usage:
  python scripts/export_sample_qdrant_payload.py --collection hydrogen_books
  python scripts/export_sample_qdrant_payload.py --collection hydrogen_books --out sample_payload.json

Requires Qdrant URL in .env (QDRANT_URL or VECTOR_DB_URL).
"""
import argparse
import json
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv()

from scripts.index_text import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue


def main():
    parser = argparse.ArgumentParser(description="Export one Qdrant point payload as JSON")
    parser.add_argument("--collection", type=str, required=True, help="Logical collection name")
    parser.add_argument("--out", type=str, default="", help="Output file (default: stdout)")
    parser.add_argument("--leaf", action="store_true", help="Export a leaf chunk (is_leaf=true) instead of any point")
    args = parser.parse_args()

    prefix = (os.getenv("QDRANT_COLLECTION_PREFIX") or "").strip()
    coll = f"{prefix.rstrip('_')}_{args.collection}" if prefix else args.collection

    client = get_qdrant_client()

    scroll_filter = None
    if args.leaf:
        scroll_filter = Filter(must=[FieldCondition(key="is_leaf", match=MatchValue(value=True))])

    points, _ = client.scroll(
        collection_name=coll,
        limit=1,
        with_payload=True,
        with_vectors=False,
        scroll_filter=scroll_filter
    )
    if not points:
        print(f"No points in collection: {coll}", file=sys.stderr)
        sys.exit(1)

    payload = points[0].payload or {}
    sample = {
        "id": points[0].id,
        "payload": dict(payload),
    }
    text = json.dumps(sample, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
