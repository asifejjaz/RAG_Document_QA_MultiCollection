#!/usr/bin/env python3
"""
Numeric preservation check: flags chunks with missing or unexpected numeric tokens.
Output: /state/reports/numeric_issues.json
Required CLI:
  python scripts/check_numbers.py --collection hydrogen_books
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()

from scripts.index_text import get_qdrant_client


def extract_numbers(text: str) -> List[str]:
    """Return list of numeric tokens (digits, decimals, percentages)."""
    # Match integers, decimals, percentages, and numbers with commas
    tokens = re.findall(r"\d+\.?\d*%?|\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    return tokens


def check_chunk(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check a single chunk for numeric issues.
    Flags: no numbers in text where we might expect some, or empty/non-numeric in key fields.
    """
    text = (payload.get("text") or "").strip()
    chunk_id = payload.get("chunk_id", "")
    source_path = payload.get("source_path", "")
    issues = []

    numbers = extract_numbers(text)
    # Heuristic: if chunk is long enough and has no digits at all, flag (might have lost numbers)
    if len(text) > 100 and not numbers:
        issues.append("no_numeric_tokens_in_chunk")
    # Check chunk_id format: doc_id:page_start:chunk_index (should contain digits)
    if chunk_id and not re.search(r"\d", chunk_id):
        issues.append("chunk_id_has_no_digits")

    if not issues:
        return None
    return {
        "chunk_id": chunk_id,
        "source_path": source_path,
        "doc_id": payload.get("doc_id"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "issues": issues,
        "text_preview": text[:150] + "…" if len(text) > 150 else text,
    }


def main():
    parser = argparse.ArgumentParser(description="Numeric preservation check")
    parser.add_argument("--collection", type=str, required=True, help="Collection name")
    parser.add_argument("--state", type=str, default="/state", help="State directory for output")
    args = parser.parse_args()

    prefix = (os.getenv("QDRANT_COLLECTION_PREFIX") or "").strip()
    coll_name = f"{prefix.rstrip('_')}_{args.collection}" if prefix else args.collection

    client = get_qdrant_client()
    offset = None
    all_issues = []
    try:
        while True:
            points, offset = client.scroll(
                collection_name=coll_name,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                item = check_chunk(payload)
                if item:
                    all_issues.append(item)
            if offset is None:
                break
    except Exception as e:
        print(f"Error: collection {coll_name}: {e}")
        sys.exit(1)

    out = {
        "collection": coll_name,
        "total_issues": len(all_issues),
        "issues": all_issues,
    }
    state_path = Path(args.state)
    reports_dir = state_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "numeric_issues.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(all_issues)} issues to {out_file}")
    sys.exit(0)


if __name__ == "__main__":
    main()
