#!/usr/bin/env python3
"""
Numeric preservation check: compare digit-like tokens in each chunk vs source page text.

Resolves source files via payload source_path or by searching --data-root for file_name.
Output: /state/reports/numeric_issues.json (extended with page comparison when source is found)

Required CLI:
  python scripts/check_numbers.py --collection hydrogen_books
  python scripts/check_numbers.py --collection hydrogen_books --data-root ./data
"""
from __future__ import annotations

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv()

from scripts.index_text import get_qdrant_client


def extract_numbers(text: str) -> List[str]:
    """Return list of numeric tokens (digits, decimals, percentages, comma-separated)."""
    if not text:
        return []
    return re.findall(r"\d+\.?\d*%?|\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)


def _counter(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _multiset_delta(a: Counter, b: Counter) -> Tuple[List[str], List[str]]:
    """Return (elements in b not fully covered by a, elements in a not in b) for digit tokens."""
    missing_in_chunk: List[str] = []
    extra_in_chunk: List[str] = []
    for tok, count in b.items():
        if a[tok] < count:
            missing_in_chunk.extend([tok] * (count - a[tok]))
    for tok, count in a.items():
        if b[tok] < count:
            extra_in_chunk.extend([tok] * (count - b[tok]))
    return missing_in_chunk, extra_in_chunk


def _pdf_page_text(path: Path, page_1indexed: int) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""
    try:
        doc = fitz.open(str(path))
        try:
            if page_1indexed < 1 or page_1indexed > len(doc):
                return ""
            return doc[page_1indexed - 1].get_text() or ""
        finally:
            doc.close()
    except Exception:
        return ""


def _docx_head_text(path: Path, max_chars: int = 50000) -> str:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        return ""
    try:
        docs = Docx2txtLoader(str(path)).load()
        return "\n".join(d.page_content for d in docs)[:max_chars]
    except Exception:
        return ""


def resolve_source_file(payload: Dict[str, Any], data_root: Optional[Path]) -> Optional[Path]:
    """Best-effort path to original file for side-by-side checks."""
    sp = (payload.get("source_path") or "").strip()
    if sp:
        p = Path(sp)
        if p.is_file():
            return p
    fn = (payload.get("file_name") or "").strip()
    if not fn or not data_root or not data_root.is_dir():
        return None
    direct = data_root / fn
    if direct.is_file():
        return direct
    for sub in data_root.iterdir():
        if sub.is_dir():
            cand = sub / fn
            if cand.is_file():
                return cand
    try:
        for cand in data_root.rglob(fn):
            if cand.is_file():
                return cand
    except OSError:
        pass
    return None


def load_page_text(path: Path, page_start: int) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf_page_text(path, page_start)
    if ext in (".docx", ".doc"):
        return _docx_head_text(path)
    return ""


def check_chunk(
    payload: Dict[str, Any],
    data_root: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Return a report dict if issues found."""
    text = (payload.get("text") or "").strip()
    chunk_id = payload.get("chunk_id", "")
    issues: List[str] = []
    numbers = extract_numbers(text)

    if len(text) > 100 and not numbers:
        issues.append("no_numeric_tokens_in_chunk")
    if chunk_id and not re.search(r"\d", str(chunk_id)):
        issues.append("chunk_id_has_no_digits")

    page_start = payload.get("page_start")
    if page_start is None:
        page_start = payload.get("page_number")
    try:
        page_start = int(page_start) if page_start is not None else 1
    except (TypeError, ValueError):
        page_start = 1

    src = resolve_source_file(payload, data_root)
    page_text = ""
    missing_in_chunk: List[str] = []
    extra_in_chunk: List[str] = []
    comparison_status = "skipped_no_source"

    if src and src.is_file():
        page_text = load_page_text(src, page_start)
        if page_text:
            comparison_status = "compared"
            pn = extract_numbers(text)
            pp = extract_numbers(page_text)
            missing_in_chunk, extra_in_chunk = _multiset_delta(_counter(pn), _counter(pp))
            if missing_in_chunk:
                issues.append("numbers_on_page_missing_in_chunk")
            if extra_in_chunk:
                issues.append("numbers_in_chunk_not_on_page")
        else:
            comparison_status = "source_page_empty"
    elif data_root:
        comparison_status = "source_file_not_found"

    if not issues:
        return None

    return {
        "chunk_id": chunk_id,
        "source_path": payload.get("source_path", ""),
        "resolved_source": str(src) if src else None,
        "doc_id": payload.get("doc_id"),
        "page_start": page_start,
        "page_end": payload.get("page_end"),
        "comparison_status": comparison_status,
        "numbers_in_chunk": numbers,
        "numbers_on_page_sample": extract_numbers(page_text)[:40] if page_text else [],
        "missing_in_chunk": missing_in_chunk[:50],
        "extra_in_chunk": extra_in_chunk[:50],
        "issues": issues,
        "text_preview": text[:150] + "…" if len(text) > 150 else text,
        "page_text_preview": (page_text[:200] + "…") if len(page_text) > 200 else page_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Numeric preservation check (chunk vs source page)")
    parser.add_argument("--collection", type=str, required=True, help="Logical collection name (prefix applied if set in env)")
    parser.add_argument("--state", type=str, default=os.getenv("STATE_ROOT", "./state"), help="State directory for output")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("DATA_ROOT", ""),
        help="Corpus root to resolve file_name (e.g. ./data). Optional but needed for page comparison.",
    )
    args = parser.parse_args()

    prefix = (os.getenv("QDRANT_COLLECTION_PREFIX") or "").strip()
    coll_name = f"{prefix.rstrip('_')}_{args.collection}" if prefix else args.collection

    data_root = Path(args.data_root).resolve() if args.data_root else None
    if data_root and not data_root.is_dir():
        data_root = None

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
                item = check_chunk(payload, data_root)
                if item:
                    all_issues.append(item)
            if offset is None:
                break
    except Exception as e:
        print(f"Error: collection {coll_name}: {e}")
        sys.exit(1)

    out = {
        "collection": coll_name,
        "data_root": str(data_root) if data_root else None,
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
