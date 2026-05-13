#!/usr/bin/env python3
"""
Retrieve top-k chunks for inspection (no answering).
Required CLI:
  python scripts/query_chunks.py --q "electrolyzer efficiency" --collection hydrogen_books --topk 8
"""
import os
import sys
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()

from scripts.index_text import get_qdrant_client, get_embeddings_model
from scripts import embed_config
from qdrant_client.models import Filter, FieldCondition, MatchValue


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks (no answering)")
    parser.add_argument("--q", type=str, required=True, help="Query text")
    parser.add_argument("--collection", type=str, required=True, help="Collection name")
    parser.add_argument("--topk", type=int, default=8, help="Number of chunks to return")
    parser.add_argument("--embedding", type=str, default=None, help="Embedding model id (default from env)")
    args = parser.parse_args()

    embedding_id = args.embedding or os.getenv("EMBED_MODEL")
    embeddings = get_embeddings_model(embedding_id)
    client = get_qdrant_client()

    # Optional collection prefix (match index_text)
    prefix = (os.getenv("QDRANT_COLLECTION_PREFIX") or "").strip()
    coll_name = f"{prefix.rstrip('_')}_{args.collection}" if prefix else args.collection

    try:
        query_vector = embeddings.embed_query(args.q)
        results = client.search(
            collection_name=coll_name,
            query_vector=query_vector,
            limit=args.topk,
            with_payload=True,
            query_filter=Filter(must=[FieldCondition(key="is_leaf", match=MatchValue(value=True))])
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Query: {args.q}\nCollection: {coll_name}\nTop-{args.topk} hits:\n")
    for i, r in enumerate(results, 1):
        p = r.payload or {}
        score = r.score
        chunk_id = p.get("chunk_id", "")
        source_path = p.get("source_path", "")
        page_start = p.get("page_start", "")
        page_end = p.get("page_end", "")
        text = p.get("text", "")
        text_preview = (text[:200] + "…") if len(text) > 200 else text
        print(f"--- Hit {i} ---")
        print(f"score:        {score:.4f}")
        print(f"chunk_id:     {chunk_id}")
        print(f"source_path:  {source_path}")
        print(f"page_start:   {page_start}")
        print(f"page_end:     {page_end}")
        print(f"page_start-page_end: {page_start}-{page_end}")
        print(f"text_preview: {text_preview}")
        print()
    sys.exit(0)


if __name__ == "__main__":
    main()
