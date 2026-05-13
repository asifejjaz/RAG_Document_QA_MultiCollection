#!/usr/bin/env python3
"""
Retrieve chunks → call Ollama → return answer with citations.
Hard rule in prompt: "Use only the provided context. If not found, say NOT FOUND. Cite as (source_path p.page_start–page_end)."
Required CLI:
  python scripts/answer_local.py --q "..." --collection hydrogen_books --model qwen2.5:7b-instruct
  python scripts/answer_local.py --q "..." --collection hydrogen_books --model llama3.1:8b-instruct
"""
import os
import sys
import argparse
import requests
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()

from scripts.index_text import get_qdrant_client, get_embeddings_model
from scripts import embed_config
from qdrant_client.models import Filter, FieldCondition, MatchValue


PROMPT_RULE = (
    "Use only the provided context. If not found, say NOT FOUND. "
    "Cite as (source_path p.page_start–page_end)."
)


def main():
    parser = argparse.ArgumentParser(description="Answer from context using Ollama")
    parser.add_argument("--q", type=str, required=True, help="Question")
    parser.add_argument("--collection", type=str, required=True, help="Collection name")
    parser.add_argument("--model", type=str, required=True, help="Ollama model (e.g. qwen2.5:7b-instruct, llama3.1:8b-instruct)")
    parser.add_argument("--topk", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--embedding", type=str, default=None, help="Embedding model id")
    args = parser.parse_args()

    embedding_id = args.embedding or os.getenv("EMBED_MODEL")
    embeddings = get_embeddings_model(embedding_id)
    client = get_qdrant_client()

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

    # If no relevant chunks, short-circuit per PROMPT_RULE
    if not results:
        print("NOT FOUND")
        sys.exit(0)

    context_parts = []
    for r in results:
        p = r.payload or {}
        text = p.get("text", "")
        source_path = p.get("source_path", "")
        page_start = p.get("page_start", "")
        page_end = p.get("page_end", "")
        context_parts.append(f"[{source_path} p.{page_start}–{page_end}]: {text}")

    context_text = "\n\n".join(context_parts)
    system = f"You are a research assistant. {PROMPT_RULE}"
    user_content = f"Context:\n{context_text}\n\nQuestion: {args.q}"

    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            },
            timeout=300,
        )
        r.raise_for_status()
        answer = r.json().get("message", {}).get("content", "")
        print(answer)
    except Exception as e:
        print(f"Ollama error: {e}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
