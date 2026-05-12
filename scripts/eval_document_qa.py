#!/usr/bin/env python3
"""Evaluate QA over an ingested collection using Azure OpenAI chat."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.index_text import get_qdrant_client, get_embeddings_model
from scripts.azure_openai_env import set_env


def main():
    embedding_id = os.getenv("EMBED_MODEL", "azure_ada")
    embeddings = get_embeddings_model(embedding_id)
    client = get_qdrant_client()
    az_model_client, chat_client, _ = set_env()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    collection = "rag_hydrogen_books"
    queries = [
        "What are the main forms of hydrogen discussed for bunkering?",
        "Which port is described as aiming to become a green hydrogen hub?",
        "What are the safety or storage challenges of hydrogen bunkering?",
    ]

    for q in queries:
        print("QUERY:", q)
        query_vector = embeddings.embed_query(q)
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=5,
            with_payload=True,
        )
        contexts = []
        for r in results:
            p = r.payload or {}
            context_line = (
                f"[{p.get('source_path','unknown')} p.{p.get('page_start','?')}-{p.get('page_end','?')}]: "
                f"{p.get('text','')}"
            )
            contexts.append(context_line)

        prompt = (
            "Use only the provided context. If not found, reply exactly NOT FOUND. "
            "Cite sources using source_path and page range."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Context:\n" + "\n\n".join(contexts) + "\n\nQuestion: " + q,
            },
        ]

        resp = chat_client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        ans = resp.choices[0].message.content
        print("ANSWER:")
        print(ans)
        print("---\n")


if __name__ == "__main__":
    main()
