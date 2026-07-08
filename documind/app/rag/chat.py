"""Grounded, cited answering with hierarchical context (leaf retrieval + parent context)."""
from app import config
from app.rag import embed, store

SYSTEM = """You are a document assistant. Answer ONLY from the provided context passages.
- Cite sources inline like [1], [2] matching the numbered passages.
- If the answer is not in the context, say clearly it isn't in the provided documents. Never invent facts.
- Preserve exact numbers, units, and dates from the sources. Be concise. Use the user's language."""


def answer(user_id: int, question: str, folder: str | None = None, doc_id: str | None = None) -> dict:
    from app.rag import llm  # local import so retrieval works even if LLM import fails
    qvec, q_tokens = embed.embed([question], input_type="query")
    hits = store.search(user_id, qvec[0], config.TOP_K, folder=folder, doc_id=doc_id)
    if not hits:
        return {"answer": "I couldn't find anything relevant in your documents. Try another folder, upload a file, or rephrase.",
                "sources": [], "usage": {"embed_tokens": q_tokens, "prompt_tokens": 0, "output_tokens": 0}}
    # Use fuller parent context for the model, but keep one entry per source passage for citations.
    blocks, sources = [], []
    for i, h in enumerate(hits):
        ctx = h.get("parent_text") or h["text"]
        blocks.append(f"[{i + 1}] (from {h['filename']} — {h['location']})\n{ctx}")
        sources.append({"n": i + 1, "filename": h["filename"], "folder": h.get("folder", ""),
                        "location": h["location"], "score": round(h["score"], 3), "snippet": h["text"][:240]})
    prompt = f"Context passages:\n{chr(10).join(blocks)}\n\nQuestion: {question}\n\nAnswer with inline [n] citations:"
    text, p_tokens, o_tokens = llm.generate(SYSTEM, prompt)
    return {"answer": text, "sources": sources,
            "usage": {"embed_tokens": q_tokens, "prompt_tokens": p_tokens, "output_tokens": o_tokens}}
