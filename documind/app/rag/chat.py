"""Grounded, cited answering over a user's documents."""
from app import config
from app.rag import embed, store, llm

SYSTEM = """You are a document assistant. Answer ONLY from the provided context passages.
- Cite the sources you use inline like [1], [2] matching the numbered passages.
- If the answer is not in the context, say clearly that it isn't in the provided documents. Never invent facts.
- Be concise and accurate. Use the user's language."""


def answer(user_id: int, question: str, doc_id: str | None = None) -> dict:
    qvec, q_tokens = embed.embed([question], input_type="query")
    hits = store.search(user_id, qvec[0], config.TOP_K, doc_id=doc_id)
    if not hits:
        return {
            "answer": "I couldn't find anything relevant in your documents. Try uploading a file first, or rephrasing.",
            "sources": [], "usage": {"embed_tokens": q_tokens, "prompt_tokens": 0, "output_tokens": 0},
        }
    context = "\n\n".join(
        f"[{i + 1}] (from {h['filename']} — {h['location']})\n{h['text']}" for i, h in enumerate(hits)
    )
    prompt = f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer with inline [n] citations:"
    text, p_tokens, o_tokens = llm.generate(SYSTEM, prompt)
    sources = [
        {"n": i + 1, "filename": h["filename"], "location": h["location"], "score": round(h["score"], 3),
         "snippet": h["text"][:240]}
        for i, h in enumerate(hits)
    ]
    return {
        "answer": text,
        "sources": sources,
        "usage": {"embed_tokens": q_tokens, "prompt_tokens": p_tokens, "output_tokens": o_tokens},
    }
