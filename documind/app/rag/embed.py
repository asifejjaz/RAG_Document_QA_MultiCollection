"""Voyage embeddings (REST)."""
import requests
from app import config

URL = "https://api.voyageai.com/v1/embeddings"


def embed(texts: list[str], input_type: str = "document") -> tuple[list[list[float]], int]:
    """Return (vectors, tokens_used). input_type: 'document' or 'query'."""
    if not texts:
        return [], 0
    vecs: list[list[float]] = []
    tokens = 0
    # Voyage caps batch size; chunk to 128
    for i in range(0, len(texts), 128):
        batch = texts[i : i + 128]
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {config.VOYAGE_API_KEY}", "content-type": "application/json"},
            json={"input": batch, "model": config.EMBED_MODEL, "input_type": input_type},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        vecs.extend(d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"]))
        tokens += data.get("usage", {}).get("total_tokens", 0)
    return vecs, tokens
