"""Embedded Qdrant vector store with per-user + per-folder isolation."""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
)
from app import config

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=config.QDRANT_PATH)
        existing = {c.name for c in _client.get_collections().collections}
        if config.COLLECTION not in existing:
            _client.create_collection(
                config.COLLECTION,
                vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
            )
    return _client


def add_chunks(user_id: int, doc_id: str, filename: str, folder: str, chunks: list[dict], vectors: list[list[float]]):
    """chunks: [{text, location, chunk_index, is_leaf, parent_text}]. Only leaf chunks get embedded/searched."""
    pts = []
    for ch, vec in zip(chunks, vectors):
        pts.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "user_id": user_id, "doc_id": doc_id, "filename": filename, "folder": folder,
                "location": ch.get("location", ""), "chunk_index": ch.get("chunk_index", 0),
                "is_leaf": True, "text": ch["text"], "parent_text": ch.get("parent_text", ""),
            },
        ))
    client().upsert(config.COLLECTION, points=pts)


def search(user_id: int, query_vec: list[float], top_k: int, folder: str | None = None, doc_id: str | None = None):
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="is_leaf", match=MatchValue(value=True))]
    if folder:
        must.append(FieldCondition(key="folder", match=MatchValue(value=folder)))
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    res = client().query_points(
        config.COLLECTION, query=query_vec, limit=top_k, query_filter=Filter(must=must)
    ).points
    return [{"score": p.score, **p.payload} for p in res]


def delete_doc(user_id: int, doc_id: str):
    client().delete(config.COLLECTION, points_selector=Filter(must=[
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
    ]))
