from fastapi import APIRouter
from backend.app.config import settings

router = APIRouter()

@router.get("")
@router.get("/")
def get_health():
    return {
        "status": "healthy",
        "service": "Eva Research AI RAG Backend",
        "config": {
            "embed_model": settings.embed_model_id,
            "default_llm": settings.default_llm_id,
            "chunk_size": settings.chunk_size,
            "qdrant_url": settings.qdrant_url
        }
    }
