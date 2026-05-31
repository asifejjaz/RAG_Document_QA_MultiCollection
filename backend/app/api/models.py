import logging
import os
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from scripts import embed_config

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/options")
def get_model_options():
    """
    Get all available embedding providers, embedding models, and answer models
    along with default configuration values.
    """
    try:
        providers = embed_config.list_embedding_provider_choices()
        
        provider_options = {}
        for p in providers:
            p_id = p["id"]
            provider_options[p_id] = {
                "embeddings": embed_config.get_embedding_options_for_provider(p_id),
                "llms": embed_config.get_llm_options_for_provider(p_id)
            }
            
        default_provider = embed_config.infer_embedding_provider_from_env()
        
        return {
            "providers": providers,
            "options": provider_options,
            "default_provider": default_provider,
            "default_llm": embed_config.get_default_llm_id(),
            "default_embedding": os.getenv("EMBED_MODEL", "openai_small")
        }
    except Exception as e:
        logger.error("Failed to get model options: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to load model options: {str(e)}")
