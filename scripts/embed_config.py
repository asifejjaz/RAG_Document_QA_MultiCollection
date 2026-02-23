"""
Embedding and LLM config: single source of truth for model options.
Provides get_embedding_options(), get_embeddings_model(), get_embedding_dimension(),
get_llm_options(), and helpers for Azure vs Ollama answer generation.
"""
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Embedding registry
# ---------------------------------------------------------------------------

def _embedding_registry() -> List[Dict[str, Any]]:
    return [
        {
            "id": "azure_ada",
            "label": "Azure OpenAI (text-embedding-ada-002)",
            "type": "azure",
            "model": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            "dimension": 1536,
            "params": {
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            },
        },
        {
            "id": "bge_m3",
            "label": "BGE-M3 (sentence-transformers)",
            "type": "sentence_transformers",
            "model": os.getenv("BGE_M3_MODEL", "BAAI/bge-m3"),
            "dimension": 1024,
            "params": {},
        },
    ]


def get_embedding_options() -> List[Dict[str, Any]]:
    """Return list of {id, label, dimension} for UI."""
    return [
        {"id": e["id"], "label": e["label"], "dimension": e["dimension"]}
        for e in _embedding_registry()
    ]


def get_embedding_dimension(embedding_id: Optional[str] = None) -> int:
    """Return vector dimension for the given embedding id (or default)."""
    rid = embedding_id or os.getenv("EMBED_MODEL", "azure_ada")
    for e in _embedding_registry():
        if e["id"] == rid:
            return e["dimension"]
    return _embedding_registry()[0]["dimension"]


def get_embeddings_model(embedding_id: Optional[str] = None):
    """
    Return a LangChain-compatible embedding client (embed_documents, embed_query).
    Lazy-loads sentence-transformers so Azure-only startup stays fast.
    """
    rid = embedding_id or os.getenv("EMBED_MODEL", "azure_ada")
    for e in _embedding_registry():
        if e["id"] != rid:
            continue
        if e["type"] == "azure":
            from langchain_openai import AzureOpenAIEmbeddings
            return AzureOpenAIEmbeddings(
                model=e["model"],
                azure_endpoint=e["params"].get("azure_endpoint"),
                api_key=e["params"].get("api_key"),
                openai_api_version=e["params"].get("api_version", "2023-05-15"),
            )
        if e["type"] == "sentence_transformers":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=e["model"],
                model_kwargs={"trust_remote_code": True},
            )
    # fallback to first
    first = _embedding_registry()[0]
    if first["type"] == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            model=first["model"],
            azure_endpoint=first["params"].get("azure_endpoint"),
            api_key=first["params"].get("api_key"),
            openai_api_version=first["params"].get("api_version", "2023-05-15"),
        )
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=first["model"],
        model_kwargs={"trust_remote_code": True},
    )


# ---------------------------------------------------------------------------
# LLM registry
# ---------------------------------------------------------------------------

def _llm_registry() -> List[Dict[str, Any]]:
    azure_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-gpt-4.1")
    primary = os.getenv("LOCAL_LLM_MODEL_PRIMARY", "qwen2.5:7b-instruct")
    secondary = os.getenv("LOCAL_LLM_MODEL_SECONDARY", "llama3.1:8b-instruct")
    return [
        {"id": "azure", "label": f"Azure ({azure_deploy})", "type": "azure", "model": azure_deploy},
        {"id": "ollama_qwen2.5", "label": f"Ollama: {primary}", "type": "ollama", "model": primary},
        {"id": "ollama_llama3.1", "label": f"Ollama: {secondary}", "type": "ollama", "model": secondary},
    ]


def get_llm_options() -> List[Dict[str, Any]]:
    """Return list of {id, label} for UI."""
    return [{"id": e["id"], "label": e["label"]} for e in _llm_registry()]


def get_default_llm_id() -> str:
    return os.getenv("DEFAULT_LLM", "azure")


def is_ollama(llm_id: Optional[str]) -> bool:
    if not llm_id:
        return False
    for e in _llm_registry():
        if e["id"] == llm_id:
            return e["type"] == "ollama"
    return False


def get_ollama_model_name(llm_id: Optional[str]) -> Optional[str]:
    """Return Ollama model name (e.g. qwen2.5:7b-instruct) for the given llm_id."""
    if not llm_id:
        return None
    for e in _llm_registry():
        if e["id"] == llm_id and e["type"] == "ollama":
            return e["model"]
    return None
