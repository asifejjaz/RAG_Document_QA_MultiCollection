"""
Embedding and LLM config: single source of truth for model options.
Provides get_embedding_options(), get_embeddings_model(), get_embedding_dimension(),
get_llm_options(), and helpers for Azure vs OpenAI.com vs Ollama answer generation.
"""
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# Sidebar "Embedding / answer provider" grouping
EMBEDDING_PROVIDER_AZURE = "azure"
EMBEDDING_PROVIDER_OPENAI = "openai"
EMBEDDING_PROVIDER_LOCAL = "local"


def get_openai_api_key() -> Optional[str]:
    """OpenAI.com API key: standard name or OPEN_AI_KEY from .env."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")


def _embedding_provider_for_entry(entry: Dict[str, Any]) -> str:
    t = entry.get("type")
    if t == "azure":
        return EMBEDDING_PROVIDER_AZURE
    if t == "openai_platform":
        return EMBEDDING_PROVIDER_OPENAI
    if t == "sentence_transformers":
        return EMBEDDING_PROVIDER_LOCAL
    return EMBEDDING_PROVIDER_AZURE


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
        {
            "id": "openai_small",
            "label": "OpenAI API (text-embedding-3-small)",
            "type": "openai_platform",
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "dimension": 1536,
            "params": {},
        },
        {
            "id": "openai_ada",
            "label": "OpenAI API (text-embedding-ada-002)",
            "type": "openai_platform",
            "model": "text-embedding-ada-002",
            "dimension": 1536,
            "params": {},
        },
    ]


def get_embedding_options() -> List[Dict[str, Any]]:
    """Return list of {id, label, dimension} for UI."""
    return [
        {"id": e["id"], "label": e["label"], "dimension": e["dimension"]}
        for e in _embedding_registry()
    ]


def get_embedding_options_for_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Embedding choices for the selected provider (Azure / OpenAI.com / Local)."""
    return [
        {"id": e["id"], "label": e["label"], "dimension": e["dimension"]}
        for e in _embedding_registry()
        if _embedding_provider_for_entry(e) == provider_id
    ]


def get_llm_options_for_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Answer-model choices for the selected provider."""
    all_llm = _llm_registry()
    if provider_id == EMBEDDING_PROVIDER_LOCAL:
        return [{"id": e["id"], "label": e["label"]} for e in all_llm if e["type"] == "ollama"]
    if provider_id == EMBEDDING_PROVIDER_OPENAI:
        return [{"id": e["id"], "label": e["label"]} for e in all_llm if e["type"] in ("openai_platform", "ollama")]
    # Azure path: Azure chat + Ollama alternatives
    return [{"id": e["id"], "label": e["label"]} for e in all_llm if e["type"] in ("azure", "ollama")]


def list_embedding_provider_choices() -> List[Dict[str, str]]:
    """Providers available in the UI (OpenAI omitted if no API key)."""
    out: List[Dict[str, str]] = [
        {"id": EMBEDDING_PROVIDER_AZURE, "label": "Azure OpenAI"},
    ]
    if get_openai_api_key():
        out.append({"id": EMBEDDING_PROVIDER_OPENAI, "label": "OpenAI (API)"})
    out.append({"id": EMBEDDING_PROVIDER_LOCAL, "label": "Local (BGE-M3 + Ollama)"})
    return out


def infer_embedding_provider_from_env() -> str:
    """Initial sidebar provider from EMBED_MODEL / DEFAULT_LLM (only valid if keys exist)."""
    em = os.getenv("EMBED_MODEL", "openai_small")
    if em in ("openai_small", "openai_ada"):
        if get_openai_api_key():
            return EMBEDDING_PROVIDER_OPENAI
        return EMBEDDING_PROVIDER_AZURE
    if em == "bge_m3":
        return EMBEDDING_PROVIDER_LOCAL
    dl = os.getenv("DEFAULT_LLM", "openai")
    if dl == "openai" and get_openai_api_key():
        return EMBEDDING_PROVIDER_OPENAI
    if dl in ("ollama_qwen2.5", "ollama_llama3.1"):
        return EMBEDDING_PROVIDER_LOCAL
    return EMBEDDING_PROVIDER_AZURE


def get_embedding_dimension(embedding_id: Optional[str] = None) -> int:
    """Return vector dimension for the given embedding id (or default)."""
    rid = embedding_id or os.getenv("EMBED_MODEL", "openai_small")
    for e in _embedding_registry():
        if e["id"] == rid:
            return e["dimension"]
    return _embedding_registry()[0]["dimension"]


def get_embeddings_model(embedding_id: Optional[str] = None):
    """
    Return a LangChain-compatible embedding client (embed_documents, embed_query).
    Lazy-loads sentence-transformers so OpenAI-only startup stays fast.
    """
    rid = embedding_id or os.getenv("EMBED_MODEL", "openai_small")
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
        if e["type"] == "openai_platform":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=e["model"],
                api_key=get_openai_api_key(),
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
    openai_chat = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    entries: List[Dict[str, Any]] = [
        {"id": "azure", "label": f"Azure ({azure_deploy})", "type": "azure", "model": azure_deploy},
        {"id": "openai", "label": f"OpenAI ({openai_chat})", "type": "openai_platform", "model": openai_chat},
        {"id": "ollama_qwen2.5", "label": f"Ollama: {primary}", "type": "ollama", "model": primary},
        {"id": "ollama_llama3.1", "label": f"Ollama: {secondary}", "type": "ollama", "model": secondary},
    ]
    return entries


def get_llm_options() -> List[Dict[str, Any]]:
    """Return list of {id, label} for UI."""
    return [{"id": e["id"], "label": e["label"]} for e in _llm_registry()]


def get_default_llm_id() -> str:
    return os.getenv("DEFAULT_LLM", "openai")


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


def is_openai_platform(llm_id: Optional[str]) -> bool:
    """True if llm_id refers to OpenAI.com API chat (not Azure)."""
    if not llm_id:
        return False
    for e in _llm_registry():
        if e["id"] == llm_id:
            return e["type"] == "openai_platform"
    return False


def is_azure(llm_id: Optional[str]) -> bool:
    """True if llm_id refers to Azure OpenAI chat."""
    if not llm_id:
        return False
    for e in _llm_registry():
        if e["id"] == llm_id:
            return e["type"] == "azure"
    return False


def get_openai_chat_model_name(llm_id: Optional[str]) -> Optional[str]:
    """Return OpenAI chat model name (e.g. gpt-4o-mini) for the given llm_id."""
    if not llm_id:
        return None
    for e in _llm_registry():
        if e["id"] == llm_id and e["type"] == "openai_platform":
            return e["model"]
    return None
