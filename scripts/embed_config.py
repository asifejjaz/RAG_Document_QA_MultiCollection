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
EMBEDDING_PROVIDER_GEMINI = "gemini"


def get_openai_api_key() -> Optional[str]:
    """OpenAI.com API key: standard name or OPEN_AI_KEY from .env."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")


def _embedding_provider_for_entry(entry: Dict[str, Any]) -> str:
    t = entry.get("type")
    if t == "azure":
        return EMBEDDING_PROVIDER_AZURE
    if t == "openai_platform":
        return EMBEDDING_PROVIDER_OPENAI
    if t == "voyage":
        return EMBEDDING_PROVIDER_GEMINI
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
        {
            "id": "voyage_3_lite",
            "label": "Voyage AI (voyage-3-lite)",
            "type": "voyage",
            "model": "voyage-3-lite",
            "dimension": 512,
            "params": {
                "api_key": os.getenv("VOYAGE_API_KEY"),
            },
        },
    ]


def get_embedding_options() -> List[Dict[str, Any]]:
    """Return list of {id, label, dimension} for UI."""
    return [
        {"id": e["id"], "label": e["label"], "dimension": e["dimension"]}
        for e in _embedding_registry()
    ]


def get_embedding_options_for_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Embedding choices for the selected provider (Azure / OpenAI.com / Gemini)."""
    return [
        {"id": e["id"], "label": e["label"], "dimension": e["dimension"]}
        for e in _embedding_registry()
        if _embedding_provider_for_entry(e) == provider_id
    ]


def get_llm_options_for_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Answer-model choices for the selected provider."""
    all_llm = _llm_registry()
    if provider_id == EMBEDDING_PROVIDER_GEMINI:
        return [{"id": e["id"], "label": e["label"]} for e in all_llm if e["type"] == "gemini"]
    if provider_id == EMBEDDING_PROVIDER_OPENAI:
        return [{"id": e["id"], "label": e["label"]} for e in all_llm if e["type"] == "openai_platform"]
    # Azure path: Azure chat
    return [{"id": e["id"], "label": e["label"]} for e in all_llm if e["type"] == "azure"]


def list_embedding_provider_choices() -> List[Dict[str, str]]:
    """Providers available in the UI (OpenAI omitted if no API key, Azure omitted if ENABLE_AZURE is False)."""
    out: List[Dict[str, str]] = []
    
    enable_azure = os.getenv("ENABLE_AZURE", "true").strip().lower() != "false"
    if enable_azure:
        out.append({"id": EMBEDDING_PROVIDER_AZURE, "label": "Azure OpenAI"})
        
    if get_openai_api_key():
        out.append({"id": EMBEDDING_PROVIDER_OPENAI, "label": "OpenAI (API)"})
    if os.getenv("GEMINI_API_KEY") and os.getenv("VOYAGE_API_KEY"):
        out.append({"id": EMBEDDING_PROVIDER_GEMINI, "label": "Google Gemini & Voyage AI"})
    return out


def infer_embedding_provider_from_env() -> str:
    """Initial sidebar provider from EMBED_MODEL / DEFAULT_LLM (only valid if keys exist)."""
    enable_azure = os.getenv("ENABLE_AZURE", "true").strip().lower() != "false"
    em = os.getenv("EMBED_MODEL", "openai_small")
    if em == "voyage_3_lite":
        return EMBEDDING_PROVIDER_GEMINI
    if em in ("openai_small", "openai_ada"):
        if get_openai_api_key():
            return EMBEDDING_PROVIDER_OPENAI
        return EMBEDDING_PROVIDER_AZURE if enable_azure else EMBEDDING_PROVIDER_GEMINI
    dl = os.getenv("DEFAULT_LLM", "openai")
    if dl == "openai" and get_openai_api_key():
        return EMBEDDING_PROVIDER_OPENAI
    if dl == "gemini_2_5_flash":
        return EMBEDDING_PROVIDER_GEMINI
    return EMBEDDING_PROVIDER_AZURE if enable_azure else EMBEDDING_PROVIDER_GEMINI


def get_embedding_dimension(embedding_id: Optional[str] = None) -> int:
    """Return vector dimension for the given embedding id (or default)."""
    rid = embedding_id or os.getenv("EMBED_MODEL", "openai_small")
    for e in _embedding_registry():
        if e["id"] == rid:
            return e["dimension"]
    return _embedding_registry()[0]["dimension"]


class VoyageLangChainEmbeddings:
    def __init__(self, model: str, api_key: str):
        self.model = model
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        
    def embed_query(self, text: str) -> List[float]:
        return self.client.embed([text], model=self.model, input_type="query").embeddings[0]
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed(texts, model=self.model, input_type="document").embeddings


def get_embeddings_model(embedding_id: Optional[str] = None):
    """
    Return a LangChain-compatible embedding client (embed_documents, embed_query).
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
        if e["type"] == "openai_platform":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=e["model"],
                api_key=get_openai_api_key(),
            )
        if e["type"] == "voyage":
            return VoyageLangChainEmbeddings(
                model=e["model"],
                api_key=e["params"].get("api_key") or os.getenv("VOYAGE_API_KEY"),
            )
    # fallback to first
    first = _embedding_registry()[0]
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        model=first["model"],
        azure_endpoint=first["params"].get("azure_endpoint"),
        api_key=first["params"].get("api_key"),
        openai_api_version=first["params"].get("api_version", "2023-05-15"),
    )


# ---------------------------------------------------------------------------
# LLM registry
# ---------------------------------------------------------------------------

def _llm_registry() -> List[Dict[str, Any]]:
    azure_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-gpt-4.1")
    openai_chat = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    entries: List[Dict[str, Any]] = [
        {"id": "azure", "label": f"Azure ({azure_deploy})", "type": "azure", "model": azure_deploy},
        {"id": "openai", "label": f"OpenAI ({openai_chat})", "type": "openai_platform", "model": openai_chat},
        {"id": "gemini_2_5_flash", "label": "Gemini (gemini-2.5-flash)", "type": "gemini", "model": "gemini-2.5-flash"},
    ]
    return entries


def get_llm_options() -> List[Dict[str, Any]]:
    """Return list of {id, label} for UI."""
    return [{"id": e["id"], "label": e["label"]} for e in _llm_registry()]


def get_default_llm_id() -> str:
    return os.getenv("DEFAULT_LLM", "gemini_2_5_flash")


def is_ollama(llm_id: Optional[str]) -> bool:
    return False


def get_ollama_model_name(llm_id: Optional[str]) -> Optional[str]:
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


def is_gemini(llm_id: Optional[str]) -> bool:
    if not llm_id:
        return False
    for e in _llm_registry():
        if e["id"] == llm_id:
            return e["type"] == "gemini"
    return False


def get_openai_chat_model_name(llm_id: Optional[str]) -> Optional[str]:
    """Return OpenAI chat model name (e.g. gpt-4o-mini) for the given llm_id."""
    if not llm_id:
        return None
    for e in _llm_registry():
        if e["id"] == llm_id and e["type"] == "openai_platform":
            return e["model"]
    return None
