import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseModel):
    # Base paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = Field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "./data")).resolve())
    state_root: Path = Field(default_factory=lambda: Path(os.getenv("STATE_ROOT", "./state")).resolve())
    sessions_dir: Path = Field(default_factory=lambda: Path(os.getenv("SESSIONS_DIR", "./sessions")).resolve())
    
    # Qdrant Config
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL") or os.getenv("VECTOR_DB_URL") or "http://localhost:6333")
    qdrant_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    qdrant_collection_prefix: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_PREFIX", "rag_").strip())

    # Embedding Config
    embed_model_id: str = Field(default_factory=lambda: os.getenv("EMBED_MODEL", "openai_small"))
    
    # LLM Config
    default_llm_id: str = Field(default_factory=lambda: os.getenv("DEFAULT_LLM", "openai"))
    openai_chat_model: str = Field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY") or "")
    
    # Azure Config
    azure_openai_endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_openai_api_key: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_openai_api_version: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"))
    azure_openai_deployment: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
    azure_openai_embedding_deployment: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "")

    # Google Gemini & Voyage AI Config
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    voyage_api_key: str = Field(default_factory=lambda: os.getenv("VOYAGE_API_KEY", ""))

    # Local LLMs (vLLM)
    vllm_base_url: str = Field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    vllm_model: str = Field(default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    # Chunking Config
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1200")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    
    # Ingestion Constraints
    min_extracted_text_chars: int = Field(default_factory=lambda: int(os.getenv("MIN_EXTRACTED_TEXT_CHARS", "40")))
    
    @field_validator("data_root", "state_root", "sessions_dir", mode="before")
    @classmethod
    def make_path(cls, v):
        if isinstance(v, str):
            return Path(v).resolve()
        return v

# Instantiate global settings
settings = Settings()

# Make sure directories exist
settings.data_root.mkdir(parents=True, exist_ok=True)
settings.state_root.mkdir(parents=True, exist_ok=True)
settings.sessions_dir.mkdir(parents=True, exist_ok=True)
(settings.state_root / "reports").mkdir(parents=True, exist_ok=True)
