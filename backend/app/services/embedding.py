import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional
from backend.app.config import settings

logger = logging.getLogger(__name__)

class BaseEmbeddingService(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key
        # Lazy load LangChain dependency
        from langchain_openai import OpenAIEmbeddings
        self._client = OpenAIEmbeddings(model=model_name, api_key=api_key)
        
    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)
        
    @property
    def dimension(self) -> int:
        return 1536


class AzureEmbeddingService(BaseEmbeddingService):
    def __init__(self, model_name: str, api_key: str, endpoint: str, api_version: str):
        self._model_name = model_name
        self._api_key = api_key
        self._endpoint = endpoint
        self._api_version = api_version
        from langchain_openai import AzureOpenAIEmbeddings
        self._client = AzureOpenAIEmbeddings(
            model=model_name,
            azure_endpoint=endpoint,
            api_key=api_key,
            openai_api_version=api_version,
        )
        
    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)
        
    @property
    def dimension(self) -> int:
        return 1536


class VoyageEmbeddingService(BaseEmbeddingService):
    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key
        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        
    def embed_query(self, text: str) -> List[float]:
        return self._client.embed([text], model=self._model_name, input_type="query").embeddings[0]
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed(texts, model=self._model_name, input_type="document").embeddings
        
    @property
    def dimension(self) -> int:
        return 512


class EmbeddingServiceFactory:
    @staticmethod
    def get_service(model_id: Optional[str] = None) -> BaseEmbeddingService:
        mid = model_id or settings.embed_model_id
        
        if mid == "azure_ada":
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                raise ValueError("Azure OpenAI details are not configured in settings/environment.")
            return AzureEmbeddingService(
                model_name=settings.azure_openai_embedding_deployment or "text-embedding-ada-002",
                api_key=settings.azure_openai_api_key,
                endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
            
        elif mid == "voyage_3_lite":
            if not settings.voyage_api_key:
                raise ValueError("Voyage API key is missing. Set VOYAGE_API_KEY in environment.")
            return VoyageEmbeddingService(
                model_name="voyage-3-lite",
                api_key=settings.voyage_api_key,
            )
            
        elif mid in ("openai_small", "openai_ada"):
            model_name = "text-embedding-3-small" if mid == "openai_small" else "text-embedding-ada-002"
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in environment.")
            return OpenAIEmbeddingService(
                model_name=model_name,
                api_key=settings.openai_api_key,
            )
            
        else:
            raise ValueError(f"Unknown embedding model ID: {mid}")
