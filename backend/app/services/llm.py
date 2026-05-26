import logging
import requests
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional, Any
from backend.app.config import settings

logger = logging.getLogger(__name__)

def clean_messages_for_text_only(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert multimodal messages (containing image URLs) into clean text-only messages."""
    cleaned = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    text_parts.append("[Image Attachment]")
            cleaned.append({"role": role, "content": "\n".join(text_parts)})
        else:
            cleaned.append({"role": role, "content": content})
    return cleaned

class BaseLLMService(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Synchronously generate response."""
        pass

    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """Stream response token-by-token."""
        pass


class OpenAILLMService(BaseLLMService):
    """Handles both OpenAI API platform and OpenAI-compatible services like vLLM."""
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        self._model_name = model_name
        self._api_key = api_key or "dummy_key"
        self._base_url = base_url
        
        import openai
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url
        )

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("OpenAI/vLLM completion failed: %s", e)
            raise

    def generate_stream(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("OpenAI/vLLM stream failed: %s", e)
            raise


class AzureLLMService(BaseLLMService):
    def __init__(self, deployment_name: str, api_key: str, endpoint: str, api_version: str):
        self._deployment_name = deployment_name
        import openai
        self._client = openai.AzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=endpoint
        )

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("Azure completion failed: %s", e)
            raise

    def generate_stream(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        try:
            stream = self._client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("Azure stream failed: %s", e)
            raise


class GeminiLLMService(BaseLLMService):
    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        import base64
        from google.genai import types
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_instruction = content
                continue

            # Map assistant role to model
            g_role = "model" if role == "assistant" else "user"

            parts = []
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:image/"):
                            header, base64_data = img_url.split(";base64,")
                            mime_type = header.split("data:")[1]
                            parts.append(
                                types.Part.from_bytes(
                                    data=base64.b64decode(base64_data),
                                    mime_type=mime_type
                                )
                            )
            else:
                parts.append(content)

            gemini_contents.append(
                types.Content(
                    role=g_role,
                    parts=[types.Part.from_text(text=p) if isinstance(p, str) else p for p in parts]
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=800
        )
        return gemini_contents, config

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        try:
            gemini_contents, config = self._convert_messages(messages)
            resp = self._client.models.generate_content(
                model=self._model_name,
                contents=gemini_contents,
                config=config
            )
            return resp.text or ""
        except Exception as e:
            logger.error("Gemini completion failed: %s", e)
            raise

    def generate_stream(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        try:
            gemini_contents, config = self._convert_messages(messages)
            response_stream = self._client.models.generate_content_stream(
                model=self._model_name,
                contents=gemini_contents,
                config=config
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("Gemini stream failed: %s", e)
            raise


class LLMServiceFactory:
    @staticmethod
    def get_service(llm_id: Optional[str] = None) -> BaseLLMService:
        target_id = llm_id or settings.default_llm_id
        
        if target_id == "azure":
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                raise ValueError("Azure OpenAI details are not configured.")
            return AzureLLMService(
                deployment_name=settings.azure_openai_deployment,
                api_key=settings.azure_openai_api_key,
                endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version
            )
            
        elif target_id == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is missing.")
            return OpenAILLMService(
                model_name=settings.openai_chat_model,
                api_key=settings.openai_api_key
            )
            
        elif target_id == "vllm":
            return OpenAILLMService(
                model_name=settings.vllm_model,
                api_key="token",  # vLLM standard dummy key
                base_url=settings.vllm_base_url
            )
            
        elif target_id == "gemini_2_5_flash":
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key is missing.")
            return GeminiLLMService(
                model_name="gemini-2.5-flash",
                api_key=settings.gemini_api_key
            )
            
        else:
            raise ValueError(f"Unknown LLM model ID: {target_id}")
