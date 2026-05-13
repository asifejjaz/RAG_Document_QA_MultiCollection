"""
OpenAI Azure clients for chat + LangChain embeddings.
Note: AutoGen integration (az_model_client) removed — not used in app.
"""
import os

import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def get_azure_openai_client():
    """Build and return Azure OpenAI SDK client (sync) and Ada embeddings."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    client = openai.AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=endpoint,
    )
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint=endpoint,
        api_key=api_key,
        openai_api_version=api_version,
    )
    return client, embeddings


# Backwards compatible alias: old set_env() returned (az_model_client, client, embeddings)
def set_env():
    """Legacy wrapper returning (None, client, embeddings)."""
    client, embeddings = get_azure_openai_client()
    return None, client, embeddings
