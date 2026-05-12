"""
Azure OpenAI clients for chat + LangChain embeddings (shared by Streamlit and legacy AutoGen path).

CLI / Streamlit chat primarily uses `client` (Azure OpenAI SDK). AutoGen helpers use `az_model_client`.
"""
import os

import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()


def set_env():
    """Build Azure chat completion client, sync chat client, and Ada embeddings (legacy default)."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model = os.getenv("AZURE_OPENAI_MODEL")

    az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=deployment,
        model=model,
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
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

    return az_model_client, client, embeddings
