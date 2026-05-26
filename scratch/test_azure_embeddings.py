import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path("c:/Users/HP/Documents/Python Scripts/Python/ai_rag/Eva_Rsearch_AI")
load_dotenv(project_root / ".env")

print("AZURE_OPENAI_ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE_OPENAI_DEPLOYMENT_NAME:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print("AZURE_OPENAI_API_VERSION:", os.getenv("AZURE_OPENAI_API_VERSION"))

try:
    from langchain_openai import AzureOpenAIEmbeddings
    
    # Try different deployment names
    deployments = ["text-embedding-ada-002", "text-embedding-3-small", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")]
    
    for dep in deployments:
        if not dep:
            continue
        print(f"\nTrying AzureOpenAIEmbeddings with deployment: {dep} ...")
        try:
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=dep,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            )
            res = embeddings.embed_query("test embedding query")
            print(f"Success! Embedding dimension: {len(res)}")
            break
        except Exception as e:
            print(f"Failed with deployment {dep}: {e}")
            
except Exception as e:
    print(f"Failed to import/run Azure OpenAI Embeddings: {e}")
