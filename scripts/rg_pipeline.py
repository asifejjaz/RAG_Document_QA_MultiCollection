import os
from dotenv import load_dotenv
import openai
from langchain_openai import AzureOpenAIEmbeddings
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from qdrant_client.models import PointStruct
from autogen_agentchat.ui import Console
from typing import List, Dict, Any, Optional, Tuple
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
import asyncio 
from .index_text import setup_collection, get_qdrant_client
from qdrant_client import QdrantClient
from autogen_agentchat.messages import ChatMessage, TextMessage
from sessions.sessionManager import SessionManager, SessionSelector

load_dotenv()  # Load environment variables from .env file

def set_env():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model = os.getenv("AZURE_OPENAI_MODEL")
    # document_key = os.getenv("document_key")
    # document_endpoint = os.getenv("document_endpoint")
    # model_id = os.getenv("document_model_id")
    
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
        azure_endpoint=endpoint
    )
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint=endpoint,
        api_key=api_key,
        openai_api_version=api_version
    )
    
    # document_analysis_client = DocumentAnalysisClient(
    #     endpoint=document_endpoint, 
    #     credential=AzureKeyCredential(document_key)
    # )
 
    return az_model_client, client, embeddings


def _reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = 60
) -> list:
    """
    Merge dense and sparse result lists using Reciprocal Rank Fusion.
    Each result must have an 'id' and a 'payload'. Scores are recomputed;
    original similarity scores are not used in RRF itself but are preserved
    in the output for logging.

    Args:
        dense_results: Results from vector search (each has .id, .score, .payload)
        sparse_results: Results from keyword scroll (each has .id, .payload)
        k: RRF constant (higher k dampens rank differences; 60 is standard)

    Returns:
        List of dicts sorted by fused score, deduped by point id
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Any] = {}

    # Score dense results by rank
    for rank, result in enumerate(dense_results, start=1):
        point_id = result.id
        scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
        payloads[point_id] = {
            "payload": result.payload,
            "dense_score": result.score,
        }

    # Score sparse results by rank
    for rank, result in enumerate(sparse_results, start=1):
        point_id = result.id
        scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
        if point_id not in payloads:
            payloads[point_id] = {"payload": result.payload, "dense_score": None}

    # Sort by fused score descending
    fused = [
        {
            "id": pid,
            "rrf_score": scores[pid],
            "payload": payloads[pid]["payload"],
            "dense_score": payloads[pid]["dense_score"],
        }
        for pid in scores
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def _swap_parent_context(fused_results: list) -> list:
    """
    For each result, if a parent_text exists in the payload, replace the
    chunk text with the parent text. This gives the LLM the broader context
    while still having matched on the precise child chunk.

    Args:
        fused_results: Output from _reciprocal_rank_fusion

    Returns:
        Same list with 'text' field swapped where applicable
    """
    for item in fused_results:
        payload = item["payload"]
        parent_text = payload.get("parent_text")
        if parent_text:
            # Swap: LLM sees parent context, but we note which child matched
            item["child_text"] = payload.get("text", "")
            payload["text"] = parent_text
    return fused_results


def retrieve_context(
    query: str,
    qdrant_client: QdrantClient,
    embeddings: AzureOpenAIEmbeddings,
    collection_name: str = "research_papers",
    top_k: int = 5,
    doc_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: dense vector search + sparse keyword search,
    merged with RRF, then parent context is swapped in.

    Args:
        query: User query string
        qdrant_client: Qdrant client
        embeddings: Embedding model
        collection_name: Target collection
        top_k: Number of final results to return
        doc_type: Optional filter by doc_type payload field

    Returns:
        List of context dicts ready to format into the LLM prompt
    """
    # --- shared filter: only search leaf (child) nodes ---
    base_filter_conditions = [
        {"key": "is_leaf", "match": {"value": True}}
    ]
    if doc_type:
        base_filter_conditions.append(
            {"key": "doc_type", "match": {"value": doc_type}}
        )
    base_filter = {"must": base_filter_conditions}

    # --- 1. Dense vector search (top 20 candidates) ---
    query_vector = embeddings.embed_query(query)
    dense_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=20,
        query_filter=base_filter,
        with_payload=True
    )

    # --- 2. Sparse keyword search using the TextIndex (top 20 candidates) ---
    sparse_results, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": base_filter_conditions + [
                {"key": "text", "match": {"value": query}}
            ]
        },
        limit=20,
        with_payload=True,
        with_vectors=False
    )

    # --- 3. Merge with Reciprocal Rank Fusion ---
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)

    # --- 4. Parent context swap on the top-k results ---
    top_fused = fused[:top_k]
    top_fused = _swap_parent_context(top_fused)

    # --- 5. Format into the structure expected downstream ---
    contexts = []
    for item in top_fused:
        payload = item["payload"]
        contexts.append({
            "text": payload.get("text", ""),
            "score": item["rrf_score"],
            "dense_score": item.get("dense_score"),
            "page_number": payload.get("page_number", "N/A"),
            "doc_id": payload.get("doc_id", "N/A"),
            "source": payload.get("file_name", "N/A"),
            "parent_id": payload.get("parent_id"),
            "child_text": item.get("child_text"),  # The original child match, if swapped
        })

    return contexts


# ============================================================================
# SINGLE AGENT RAG WITH CONVERSATION HISTORY
# ============================================================================
class RAGRetriever:
    """RAG retrieval helper for single agent approach"""

    def __init__(self, qdrant_client: QdrantClient, embeddings: AzureOpenAIEmbeddings):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings

    def retrieve(
        self,
        query: str,
        collection_name: str = "research_papers",
        top_k: int = 5,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Thin wrapper: delegates to the module-level hybrid retrieve_context.
        """
        return retrieve_context(
            query=query,
            qdrant_client=self.qdrant_client,
            embeddings=self.embeddings,
            collection_name=collection_name,
            top_k=top_k,
            doc_type=doc_type
        )

async def create_rag_agent(
    model_client: AzureOpenAIChatCompletionClient,
    buffer_size: int = 10
):
    """
    Create single AutoGen agent for RAG pipeline with conversation history
    """
    
    # Use buffered context to maintain conversation history
    model_context = BufferedChatCompletionContext(buffer_size=buffer_size)
    print(f"📝 Using buffered context (last {buffer_size} messages)")
    
    # Single Marius agent
    agent = AssistantAgent(
        name="Marius",
        model_client=model_client,
        model_context=model_context,
        system_message="""You are Marius, a friendly and knowledgeable HR Assistant. 

Your personality traits:
- Warm and approachable
- Direct and clear in your answers
- Professional but conversational
- Natural in your responses

Guidelines:
- Never say "based on the provided context" or similar phrases
- Never mention that you're looking up information
- Speak as if you naturally know the company policies
- Use conversational language while maintaining professionalism
- If you don't know something, simply say "I don't have that information" without explanations
- Incorporate relevant details naturally into your responses
- When context is provided in the user's message, use it as your authoritative source
- Answer questions directly and helpfully"""
    )
    
    return agent


# =================================================================
# RAG PIPELINE WITH SESSION SUPPORT - SINGLE AGENT
# ============================================================================

async def run_rag_pipeline(
    model_client: AzureOpenAIChatCompletionClient,
    qdrant_client: QdrantClient,
    embedding: AzureOpenAIEmbeddings,
    user_query: str,
    session_manager: SessionManager,
    buffer_size: int = 10
):
    """Run the RAG pipeline with single agent and session support"""
    top_k = 5  # Number of context chunks to retrieve  
    
    # Retrieve context using the standalone function (not through RAGRetriever class)
    contexts = retrieve_context(
        query=user_query,
        qdrant_client=qdrant_client,
        embeddings=embedding,
        collection_name="research_papers",
        top_k=top_k,
        doc_type=None  # Add this if you want to filter by doc_type
    )
    
    # Format context as text
    context_text = "\n\n".join([
        f"[Source: {ctx['source']}, Page: {ctx['page_number']}, Score: {ctx['score']:.3f}]\n{ctx['text']}"
        for ctx in contexts
    ])

    # Get conversation history from session manager
    conversation_history = session_manager.get_history()
    history_text = ""
    if conversation_history:
        history_text = "\nPrevious conversation:\n" + "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[-3:]  # Include last 3 messages for context
        ])
    
    # Create enhanced prompt with context and conversation history
    enhanced_query = f"""Background information:
{context_text}
{history_text}

Question: {user_query}

Please provide a clear, accurate answer based on the context and previous conversation above."""
    
    # Create single agent
    agent = await create_rag_agent(model_client, buffer_size=buffer_size)
    
    print("\n" + "="*60)
    print(f"Query: {user_query}")
    print("="*60 + "\n")
    
    # Add query to session
    # session_manager.add_message("user", user_query)
    
    # Run the agent and stream to console
    stream = agent.run_stream(task=enhanced_query)
    result = await Console(stream)
    
    # Add response to session
    if result and hasattr(result, 'messages') and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, 'content'):
            session_manager.add_message("assistant", str(last_message.content))
    
    return result

