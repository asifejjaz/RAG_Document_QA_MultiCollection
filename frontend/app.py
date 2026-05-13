import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` and `scripts` can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import time
import re
from datetime import datetime, timedelta
import os
import tempfile
from dotenv import load_dotenv
import atexit
from typing import Optional, List, Dict

# Import from refactored scripts
from scripts.index_text import (
    get_qdrant_client,
    get_embeddings_model,
    get_collection_names_for_dimension,
    get_collection_vector_size,
    delete_qdrant_collection,
    process_file,
)
from scripts import embed_config

from scripts.report_inventory import (
    generate_inventory_report
)

# Session management
from sessions.sessionManager import SessionManager

load_dotenv()


def generate_answer(messages: List[Dict], model_id: Optional[str] = None) -> str:
    """Generate answer using Azure, OpenAI.com API, or Ollama based on model_id."""
    import requests
    mid = model_id or st.session_state.get("selected_llm_id") or embed_config.get_default_llm_id()
    if embed_config.is_ollama(mid):
        model_name = embed_config.get_ollama_model_name(mid)
        base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        # Build Ollama messages from OpenAI-format messages
        ollama_messages = []
        for m in messages:
            role = "system" if m["role"] == "system" else m["role"]
            ollama_messages.append({"role": role, "content": m["content"]})
        try:
            r = requests.post(
                f"{base_url}/api/chat",
                json={"model": model_name, "messages": ollama_messages, "stream": False},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            return f"Ollama error: {e}"
    if embed_config.is_openai_platform(mid):
        oclient = st.session_state.get("openai_platform_client")
        if not oclient:
            return "OpenAI chat is not configured. Set OPENAI_API_KEY or OPEN_AI_KEY in .env and restart the app."
        chat_model = embed_config.get_openai_chat_model_name(mid) or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        try:
            resp = oclient.chat.completions.create(
                model=chat_model,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error: {e}"
    # Azure OpenAI
    if embed_config.is_azure(mid):
        client = st.session_state.get("client")
        if not client:
            try:
                from scripts.azure_openai_env import get_azure_openai_client
                client, _ = get_azure_openai_client()
                st.session_state.client = client
            except ImportError as e:
                return f"Azure support unavailable: {e}. Install required packages or switch to OpenAI/Ollama."
            except Exception as e:
                return f"Failed to initialize Azure OpenAI client: {e}"
        if not client:
            return "No Azure chat client configured. Set Azure OpenAI variables in .env, or choose OpenAI / Ollama in the sidebar."
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Azure OpenAI API error: {e}"
    return "Unknown LLM provider selected."


def is_smalltalk(text: str) -> bool:
    """Return True for greetings and smalltalk that shouldn't use RAG."""
    if not text:
        return False
    cleaned = text.strip().lower()
    if len(cleaned) <= 3:
        return cleaned in {"hi", "hey", "yo"}
    patterns = [
        r"^(hi|hello|hey|yo|hiya|sup|greetings)\b",
        r"^(good (morning|afternoon|evening))\b",
        r"^(how are you|how's it going|how are things)\b",
        r"^(who are you|what can you do|help)\b",
    ]
    return any(re.match(p, cleaned) for p in patterns)

# ============================================================================
# INTEGRATION HELPERS (one folder = one Qdrant collection)
# ============================================================================

def get_available_folders(embedding_id: Optional[str] = None) -> List[str]:
    """Get list of Qdrant collection names compatible with current embedding dimension."""
    try:
        qdrant_client = get_qdrant_client()
        eid = embedding_id or st.session_state.get("selected_embedding_id") or os.getenv("EMBED_MODEL", "openai_small")
        dimension = embed_config.get_embedding_dimension(eid)
        return get_collection_names_for_dimension(qdrant_client, dimension)
    except Exception:
        return []


# Leaf chunks only (matches dense retrieval in scripts/rg_pipeline.py)
_LEAF_FILTER = {"must": [{"key": "is_leaf", "match": {"value": True}}]}


def retrieve_context_with_folder(
    query: str,
    qdrant_client,
    embeddings,
    collection_name: Optional[str],
    top_k: int = 5,
    embedding_id: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve context from one or all Qdrant collections.
    Only searches collections whose vector size matches the current embedding dimension.
    """
    try:
        eid = embedding_id or st.session_state.get("selected_embedding_id") or os.getenv("EMBED_MODEL", "openai_small")
        dimension = embed_config.get_embedding_dimension(eid)
        if collection_name:
            vs = get_collection_vector_size(qdrant_client, collection_name)
            if vs is not None and vs != dimension:
                st.error(
                    f"Embedding dimension mismatch: collection **{collection_name}** uses vector size **{vs}**, "
                    f"but the selected embedding expects **{dimension}**. Change the embedding model in the sidebar "
                    "or choose a collection built with the same model."
                )
                return []
        query_vector = embeddings.embed_query(query)
        all_results = []

        if collection_name:
            # Single collection: dense search on leaf chunks
            try:
                results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=_LEAF_FILTER,
                    with_payload=True
                )
                all_results = [(r, collection_name) for r in results]
            except Exception:
                all_results = []
        else:
            # All Folders: only collections matching current embedding dimension
            collection_names = get_collection_names_for_dimension(qdrant_client, dimension)
            for coll_name in collection_names:
                try:
                    results = qdrant_client.search(
                        collection_name=coll_name,
                        query_vector=query_vector,
                        limit=top_k,
                        query_filter=_LEAF_FILTER,
                        with_payload=True
                    )
                    all_results.extend([(r, coll_name) for r in results])
                except Exception:
                    continue
            all_results.sort(key=lambda x: x[0].score, reverse=True)
            all_results = all_results[:top_k]

        contexts = []
        for result, coll_name in all_results:
            payload = result.payload or {}
            # Prefer file_name; fall back to basename of source_path (index_text stores source_path)
            source_path = payload.get("source_path", "")
            file_name = payload.get("file_name") or (Path(source_path).name if source_path else "Unknown")
            page_num = payload.get("page_number") or payload.get("page_start") or 0
            contexts.append({
                "text": payload.get("text", ""),
                "source": source_path or file_name,
                "page_number": page_num,
                "score": result.score,
                "doc_type": payload.get("doc_type", "document"),
                "folder_name": coll_name,
                "file_name": file_name
            })
        return contexts

    except Exception as e:
        err = str(e).strip() or type(e).__name__
        if "connection" in err.lower() or "embed" in err.lower():
            st.error(f"Error retrieving context: {err}. Check Azure OpenAI endpoint and API key (embeddings).")
        else:
            st.error(f"Error retrieving context: {err}")
        return []


def get_folder_documents_ui(folder_name: str) -> List[Dict]:
    """Get all documents in a folder (folder = one Qdrant collection)."""
    try:
        qdrant_client = get_qdrant_client()
        scroll_result = qdrant_client.scroll(
            collection_name=folder_name,
            limit=10000,
            with_payload=["file_name", "doc_id", "doc_type"]
        )
        docs = {}
        for point in scroll_result[0]:
            payload = point.payload or {}
            file_name = payload.get("file_name", "Unknown")
            doc_id = payload.get("doc_id")
            if file_name not in docs:
                docs[file_name] = {
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "doc_type": payload.get("doc_type", "document"),
                    "chunks": 0
                }
            docs[file_name]["chunks"] += 1
        return list(docs.values())
    except Exception as e:
        st.error(f"Error fetching folder documents: {e}")
        return []


def get_folder_statistics(folder_name: str) -> Dict:
    """Get statistics for a folder (folder = one Qdrant collection)."""
    try:
        qdrant_client = get_qdrant_client()
        scroll_result = qdrant_client.scroll(
            collection_name=folder_name,
            limit=10000,
            with_payload=True
        )
        points = scroll_result[0]
        unique_files = set()
        doc_types = {}
        for point in points:
            payload = point.payload or {}
            file_name = payload.get("file_name")
            doc_type = payload.get("doc_type", "document")
            if file_name:
                unique_files.add(file_name)
            if doc_type:
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        return {
            "total_chunks": len(points),
            "total_files": len(unique_files),
            "doc_types": doc_types,
            "files": sorted(list(unique_files))
        }
    except Exception as e:
        st.error(f"Error fetching folder stats: {e}")
        return {}


def ingest_file_to_collection(
    file_path: str,
    collection_name: str,
    embedding_id: Optional[str] = None,
    logical_file_name: Optional[str] = None,
) -> Dict:
    """
    Ingest a file using the new refactored system.
    Uses selected embedding from UI (or embedding_id) so collection dimension matches.
    logical_file_name: original upload filename (stored in Qdrant metadata instead of temp path name).
    """
    try:
        eid = embedding_id or st.session_state.get("selected_embedding_id") or os.getenv("EMBED_MODEL")
        embeddings_model = get_embeddings_model(eid)
        qdrant_client = get_qdrant_client()
        
        result = process_file(
            file_path=file_path,
            collection_name=collection_name,
            embeddings_model=embeddings_model,
            client=qdrant_client,
            embedding_id=eid,
            logical_file_name=logical_file_name,
        )
        
        return result
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME MANAGEMENT
# ============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'system'

def get_theme_css(theme):
    """Generate CSS based on current theme"""
    if theme == 'dark':
        return """
        <style>
            html { color-scheme: dark; }
            :root {
                --bg-primary: #0b0f12;
                --bg-secondary: #0d1117;
                --text-primary: #e6eef3;
                --text-secondary: #aebfcc;
                --accent-primary: #66b2ff;
                --accent-secondary: #2aa2ff;
                --border-color: rgba(255,255,255,0.04);
                --chat-user-bg: #1c2635;
                --chat-assistant-bg: #0d1117;
            }
            .main, .stApp, body, html { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
                background: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
            }
            section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] a { color: var(--accent-primary) !important; }
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] label { color: var(--text-secondary) !important; }
            .stChatMessage, [data-testid="stChatMessage"] { background-color: var(--chat-assistant-bg) !important; border-radius: 12px !important; padding: 12px !important; margin: 8px 0 !important; color: var(--text-primary) !important; }
            .stChatMessage *, [data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
            .law-firm-header { background: linear-gradient(90deg,#06263e 0%, #0b3a66 100%) !important; color: #dbeeff !important; padding: 18px 16px; margin: -1rem -1rem 1rem -1rem; font-family: "Georgia", serif; border-bottom: 2px solid rgba(255,255,255,0.04) !important; }
            .law-firm-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.6px; }
            .main-title { font-size: 2.4rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #2aa2ff 10%, #66b2ff 60%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 6px 18px rgba(10,20,30,0.6); margin-bottom: 10px; }
            .source-box { background-color: rgba(8,20,30,0.6) !important; border-left: 4px solid #2b6cb0 !important; padding: 10px; margin: 10px 0; border-radius: 6px; font-size: 0.9rem; color: #cfe9ff !important; }
            .folder-badge { background: linear-gradient(135deg, #2a5298 0%, #1e3a6b 100%); color: #e6f2ff; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
            section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: linear-gradient(180deg,#061018 0%, #071426 100%) !important; color: #dbeeff !important; }
            section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] .stMarkdown * { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] [data-testid="stSelectbox"] * { color: var(--text-primary) !important; }
            .stSelectbox div[data-baseweb="select"] > div { background: rgba(255,255,255,0.04) !important; }
            .stTextInput>div>input, .stTextArea>div>textarea { background: rgba(255,255,255,0.04) !important; color: #e6eef3 !important; border: 1px solid rgba(255,255,255,0.08) !important; }
            [data-testid="stChatInput"] textarea { background: rgba(255,255,255,0.04) !important; color: #e6eef3 !important; border: 1px solid rgba(255,255,255,0.08) !important; }
            textarea, input[type="text"], .stTextArea>div>textarea, .stTextInput>div>input { background: rgba(255,255,255,0.02) !important; color: #ffffff !important; border: 1px solid var(--border-color) !important; border-radius: 10px !important; padding: 10px !important; }
            .stButton>button { background: linear-gradient(180deg,#0f394f 0%, #072b3f 100%) !important; color: #e6eef3 !important; border: 1px solid rgba(255,255,255,0.04) !important; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
        """
    elif theme == 'light':
        return """
        <style>
            html { color-scheme: light; }
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f0f2f6;
                --text-primary: #262730;
                --text-secondary: #666666;
                --accent-primary: #003876;
                --accent-secondary: #0056b3;
                --border-color: #e0e0e0;
                --chat-user-bg: #f0f2f6;
                --chat-assistant-bg: #ffffff;
            }
            .main, .stApp, body, html { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: #f6f7fb !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] .stMarkdown * { color: var(--text-primary) !important; }
            .stChatMessage, [data-testid="stChatMessage"] { background-color: var(--chat-user-bg) !important; border-radius: 10px !important; padding: 10px !important; margin: 5px 0 !important; color: var(--text-primary) !important; }
            .stChatMessage *, [data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
            .law-firm-header { background-color: #003876 !important; color: white !important; padding: 15px; margin: -1rem -1rem 1rem -1rem; font-family: "Georgia", serif; border-bottom: 3px solid #002850 !important; }
            .main-title { font-size: 2.5em; font-weight: bold; text-align: center; background: linear-gradient(135deg, #003876 0%, #0056b3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
            .source-box { background-color: #e8f4f8 !important; border-left: 4px solid #667eea !important; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 0.9em; color: #1a1a2e !important; }
            .folder-badge { background: linear-gradient(135deg, #003876 0%, #0056b3 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
            .stSelectbox div[data-baseweb="select"] > div { background: #ffffff !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
            .stSelectbox div[data-baseweb="select"] * { color: var(--text-primary) !important; }
            .stTextInput>div>input, .stTextArea>div>textarea { background: #ffffff !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
            [data-testid="stChatInput"] textarea { background: #ffffff !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
        """
    # system
    return """
    <style>
        html { color-scheme: light dark; }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #0b0f12;
                --bg-secondary: #0d1117;
                --text-primary: #e6eef3;
                --text-secondary: #aebfcc;
                --accent-primary: #66b2ff;
                --accent-secondary: #2aa2ff;
                --border-color: rgba(255,255,255,0.04);
                --chat-user-bg: #1c2635;
                --chat-assistant-bg: #0d1117;
            }
            .main, .stApp, body, html { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: var(--bg-secondary) !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] a { color: var(--accent-primary) !important; }
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] label { color: var(--text-secondary) !important; }
            .stChatMessage, [data-testid="stChatMessage"] { background-color: var(--chat-assistant-bg) !important; border-radius: 12px !important; padding: 12px !important; margin: 8px 0 !important; color: var(--text-primary) !important; }
            .stChatMessage *, [data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
            .law-firm-header { background: linear-gradient(90deg,#06263e 0%, #0b3a66 100%) !important; color: #dbeeff !important; padding: 18px 16px; margin: -1rem -1rem 1rem -1rem; font-family: "Georgia", serif; border-bottom: 2px solid rgba(255,255,255,0.04) !important; }
            .law-firm-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.6px; }
            .main-title { font-size: 2.4rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #2aa2ff 10%, #66b2ff 60%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 6px 18px rgba(10,20,30,0.6); margin-bottom: 10px; }
            .source-box { background-color: rgba(8,20,30,0.6) !important; border-left: 4px solid #2b6cb0 !important; padding: 10px; margin: 10px 0; border-radius: 6px; font-size: 0.9rem; color: #cfe9ff !important; }
            .folder-badge { background: linear-gradient(135deg, #2a5298 0%, #1e3a6b 100%); color: #e6f2ff; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
            .stButton>button { background: linear-gradient(180deg,#0f394f 0%, #072b3f 100%) !important; color: #e6eef3 !important; border: 1px solid rgba(255,255,255,0.04) !important; }
        }
        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f0f2f6;
                --text-primary: #262730;
                --text-secondary: #666666;
                --accent-primary: #003876;
                --accent-secondary: #0056b3;
                --border-color: #e0e0e0;
                --chat-user-bg: #f0f2f6;
                --chat-assistant-bg: #ffffff;
            }
            .main, .stApp, body, html { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: #f6f7fb !important; color: var(--text-primary) !important; }
            section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
            .stChatMessage, [data-testid="stChatMessage"] { background-color: var(--chat-user-bg) !important; border-radius: 10px !important; padding: 10px !important; margin: 5px 0 !important; color: var(--text-primary) !important; }
            .stChatMessage *, [data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
            .law-firm-header { background-color: #003876 !important; color: white !important; padding: 15px; margin: -1rem -1rem 1rem -1rem; font-family: "Georgia", serif; border-bottom: 3px solid #002850 !important; }
            .main-title { font-size: 2.5em; font-weight: bold; text-align: center; background: linear-gradient(135deg, #003876 0%, #0056b3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
            .source-box { background-color: #e8f4f8 !important; border-left: 4px solid #667eea !important; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 0.9em; color: #1a1a2e !important; }
            .folder-badge { background: linear-gradient(135deg, #003876 0%, #0056b3 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
        }
    </style>
    """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _delete_if_empty(session_manager: SessionManager, session_id: str):
    """Delete the session if it exists and has no messages."""
    try:
        if not session_id:
            return False
        sess = session_manager.get_session(session_id)
        if sess is None:
            return False
        if sess.get("message_count", 0) == 0:
            session_manager.delete_session(session_id)
            return True
    except Exception:
        pass
    return False


def cleanup_session():
    """Cleanup function to handle session on exit"""
    if 'session_manager' in st.session_state and st.session_state.get('current_session_id'):
        try:
            st.session_state.session_manager.save_session()
        except:
            pass

atexit.register(cleanup_session)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

# Default model selection (from config/env)
if "embedding_provider" not in st.session_state:
    st.session_state.embedding_provider = embed_config.infer_embedding_provider_from_env()
_allowed_providers = [p["id"] for p in embed_config.list_embedding_provider_choices()]
if st.session_state.embedding_provider not in _allowed_providers:
    st.session_state.embedding_provider = _allowed_providers[0]

if "selected_embedding_id" not in st.session_state:
    st.session_state.selected_embedding_id = os.getenv("EMBED_MODEL", "openai_small")
if "selected_llm_id" not in st.session_state:
    st.session_state.selected_llm_id = embed_config.get_default_llm_id()

# Align embedding + LLM with current provider (before init and on every rerun)
_prov = st.session_state.embedding_provider
_emb_opts = embed_config.get_embedding_options_for_provider(_prov)
_emb_ids = [o["id"] for o in _emb_opts]
if _emb_ids and st.session_state.selected_embedding_id not in _emb_ids:
    st.session_state.selected_embedding_id = _emb_ids[0]
_llm_opts = embed_config.get_llm_options_for_provider(_prov)
_llm_ids = [o["id"] for o in _llm_opts]
if _llm_ids and st.session_state.selected_llm_id not in _llm_ids:
    st.session_state.selected_llm_id = _llm_ids[0]

if 'initialized' not in st.session_state:
    with st.spinner("🚀 Initializing RAG system..."):
        try:
            # Initialize using refactored system (embedding from config/UI selection)
            qdrant_client = get_qdrant_client()
            embeddings = get_embeddings_model(st.session_state.get("selected_embedding_id"))
            
            # Azure OpenAI for chat (only if selected as LLM)
            azure_client = None
            if st.session_state.get("selected_llm_id") == "azure":
                try:
                    from scripts.azure_openai_env import get_azure_openai_client
                    azure_client, _ = get_azure_openai_client()
                except ImportError as e:
                    st.error(f"Azure support unavailable: {e}. Install autogen-ext or switch to OpenAI/Ollama.")
                    st.stop()
            
            # OpenAI.com API (platform.openai.com) for chat + optional embeddings
            openai_platform_client = None
            if embed_config.get_openai_api_key():
                import openai as openai_sdk
                openai_platform_client = openai_sdk.OpenAI(api_key=embed_config.get_openai_api_key())
            
            # Validate: Qdrant + embeddings always required; Azure client optional
            if not embeddings or not qdrant_client:
                raise ValueError("Failed to initialize Qdrant or embeddings")
            
            # Health check
            try:
                collections = qdrant_client.get_collections()
                if not collections:
                    st.warning("⚠️ No collections found. Please ingest documents first.")
            except Exception as e:
                st.error(f"⚠️ Qdrant connection issue: {e}")
                st.stop()
            
            # Initialize session manager (use project-root sessions directory)
            session_manager = SessionManager(sessions_dir=Path(__file__).resolve().parents[1] / "sessions")
            
            # Store in session state
            st.session_state.client = azure_client
            st.session_state.openai_platform_client = openai_platform_client
            st.session_state.embeddings = embeddings
            st.session_state.qdrant = qdrant_client
            st.session_state.session_manager = session_manager
            st.session_state.selected_folder = "All Folders"
            
            # Load or create session
            sessions = session_manager.list_sessions() or []
            recent_session = None
            
            if sessions:
                sessions.sort(key=lambda x: x.get('last_active', ''), reverse=True)
                recent_session = sessions[0] if sessions else None
                
                if recent_session and recent_session.get('message_count', 0) == 0:
                    _delete_if_empty(session_manager, recent_session.get('session_id'))
                    recent_session = None
                
                if recent_session:
                    last_active = datetime.fromisoformat(recent_session['last_active'])
                    if datetime.now() - last_active < timedelta(hours=1):
                        session_manager.load_session(recent_session['session_id'])
                        st.session_state.current_session_id = recent_session['session_id']
                    else:
                        session_id = session_manager.create_session("streamlit_user")
                        st.session_state.current_session_id = session_id
                else:
                    session_id = session_manager.create_session("streamlit_user")
                    st.session_state.current_session_id = session_id
            
            st.session_state.initialized = True
            st.success("✅ System initialized!")
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            st.stop()

# Initialize messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_loaded = False

if not st.session_state.messages_loaded and hasattr(st.session_state, 'session_manager'):
    history = st.session_state.session_manager.get_history()
    st.session_state.messages = []
    for msg in history:
        st.session_state.messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    st.session_state.messages_loaded = True

# ============================================================================
# COLLECTION STATS
# ============================================================================

def get_collection_stats(collection_name: Optional[str] = None):
    """Get statistics. If collection_name is None, aggregate over all collections (folders)."""
    try:
        qdrant_client = get_qdrant_client()
        if collection_name:
            info = qdrant_client.get_collection(collection_name)
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=["doc_id"]
            )
            unique_docs = {p.payload.get("doc_id") for p in scroll_result[0] if p.payload and p.payload.get("doc_id")}
            return {
                "total_documents": len(unique_docs),
                "total_chunks": info.points_count,
                "total_folders": 1,
                "vector_size": info.config.params.vectors.size
            }
        collections = qdrant_client.get_collections().collections
        total_chunks = 0
        total_docs = 0
        vector_size = 1536
        for c in collections:
            try:
                info = qdrant_client.get_collection(c.name)
                total_chunks += info.points_count
                vector_size = info.config.params.vectors.size
                scroll_result = qdrant_client.scroll(collection_name=c.name, limit=10000, with_payload=["doc_id"])
                total_docs += len({p.payload.get("doc_id") for p in scroll_result[0] if p.payload and p.payload.get("doc_id")})
            except Exception:
                continue
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_folders": len(collections),
            "vector_size": vector_size
        }
    except Exception:
        return {"total_documents": 0, "total_chunks": 0, "total_folders": 0, "vector_size": 1536}

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
        <div class="law-firm-header">
            <div class="law-firm-title">Research</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection: provider first, then filtered embedding + answer models
    st.markdown("### ⚙️ Models")
    provider_choices = embed_config.list_embedding_provider_choices()
    prov_labels = [c["label"] for c in provider_choices]
    prov_ids = [c["id"] for c in provider_choices]
    if st.session_state.embedding_provider not in prov_ids:
        st.session_state.embedding_provider = prov_ids[0]
    prov_idx = prov_ids.index(st.session_state.embedding_provider)
    selected_prov_label = st.selectbox(
        "Embedding / answer provider",
        options=prov_labels,
        index=prov_idx,
        key="embedding_provider_select",
        help="Azure: Ada embeddings + Azure/Ollama chat. OpenAI: OpenAI embeddings + OpenAI/Ollama chat. Local: BGE-M3 + Ollama only.",
    )
    new_prov = prov_ids[prov_labels.index(selected_prov_label)]
    prov_changed = new_prov != st.session_state.embedding_provider
    st.session_state.embedding_provider = new_prov

    embed_opts = embed_config.get_embedding_options_for_provider(new_prov)
    embed_labels = [o["label"] for o in embed_opts]
    embed_ids = [o["id"] for o in embed_opts]
    if not embed_ids:
        st.caption("No embedding models for this provider.")
    else:
        if prov_changed or st.session_state.selected_embedding_id not in embed_ids:
            st.session_state.selected_embedding_id = embed_ids[0]
            if st.session_state.get("initialized"):
                st.session_state.embeddings = get_embeddings_model(embed_ids[0])
        current_embed_id = st.session_state.get("selected_embedding_id") or embed_ids[0]
        embed_idx = embed_ids.index(current_embed_id) if current_embed_id in embed_ids else 0
        new_embed_label = st.selectbox(
            "Embedding model",
            options=embed_labels,
            index=embed_idx,
            key="embedding_selector",
        )
        new_embed_id = embed_ids[embed_labels.index(new_embed_label)]
        if new_embed_id != st.session_state.get("selected_embedding_id"):
            st.session_state.selected_embedding_id = new_embed_id
            if st.session_state.get("initialized"):
                st.session_state.embeddings = get_embeddings_model(new_embed_id)

    llm_opts = embed_config.get_llm_options_for_provider(new_prov)
    llm_labels = [o["label"] for o in llm_opts]
    llm_ids = [o["id"] for o in llm_opts]
    if not llm_ids:
        st.caption("No answer models for this provider.")
    else:
        if prov_changed or st.session_state.selected_llm_id not in llm_ids:
            st.session_state.selected_llm_id = llm_ids[0]
        current_llm_id = st.session_state.get("selected_llm_id") or llm_ids[0]
        llm_idx = llm_ids.index(current_llm_id) if current_llm_id in llm_ids else 0
        selected_llm_label = st.selectbox(
            "Answer model",
            options=llm_labels,
            index=llm_idx,
            key="llm_selector",
        )
        if selected_llm_label in llm_labels:
            st.session_state.selected_llm_id = llm_ids[llm_labels.index(selected_llm_label)]
    
    st.markdown("---")
    st.markdown("### 📁 Select Collection")
    
    available_folders = ["All Folders"] + get_available_folders()
    current = st.session_state.get("selected_folder", "All Folders")
    idx = available_folders.index(current) if current in available_folders else 0
    selected_folder = st.selectbox(
        "Filter by collection:",
        options=available_folders,
        index=idx,
        key="folder_selector",
        help="One folder = one collection. Choose a collection to search within (isolation)."
    )
    
    if selected_folder != st.session_state.get('selected_folder'):
        st.session_state.selected_folder = selected_folder
    
    if (
        selected_folder != "All Folders"
        and st.session_state.get("initialized")
        and hasattr(st.session_state, "qdrant")
    ):
        _eid = st.session_state.get("selected_embedding_id") or os.getenv("EMBED_MODEL", "openai_small")
        _dim = embed_config.get_embedding_dimension(_eid)
        _vs = get_collection_vector_size(st.session_state.qdrant, selected_folder)
        if _vs is not None and _vs != _dim:
            st.warning(
                f"⚠️ Collection vector size is **{_vs}** but the selected embedding expects **{_dim}**. "
                "Chat retrieval will be blocked until they match."
            )
    
    # Show folder details
    if selected_folder != "All Folders":
        with st.expander("📊 Collection Details", expanded=False):
            stats = get_folder_statistics(selected_folder)
            if stats:
                st.metric("Files", stats['total_files'])
                st.metric("Chunks", stats['total_chunks'])
                
                if stats.get('doc_types'):
                    st.write("**Document Types:**")
                    for doc_type, count in stats['doc_types'].items():
                        st.write(f"- {doc_type}: {count}")
                
                if stats.get('files'):
                    st.write("**Files:**")
                    for file in stats['files']:
                        st.write(f"• {file}")
    
    # Remove Qdrant collection (vectors only; disk files under data/ are unchanged)
    folders_for_delete = get_available_folders()
    with st.expander("🗑️ Remove collection", expanded=False):
        st.warning(
            "Deletes the **Qdrant collection** and **all indexed vectors**. "
            "The collection disappears from this list. Files on disk (e.g. under `data/`) are **not** deleted."
        )
        if not folders_for_delete:
            st.caption("No collections match the current embedding dimension — nothing to remove.")
        else:
            coll_to_delete = st.selectbox(
                "Collection to delete",
                options=folders_for_delete,
                key="delete_collection_select",
                help="Full Qdrant collection name (same as in Filter by collection).",
            )
            confirm_delete = st.checkbox(
                "I understand this cannot be undone",
                key="delete_collection_confirm",
            )
            if st.button(
                "Delete this collection",
                type="primary",
                disabled=not confirm_delete,
                key="delete_collection_button",
            ):
                if not st.session_state.get("initialized") or not hasattr(st.session_state, "qdrant"):
                    st.error("App is not ready. Wait for initialization to finish.")
                else:
                    try:
                        delete_qdrant_collection(st.session_state.qdrant, coll_to_delete)
                        if st.session_state.get("selected_folder") == coll_to_delete:
                            st.session_state.selected_folder = "All Folders"
                        st.success(f"Deleted collection: **{coll_to_delete}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete collection: {e}")
    
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", key="new_chat", use_container_width=True):
        current = st.session_state.get('current_session_id')
        
        if current and hasattr(st.session_state, 'session_manager'):
            try:
                st.session_state.session_manager.save_session(current)
                session = st.session_state.session_manager.get_session(current)
                if session and session.get('message_count', 0) == 0:
                    st.session_state.session_manager.delete_session(current)
            except:
                pass
        
        try:
            new_session = st.session_state.session_manager.create_session("streamlit_user")
            st.session_state.current_session_id = new_session
            st.session_state.messages = []
            st.session_state.messages_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"Failed to create new session: {e}")
    
    st.markdown("---")
    
    # Previous Chats
    st.markdown("### 💬 Previous Chats")
    
    if hasattr(st.session_state, 'session_manager'):
        sessions = st.session_state.session_manager.list_sessions()
        sessions.sort(key=lambda x: x['last_active'], reverse=True)
        
        for session in sessions[:10]:
            history_preview = st.session_state.session_manager.get_history(session['session_id'])
            title = "New Conversation"
            if history_preview:
                first_msg = history_preview[0]['content']
                title = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
            
            last_active = datetime.fromisoformat(session['last_active'])
            date_str = last_active.strftime("%b %d, %I:%M %p")
            
            is_current = session['session_id'] == st.session_state.get('current_session_id')
            
            col1, col2 = st.columns([8, 2])
            
            with col1:
                button_label = f"{'📍' if is_current else '💬'} {title}"
                if st.button(
                    button_label,
                    key=f"session_{session['session_id']}",
                    use_container_width=True,
                    help=f"Last active: {date_str}",
                    disabled=is_current
                ):
                    active = st.session_state.get('current_session_id')
                    if active and active != session['session_id']:
                        _delete_if_empty(st.session_state.session_manager, active)
                    
                    st.session_state.session_manager.load_session(session['session_id'])
                    st.session_state.current_session_id = session['session_id']
                    
                    st.session_state.messages = []
                    loaded_history = st.session_state.session_manager.get_history()
                    for msg in loaded_history:
                        st.session_state.messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    st.session_state.messages_loaded = True
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session['session_id']}", use_container_width=True):
                    if session['session_id'] == st.session_state.get('current_session_id'):
                        new_session = st.session_state.session_manager.create_session("streamlit_user")
                        st.session_state.current_session_id = new_session
                        st.session_state.messages = []
                        st.session_state.messages_loaded = True
                    st.session_state.session_manager.delete_session(session['session_id'])
                    st.rerun()
            
            st.markdown("<hr style='margin: 4px 0; opacity: 0.1'>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Collection Stats
    with st.expander("📊 Collection Stats"):
        stats = get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats['total_documents'])
            st.metric("Collections", stats['total_folders'])
        with col2:
            st.metric("Chunks", stats['total_chunks'])
    
    # Document Management
    with st.expander("📤 Upload Documents"):
        st.markdown("### Upload to Collection")
        
        # Select target collection
        existing_collections = get_available_folders()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            use_existing = st.checkbox("Use existing collection", value=True)
        
        if use_existing and existing_collections:
            target_collection = st.selectbox(
                "Select collection:",
                options=existing_collections,
                key="upload_collection_select"
            )
        else:
            # When no collections exist, default to research_papers (plan: optional default)
            default_new = "research_papers" if not existing_collections else ""
            target_collection = st.text_input(
                "New collection name:",
                value=default_new,
                key="upload_collection_new",
                placeholder="e.g., research_papers or my_docs"
            )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF or DOCX files",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        if uploaded_files and st.button("📥 Upload & Index", use_container_width=True):
            if not target_collection:
                st.error("Please specify a collection name")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total = len(uploaded_files)
                for idx, uploaded_file in enumerate(uploaded_files):
                    tmp_path = None
                    try:
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=Path(uploaded_file.name).suffix
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Ingest using refactored system
                        result = ingest_file_to_collection(
                            tmp_path,
                            target_collection,
                            logical_file_name=uploaded_file.name,
                        )
                        
                        if result.get('status') == 'success':
                            st.success(f"✅ {uploaded_file.name}: {result.get('chunks_upserted', 0)} chunks")
                        elif result.get('status') == 'skipped':
                            st.warning(f"⚠️ {uploaded_file.name}: {result.get('error', 'Skipped')}")
                        else:
                            st.error(f"❌ {uploaded_file.name}: {result.get('error', 'Failed')}")
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        # Always cleanup temp file
                        if tmp_path:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    
                    progress_bar.progress((idx + 1) / total)
                
                status_text.text("Upload complete!")
                st.balloons()
                
                # Refresh to show new collection
                time.sleep(1)
                st.rerun()
    
    # View Inventory Report
    with st.expander("📊 View Inventory"):
        if st.button("Generate Report", use_container_width=True):
            try:
                with st.spinner("Generating inventory report..."):
                    # Use refactored report system
                    state_dir = os.getenv("STATE_ROOT", "./state")
                    report = generate_inventory_report(state_dir)
                    
                    st.markdown("### Inventory Summary")
                    overall = report['overall']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Collections", overall['total_collections'])
                    with col2:
                        st.metric("Files", overall['total_files'])
                    with col3:
                        st.metric("Chunks", overall['total_chunks'])
                    
                    # Show collections
                    if report.get('collections'):
                        st.markdown("### Collections")
                        for coll_name, coll_data in report['collections'].items():
                            with st.container():
                                st.markdown(f"**📁 {coll_name}**")
                                st.write(f"Files: {coll_data['total_files']}, Chunks: {coll_data['total_chunks']}")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.markdown('<div class="main-title">💬 Eva</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Your personal Research Assistant</p>", unsafe_allow_html=True)

# Show current filter
if st.session_state.get('selected_folder', 'All Folders') != 'All Folders':
    st.markdown(f"""
        <div class="folder-badge">
            📁 Searching in: {st.session_state.selected_folder}
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Ensure active session
    if not st.session_state.get('current_session_id'):
        session_id = st.session_state.session_manager.create_session("streamlit_user")
        st.session_state.current_session_id = session_id
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.session_manager.add_message("user", prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            try:
                folder_filter = st.session_state.get("selected_folder", "All Folders")
                # One folder = one Qdrant collection; pass that collection or None for "All Folders"
                search_collection = None if folder_filter == "All Folders" else folder_filter

                # Short-circuit greetings/smalltalk (no retrieval, no citations)
                contexts = None
                if is_smalltalk(prompt):
                    smalltalk_msg = (
                        "Hi! I'm Eva, your research assistant. "
                        "Ask a question about your documents or choose a collection to search."
                    )
                    message_placeholder.markdown(smalltalk_msg)
                    st.session_state.messages.append({"role": "assistant", "content": smalltalk_msg})
                    st.session_state.session_manager.add_message("assistant", smalltalk_msg)
                else:
                    # Retrieve context
                    contexts = retrieve_context_with_folder(
                        query=prompt,
                        qdrant_client=st.session_state.qdrant,
                        embeddings=st.session_state.embeddings,
                        collection_name=search_collection,
                        top_k=5
                    )
                
                if contexts is None:
                    pass
                elif not contexts:
                    no_results_msg = f"I couldn't find relevant information"
                    if folder_filter != "All Folders":
                        no_results_msg += f" in '{folder_filter}'"
                    no_results_msg += "."
                    
                    message_placeholder.markdown(no_results_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
                    st.session_state.session_manager.add_message("assistant", no_results_msg)
                else:
                    # Format context
                    context_text = "\n\n".join([
                        f"[From {ctx['file_name']} (Page {ctx['page_number']})]:\n{ctx['text']}"
                        for ctx in contexts
                    ])
                    
                    # Generate answer
                    messages = [
                        {
                            "role": "system",
                            "content": """You are Eva, a research assistant. Use ONLY the provided context.

                            Rules:
                            - If the answer is not supported by the context, reply with I do not have enough information, do not make up an answer.
                            - Cite every factual claim with source file name and page, e.g. [filename, p. 12]
                            - Do not invent numbers, citations, or sources
                            - Be concise and professional"""
                        },
                        {
                            "role": "user",
                            "content": f"""Context:\n{context_text}\n\nQuestion: {prompt}"""
                        }
                    ]
                    
                    answer = generate_answer(messages, model_id=st.session_state.get("selected_llm_id"))
                    message_placeholder.markdown(answer)
                    
                    # Show sources
                    with st.expander("📚 Sources", expanded=False):
                        for i, ctx in enumerate(contexts[:3], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}:</strong> {ctx['file_name']}<br>
                                <strong>Collection:</strong> {ctx['folder_name']}<br>
                                <strong>Page:</strong> {ctx.get('page_number', 'N/A')}<br>
                                <strong>Relevance:</strong> {ctx['score']:.2%}<br>
                                <em>{ctx['text'][:200]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.session_manager.add_message("assistant", answer)
                    
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
