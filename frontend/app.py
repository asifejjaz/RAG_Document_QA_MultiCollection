import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` and `scripts` can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
from datetime import datetime, timedelta
import os
import tempfile
from dotenv import load_dotenv
import atexit
from typing import Optional, List, Dict
import json

# Import from refactored scripts
from scripts.index_text import (
    get_qdrant_client,
    get_embeddings_model,
    process_file,
    generate_file_metadata,
    generate_doc_id
)

from scripts.report_inventory import (
    generate_inventory_report
)

# Import your existing RAG functions (keeping backward compatibility)
from scripts.rg_pipeline import set_env, retrieve_context, RAGRetriever, create_rag_agent


# Session management
from sessions.sessionManager import SessionManager

load_dotenv()
QDcollection = "research_papers"

# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def get_available_folders() -> List[str]:
    """Get list of unique folders in the collection"""
    try:
        qdrant_client = get_qdrant_client()
        scroll_result = qdrant_client.scroll(
            collection_name=QDcollection,
            limit=10000,
            with_payload=["folder_name", "collection"]
        )
        
        folders = set()
        for point in scroll_result[0]:
            if point.payload:
                # Use collection name as folder
                collection = point.payload.get("collection")
                folder_name = point.payload.get("folder_name")
                
                # Prefer collection name (from new system)
                if collection and collection != "Unknown":
                    folders.add(collection)
                elif folder_name and folder_name != "Unknown":
                    folders.add(folder_name)
        
        return sorted(list(folders))
    except Exception as e:
        st.error(f"Error fetching folders: {e}")
        return []


def retrieve_context_with_folder(
    query: str,
    qdrant_client,
    embeddings,
    collection_name: str,
    folder_filter: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve context with optional folder filtering
    Compatible with both old and new metadata structures
    """
    try:
        query_vector = embeddings.embed_query(query)
        
        # Build filter for folder/collection
        search_filter = None
        if folder_filter and folder_filter != "All Folders":
            # Try both collection and folder_name fields for compatibility
            search_filter = {
                "should": [
                    {"key": "collection", "match": {"value": folder_filter}},
                    {"key": "folder_name", "match": {"value": folder_filter}}
                ]
            }
        
        # Search with filter
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        )
        
        contexts = []
        for result in results:
            # Extract folder/collection name (prefer new structure)
            folder = result.payload.get("collection") or result.payload.get("folder_name", "Unknown")
            file_name = result.payload.get("file_name", "Unknown")
            
            contexts.append({
                "text": result.payload.get("text", ""),
                "source": result.payload.get("source", file_name),
                "page_number": result.payload.get("page_number", 0),
                "score": result.score,
                "doc_type": result.payload.get("doc_type", "document"),
                "folder_name": folder,
                "file_name": file_name
            })
        
        return contexts
        
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return []


def get_folder_documents_ui(folder_name: str) -> List[Dict]:
    """Get all documents in a specific folder (UI version)"""
    try:
        qdrant_client = get_qdrant_client()
        scroll_result = qdrant_client.scroll(
            collection_name=QDcollection,
            scroll_filter={
                "should": [
                    {"key": "collection", "match": {"value": folder_name}},
                    {"key": "folder_name", "match": {"value": folder_name}}
                ]
            },
            limit=10000,
            with_payload=["file_name", "doc_id", "doc_type"]
        )
        
        docs = {}
        for point in scroll_result[0]:
            file_name = point.payload.get("file_name", "Unknown")
            doc_id = point.payload.get("doc_id")
            
            if file_name not in docs:
                docs[file_name] = {
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "doc_type": point.payload.get("doc_type", "document"),
                    "chunks": 0
                }
            docs[file_name]["chunks"] += 1
        
        return list(docs.values())
        
    except Exception as e:
        st.error(f"Error fetching folder documents: {e}")
        return []


def get_folder_statistics(folder_name: str) -> Dict:
    """Get statistics for a folder"""
    try:
        qdrant_client = get_qdrant_client()
        scroll_result = qdrant_client.scroll(
            collection_name=QDcollection,
            scroll_filter={
                "should": [
                    {"key": "collection", "match": {"value": folder_name}},
                    {"key": "folder_name", "match": {"value": folder_name}}
                ]
            },
            limit=10000,
            with_payload=True
        )
        
        points = scroll_result[0]
        unique_files = set()
        doc_types = {}
        
        for point in points:
            file_name = point.payload.get("file_name")
            doc_type = point.payload.get("doc_type", "document")
            
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


def ingest_file_to_collection(file_path: str, collection_name: str) -> Dict:
    """
    Ingest a file using the new refactored system
    
    Args:
        file_path: Path to file
        collection_name: Target collection (will be used as folder name)
        
    Returns:
        Result dictionary
    """
    try:
        embeddings_model = get_embeddings_model()
        qdrant_client = get_qdrant_client()
        
        # Process file using refactored pipeline
        result = process_file(
            file_path=file_path,
            collection_name=collection_name,
            embeddings_model=embeddings_model,
            client=qdrant_client
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
    st.session_state.theme = 'light'

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
            .stChatMessage { background-color: var(--chat-assistant-bg) !important; border-radius: 12px !important; padding: 12px !important; margin: 8px 0 !important; color: var(--text-primary) !important; }
            .law-firm-header { background: linear-gradient(90deg,#06263e 0%, #0b3a66 100%) !important; color: #dbeeff !important; padding: 18px 16px; margin: -1rem -1rem 1rem -1rem; font-family: "Georgia", serif; border-bottom: 2px solid rgba(255,255,255,0.04) !important; }
            .law-firm-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.6px; }
            .main-title { font-size: 2.4rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #2aa2ff 10%, #66b2ff 60%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 6px 18px rgba(10,20,30,0.6); margin-bottom: 10px; }
            .source-box { background-color: rgba(8,20,30,0.6) !important; border-left: 4px solid #2b6cb0 !important; padding: 10px; margin: 10px 0; border-radius: 6px; font-size: 0.9rem; color: #cfe9ff !important; }
            .folder-badge { background: linear-gradient(135deg, #2a5298 0%, #1e3a6b 100%); color: #e6f2ff; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
            section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: linear-gradient(180deg,#061018 0%, #071426 100%) !important; color: #dbeeff !important; }
            textarea, input[type="text"], .stTextArea>div>textarea, .stTextInput>div>input { background: rgba(255,255,255,0.02) !important; color: #ffffff !important; border: 1px solid var(--border-color) !important; border-radius: 10px !important; padding: 10px !important; }
            .stButton>button { background: linear-gradient(180deg,#0f394f 0%, #072b3f 100%) !important; color: #e6eef3 !important; border: 1px solid rgba(255,255,255,0.04) !important; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
        """
    else:
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
            .stChatMessage { background-color: var(--bg-secondary) !important; border-radius: 10px !important; padding: 10px !important; margin: 5px 0 !important; }
            .law-firm-header { background-color: #003876 !important; color: white !important; padding: 15px; margin: -1rem -1rem 1rem -1rem; font-family: "Georgia", serif; border-bottom: 3px solid #002850 !important; }
            .main-title { font-size: 2.5em; font-weight: bold; text-align: center; background: linear-gradient(135deg, #003876 0%, #0056b3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
            .source-box { background-color: #e8f4f8 !important; border-left: 4px solid #667eea !important; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 0.9em; }
            .folder-badge { background: linear-gradient(135deg, #003876 0%, #0056b3 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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

if 'initialized' not in st.session_state:
    with st.spinner("🚀 Initializing RAG system..."):
        try:
            # Initialize using refactored system
            qdrant_client = get_qdrant_client()
            embeddings = get_embeddings_model()
            
            # Initialize Azure OpenAI for chat
            az_model_client, client, _ = set_env()
            
            # Validate components
            if not all([az_model_client, client, embeddings, qdrant_client]):
                raise ValueError("Failed to initialize required components")
            
            # Health check
            try:
                collections = qdrant_client.get_collections()
                if not collections:
                    st.warning("⚠️ No collections found. Please ingest documents first.")
            except Exception as e:
                st.error(f"⚠️ Qdrant connection issue: {e}")
                st.stop()
            
            # Initialize session manager
            session_manager = SessionManager(sessions_dir="sessions")
            
            # Store in session state
            st.session_state.az_model_client = az_model_client
            st.session_state.client = client
            st.session_state.embeddings = embeddings
            st.session_state.qdrant = qdrant_client
            st.session_state.session_manager = session_manager
            st.session_state.agent = None
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

def get_collection_stats():
    """Get statistics about the collection"""
    try:
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(QDcollection)
        
        all_points = qdrant_client.scroll(
            collection_name=QDcollection,
            limit=10000,
            with_payload=True
        )
        
        unique_docs = set()
        unique_folders = set()
        for point in all_points[0]:
            if point.payload:
                if "doc_id" in point.payload:
                    unique_docs.add(point.payload["doc_id"])
                
                # Check both collection and folder_name
                collection = point.payload.get("collection")
                folder = point.payload.get("folder_name")
                
                if collection and collection != "Unknown":
                    unique_folders.add(collection)
                elif folder and folder != "Unknown":
                    unique_folders.add(folder)
        
        return {
            "total_documents": len(unique_docs),
            "total_chunks": collection_info.points_count,
            "total_folders": len(unique_folders),
            "vector_size": collection_info.config.params.vectors.size
        }
    except:
        return {"total_documents": 0, "total_chunks": 0, "total_folders": 0, "vector_size": 1536}

# ============================================================================
# ASYNC HELPERS
# ============================================================================

async def reset_agent():
    """Reset the agent instance"""
    st.session_state.agent = await create_rag_agent(
        st.session_state.az_model_client,
        buffer_size=10
    )

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
        <div class="law-firm-header">
            <div class="law-firm-title">Research</div>
        </div>
    """, unsafe_allow_html=True)
    
    # FOLDER SELECTION
    st.markdown("### 📁 Select Collection")
    
    available_folders = ["All Folders"] + get_available_folders()
    
    selected_folder = st.selectbox(
        "Filter by collection:",
        options=available_folders,
        index=available_folders.index(st.session_state.get('selected_folder', 'All Folders')),
        key="folder_selector",
        help="Choose a collection to search within"
    )
    
    if selected_folder != st.session_state.get('selected_folder'):
        st.session_state.selected_folder = selected_folder
    
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
    
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", key="new_chat", use_container_width=True):
        current = st.session_state.get('current_session_id')
        
        if current and hasattr(st.session_state, 'session_manager'):
            try:
                st.session_state.session_manager._save_session(current)
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
            asyncio.run(reset_agent())
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
                    asyncio.run(reset_agent())
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session['session_id']}", use_container_width=True):
                    if session['session_id'] == st.session_state.get('current_session_id'):
                        new_session = st.session_state.session_manager.create_session("streamlit_user")
                        st.session_state.current_session_id = new_session
                        st.session_state.messages = []
                        st.session_state.messages_loaded = True
                        asyncio.run(reset_agent())
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
            target_collection = st.text_input(
                "New collection name:",
                key="upload_collection_new",
                placeholder="e.g., hydrogen_books"
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
                        result = ingest_file_to_collection(tmp_path, target_collection)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                        if result.get('status') == 'success':
                            st.success(f"✅ {uploaded_file.name}: {result.get('chunks_upserted', 0)} chunks")
                        else:
                            st.error(f"❌ {uploaded_file.name}: {result.get('error', 'Failed')}")
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    
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
                    report = generate_inventory_report("/state")
                    
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
                folder_filter = st.session_state.get('selected_folder', 'All Folders')
                
                # Retrieve context
                contexts = retrieve_context_with_folder(
                    query=prompt,
                    qdrant_client=st.session_state.qdrant,
                    embeddings=st.session_state.embeddings,
                    collection_name=QDcollection,
                    folder_filter=folder_filter if folder_filter != "All Folders" else None,
                    top_k=5
                )
                
                if not contexts:
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
                            "content": """You are Eva, a knowledgeable Research Assistant.
                            
                            Provide clear answers based on the context. Cite sources when relevant.
                            Be concise and professional."""
                        },
                        {
                            "role": "user",
                            "content": f"""Context:\n{context_text}\n\nQuestion: {prompt}"""
                        }
                    ]
                    
                    response = st.session_state.client.chat.completions.create(
                        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    answer = response.choices[0].message.content
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