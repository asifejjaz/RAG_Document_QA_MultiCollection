import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.config import settings
# Ensure project root is in path so sessions module can be imported
if str(settings.project_root) not in sys.path:
    sys.path.insert(0, str(settings.project_root))

from sessions.sessionManager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate SessionManager with path from settings
session_manager = SessionManager(sessions_dir=settings.sessions_dir)

class CreateSessionRequest(BaseModel):
    user_name: Optional[str] = None

@router.get("")
@router.get("/")
def list_sessions(active_session_id: Optional[str] = None):
    """
    List all chat sessions on disk sorted by last active.
    Cleans up empty sessions (message_count == 0) unless they match the active_session_id.
    """
    try:
        # Reload sessions from disk to ensure sync
        session_manager.sessions = {}
        session_manager._load_all_sessions()
        
        sessions = session_manager.list_sessions()
        
        cleaned_sessions = []
        for s in sessions:
            sess_id = s["session_id"]
            full_sess = session_manager.get_session(sess_id)
            msg_count = full_sess.get("message_count", 0) if full_sess else 0
            
            if msg_count == 0 and sess_id != active_session_id:
                # Delete empty session only if it is older than 5 minutes to prevent race conditions during creation
                last_active_str = full_sess.get("last_active") if full_sess else None
                try:
                    if last_active_str:
                        dt = datetime.fromisoformat(last_active_str)
                        if (datetime.now() - dt).total_seconds() > 300:
                            session_manager.delete_session(sess_id)
                        else:
                            cleaned_sessions.append(s)
                    else:
                        session_manager.delete_session(sess_id)
                except Exception:
                    session_manager.delete_session(sess_id)
            else:
                cleaned_sessions.append(s)
        
        # Add preview of the first message for UX
        result = []
        for s in cleaned_sessions:
            history = session_manager.get_history(s["session_id"])
            preview = "New Conversation"
            if history:
                first_content = history[0]["content"]
                preview = first_content[:40] + "..." if len(first_content) > 40 else first_content
            
            result.append({
                **s,
                "preview": preview
            })
        return result
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
@router.post("/")
def create_session(req: CreateSessionRequest):
    """
    Create a new chat session.
    """
    try:
        session_id = session_manager.create_session(req.user_name)
        return {
            "session_id": session_id,
            "status": "success"
        }
    except Exception as e:
        logger.error("Failed to create session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/history")
def get_session_history(session_id: str):
    """
    Get conversation history/messages for a session.
    """
    try:
        history = session_manager.get_history(session_id)
        # Format for React frontend consumption
        # React expects: id, role, content, citations, feedbackSubmitted, originalQuery
        formatted_messages = []
        for i, msg in enumerate(history):
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp")
            
            # Retrieve persistent ID, metadata citations, feedback, original query
            msg_id = msg.get("id") or f"{session_id}-{role}-{i}"
            metadata = msg.get("metadata", {})
            citations = metadata.get("citations", [])
            feedback_submitted = metadata.get("feedbackSubmitted")
            original_query = metadata.get("originalQuery", "")
            
            formatted_messages.append({
                "id": msg_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "citations": citations,
                "feedbackSubmitted": feedback_submitted,
                "originalQuery": original_query
            })
        return formatted_messages
    except Exception as e:
        logger.error("Failed to get history for session '%s': %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """
    Delete a session and its files permanently.
    """
    try:
        success = session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "success", "message": f"Session '{session_id}' deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session '%s': %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e))
