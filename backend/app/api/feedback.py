import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.config import settings
from backend.app.api.sessions import session_manager

logger = logging.getLogger(__name__)

router = APIRouter()

class FeedbackPayload(BaseModel):
    message_id: str
    session_id: str
    prompt: str
    answer: str
    feedback: str = Field(..., description="Either 'up' or 'down'")
    comment: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = []
    timestamp: Optional[str] = None


@router.post("")
@router.post("/")
def submit_feedback(payload: FeedbackPayload):
    """
    Submits user telemetry feedback (Thumbs Up/Down + optional comment)
    and saves it to user_feedback.jsonl.
    """
    if payload.feedback not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Feedback must be either 'up' or 'down'.")

    try:
        report_dir = settings.state_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / "user_feedback.jsonl"
        
        # Inject timestamp if not provided
        data = payload.model_dump()
        if not data.get("timestamp"):
            data["timestamp"] = datetime.utcnow().isoformat()
            
        # Write serialized JSON line
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
        # Write feedback to the session JSON file on disk
        session_manager.update_message(
            session_id=payload.session_id,
            msg_id=payload.message_id,
            updates={"metadata": {"feedbackSubmitted": payload.feedback}}
        )
            
        logger.info("Feedback successfully logged for message %s (feedback: %s)", payload.message_id, payload.feedback)
        return {"status": "success", "message": "Feedback logged successfully."}
    except Exception as e:
        logger.error("Failed to log feedback: %s", e)
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")
