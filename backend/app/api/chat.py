import re
import json
import queue
import logging
import threading
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.app.agents.graph import rag_graph
from backend.app.api.sessions import session_manager

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    collection_name: Optional[str] = None
    session_id: Optional[str] = None
    llm_id: Optional[str] = None
    embedding_id: Optional[str] = None
    user_msg_id: Optional[str] = None
    assistant_msg_id: Optional[str] = None


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


@router.post("/stream")
@router.post("/stream/")
def stream_chat(req: ChatRequest):
    """
    POST stream chat endpoint that runs the self-corrective LangGraph workflow 
    in a background thread and streams intermediate events (token generation, 
    search citations, query rewrites) using Server-Sent Events (SSE).
    """
    if is_smalltalk(req.message):
        smalltalk_msg = (
            "Hi! I'm Eva, your research assistant. "
            "Ask a question about your documents or choose a collection to search."
        )
        
        # Save user message to session history
        if req.session_id:
            try:
                session_manager.add_message(
                    role="user",
                    content=req.message,
                    session_id=req.session_id,
                    msg_id=req.user_msg_id
                )
            except Exception as se:
                logger.error("Failed to add user message to session '%s': %s", req.session_id, se)
                
        # Save assistant message to session history
        if req.session_id:
            try:
                session_manager.add_message(
                    role="assistant",
                    content=smalltalk_msg,
                    session_id=req.session_id,
                    msg_id=req.assistant_msg_id,
                    metadata={"originalQuery": req.message}
                )
            except Exception as se:
                logger.error("Failed to add assistant response to session '%s': %s", req.session_id, se)
                
        def smalltalk_generator():
            # Yield response word by word/whitespace split with tiny delay
            tokens = re.split(r'(\s+)', smalltalk_msg)
            for t in tokens:
                if t:
                    import time
                    time.sleep(0.01)
                    yield f"data: {json.dumps({'type': 'token', 'content': t})}\n\n"
            yield f"data: {json.dumps(None)}\n\n"
            
        return StreamingResponse(smalltalk_generator(), media_type="text/event-stream")

    q = queue.Queue()

    # Callbacks to put events in the queue
    def on_token(token: str):
        q.put({"type": "token", "content": token})

    def on_retrieval(chunks: list):
        q.put({"type": "citations", "chunks": chunks})

    def on_query_rewrite(rewritten: str):
        q.put({"type": "query_update", "query": rewritten})

    # Save user message to session history
    if req.session_id:
        try:
            session_manager.add_message(
                role="user",
                content=req.message,
                session_id=req.session_id,
                msg_id=req.user_msg_id
            )
        except Exception as se:
            logger.error("Failed to add user message to session '%s': %s", req.session_id, se)

    # Structure inputs for LangGraph State
    messages = []
    for msg in req.history:
        role = msg.get("role")
        content = msg.get("content")
        if role and content:
            messages.append({"role": role, "content": content})

    # Append current message
    messages.append({"role": "user", "content": req.message})

    state_inputs = {
        "messages": messages,
        "query": req.message,
        "original_query": req.message,
        "collection_name": req.collection_name,
        "retrieved_chunks": [],
        "generation": "",
        "grade": "",
        "loop_count": 0
    }

    config = {
        "configurable": {
            "on_token": on_token,
            "on_retrieval": on_retrieval,
            "on_query_rewrite": on_query_rewrite,
            "llm_id": req.llm_id,
            "embedding_id": req.embedding_id
        }
    }

    # Worker thread runner
    def run_graph():
        try:
            logger.info("Starting LangGraph execution for query: '%s'", req.message)
            res_state = rag_graph.invoke(state_inputs, config=config)
            logger.info("LangGraph execution completed successfully")
            
            # Save assistant response to session history
            if req.session_id and res_state:
                final_ans = res_state.get("generation", "")
                if final_ans:
                    try:
                        chunks = res_state.get("retrieved_chunks", [])
                        orig_query = res_state.get("original_query", req.message)
                        metadata = {
                            "citations": chunks,
                            "originalQuery": orig_query
                        }
                        session_manager.add_message(
                            role="assistant",
                            content=final_ans,
                            session_id=req.session_id,
                            msg_id=req.assistant_msg_id,
                            metadata=metadata
                        )
                    except Exception as se:
                        logger.error("Failed to add assistant response to session '%s': %s", req.session_id, se)
        except Exception as e:
            logger.error("Exception during LangGraph run: %s", e)
            q.put({"type": "error", "message": f"Graph execution error: {str(e)}"})
        finally:
            # Sentinel to close stream
            q.put(None)

    # Launch graph execution thread
    threading.Thread(target=run_graph, daemon=True).start()

    # Yield items from queue as SSE events
    def event_generator():
        while True:
            try:
                item = q.get(timeout=600)  # 10 minute timeout
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                logger.warning("Event stream timed out waiting for queue elements")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Response processing timed out'})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
