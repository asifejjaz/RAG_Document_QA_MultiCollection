import json
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)

def test_submit_feedback(tmp_path):
    """
    Test that submitting feedback via POST /api/feedback successfully writes
    to the user_feedback.jsonl file under the settings.state_root directory.
    """
    # Temporarily override state_root for testing
    original_state_root = settings.state_root
    settings.state_root = tmp_path
    
    payload = {
        "message_id": "test-msg-uuid-1",
        "session_id": "test-session-uuid-1",
        "prompt": "Explain hydrogen synthesis.",
        "answer": "Hydrogen can be synthesized via water electrolysis.",
        "feedback": "up",
        "comment": "Highly accurate citation.",
        "retrieved_chunks": [
            {"source": "chemistry_basics.pdf", "text": "Water electrolysis produces oxygen and hydrogen.", "page_number": 12}
        ]
    }
    
    try:
        response = client.post("/api/feedback", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Verify telemetry file existence and contents
        report_file = tmp_path / "reports" / "user_feedback.jsonl"
        assert report_file.exists()
        
        with open(report_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            logged_data = json.loads(lines[0])
            assert logged_data["message_id"] == "test-msg-uuid-1"
            assert logged_data["feedback"] == "up"
            assert logged_data["comment"] == "Highly accurate citation."
            assert logged_data["retrieved_chunks"][0]["source"] == "chemistry_basics.pdf"
            
    finally:
        # Restore settings
        settings.state_root = original_state_root
