"""DocuMind configuration. Rename PRODUCT_NAME here to rebrand."""
import os
from pathlib import Path

PRODUCT_NAME = "DocuMind"
TAGLINE = "Ask your documents. Get answers you can trace."

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
QDRANT_PATH = str(DATA / "qdrant")
DB_PATH = str(DATA / "documind.db")
for d in (DATA, UPLOADS):
    d.mkdir(parents=True, exist_ok=True)

# Secrets (source /root/.secrets/cognitionsync-apis before launch)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")

# Models
EMBED_MODEL = "voyage-3-lite"
EMBED_DIM = 512
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
GEMINI_VISION_MODEL = "gemini-2.5-flash"

# RAG params
CHUNK_CHARS = 1100
CHUNK_OVERLAP = 150
TOP_K = 6
MAX_UPLOAD_MB = 25

COLLECTION = "chunks"
