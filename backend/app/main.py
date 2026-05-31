import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from backend.app.config import settings
from backend.app.api.health import router as health_router
from backend.app.api.collections import router as collections_router
from backend.app.api.chat import router as chat_router
from backend.app.api.feedback import router as feedback_router
from backend.app.api.models import router as models_router
from backend.app.api.sessions import router as sessions_router

# Setup logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("backend")

app = FastAPI(
    title="Eva Research AI RAG API",
    description="Decoupled React-ready FastAPI agentic search API powered by LangGraph.",
    version="2.0.0"
)

# CORS Configuration
# Allow local React dev environment and any production client domain (e.g., Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in strict settings if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router, prefix="/api/health", tags=["health"])
app.include_router(collections_router, prefix="/api/collections", tags=["collections"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["feedback"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])

# Serve image assets
assets_dir = settings.data_root / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# Optional fallback to serve Vite React build files if compiled locally inside the backend
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    logger.info("Serving compiled frontend static files from %s", static_dir)
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.info("Frontend static dir not found at %s. Running in standalone API mode.", static_dir)
    @app.get("/")
    def read_root():
        return RedirectResponse(url="/api/health")
