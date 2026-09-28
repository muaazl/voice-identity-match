"""
VoiceGate FastAPI Application Entrypoint.
Serves REST and WebSocket endpoints for real-time voice biometrics.
Optimized for low-latency CPU execution via ONNX Runtime.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from app.core.audio.pipeline import AudioPipeline
from app.core.engine.session import SessionManager
from app.api.routes_identities import router as identities_router
from app.api.routes_verify import router as verify_router
from app.api.routes_sessions import router as sessions_router
from app.api.routes_audio import router as audio_router
from app.api.routes_ws import router as ws_router
from app.api.schemas import HealthResponse
from app.config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Pre-loads ONNX models, initializes the session manager
    and engine subsystems, and runs an in-memory warm-up pass.
    """
    logger.info("Initializing VoiceGate Audio ML and Engine subsystems...")
    models_dir = Path(settings.models_dir)

    # Ensure models are present (auto-downloads on first boot if missing)
    required_models = ["dtln_model_1.onnx", "dtln_model_2.onnx", "silero_vad.onnx", "campplus.onnx"]
    if not all((models_dir / m).exists() for m in required_models):
        logger.info("ONNX models missing. Automatically fetching checkpoints...")
        from scripts.download_models import main as download_all_models
        download_all_models(str(models_dir))

    # 1. Initialize Core Audio Pipeline (Zero-PyTorch ONNX)
    app.state.pipeline = AudioPipeline(
        models_dir=models_dir, 
        num_threads=settings.num_threads,
        vad_threshold=settings.vad_threshold
    )

    # 2. Initialize Multi-Tenant Session Orchestrator
    app.state.session_manager = SessionManager(
        default_session_id=settings.default_room_code, 
        embedding_dim=settings.embedding_dim
    )

    # Prime default session
    app.state.session_manager.get_or_create_session(settings.default_room_code)

    # 3. Model Warm-up Pass (Eliminates runtime cold-start JIT delay)
    logger.info("Executing model warm-up forward pass...")
    try:
        synthetic_warmup = np.zeros(16000, dtype=np.float32)
        _ = app.state.pipeline.process(synthetic_warmup)
        logger.info("Warm-up complete: ONNX Runtime graph primed on CPU.")
    except Exception as e:
        logger.warning(f"Warm-up pass encountered an issue (non-fatal): {e}")

    logger.info("VoiceGate server is ready to accept biometric requests.")
    yield
    logger.info("Shutting down VoiceGate server...")
    if hasattr(app.state, "pipeline"):
        del app.state.pipeline


# Create FastAPI application
app = FastAPI(
    title="VoiceGate API",
    description="Real-time Voice Biometric Authentication Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (Permissive for local development and web clients)
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(audio_router)
app.include_router(sessions_router)
app.include_router(identities_router)
app.include_router(verify_router)
app.include_router(ws_router)



@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health and diagnostics endpoint."""
    pipeline_loaded = hasattr(app.state, "pipeline") and app.state.pipeline is not None
    session_manager = getattr(app.state, "session_manager", None)

    enrolled = 0
    if session_manager:
        # Aggregate across all active sessions
        for session_id in session_manager.list_active_sessions():
            session = session_manager.get_session(session_id)
            if session:
                enrolled += session.registry.count()

    return HealthResponse(
        status="healthy" if pipeline_loaded else "initializing",
        models_loaded=pipeline_loaded,
        embedding_dim=settings.embedding_dim,
        enrolled_identities=enrolled,
    )



# Mount static assets from app/static/
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>VoiceGate Backend Online</h1>")
else:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index_placeholder():
        return "<h1>VoiceGate Backend Online</h1>"


if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    logger.info(f"Starting server on {host}:{settings.port}...")
    uvicorn.run("app.main:app", host=host, port=settings.port, reload=settings.debug, workers=1)
