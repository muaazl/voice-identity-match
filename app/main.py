"""
VoiceMimic FastAPI Application Entrypoint.
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
from app.core.engine.registry import VectorRegistry
from app.core.engine.game import GameEngine
from app.api.round_cache import RoundCache
from app.api.routes_players import router as players_router
from app.api.routes_game import router as game_router
from app.api.routes_ws import router as ws_router
from app.api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("VoiceMimic")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Pre-loads ONNX models, initializes the vector
    registry and game engine, and runs an in-memory warm-up pass.
    """
    logger.info("Initializing VoiceMimic Audio ML and Engine subsystems...")
    models_dir = Path("onnx_models")

    # Ensure models are present (auto-downloads on first boot if missing)
    required_models = ["dtln_model_1.onnx", "dtln_model_2.onnx", "silero_vad.onnx", "campplus.onnx"]
    if not all((models_dir / m).exists() for m in required_models):
        logger.info("ONNX models missing. Automatically fetching checkpoints...")
        from scripts.download_models import main as download_all_models
        download_all_models(str(models_dir))

    # 1. Initialize Core Audio Pipeline (Zero-PyTorch ONNX)
    app.state.pipeline = AudioPipeline(models_dir=models_dir, num_threads=2)

    # 2. Initialize Biometric Vector Registry & Game Engine
    app.state.registry = VectorRegistry(embedding_dim=192)
    app.state.game = GameEngine(app.state.registry)
    app.state.round_cache = RoundCache(max_size=500, ttl_seconds=3600.0)

    # 3. Model Warm-up Pass (Eliminates runtime cold-start JIT delay)
    logger.info("Executing model warm-up forward pass...")
    try:
        synthetic_warmup = np.zeros(16000, dtype=np.float32)
        _ = app.state.pipeline.process(synthetic_warmup)
        logger.info("Warm-up complete: ONNX Runtime graph primed on CPU.")
    except Exception as e:
        logger.warning(f"Warm-up pass encountered an issue (non-fatal): {e}")

    logger.info("VoiceMimic server is ready to accept biometric requests.")
    yield
    logger.info("Shutting down VoiceMimic server...")


# Create FastAPI application
app = FastAPI(
    title="VoiceMimic API",
    description="Real-time Voice Biometric Party Game Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (Permissive for local development and web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(players_router)
app.include_router(game_router)
app.include_router(ws_router)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health and diagnostics endpoint."""
    pipeline_loaded = hasattr(app.state, "pipeline") and app.state.pipeline is not None
    registry = getattr(app.state, "registry", None)
    round_cache = getattr(app.state, "round_cache", None)

    return HealthResponse(
        status="healthy" if pipeline_loaded else "initializing",
        models_loaded=pipeline_loaded,
        embedding_dim=192,
        enrolled_players=registry.count() if registry else 0,
        active_rounds=round_cache.count() if round_cache else 0,
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
        return HTMLResponse("<h1>VoiceMimic Backend Online</h1>")
else:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index_placeholder():
        return "<h1>VoiceMimic Backend Online</h1>"


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    logger.info(f"Starting server on {host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=False, workers=1)
