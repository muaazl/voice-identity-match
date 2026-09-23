"""
Stateless Audio ML Extraction API.
Accepts raw audio chunks, denoises with DTLN, performs VAD, and extracts 192-D speaker embeddings with CAM++.
Completely stateless: no session, room, or player data is retained on server.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from .schemas import AudioEmbedResponse, AudioMetricsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/audio", tags=["Audio"])


@router.post("/embed", response_model=AudioEmbedResponse)
async def extract_embedding(
    request: Request,
    file: UploadFile = File(..., description="Audio utterance WAV/PCM container"),
):
    """
    Stateless endpoint: Process speech sample and return 192-D unit-normalized embedding vector.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) < 320:
        raise HTTPException(
            status_code=400, detail="Audio file too small or empty. Please record audio."
        )

    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=500, detail="Audio ML pipeline not initialized.")

    try:
        meta = pipeline.process_with_metadata(audio_bytes)
    except Exception as e:
        logger.error(f"Stateless audio processing failure: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Failed to process audio: {str(e)}")

    embedding_list = [round(float(v), 6) for v in meta["embedding"]]

    return AudioEmbedResponse(
        embedding=embedding_list,
        audio_metrics=AudioMetricsResponse(
            raw_duration_sec=meta["raw_duration_sec"],
            speech_duration_sec=meta["speech_duration_sec"],
            speech_ratio=meta["speech_ratio"],
        ),
        status="success",
    )
