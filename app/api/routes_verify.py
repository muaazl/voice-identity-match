"""
VoiceGate Verification API Endpoints.
Handles 1:1 and 1:N speaker verification.
"""

import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends

from .schemas import (
    VerifyResponse,
    CandidateMatchResponse,
    AudioMetricsResponse,
)
from .dependencies import get_session
from app.core.engine.session import AuthSession

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Verification"])


@router.post("/verify", response_model=VerifyResponse)
async def verify_speaker(
    request: Request,
    file: UploadFile = File(..., description="Audio sample for verification"),
    identity_id: Optional[str] = Form(None, description="Optional ID for 1:1 verification"),
    threshold: float = Form(0.65, description="Match threshold"),
    session: AuthSession = Depends(get_session),
):
    """
    Verify a speaker against enrolled identities.
    If identity_id is provided, performs 1:1 authentication.
    Otherwise, performs 1:N identification.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) < 320:
        raise HTTPException(status_code=400, detail="Audio file too small or empty.")

    pipeline = request.app.state.pipeline
    engine = session.engine

    try:
        meta = pipeline.process_with_metadata(audio_bytes)
    except Exception as e:
        logger.error(f"Audio processing failure in verify: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {str(e)}")

    query_embedding = meta["embedding"]
    try:
        result = engine.verify_speaker(
            query_embedding=query_embedding,
            target_identity_id=identity_id,
            threshold=threshold
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    rankings_response = [
        CandidateMatchResponse(
            identity_id=r.identity_id,
            name=r.name,
            similarity=round(r.similarity, 4),
            similarity_percent=round(max(0.0, r.similarity) * 100, 1),
            probability_percent=round(r.probability * 100, 1) if r.probability is not None else None,
        )
        for r in result.rankings
    ]

    audio_metrics = AudioMetricsResponse(
        raw_duration_sec=meta["raw_duration_sec"],
        speech_duration_sec=meta["speech_duration_sec"],
        speech_ratio=meta["speech_ratio"],
    )

    return VerifyResponse(
        verdict=result.verdict,
        identity_id=result.identity_id,
        name=result.name,
        confidence=round(result.confidence, 4),
        confidence_percent=round(max(0.0, result.confidence) * 100, 1),
        is_certain=result.is_certain,
        margin=round(result.margin, 4),
        rankings=rankings_response,
        audio_metrics=audio_metrics,
    )
