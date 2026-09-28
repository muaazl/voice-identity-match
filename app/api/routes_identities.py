"""
Identity Enrollment & Registry Management Endpoints.
"""

import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends

from .schemas import IdentityEnrollResponse, IdentityListResponse, IdentityItem
from .dependencies import get_session
from app.core.engine.session import AuthSession

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/identities", tags=["Identities"])


@router.post("/enroll", response_model=IdentityEnrollResponse)
async def enroll_identity(
    request: Request,
    name: str = Form(..., description="Display name for the identity"),
    file: UploadFile = File(..., description="Voice sample (WAV/PCM/Audio container)"),
    session: AuthSession = Depends(get_session),
):
    """
    Enroll a new identity profile from a speech sample.
    Extracts 192-D speaker centroid using DTLN, Silero VAD, and CAM++.
    Scoped to the active AuthSession.
    """
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Identity name cannot be empty.")

    audio_bytes = await file.read()
    if len(audio_bytes) < 320:
        raise HTTPException(
            status_code=400, detail="Audio file too small or empty. Please record audio."
        )

    pipeline = request.app.state.pipeline
    registry = session.registry

    try:
        meta = pipeline.process_with_metadata(audio_bytes)
    except Exception as e:
        logger.error(f"Audio processing failure during enrollment: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Failed to process audio: {str(e)}")

    identity_id = f"id_{uuid.uuid4().hex[:8]}"
    profile = registry.register_identity(identity_id, clean_name, meta["embedding"])

    return IdentityEnrollResponse(
        identity_id=profile.identity_id,
        name=profile.name,
        sample_count=profile.sample_count,
        audio_duration_sec=meta["raw_duration_sec"],
        speech_duration_sec=meta["speech_duration_sec"],
        message=f"Identity '{clean_name}' successfully enrolled in session '{session.session_id}'.",
        embedding=[round(float(v), 6) for v in meta["embedding"]],
    )


@router.get("", response_model=IdentityListResponse)
async def list_identities(session: AuthSession = Depends(get_session)):
    """List all currently enrolled identities in the active AuthSession."""
    registry = session.registry
    identities = registry.list_identities()
    items = [
        IdentityItem(
            identity_id=p.identity_id,
            name=p.name,
            sample_count=p.sample_count,
        )
        for p in identities
    ]
    return IdentityListResponse(identities=items, total=len(items))


@router.delete("/{identity_id}")
async def delete_identity(identity_id: str, session: AuthSession = Depends(get_session)):
    """Remove an identity profile from the active AuthSession registry."""
    registry = session.registry
    removed = registry.remove_identity(identity_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Identity '{identity_id}' not found.")
    return {"message": f"Identity '{identity_id}' deleted successfully.", "identity_id": identity_id}
