"""
Player Enrollment & Registry Management Endpoints.
"""

import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends

from .schemas import PlayerRegisterResponse, PlayerListResponse, PlayerItem
from .dependencies import get_room
from app.core.engine.room import GameRoom

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/players", tags=["Players"])


@router.post("/register", response_model=PlayerRegisterResponse)
async def register_player(
    request: Request,
    player_name: str = Form(..., description="Display name for the player"),
    file: UploadFile = File(..., description="Voice sample (WAV/PCM/Audio container)"),
    room: GameRoom = Depends(get_room),
):
    """
    Register a new player profile from a 10-15s speech sample.
    Extracts 192-D speaker centroid using DTLN, Silero VAD, and CAM++.
    Scoped to the active GameRoom.
    """
    clean_name = player_name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Player name cannot be empty.")

    audio_bytes = await file.read()
    if len(audio_bytes) < 320:
        raise HTTPException(
            status_code=400, detail="Audio file too small or empty. Please record audio."
        )

    pipeline = request.app.state.pipeline
    registry = room.registry

    try:
        meta = pipeline.process_with_metadata(audio_bytes)
    except Exception as e:
        logger.error(f"Audio processing failure during registration: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Failed to process audio: {str(e)}")

    player_id = f"p_{uuid.uuid4().hex[:8]}"
    profile = registry.register_player(player_id, clean_name, meta["embedding"])

    return PlayerRegisterResponse(
        player_id=profile.player_id,
        name=profile.name,
        sample_count=profile.sample_count,
        score=profile.score,
        audio_duration_sec=meta["raw_duration_sec"],
        speech_duration_sec=meta["speech_duration_sec"],
        message=f"Player '{clean_name}' successfully enrolled in room '{room.room_code}'!",
        embedding=[round(float(v), 6) for v in meta["embedding"]],
    )



@router.get("", response_model=PlayerListResponse)
async def list_players(room: GameRoom = Depends(get_room)):
    """List all currently enrolled players in the active GameRoom."""
    registry = room.registry
    players = registry.list_players()
    items = [
        PlayerItem(
            player_id=p.player_id,
            name=p.name,
            sample_count=p.sample_count,
            score=p.score,
        )
        for p in players
    ]
    return PlayerListResponse(players=items, total=len(items))


@router.delete("/{player_id}")
async def delete_player(player_id: str, room: GameRoom = Depends(get_room)):
    """Remove a player profile from the active GameRoom registry."""
    registry = room.registry
    removed = registry.remove_player(player_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found.")
    return {"message": f"Player '{player_id}' deleted successfully.", "player_id": player_id}

