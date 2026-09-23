"""
Room & Session Management API Endpoints.
Handles room creation, validation, status, and listing.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Depends

from .schemas import RoomCreateRequest, RoomCreateResponse, RoomStatusResponse
from app.core.engine.room import RoomManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rooms", tags=["Rooms"])


def _get_room_manager(request: Request) -> RoomManager:
    manager = getattr(request.app.state, "room_manager", None)
    if not manager:
        raise HTTPException(status_code=500, detail="Room manager not initialized.")
    return manager


@router.post("/create", response_model=RoomCreateResponse)
async def create_room(
    request: Request,
    body: Optional[RoomCreateRequest] = None,
    manager: RoomManager = Depends(_get_room_manager),
):
    """
    Create a new isolated game room with a 4-letter code.
    If a custom code is passed, it is sanitized to uppercase.
    """
    requested_code = body.room_code if body else None
    room = manager.create_room(room_code=requested_code)

    return RoomCreateResponse(
        room_code=room.room_code,
        created_at=room.created_at,
        message=f"Room '{room.room_code}' created successfully.",
    )


@router.get("/{room_code}/status", response_model=RoomStatusResponse)
async def get_room_status(
    room_code: str,
    manager: RoomManager = Depends(_get_room_manager),
):
    """
    Check if a room exists, its active player count, and timestamp.
    """
    code = room_code.strip().upper()
    room = manager.get_room(code)

    if not room:
        return RoomStatusResponse(
            room_code=code,
            exists=False,
            player_count=0,
            active_rounds=0,
            created_at=None,
            last_active_at=None,
        )

    return RoomStatusResponse(
        room_code=room.room_code,
        exists=True,
        player_count=room.registry.count(),
        active_rounds=room.round_cache.count(),
        created_at=room.created_at,
        last_active_at=room.last_active_at,
    )


@router.get("", tags=["Rooms"])
async def list_active_rooms(
    manager: RoomManager = Depends(_get_room_manager),
):
    """List all currently active rooms (excluding expired)."""
    return {
        "active_rooms": manager.list_active_rooms(),
        "total": manager.count(),
    }
