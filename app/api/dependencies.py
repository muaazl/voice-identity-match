"""
FastAPI dependencies for VoiceMimic.
Provides multi-tenant room resolution from headers, query params, or defaults.
"""

from typing import Optional
from fastapi import Request, Header, Query, HTTPException

from app.core.engine.room import GameRoom, RoomManager


def get_room_manager(request: Request) -> RoomManager:
    manager = getattr(request.app.state, "room_manager", None)
    if not manager:
        raise HTTPException(status_code=500, detail="Room manager not initialized on server.")
    return manager


async def get_room(
    request: Request,
    x_room_code: Optional[str] = Header(None, alias="X-Room-Code"),
    room: Optional[str] = Query(None),
) -> GameRoom:
    """
    Resolve the active GameRoom for the request.
    Precedence:
    1. HTTP Header 'X-Room-Code'
    2. Query Parameter '?room=...'
    3. Fallback to 'DEFAULT'
    """
    room_manager = get_room_manager(request)

    # Determine requested room code
    code = (x_room_code or room or "DEFAULT").strip().upper()

    # Verify room exists (do not auto-create uncreated rooms)
    game_room = room_manager.get_room(code)
    if not game_room:
        raise HTTPException(
            status_code=404,
            detail=f"Room '{code}' has not been created yet or has expired. Please create it first.",
        )
    game_room.touch()
    return game_room

