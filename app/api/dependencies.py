"""
FastAPI dependencies for VoiceGate.
Provides multi-tenant session resolution from headers, query params, or defaults.
"""

from typing import Optional
from fastapi import Request, Header, Query, HTTPException

from app.core.engine.session import AuthSession, SessionManager


def get_session_manager(request: Request) -> SessionManager:
    manager = getattr(request.app.state, "session_manager", None)
    if not manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized on server.")
    return manager


async def get_session(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    session: Optional[str] = Query(None),
) -> AuthSession:
    """
    Resolve the active AuthSession for the request.
    Precedence:
    1. HTTP Header 'X-Session-ID'
    2. Query Parameter '?session=...'
    3. Fallback to 'DEFAULT'
    """
    session_manager = get_session_manager(request)

    # Determine requested session ID
    code = (x_session_id or session or "DEFAULT").strip().upper()

    # Verify session exists (do not auto-create uncreated sessions)
    auth_session = session_manager.get_session(code)
    if not auth_session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{code}' has not been created yet or has expired. Please create it first.",
        )
    auth_session.touch()
    return auth_session
