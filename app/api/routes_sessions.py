"""
AuthSession Management Endpoints.
"""

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatusResponse,
)

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.post("/create", response_model=SessionCreateResponse)
async def create_session(request: Request, body: SessionCreateRequest = SessionCreateRequest()):
    """
    Initialize a new isolated AuthSession.
    Returns the session ID required for all subsequent API calls in that session.
    """
    manager = request.app.state.session_manager
    session = manager.create_session(body.session_id)
    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        message=f"Session '{session.session_id}' created successfully.",
    )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(request: Request, session_id: str):
    """
    Check the status and TTL of an AuthSession.
    """
    manager = request.app.state.session_manager
    code = session_id.upper().strip()

    # Access without touching TTL using internal dict to avoid eviction side-effects here
    session = manager._sessions.get(code)
    
    if session and not session.is_expired():
        return SessionStatusResponse(
            session_id=session.session_id,
            exists=True,
            identity_count=session.registry.count(),
            created_at=session.created_at,
            last_active_at=session.last_active_at,
        )
    else:
        return SessionStatusResponse(
            session_id=code,
            exists=False,
            identity_count=0,
        )


@router.get("", response_model=dict)
async def list_sessions(request: Request):
    """
    List all currently active AuthSessions (admin diagnostic).
    """
    manager = request.app.state.session_manager
    sessions = manager.list_active_sessions()
    return {"active_sessions": sessions, "count": len(sessions)}
