"""
Multi-Tenant Session Management for VoiceGate.
Provides isolated VectorRegistry and VerificationEngine per auth session.
"""

import time
import random
import threading
import logging
from typing import Dict, List, Optional, Any

from .registry import VectorRegistry
from .verification import VerificationEngine

logger = logging.getLogger(__name__)

# Character set for 4-letter session IDs
SESSION_ID_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class AuthSession:
    """
    Isolated session encapsulating identities and verification mechanics.
    """

    def __init__(self, session_id: str, embedding_dim: int = 192):
        self.session_id = session_id.upper().strip()
        self.embedding_dim = embedding_dim
        self.registry = VectorRegistry(embedding_dim=embedding_dim)
        self.engine = VerificationEngine(self.registry)
        self.created_at = time.time()
        self.last_active_at = time.time()

    def touch(self) -> None:
        """Mark session as actively in use to reset idle TTL."""
        self.last_active_at = time.time()

    def is_expired(self, max_idle_seconds: float = 7200.0) -> bool:
        """Check if session has exceeded maximum idle time (default 2 hours)."""
        return (time.time() - self.last_active_at) > max_idle_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session status metadata."""
        return {
            "session_id": self.session_id,
            "identity_count": self.registry.count(),
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
        }


class SessionManager:
    """
    Thread-safe registry of active AuthSessions with automatic lifecycle management.
    """

    def __init__(self, default_session_id: str = "DEFAULT", embedding_dim: int = 192):
        self._lock = threading.RLock()
        self._embedding_dim = embedding_dim
        self._default_session_id = default_session_id.upper()
        self._sessions: Dict[str, AuthSession] = {}

        # Prime default session for backwards compatibility
        self.get_or_create_session(self._default_session_id)

    def _generate_unique_id(self, length: int = 4) -> str:
        """Generate a random unique uppercase alphanumeric session ID."""
        for _ in range(100):
            code = "".join(random.choices(SESSION_ID_CHARS, k=length))
            if code not in self._sessions:
                return code
        return "".join(random.choices(SESSION_ID_CHARS, k=length + 2))

    def create_session(self, session_id: Optional[str] = None) -> AuthSession:
        """
        Create a new auth session.
        If session_id is None, generates a 4-letter unique code.
        """
        with self._lock:
            self.cleanup_expired_sessions()

            if session_id:
                code = session_id.upper().strip()
                if not code:
                    code = self._generate_unique_id()
            else:
                code = self._generate_unique_id()

            session = AuthSession(session_id=code, embedding_dim=self._embedding_dim)
            self._sessions[code] = session
            logger.info(f"Created new AuthSession '{code}'")
            return session

    def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Retrieve an existing session by ID, or None if not found/expired."""
        code = session_id.upper().strip() if session_id else self._default_session_id
        with self._lock:
            session = self._sessions.get(code)
            if not session:
                return None
            if code != self._default_session_id and session.is_expired():
                del self._sessions[code]
                logger.info(f"Evicted expired AuthSession '{code}' on access.")
                return None
            session.touch()
            return session

    def get_or_create_session(self, session_id: str) -> AuthSession:
        """Retrieve existing session or create a new one if it does not exist."""
        code = session_id.upper().strip() if session_id else self._default_session_id
        with self._lock:
            session = self._sessions.get(code)
            if session:
                if code != self._default_session_id and session.is_expired():
                    del self._sessions[code]
                    logger.info(f"Evicted expired AuthSession '{code}' on access.")
                else:
                    session.touch()
                    return session
            session = AuthSession(session_id=code, embedding_dim=self._embedding_dim)
            self._sessions[code] = session
            logger.info(f"Initialized AuthSession '{code}'")
            return session

    def delete_session(self, session_id: str) -> bool:
        """Remove a session by ID. Default session cannot be deleted."""
        code = session_id.upper().strip()
        if code == self._default_session_id:
            return False

        with self._lock:
            if code in self._sessions:
                del self._sessions[code]
                logger.info(f"Deleted AuthSession '{code}'")
                return True
            return False

    def list_active_sessions(self) -> List[str]:
        """Return list of all non-expired session IDs."""
        with self._lock:
            self.cleanup_expired_sessions()
            return list(self._sessions.keys())

    def count(self) -> int:
        """Return total active sessions count."""
        with self._lock:
            return len(self._sessions)

    def cleanup_expired_sessions(self, max_idle_seconds: float = 7200.0) -> int:
        """
        Evict idle sessions that exceeded max_idle_seconds (default 2 hours).
        Keeps the default session intact.
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                code
                for code, s in self._sessions.items()
                if code != self._default_session_id and (now - s.last_active_at) > max_idle_seconds
            ]
            for code in expired_keys:
                del self._sessions[code]
            if expired_keys:
                logger.info(f"Pruned {len(expired_keys)} idle sessions: {expired_keys}")
            return len(expired_keys)
