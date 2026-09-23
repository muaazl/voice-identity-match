"""
Multi-Room & Session Management for VoiceMimic.
Provides isolated VectorRegistry, GameEngine, and RoundCache per game room.
"""

import time
import random
import string
import threading
import logging
from typing import Dict, List, Optional

from .registry import VectorRegistry
from .game import GameEngine
from app.api.round_cache import RoundCache

logger = logging.getLogger(__name__)

# Character set for 4-letter room codes (excludes visually ambiguous chars 0/O, 1/I)
ROOM_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class GameRoom:
    """
    Isolated game session encapsulating player biometrics, game mechanics, and round caches.
    """

    def __init__(self, room_code: str, embedding_dim: int = 192):
        self.room_code = room_code.upper().strip()
        self.embedding_dim = embedding_dim
        self.registry = VectorRegistry(embedding_dim=embedding_dim)
        self.game = GameEngine(self.registry)
        self.round_cache = RoundCache(max_size=500, ttl_seconds=3600.0)
        self.created_at = time.time()
        self.last_active_at = time.time()

    def touch(self) -> None:
        """Mark room as actively in use to reset idle TTL."""
        self.last_active_at = time.time()

    def is_expired(self, max_idle_seconds: float = 7200.0) -> bool:
        """Check if room has exceeded maximum idle time (default 2 hours)."""
        return (time.time() - self.last_active_at) > max_idle_seconds

    def to_dict(self) -> dict:
        """Serialize room status metadata."""
        return {
            "room_code": self.room_code,
            "player_count": self.registry.count(),
            "active_rounds": self.round_cache.count(),
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
        }


class RoomManager:
    """
    Thread-safe registry of active GameRooms with automatic lifecycle management.
    """

    def __init__(self, default_room_code: str = "DEFAULT", embedding_dim: int = 192):
        self._lock = threading.RLock()
        self._embedding_dim = embedding_dim
        self._default_room_code = default_room_code.upper()
        self._rooms: Dict[str, GameRoom] = {}

        # Prime default room for backwards compatibility
        self.get_or_create_room(self._default_room_code)

    def _generate_unique_code(self, length: int = 4) -> str:
        """Generate a random unique uppercase alphanumeric room code."""
        for _ in range(100):
            code = "".join(random.choices(ROOM_CODE_CHARS, k=length))
            if code not in self._rooms:
                return code
        # Fallback to longer code if space is dense
        return "".join(random.choices(ROOM_CODE_CHARS, k=length + 2))

    def create_room(self, room_code: Optional[str] = None) -> GameRoom:
        """
        Create a new game room.
        If room_code is None, generates a 4-letter unique code.
        """
        with self._lock:
            self.cleanup_expired_rooms()

            if room_code:
                code = room_code.upper().strip()
                if not code:
                    code = self._generate_unique_code()
            else:
                code = self._generate_unique_code()

            room = GameRoom(room_code=code, embedding_dim=self._embedding_dim)
            self._rooms[code] = room
            logger.info(f"Created new GameRoom '{code}'")
            return room

    def get_room(self, room_code: str) -> Optional[GameRoom]:
        """Retrieve an existing room by code, or None if not found/expired."""
        code = room_code.upper().strip() if room_code else self._default_room_code
        with self._lock:
            room = self._rooms.get(code)
            if not room:
                return None
            if code != self._default_room_code and room.is_expired():
                del self._rooms[code]
                logger.info(f"Evicted expired GameRoom '{code}' on access.")
                return None
            room.touch()
            return room

    def get_or_create_room(self, room_code: str) -> GameRoom:
        """Retrieve existing room or create a new one if it does not exist."""
        code = room_code.upper().strip() if room_code else self._default_room_code
        with self._lock:
            room = self.get_room(code)
            if room:
                return room
            room = GameRoom(room_code=code, embedding_dim=self._embedding_dim)
            self._rooms[code] = room
            logger.info(f"Initialized GameRoom '{code}'")
            return room

    def delete_room(self, room_code: str) -> bool:
        """Remove a room by code. Default room cannot be deleted."""
        code = room_code.upper().strip()
        if code == self._default_room_code:
            return False

        with self._lock:
            if code in self._rooms:
                del self._rooms[code]
                logger.info(f"Deleted GameRoom '{code}'")
                return True
            return False

    def list_active_rooms(self) -> List[str]:
        """Return list of all non-expired room codes."""
        with self._lock:
            self.cleanup_expired_rooms()
            return list(self._rooms.keys())

    def count(self) -> int:
        """Return total active rooms count."""
        with self._lock:
            return len(self._rooms)

    def cleanup_expired_rooms(self, max_idle_seconds: float = 7200.0) -> int:
        """
        Evict idle rooms that exceeded max_idle_seconds (default 2 hours).
        Keeps the default room intact.
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                code
                for code, r in self._rooms.items()
                if code != self._default_room_code and (now - r.last_active_at) > max_idle_seconds
            ]
            for code in expired_keys:
                del self._rooms[code]
            if expired_keys:
                logger.info(f"Pruned {len(expired_keys)} idle rooms: {expired_keys}")
            return len(expired_keys)
