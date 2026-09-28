"""
Thread-safe in-memory cache for game rounds and active query embeddings.
Enables round tracking and true-speaker claim resolution with EMA updates.
"""

import time
import threading
from typing import Optional, Dict, Any
import numpy as np


class RoundCache:
    """Thread-safe round storage with TTL eviction."""

    def __init__(self, max_size: int = 500, ttl_seconds: float = 3600.0):
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def put(
        self,
        round_id: str,
        query_embedding: np.ndarray,
        predicted_player_id: Optional[str] = None,
        confidence: float = 0.0,
    ) -> None:
        """Store a round embedding and metadata."""
        with self._lock:
            self._evict_expired()
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
                del self._cache[oldest_key]

            self._cache[round_id] = {
                "round_id": round_id,
                "embedding": query_embedding.copy(),
                "predicted_player_id": predicted_player_id,
                "confidence": confidence,
                "created_at": time.time(),
            }

    def get(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve round data if present and not expired."""
        with self._lock:
            entry = self._cache.get(round_id)
            if not entry:
                return None
            if time.time() - entry["created_at"] > self._ttl_seconds:
                del self._cache[round_id]
                return None
            return entry

    def pop(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and remove a round from the cache."""
        with self._lock:
            entry = self._cache.get(round_id)
            if not entry:
                return None
            
            del self._cache[round_id]

            if time.time() - entry["created_at"] > self._ttl_seconds:
                return None
            return entry

    def count(self) -> int:
        """Return total active rounds in cache."""
        with self._lock:
            self._evict_expired()
            return len(self._cache)

    def clear(self) -> None:
        """Clear all active rounds."""
        with self._lock:
            self._cache.clear()

    def _evict_expired(self) -> None:
        """Remove entries exceeding TTL."""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items() if (now - v["created_at"]) > self._ttl_seconds
        ]
        for k in expired_keys:
            del self._cache[k]
