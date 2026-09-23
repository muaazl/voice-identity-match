"""
In-Memory Thread-Safe Biometric Vector Registry.
Maintains enrolled player speaker centroids, calculates cosine similarity,
and updates speaker profiles via Exponential Moving Average (EMA).
"""

import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .models import PlayerProfile

logger = logging.getLogger(__name__)


def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure vector is float32 with unit L2 norm."""
    v = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(v)
    if norm > eps:
        return (v / norm).astype(np.float32)
    return v.astype(np.float32)


class VectorRegistry:
    """Thread-safe in-memory storage for player centroids and similarity computation."""

    def __init__(self, embedding_dim: int = 192):
        self._lock = threading.RLock()
        self._embedding_dim = embedding_dim
        # Stored internally as dictionary of PlayerProfile
        self._players: Dict[str, PlayerProfile] = {}

    def register_player(self, player_id: str, name: str, embedding: np.ndarray) -> PlayerProfile:
        """
        Enroll a new player with their initial unit-normalized speaker centroid.

        Args:
            player_id: Unique string identifier for player.
            name: Display name.
            embedding: 1D array of speaker embedding.

        Returns:
            Enrolled PlayerProfile.
        """
        normalized_centroid = _normalize_vector(embedding)
        if len(normalized_centroid) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {len(normalized_centroid)}"
            )

        with self._lock:
            existing = self._players.get(player_id)
            if existing:
                existing.name = name
                existing.centroid = normalized_centroid
                existing.sample_count = 1
                profile = existing
            else:
                profile = PlayerProfile(
                    player_id=player_id,
                    name=name,
                    centroid=normalized_centroid,
                    sample_count=1,
                    score=0,
                )
                self._players[player_id] = profile

            logger.info(f"Registered player '{name}' (ID: {player_id})")
            return profile

    def get_player(self, player_id: str) -> Optional[PlayerProfile]:
        """Retrieve player profile by ID."""
        with self._lock:
            return self._players.get(player_id)

    def list_players(self) -> List[PlayerProfile]:
        """List all enrolled player profiles."""
        with self._lock:
            return list(self._players.values())

    def remove_player(self, player_id: str) -> bool:
        """Remove a player from registry."""
        with self._lock:
            if player_id in self._players:
                del self._players[player_id]
                logger.info(f"Removed player ID: {player_id}")
                return True
            return False

    def count(self) -> int:
        """Return total enrolled players."""
        with self._lock:
            return len(self._players)

    @staticmethod
    def calculate_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two speaker vectors.
        Since both are L2-normalized, cosine similarity equals the dot product.

        Returns:
            Cosine similarity float clipped to [-1.0, 1.0].
        """
        va = _normalize_vector(embedding_a)
        vb = _normalize_vector(embedding_b)
        sim = float(np.dot(va, vb))
        return max(-1.0, min(1.0, sim))

    def update_centroid(
        self, player_id: str, new_embedding: np.ndarray, alpha: float = 0.85
    ) -> np.ndarray:
        """
        Update stored speaker centroid using Exponential Moving Average (EMA):
            c_new = Normalize(alpha * c_old + (1 - alpha) * new_embedding)

        Args:
            player_id: ID of the player to update.
            new_embedding: Verified new speech embedding.
            alpha: Smoothing factor in [0.0, 1.0]. Default 0.85.

        Returns:
            The updated, unit-normalized centroid.
        """
        new_vec = _normalize_vector(new_embedding)
        if len(new_vec) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {len(new_vec)}"
            )

        with self._lock:
            player = self._players.get(player_id)
            if not player:
                raise KeyError(f"Player ID '{player_id}' not found in registry.")

            # Compute EMA
            updated = alpha * player.centroid + (1.0 - alpha) * new_vec
            normalized_updated = _normalize_vector(updated)

            player.centroid = normalized_updated
            player.sample_count += 1

            logger.info(
                f"Updated centroid for '{player.name}' (ID: {player_id}, sample #{player.sample_count})"
            )
            return player.centroid

    def get_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve all enrolled centroids stacked into a single 2D matrix (N, D).
        Enables vectorized 1-of-N cosine scoring via a single BLAS GEMV call.

        Returns:
            Tuple of (list_of_player_ids, matrix_of_centroids).
        """
        with self._lock:
            if not self._players:
                return [], np.empty((0, self._embedding_dim), dtype=np.float32)

            player_ids = list(self._players.keys())
            matrix = np.stack([self._players[pid].centroid for pid in player_ids]).astype(
                np.float32
            )
            return player_ids, matrix

    def reset(self) -> None:
        """Clear all enrolled players."""
        with self._lock:
            self._players.clear()
            logger.info("VectorRegistry reset: all players cleared.")
