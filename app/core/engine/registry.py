"""
In-Memory Thread-Safe Biometric Vector Registry.
Maintains enrolled identity speaker centroids, calculates cosine similarity,
and updates speaker profiles via Exponential Moving Average (EMA).
"""

import threading
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import IdentityProfile
from .utils import _normalize_vector

logger = logging.getLogger(__name__)

class VectorRegistry:
    """Thread-safe in-memory storage for identity centroids and similarity computation."""

    def __init__(self, embedding_dim: int = 192):
        self._lock = threading.RLock()
        self._embedding_dim = embedding_dim
        # Stored internally as dictionary of IdentityProfile
        self._identities: Dict[str, IdentityProfile] = {}
        self._matrix_cache: Optional[Tuple[List[str], np.ndarray]] = None

    def _invalidate_cache(self):
        self._matrix_cache = None

    def register_identity(self, identity_id: str, name: str, embedding: np.ndarray) -> IdentityProfile:
        """
        Enroll a new identity with their initial unit-normalized speaker centroid.
        """
        normalized_centroid = _normalize_vector(embedding)
        if len(normalized_centroid) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {len(normalized_centroid)}"
            )

        with self._lock:
            existing = self._identities.get(identity_id)
            if existing:
                existing.name = name
                existing.centroid = normalized_centroid
                existing.sample_count = 1
                profile = existing
            else:
                profile = IdentityProfile(
                    identity_id=identity_id,
                    name=name,
                    centroid=normalized_centroid,
                    sample_count=1,
                )
                self._identities[identity_id] = profile

            self._invalidate_cache()
            logger.info(f"Registered identity '{name}' (ID: {identity_id})")
            return profile

    def get_identity(self, identity_id: str) -> Optional[IdentityProfile]:
        """Retrieve identity profile by ID."""
        with self._lock:
            return self._identities.get(identity_id)

    def list_identities(self) -> List[IdentityProfile]:
        """List all enrolled identity profiles."""
        with self._lock:
            return list(self._identities.values())

    def remove_identity(self, identity_id: str) -> bool:
        """Remove an identity from registry."""
        with self._lock:
            if identity_id in self._identities:
                del self._identities[identity_id]
                self._invalidate_cache()
                logger.info(f"Removed identity ID: {identity_id}")
                return True
            return False

    def count(self) -> int:
        """Return total enrolled identities."""
        with self._lock:
            return len(self._identities)

    @staticmethod
    def calculate_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two speaker vectors."""
        va = _normalize_vector(embedding_a)
        vb = _normalize_vector(embedding_b)
        sim = float(np.dot(va, vb))
        return max(-1.0, min(1.0, sim))

    def update_centroid(
        self, identity_id: str, new_embedding: np.ndarray, alpha: float = 0.85
    ) -> np.ndarray:
        """
        Update stored speaker centroid using Exponential Moving Average (EMA).
        """
        new_vec = _normalize_vector(new_embedding)
        if len(new_vec) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {len(new_vec)}"
            )

        with self._lock:
            identity = self._identities.get(identity_id)
            if not identity:
                raise KeyError(f"Identity ID '{identity_id}' not found in registry.")

            # Compute EMA
            updated = alpha * identity.centroid + (1.0 - alpha) * new_vec
            normalized_updated = _normalize_vector(updated)

            identity.centroid = normalized_updated
            identity.sample_count += 1
            self._invalidate_cache()

            logger.info(
                f"Updated centroid for '{identity.name}' (ID: {identity_id}, sample #{identity.sample_count})"
            )
            return identity.centroid

    def get_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve all enrolled centroids stacked into a single 2D matrix (N, D).
        """
        with self._lock:
            if self._matrix_cache is not None:
                return self._matrix_cache

            if not self._identities:
                self._matrix_cache = ([], np.empty((0, self._embedding_dim), dtype=np.float32))
                return self._matrix_cache

            identity_ids = list(self._identities.keys())
            matrix = np.stack([self._identities[pid].centroid for pid in identity_ids]).astype(
                np.float32
            )
            self._matrix_cache = (identity_ids, matrix)
            return self._matrix_cache

    def reset(self) -> None:
        """Clear all enrolled identities."""
        with self._lock:
            self._identities.clear()
            self._invalidate_cache()
            logger.info("VectorRegistry reset: all identities cleared.")
