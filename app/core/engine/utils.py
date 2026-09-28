"""
Utility functions for the game engine.
"""

import numpy as np

def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure vector is float32 with unit L2 norm."""
    v = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(v)
    if norm > eps:
        return (v / norm).astype(np.float32)
    return v.astype(np.float32)
