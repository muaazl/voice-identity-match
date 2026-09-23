"""
Data models and schemas for the biometric vector registry and game engine.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class PlayerProfile:
    """Biometric profile and state for an enrolled player."""
    player_id: str
    name: str
    centroid: np.ndarray  # Shape (192,), float32, L2-normalized
    sample_count: int = 1
    score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "sample_count": self.sample_count,
            "score": self.score,
            "centroid_dim": len(self.centroid),
            "centroid_norm": float(np.linalg.norm(self.centroid)),
        }


@dataclass
class CandidateMatch:
    """Ranked candidate similarity in Mode A."""
    player_id: str
    name: str
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IdentificationResult:
    """Result of Mode A 1-of-N speaker classification."""
    winner_player_id: Optional[str]
    winner_name: Optional[str]
    confidence: float
    is_certain: bool
    margin: float
    rankings: List[CandidateMatch]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner_player_id": self.winner_player_id,
            "winner_name": self.winner_name,
            "confidence": round(self.confidence, 4),
            "is_certain": self.is_certain,
            "margin": round(self.margin, 4),
            "rankings": [r.to_dict() for r in self.rankings],
        }


@dataclass
class ImpostorEvaluationResult:
    """Result of Mode B Impostor Challenge matching."""
    target_player_id: str
    target_name: str
    similarity_score: float
    status: str  # "SYSTEM_FOOLED" | "CLOSE_MIMIC" | "POOR_ATTEMPT"
    message: str
    points: int
    security_breached: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_player_id": self.target_player_id,
            "target_name": self.target_name,
            "similarity_score": round(self.similarity_score, 4),
            "status": self.status,
            "message": self.message,
            "points": self.points,
            "security_breached": self.security_breached,
        }
