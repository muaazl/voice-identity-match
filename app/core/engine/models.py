"""
Data models and schemas for the biometric vector registry and verification engine.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class IdentityProfile:
    """Biometric profile and state for an enrolled identity."""
    identity_id: str
    name: str
    centroid: np.ndarray  # Shape (192,), float32, L2-normalized
    sample_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "name": self.name,
            "sample_count": self.sample_count,
            "centroid_dim": len(self.centroid),
            "centroid_norm": float(np.linalg.norm(self.centroid)),
        }


@dataclass
class CandidateMatch:
    """Ranked candidate similarity."""
    identity_id: str
    name: str
    similarity: float
    probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.probability is not None:
            d["probability_percent"] = round(self.probability * 100, 1)
        return d


@dataclass
class VerificationResult:
    """Result of speaker verification."""
    verdict: str  # "MATCH", "NO_MATCH", "UNCERTAIN"
    identity_id: Optional[str]
    name: Optional[str]
    confidence: float
    is_certain: bool
    margin: float
    rankings: List[CandidateMatch]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "identity_id": self.identity_id,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "is_certain": self.is_certain,
            "margin": round(self.margin, 4),
            "rankings": [r.to_dict() for r in self.rankings],
        }
