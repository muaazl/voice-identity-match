"""Engine package: Vector registry, scoring logic, and data models."""

from .models import (
    PlayerProfile,
    CandidateMatch,
    IdentificationResult,
    ImpostorEvaluationResult,
)
from .registry import VectorRegistry
from .game import GameEngine

__all__ = [
    "PlayerProfile",
    "CandidateMatch",
    "IdentificationResult",
    "ImpostorEvaluationResult",
    "VectorRegistry",
    "GameEngine",
]
