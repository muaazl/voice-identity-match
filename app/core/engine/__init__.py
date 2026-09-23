"""Engine package: Vector registry, scoring logic, and data models."""

from .models import (
    PlayerProfile,
    CandidateMatch,
    IdentificationResult,
    ImpostorEvaluationResult,
)
from .registry import VectorRegistry
from .game import GameEngine
from .room import GameRoom, RoomManager

__all__ = [
    "PlayerProfile",
    "CandidateMatch",
    "IdentificationResult",
    "ImpostorEvaluationResult",
    "VectorRegistry",
    "GameEngine",
    "GameRoom",
    "RoomManager",
]

