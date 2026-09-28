"""Engine package: Vector registry, scoring logic, and data models."""

from .models import (
    PlayerProfile,
    CandidateMatch,
    IdentificationResult,
    MimicResult,
)
from .registry import VectorRegistry
from .game import GameEngine
from .room import GameRoom, RoomManager

__all__ = [
    "PlayerProfile",
    "CandidateMatch",
    "IdentificationResult",
    "MimicResult",
    "VectorRegistry",
    "GameEngine",
    "GameRoom",
    "RoomManager",
]

