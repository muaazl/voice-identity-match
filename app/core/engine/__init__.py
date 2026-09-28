"""Engine package: Vector registry, verification logic, and data models."""

from .models import (
    IdentityProfile,
    CandidateMatch,
    VerificationResult,
)
from .registry import VectorRegistry
from .verification import VerificationEngine
from .session import AuthSession, SessionManager

__all__ = [
    "IdentityProfile",
    "CandidateMatch",
    "VerificationResult",
    "VectorRegistry",
    "VerificationEngine",
    "AuthSession",
    "SessionManager",
]
