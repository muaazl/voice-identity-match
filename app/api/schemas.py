"""
Pydantic Request and Response Schemas for VoiceGate API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# Identity Enrollment
# -------------------------------------------------------------
class IdentityEnrollResponse(BaseModel):
    identity_id: str
    name: str
    sample_count: int
    audio_duration_sec: float
    speech_duration_sec: float
    message: str
    embedding: Optional[List[float]] = None


# -------------------------------------------------------------
# Stateless Audio Embedding Extraction
# -------------------------------------------------------------
class AudioMetricsResponse(BaseModel):
    raw_duration_sec: float
    speech_duration_sec: float
    speech_ratio: float


class AudioEmbedResponse(BaseModel):
    embedding: List[float]
    audio_metrics: AudioMetricsResponse
    status: str = "success"


class IdentityItem(BaseModel):
    identity_id: str
    name: str
    sample_count: int


class IdentityListResponse(BaseModel):
    identities: List[IdentityItem]
    total: int


# -------------------------------------------------------------
# Verification (1:1 and 1:N)
# -------------------------------------------------------------
class CandidateMatchResponse(BaseModel):
    identity_id: str
    name: str
    similarity: float
    similarity_percent: float
    probability_percent: Optional[float] = None


class VerifyResponse(BaseModel):
    verdict: str
    identity_id: Optional[str]
    name: Optional[str]
    confidence: float
    confidence_percent: float
    is_certain: bool
    margin: float
    rankings: List[CandidateMatchResponse]
    audio_metrics: AudioMetricsResponse


# -------------------------------------------------------------
# Health Check
# -------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    embedding_dim: int
    enrolled_identities: int


# -------------------------------------------------------------
# Session Management
# -------------------------------------------------------------
class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: float
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    exists: bool
    identity_count: int
    created_at: Optional[float] = None
    last_active_at: Optional[float] = None
