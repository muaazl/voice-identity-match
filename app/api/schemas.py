"""
Pydantic Request and Response Schemas for VoiceMimic API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# Player Registration
# -------------------------------------------------------------
class PlayerRegisterResponse(BaseModel):
    player_id: str
    name: str
    sample_count: int
    score: int
    audio_duration_sec: float
    speech_duration_sec: float
    message: str
    embedding: Optional[List[float]] = None


# -------------------------------------------------------------
# Stateless Audio Embedding Extraction
# -------------------------------------------------------------
class AudioEmbedResponse(BaseModel):
    embedding: List[float]
    audio_metrics: "AudioMetricsResponse"
    status: str = "success"



class PlayerItem(BaseModel):
    player_id: str
    name: str
    sample_count: int
    score: int


class PlayerListResponse(BaseModel):
    players: List[PlayerItem]
    total: int


# -------------------------------------------------------------
# Mode A: Guess-Who (Blind Identifier)
# -------------------------------------------------------------
class CandidateMatchResponse(BaseModel):
    player_id: str
    name: str
    similarity: float
    similarity_percent: float
    probability_percent: Optional[float] = None


class AudioMetricsResponse(BaseModel):
    raw_duration_sec: float
    speech_duration_sec: float
    speech_ratio: float


class GuessWhoResponse(BaseModel):
    round_id: str
    predicted_player_id: Optional[str]
    predicted_name: Optional[str]
    confidence: float
    confidence_percent: float
    is_certain: bool
    needs_confirmation: bool
    margin: float
    rankings: List[CandidateMatchResponse]
    audio_metrics: AudioMetricsResponse


# -------------------------------------------------------------
# Mode A: Confirm / Claim Speaker
# -------------------------------------------------------------
class ConfirmSpeakerRequest(BaseModel):
    actual_player_id: str
    round_id: Optional[str] = None
    points: int = Field(default=50, ge=0, le=500)


class ConfirmSpeakerResponse(BaseModel):
    player_id: str
    name: str
    points_awarded: int
    total_score: int
    sample_count: int
    message: str


# -------------------------------------------------------------
# Mode B: Impostor Challenge
# -------------------------------------------------------------
class MimicChallengeResponse(BaseModel):
    target_player_id: str
    target_name: str
    similarity_score: float
    similarity_percent: float
    status: str  # "SYSTEM_FOOLED" | "CLOSE_MIMIC" | "POOR_ATTEMPT"
    message: str
    points: int
    security_breached: bool
    thresholds: Dict[str, float]


# -------------------------------------------------------------
# Scoreboard & Administration
# -------------------------------------------------------------
class ScoreboardEntry(BaseModel):
    player_id: str
    name: str
    score: int
    sample_count: int


class ScoreboardResponse(BaseModel):
    players: List[ScoreboardEntry]
    total_players: int
    active_rounds: int


class ResetRequest(BaseModel):
    reset_scores_only: bool = False


class ResetResponse(BaseModel):
    message: str
    reset_scores_only: bool
    total_players: int


# -------------------------------------------------------------
# Health Check
# -------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    embedding_dim: int
    enrolled_players: int
    active_rounds: int


# -------------------------------------------------------------
# Room & Session Management
# -------------------------------------------------------------
class RoomCreateRequest(BaseModel):
    room_code: Optional[str] = None


class RoomCreateResponse(BaseModel):
    room_code: str
    created_at: float
    message: str


class RoomStatusResponse(BaseModel):
    room_code: str
    exists: bool
    player_count: int
    active_rounds: int
    created_at: Optional[float] = None
    last_active_at: Optional[float] = None

