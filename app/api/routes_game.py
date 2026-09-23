"""
Game Mechanics & Biometric Challenge API Endpoints.
Handles Mode A (Blind Identifier), Mode B (Impostor Challenge), and Scoreboards.
"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends

from .schemas import (
    GuessWhoResponse,
    CandidateMatchResponse,
    AudioMetricsResponse,
    ConfirmSpeakerRequest,
    ConfirmSpeakerResponse,
    MimicChallengeResponse,
    ScoreboardResponse,
    ScoreboardEntry,
    ResetRequest,
    ResetResponse,
)
from .dependencies import get_room
from app.core.engine.room import GameRoom

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/game", tags=["Game"])


@router.post("/guess-who", response_model=GuessWhoResponse)
async def guess_who(
    request: Request,
    file: UploadFile = File(..., description="Utterance of the unknown speaker"),
    threshold: float = Form(0.65, description="Certainty threshold for Mode A"),
    room: GameRoom = Depends(get_room),
):
    """
    Mode A: Blind Identifier.
    Accepts speech from an unknown player, identifies candidate using 1-of-N cosine similarity,
    and caches the round state for true-speaker claims.
    Scoped to the active GameRoom.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) < 320:
        raise HTTPException(status_code=400, detail="Audio file too small or empty.")

    pipeline = request.app.state.pipeline
    game = room.game
    round_cache = room.round_cache

    try:
        meta = pipeline.process_with_metadata(audio_bytes)
    except Exception as e:
        logger.error(f"Audio processing failure in guess-who: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {str(e)}")

    query_embedding = meta["embedding"]
    result = game.identify_speaker(query_embedding, threshold=threshold)

    # Cache round embedding for dispute resolution / EMA claim
    round_id = f"rnd_{uuid.uuid4().hex[:12]}"
    round_cache.put(
        round_id=round_id,
        query_embedding=query_embedding,
        predicted_player_id=result.winner_player_id,
        confidence=result.confidence,
    )

    rankings_response = [
        CandidateMatchResponse(
            player_id=r.player_id,
            name=r.name,
            similarity=round(r.similarity, 4),
            similarity_percent=round(max(0.0, r.similarity) * 100, 1),
            probability_percent=round(r.probability * 100, 1) if r.probability is not None else None,
        )
        for r in result.rankings
    ]

    audio_metrics = AudioMetricsResponse(
        raw_duration_sec=meta["raw_duration_sec"],
        speech_duration_sec=meta["speech_duration_sec"],
        speech_ratio=meta["speech_ratio"],
    )

    return GuessWhoResponse(
        round_id=round_id,
        predicted_player_id=result.winner_player_id,
        predicted_name=result.winner_name,
        confidence=round(result.confidence, 4),
        confidence_percent=round(max(0.0, result.confidence) * 100, 1),
        is_certain=result.is_certain,
        needs_confirmation=not result.is_certain,
        margin=round(result.margin, 4),
        rankings=rankings_response,
        audio_metrics=audio_metrics,
    )


@router.post("/confirm-speaker", response_model=ConfirmSpeakerResponse)
async def confirm_speaker(body: ConfirmSpeakerRequest, room: GameRoom = Depends(get_room)):
    """
    Mode A: True Speaker Claim & Adaptive Profile Update.
    Awards points to the true speaker and adaptively updates their voice centroid via EMA.
    Scoped to the active GameRoom.
    """
    game = room.game
    registry = room.registry
    round_cache = room.round_cache

    player = registry.get_player(body.actual_player_id)
    if not player:
        raise HTTPException(
            status_code=404, detail=f"Player '{body.actual_player_id}' not found in room '{room.room_code}'."
        )

    # Check if we have the cached query embedding for this round
    cached_round = round_cache.pop(body.round_id) if body.round_id else None

    if cached_round and "embedding" in cached_round:
        claim_result = game.claim_round(
            true_player_id=body.actual_player_id,
            query_embedding=cached_round["embedding"],
            points=body.points,
        )
        message = (
            f"Speaker confirmed as '{claim_result['name']}'. "
            f"+{body.points} points awarded and voice profile updated via EMA!"
        )
        sample_count = claim_result["sample_count"]
        total_score = claim_result["total_score"]
    else:
        # Fallback: Just award points if round was already claimed or missing
        total_score = game.add_score(body.actual_player_id, body.points)
        sample_count = player.sample_count
        message = f"Points (+{body.points}) awarded to '{player.name}'."

    return ConfirmSpeakerResponse(
        player_id=player.player_id,
        name=player.name,
        points_awarded=body.points,
        total_score=total_score,
        sample_count=sample_count,
        message=message,
    )


@router.post("/mimic-challenge", response_model=MimicChallengeResponse)
async def mimic_challenge(
    request: Request,
    target_player_id: str = Form(..., description="ID of player to impersonate"),
    file: UploadFile = File(..., description="Impersonation audio sample"),
    mimic_threshold: float = Form(0.60, description="Minimum score to earn partial points"),
    match_threshold: float = Form(0.82, description="Biometric firewall boundary for max points"),
    room: GameRoom = Depends(get_room),
):
    """
    Mode B: Impostor Challenge.
    Measures acoustic proximity to the target player's centroid and evaluates biometric breach.
    Scoped to the active GameRoom.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) < 320:
        raise HTTPException(status_code=400, detail="Audio file too small or empty.")

    pipeline = request.app.state.pipeline
    game = room.game
    registry = room.registry

    target_player = registry.get_player(target_player_id)
    if not target_player:
        raise HTTPException(
            status_code=404, detail=f"Target player '{target_player_id}' not found in room '{room.room_code}'."
        )

    try:
        embedding = pipeline.process(audio_bytes)
    except Exception as e:
        logger.error(f"Audio processing failure in mimic-challenge: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {str(e)}")

    eval_result = game.evaluate_impostor(
        target_player_id=target_player_id,
        query_embedding=embedding,
        mimic_threshold=mimic_threshold,
        match_threshold=match_threshold,
    )

    return MimicChallengeResponse(
        target_player_id=eval_result.target_player_id,
        target_name=eval_result.target_name,
        similarity_score=round(eval_result.similarity_score, 4),
        similarity_percent=round(max(0.0, eval_result.similarity_score) * 100, 1),
        status=eval_result.status,
        message=eval_result.message,
        points=eval_result.points,
        security_breached=eval_result.security_breached,
        thresholds={
            "mimic_threshold": mimic_threshold,
            "match_threshold": match_threshold,
        },
    )


@router.get("/scoreboard", response_model=ScoreboardResponse)
async def get_scoreboard(room: GameRoom = Depends(get_room)):
    """Retrieve current game leaderboard and active round counts for the active GameRoom."""
    game = room.game
    registry = room.registry
    round_cache = room.round_cache

    entries = [ScoreboardEntry(**item) for item in game.get_scoreboard()]
    return ScoreboardResponse(
        players=entries,
        total_players=registry.count(),
        active_rounds=round_cache.count(),
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_game(body: ResetRequest = ResetRequest(), room: GameRoom = Depends(get_room)):
    """Reset player scores or completely clear game roster and active rounds for the active GameRoom."""
    game = room.game
    registry = room.registry
    round_cache = room.round_cache

    if body.reset_scores_only:
        game.reset_scores()
        msg = f"Player scores in room '{room.room_code}' reset to zero. Biometric profiles retained."
    else:
        registry.reset()
        game.reset_scores()
        round_cache.clear()
        msg = f"Complete game reset for room '{room.room_code}': all players, scores, and active rounds cleared."

    return ResetResponse(
        message=msg,
        reset_scores_only=body.reset_scores_only,
        total_players=registry.count(),
    )

