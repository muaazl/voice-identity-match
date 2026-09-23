"""
Game Rules & Match Logic Engine for VoiceMimic.
Orchestrates:
- Mode A (Blind Identifier): 1-of-N cosine similarity matching, certainty thresholding, and true-speaker claims.
- Mode B (Impostor Challenge): Proximity scoring vs biometric security firewall.
- Scoreboard and player point tracking.
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np

from .registry import VectorRegistry, _normalize_vector
from .models import (
    CandidateMatch,
    IdentificationResult,
    ImpostorEvaluationResult,
)

logger = logging.getLogger(__name__)


class GameEngine:
    """Game mechanics and rule evaluation engine."""

    def __init__(self, registry: VectorRegistry):
        self.registry = registry

    def identify_speaker(
        self, query_embedding: np.ndarray, threshold: float = 0.65
    ) -> IdentificationResult:
        """
        Mode A: 1-of-N Cosine Similarity Speaker Identification.

        Args:
            query_embedding: 1D array of speaker embedding.
            threshold: Minimum cosine score to consider identification certain.

        Returns:
            IdentificationResult with winner, confidence, margin, and rankings.
        """
        player_ids, matrix = self.registry.get_matrix()

        if len(player_ids) == 0:
            return IdentificationResult(
                winner_player_id=None,
                winner_name=None,
                confidence=0.0,
                is_certain=False,
                margin=0.0,
                rankings=[],
            )

        q = _normalize_vector(query_embedding)

        # Single vectorized BLAS GEMV: (N, D) @ (D,) -> (N,)
        similarities = np.dot(matrix, q)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Sort in descending order
        sorted_indices = np.argsort(similarities)[::-1]

        rankings: List[CandidateMatch] = []
        for idx in sorted_indices:
            pid = player_ids[idx]
            player = self.registry.get_player(pid)
            name = player.name if player else pid
            score = float(similarities[idx])
            rankings.append(CandidateMatch(player_id=pid, name=name, similarity=round(score, 4)))

        top_idx = sorted_indices[0]
        top_pid = player_ids[top_idx]
        top_player = self.registry.get_player(top_pid)
        top_name = top_player.name if top_player else top_pid
        confidence = float(similarities[top_idx])

        # Compute runner-up margin
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            margin = float(similarities[top_idx] - similarities[second_idx])
        else:
            margin = confidence

        is_certain = confidence >= threshold

        logger.info(
            f"Mode A Match: Winner='{top_name}' ({top_pid}), "
            f"Confidence={confidence:.4f}, Margin={margin:.4f}, Certain={is_certain}"
        )

        return IdentificationResult(
            winner_player_id=top_pid,
            winner_name=top_name,
            confidence=confidence,
            is_certain=is_certain,
            margin=margin,
            rankings=rankings,
        )

    def claim_round(
        self,
        true_player_id: str,
        query_embedding: np.ndarray,
        points: int = 50,
        alpha: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Award points to the true speaker and adaptively update their profile via EMA.
        Called in Mode A when the true speaker reveals/confirms identity.

        Args:
            true_player_id: Verified player ID who spoke.
            query_embedding: The utterance embedding.
            points: Points awarded for round claim.
            alpha: EMA smoothing parameter.

        Returns:
            Dict containing updated player status and point total.
        """
        player = self.registry.get_player(true_player_id)
        if not player:
            raise KeyError(f"Player ID '{true_player_id}' not found.")

        # Update centroid via EMA
        self.registry.update_centroid(true_player_id, query_embedding, alpha=alpha)

        # Award points
        player.score += points
        logger.info(
            f"Round claimed by '{player.name}' (ID: {true_player_id}). "
            f"+{points} pts, Total: {player.score}, Samples: {player.sample_count}"
        )

        return {
            "player_id": true_player_id,
            "name": player.name,
            "points_awarded": points,
            "total_score": player.score,
            "sample_count": player.sample_count,
        }

    def evaluate_impostor(
        self,
        target_player_id: str,
        query_embedding: np.ndarray,
        mimic_threshold: float = 0.60,
        match_threshold: float = 0.82,
    ) -> ImpostorEvaluationResult:
        """
        Mode B: Impostor Challenge evaluation against target player's centroid.

        Scoring Rules:
        - score >= match_threshold: "System Fooled! Impersonation Successful" (100 points, Security breached)
        - mimic_threshold <= score < match_threshold: "Close Mimic! System Detected Difference" (1-99 partial points)
        - score < mimic_threshold: "Poor Attempt" (0 points)

        Args:
            target_player_id: Player being impersonated.
            query_embedding: The mimic's voice embedding.
            mimic_threshold: Minimum threshold to receive partial mimic points.
            match_threshold: Biometric authentication boundary for system fooling.

        Returns:
            ImpostorEvaluationResult.
        """
        target = self.registry.get_player(target_player_id)
        if not target:
            raise KeyError(f"Target player ID '{target_player_id}' not found in registry.")

        score = self.registry.calculate_similarity(query_embedding, target.centroid)

        if score >= match_threshold:
            status = "SYSTEM_FOOLED"
            message = "System Fooled! Impersonation Successful"
            points = 100
            security_breached = True
        elif score >= mimic_threshold:
            status = "CLOSE_MIMIC"
            message = "Close Mimic! System Detected Difference"
            security_breached = False
            # Linear scaling between mimic_threshold and match_threshold
            span = max(match_threshold - mimic_threshold, 1e-6)
            progress = (score - mimic_threshold) / span
            points = int(round(1 + progress * 98))  # 1 to 99 points
            points = max(1, min(99, points))
        else:
            status = "POOR_ATTEMPT"
            message = "Poor Attempt"
            points = 0
            security_breached = False

        logger.info(
            f"Mode B Evaluation: Target='{target.name}', Score={score:.4f}, "
            f"Status={status}, Points={points}, Breached={security_breached}"
        )

        return ImpostorEvaluationResult(
            target_player_id=target_player_id,
            target_name=target.name,
            similarity_score=score,
            status=status,
            message=message,
            points=points,
            security_breached=security_breached,
        )

    def add_score(self, player_id: str, points: int) -> int:
        """Award arbitrary points to a player."""
        player = self.registry.get_player(player_id)
        if not player:
            raise KeyError(f"Player ID '{player_id}' not found.")
        player.score += points
        return player.score

    def get_scoreboard(self) -> List[Dict[str, Any]]:
        """Return all players sorted by score descending."""
        players = self.registry.list_players()
        sorted_players = sorted(players, key=lambda p: p.score, reverse=True)
        return [
            {
                "player_id": p.player_id,
                "name": p.name,
                "score": p.score,
                "sample_count": p.sample_count,
            }
            for p in sorted_players
        ]

    def reset_scores(self) -> None:
        """Reset all player scores to 0 without deleting biometric profiles."""
        for player in self.registry.list_players():
            player.score = 0
        logger.info("All player scores reset to 0.")
