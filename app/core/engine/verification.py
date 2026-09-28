"""
Verification Engine for VoiceGate.
Handles 1:1 and 1:N voice authentication via cosine similarity.
"""

import logging
from typing import Optional, List
import numpy as np

from .registry import VectorRegistry
from .utils import _normalize_vector
from .models import CandidateMatch, VerificationResult

logger = logging.getLogger(__name__)


class VerificationEngine:
    """Core verification mechanics."""

    def __init__(self, registry: VectorRegistry):
        self.registry = registry

    def verify_speaker(
        self, query_embedding: np.ndarray, target_identity_id: Optional[str] = None, threshold: float = 0.65
    ) -> VerificationResult:
        """
        Verify speaker identity using Cosine Similarity.
        If target_identity_id is provided, performs 1:1 authentication.
        Otherwise, performs 1:N identification against all enrolled identities.

        Args:
            query_embedding: 1D array of speaker embedding.
            target_identity_id: Optional ID for 1:1 matching.
            threshold: Minimum cosine score to consider a match.

        Returns:
            VerificationResult with verdict, confidence, and rankings.
        """
        q = _normalize_vector(query_embedding)

        if target_identity_id:
            # 1:1 Authentication
            target = self.registry.get_identity(target_identity_id)
            if not target:
                raise ValueError(f"Target identity {target_identity_id} not found.")
            
            score = self.registry.calculate_similarity(q, target.centroid)
            is_match = score >= threshold
            verdict = "MATCH" if is_match else "NO_MATCH"
            
            match = CandidateMatch(
                identity_id=target.identity_id,
                name=target.name,
                similarity=round(score, 4),
                probability=1.0 if is_match else 0.0
            )

            logger.info(
                f"1:1 Verify: Target='{target.name}', Score={score:.4f}, Verdict={verdict}"
            )

            return VerificationResult(
                verdict=verdict,
                identity_id=target.identity_id if is_match else None,
                name=target.name if is_match else None,
                confidence=score,
                is_certain=is_match,
                margin=score,
                rankings=[match]
            )
            
        else:
            # 1:N Identification
            identity_ids, matrix = self.registry.get_matrix()

            if len(identity_ids) == 0:
                return VerificationResult(
                    verdict="NO_MATCH",
                    identity_id=None,
                    name=None,
                    confidence=0.0,
                    is_certain=False,
                    margin=0.0,
                    rankings=[],
                )

            # Single vectorized BLAS GEMV: (N, D) @ (D,) -> (N,)
            similarities = np.dot(matrix, q)
            similarities = np.clip(similarities, -1.0, 1.0)

            # Contrastive softmax calibration with temperature tau=0.08
            tau = 0.08
            exp_sims = np.exp((similarities - np.max(similarities)) / tau)
            probs = exp_sims / np.sum(exp_sims)

            # Sort in descending order
            sorted_indices = np.argsort(similarities)[::-1]

            rankings: List[CandidateMatch] = []
            for idx in sorted_indices:
                pid = identity_ids[idx]
                identity = self.registry.get_identity(pid)
                name = identity.name if identity else pid
                score = float(similarities[idx])
                prob = float(probs[idx])
                rankings.append(
                    CandidateMatch(
                        identity_id=pid,
                        name=name,
                        similarity=round(score, 4),
                        probability=round(prob, 4),
                    )
                )

            top_idx = sorted_indices[0]
            top_pid = identity_ids[top_idx]
            top_identity = self.registry.get_identity(top_pid)
            top_name = top_identity.name if top_identity else top_pid
            confidence = float(similarities[top_idx])

            # Compute runner-up margin
            if len(sorted_indices) > 1:
                second_idx = sorted_indices[1]
                margin = float(similarities[top_idx] - similarities[second_idx])
            else:
                margin = confidence

            top_prob = float(probs[top_idx])
            is_certain = (confidence >= threshold) and (top_prob >= 0.50 or len(identity_ids) == 1)
            
            if is_certain:
                verdict = "MATCH"
            elif confidence >= (threshold - 0.1):
                verdict = "UNCERTAIN"
            else:
                verdict = "NO_MATCH"

            logger.info(
                f"1:N Verify: Winner='{top_name}' ({top_pid}), "
                f"Confidence={confidence:.4f}, Margin={margin:.4f}, Verdict={verdict}"
            )

            return VerificationResult(
                verdict=verdict,
                identity_id=top_pid if verdict == "MATCH" else None,
                name=top_name if verdict == "MATCH" else None,
                confidence=confidence,
                is_certain=is_certain,
                margin=margin,
                rankings=rankings,
            )
