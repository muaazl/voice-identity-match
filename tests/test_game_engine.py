"""
Unit tests for In-Memory Vector Registry and Game Engine (Phase 2).
Tests cosine similarity math, EMA centroid updates, Mode A & Mode B rules,
and thread safety under concurrent access.
"""

import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from app.core.engine.registry import VectorRegistry
from app.core.engine.game import GameEngine


def create_random_unit_vector(dim: int = 192, seed: int = None) -> np.ndarray:
    """Generate a reproducible random unit vector on the S^{dim-1} hypersphere."""
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.normal(0, 1, dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_registry_registration_and_lookup():
    registry = VectorRegistry(embedding_dim=192)
    vec1 = create_random_unit_vector(192, seed=42)

    profile = registry.register_player("player_1", "Alice", vec1)
    assert profile.player_id == "player_1"
    assert profile.name == "Alice"
    assert profile.sample_count == 1
    assert profile.score == 0
    assert np.allclose(profile.centroid, vec1, atol=1e-5)
    assert np.linalg.norm(profile.centroid) == pytest.approx(1.0, rel=1e-5)

    fetched = registry.get_player("player_1")
    assert fetched is not None
    assert fetched.name == "Alice"
    assert registry.count() == 1

    # Remove player
    assert registry.remove_player("player_1") is True
    assert registry.count() == 0
    assert registry.get_player("player_1") is None


def test_cosine_similarity_math():
    registry = VectorRegistry(embedding_dim=192)
    v1 = create_random_unit_vector(192, seed=1)

    # Identical vectors -> similarity = 1.0
    sim_identical = registry.calculate_similarity(v1, v1)
    assert sim_identical == pytest.approx(1.0, rel=1e-5)

    # Opposite vectors -> similarity = -1.0
    sim_opposite = registry.calculate_similarity(v1, -v1)
    assert sim_opposite == pytest.approx(-1.0, rel=1e-5)

    # Orthogonal vectors -> similarity = 0.0
    v_ortho = np.zeros(192, dtype=np.float32)
    v_ortho[0] = 1.0
    v_ortho2 = np.zeros(192, dtype=np.float32)
    v_ortho2[1] = 1.0
    sim_ortho = registry.calculate_similarity(v_ortho, v_ortho2)
    assert sim_ortho == pytest.approx(0.0, abs=1e-6)


def test_ema_centroid_update_preserves_unit_norm():
    registry = VectorRegistry(embedding_dim=192)
    v_initial = create_random_unit_vector(192, seed=10)
    registry.register_player("player_1", "Bob", v_initial)

    v_new = create_random_unit_vector(192, seed=20)
    alpha = 0.85
    updated_centroid = registry.update_centroid("player_1", v_new, alpha=alpha)

    # Unit norm must be strictly conserved
    assert np.linalg.norm(updated_centroid) == pytest.approx(1.0, rel=1e-5)

    # Mathematical expectation
    expected_unnorm = alpha * v_initial + (1 - alpha) * v_new
    expected_norm = expected_unnorm / np.linalg.norm(expected_unnorm)
    assert np.allclose(updated_centroid, expected_norm, atol=1e-5)

    # Sample count incremented
    p = registry.get_player("player_1")
    assert p.sample_count == 2


def test_mode_a_identification():
    registry = VectorRegistry(embedding_dim=192)
    game = GameEngine(registry)

    # Empty registry edge case
    empty_result = game.identify_speaker(create_random_unit_vector(192))
    assert empty_result.winner_player_id is None
    assert empty_result.confidence == 0.0
    assert empty_result.is_certain is False

    # Enroll 3 distinct players
    v_alice = create_random_unit_vector(192, seed=100)
    v_bob = create_random_unit_vector(192, seed=200)
    v_carol = create_random_unit_vector(192, seed=300)

    registry.register_player("p_alice", "Alice", v_alice)
    registry.register_player("p_bob", "Bob", v_bob)
    registry.register_player("p_carol", "Carol", v_carol)

    # Query with a vector having 0.90 similarity to Alice
    v_ortho_alice = np.random.normal(0, 1, 192).astype(np.float32)
    v_ortho_alice -= np.dot(v_ortho_alice, v_alice) * v_alice
    v_ortho_alice /= np.linalg.norm(v_ortho_alice)
    q_alice = 0.90 * v_alice + np.sqrt(1 - 0.90**2) * v_ortho_alice

    result = game.identify_speaker(q_alice, threshold=0.65)
    assert result.winner_player_id == "p_alice"
    assert result.winner_name == "Alice"
    assert result.confidence == pytest.approx(0.90, abs=1e-3)
    assert result.is_certain is True
    assert result.margin > 0.40
    assert len(result.rankings) == 3
    assert result.rankings[0].player_id == "p_alice"


def test_mode_a_claim_round():
    registry = VectorRegistry(embedding_dim=192)
    game = GameEngine(registry)

    v_bob = create_random_unit_vector(192, seed=42)
    registry.register_player("p_bob", "Bob", v_bob)

    # Bob claims round
    q = create_random_unit_vector(192, seed=43)
    claim_info = game.claim_round("p_bob", q, points=50, alpha=0.85)

    assert claim_info["player_id"] == "p_bob"
    assert claim_info["points_awarded"] == 50
    assert claim_info["total_score"] == 50
    assert claim_info["sample_count"] == 2

    bob = registry.get_player("p_bob")
    assert bob.score == 50
    assert np.linalg.norm(bob.centroid) == pytest.approx(1.0, rel=1e-5)


def test_mode_b_impostor_tiers():
    registry = VectorRegistry(embedding_dim=192)
    game = GameEngine(registry)

    v_target = create_random_unit_vector(192, seed=777)
    registry.register_player("p_target", "TargetUser", v_target)

    # Orthogonal basis vector for exact similarity synthesis
    v_ortho = np.random.normal(0, 1, 192).astype(np.float32)
    v_ortho -= np.dot(v_ortho, v_target) * v_target
    v_ortho /= np.linalg.norm(v_ortho)

    # Tier 1: System Fooled (score >= 0.82) -> test with 0.88
    s_fool = 0.88
    q_fool = s_fool * v_target + np.sqrt(1 - s_fool**2) * v_ortho
    res_fool = game.evaluate_impostor("p_target", q_fool, mimic_threshold=0.60, match_threshold=0.82)
    assert res_fool.status == "SYSTEM_FOOLED"
    assert res_fool.points == 100
    assert res_fool.security_breached is True
    assert "System Fooled" in res_fool.message

    # Tier 2: Close Mimic (0.60 <= score < 0.82) -> test with 0.72
    s_close = 0.72
    q_close = s_close * v_target + np.sqrt(1 - s_close**2) * v_ortho
    res_close = game.evaluate_impostor("p_target", q_close, mimic_threshold=0.60, match_threshold=0.82)
    assert res_close.status == "CLOSE_MIMIC"
    assert 1 <= res_close.points <= 99
    assert res_close.security_breached is False
    assert "Close Mimic" in res_close.message

    # Tier 3: Poor Attempt (score < 0.60) -> test with 0.35
    s_poor = 0.35
    q_poor = s_poor * v_target + np.sqrt(1 - s_poor**2) * v_ortho
    res_poor = game.evaluate_impostor("p_target", q_poor, mimic_threshold=0.60, match_threshold=0.82)
    assert res_poor.status == "POOR_ATTEMPT"
    assert res_poor.points == 0
    assert res_poor.security_breached is False
    assert "Poor Attempt" in res_poor.message


def test_scoreboard_ordering():
    registry = VectorRegistry(embedding_dim=192)
    game = GameEngine(registry)

    registry.register_player("p1", "Player One", create_random_unit_vector(192))
    registry.register_player("p2", "Player Two", create_random_unit_vector(192))
    registry.register_player("p3", "Player Three", create_random_unit_vector(192))

    game.add_score("p1", 30)
    game.add_score("p2", 100)
    game.add_score("p3", 75)

    scoreboard = game.get_scoreboard()
    assert scoreboard[0]["name"] == "Player Two"
    assert scoreboard[0]["score"] == 100
    assert scoreboard[1]["name"] == "Player Three"
    assert scoreboard[1]["score"] == 75
    assert scoreboard[2]["name"] == "Player One"
    assert scoreboard[2]["score"] == 30

    game.reset_scores()
    scoreboard_reset = game.get_scoreboard()
    assert all(p["score"] == 0 for p in scoreboard_reset)


def test_thread_safety_concurrency():
    """Verify thread safety with concurrent reads, writes, and updates."""
    registry = VectorRegistry(embedding_dim=192)
    game = GameEngine(registry)

    # Initial players
    for i in range(10):
        registry.register_player(f"p_{i}", f"User {i}", create_random_unit_vector(192))

    def worker_action(idx: int):
        q = create_random_unit_vector(192)
        # Perform identify
        _ = game.identify_speaker(q)
        # Perform centroid update
        target_pid = f"p_{idx % 10}"
        _ = registry.update_centroid(target_pid, q)
        # Perform impostor evaluation
        _ = game.evaluate_impostor(target_pid, q)
        # Add score
        _ = game.add_score(target_pid, 5)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_action, i) for i in range(100)]
        for f in futures:
            f.result()

    # Verify all centroids remain unit length and scores correctly accumulated
    for p in registry.list_players():
        assert np.linalg.norm(p.centroid) == pytest.approx(1.0, rel=1e-5)
        assert p.sample_count > 1
        assert p.score > 0
