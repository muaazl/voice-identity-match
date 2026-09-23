"""
Integration tests for VoiceMimic FastAPI Service and Endpoints (Phase 3).
Tests player registration, Mode A (Guess-Who & Confirm), Mode B (Mimic Challenge),
and Scoreboard management using FastAPI TestClient.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def make_wav_bytes(duration_sec: float = 1.5, freq: float = 200.0) -> bytes:
    """Generate a clean synthetic WAV file in-memory."""
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Voiced harmonic signal
    audio = (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
        + 0.1 * np.sin(2 * np.pi * (freq * 3) * t)
    ).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """Create TestClient with lifespan context active."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["embedding_dim"] == 192


def test_player_registration(client):
    # Reset game state before test
    client.post("/api/game/reset", json={"reset_scores_only": False})

    wav_alice = make_wav_bytes(duration_sec=2.0, freq=220.0)
    response = client.post(
        "/api/players/register",
        data={"player_name": "Alice"},
        files={"file": ("alice.wav", wav_alice, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice"
    assert data["sample_count"] == 1
    assert data["score"] == 0
    assert "p_" in data["player_id"]
    alice_id = data["player_id"]

    # Register second player Bob
    wav_bob = make_wav_bytes(duration_sec=2.0, freq=130.0)
    res_bob = client.post(
        "/api/players/register",
        data={"player_name": "Bob"},
        files={"file": ("bob.wav", wav_bob, "audio/wav")},
    )
    assert res_bob.status_code == 200

    # List players
    list_res = client.get("/api/players")
    assert list_res.status_code == 200
    players_data = list_res.json()
    assert players_data["total"] == 2


def test_mode_a_guess_who_and_confirm(client):
    # Use Alice's voice frequency
    wav_query = make_wav_bytes(duration_sec=2.0, freq=220.0)

    # 1. Run guess-who
    gw_res = client.post(
        "/api/game/guess-who",
        data={"threshold": 0.50},
        files={"file": ("query.wav", wav_query, "audio/wav")},
    )
    assert gw_res.status_code == 200
    gw_data = gw_res.json()
    assert "round_id" in gw_data
    assert gw_data["predicted_player_id"] is not None
    assert len(gw_data["rankings"]) == 2
    round_id = gw_data["round_id"]
    winner_id = gw_data["predicted_player_id"]

    # 2. Confirm speaker
    conf_res = client.post(
        "/api/game/confirm-speaker",
        json={"actual_player_id": winner_id, "round_id": round_id, "points": 50},
    )
    assert conf_res.status_code == 200
    conf_data = conf_res.json()
    assert conf_data["player_id"] == winner_id
    assert conf_data["points_awarded"] == 50
    assert conf_data["total_score"] == 50
    assert conf_data["sample_count"] == 2  # Incremented from EMA update


def test_mode_b_mimic_challenge(client):
    # Get players
    list_res = client.get("/api/players")
    players = list_res.json()["players"]
    target_id = players[0]["player_id"]

    wav_mimic = make_wav_bytes(duration_sec=1.5, freq=210.0)
    mimic_res = client.post(
        "/api/game/mimic-challenge",
        data={
            "target_player_id": target_id,
            "mimic_threshold": 0.50,
            "match_threshold": 0.85,
        },
        files={"file": ("mimic.wav", wav_mimic, "audio/wav")},
    )
    assert mimic_res.status_code == 200
    mimic_data = mimic_res.json()
    assert mimic_data["target_player_id"] == target_id
    assert mimic_data["status"] in ["SYSTEM_FOOLED", "CLOSE_MIMIC", "POOR_ATTEMPT"]
    assert "similarity_score" in mimic_data
    assert "points" in mimic_data


def test_scoreboard_and_reset(client):
    sb_res = client.get("/api/game/scoreboard")
    assert sb_res.status_code == 200
    sb_data = sb_res.json()
    assert sb_data["total_players"] == 2
    assert len(sb_data["players"]) == 2

    # Reset scores only
    reset_scores = client.post("/api/game/reset", json={"reset_scores_only": True})
    assert reset_scores.status_code == 200
    sb_after = client.get("/api/game/scoreboard").json()
    assert all(p["score"] == 0 for p in sb_after["players"])
    assert sb_after["total_players"] == 2

    # Complete reset
    reset_all = client.post("/api/game/reset", json={"reset_scores_only": False})
    assert reset_all.status_code == 200
    sb_final = client.get("/api/game/scoreboard").json()
    assert sb_final["total_players"] == 0


def test_frontend_static_serving(client):
    # Test index.html served at root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "VoiceMimic" in res_root.text
    assert "<canvas" in res_root.text

    # Test static assets
    assert client.get("/static/css/theme.css").status_code == 200
    assert client.get("/static/js/audio_recorder.js").status_code == 200
    assert client.get("/static/js/api_client.js").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200

