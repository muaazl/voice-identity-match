"""
Tests for Multi-Room & Multi-Lobby Session Isolation in VoiceMimic.
Verifies room creation, code generation, per-room data isolation, and API scoping.
"""

import io
import time
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.core.engine.room import RoomManager, GameRoom


def make_wav_bytes(duration_sec: float = 1.5, freq: float = 200.0) -> bytes:
    """Generate a clean synthetic WAV file in-memory."""
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
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


def test_room_manager_lifecycle():
    manager = RoomManager(default_room_code="DEFAULT", embedding_dim=192)

    # 1. Default room is primed
    assert manager.get_room("DEFAULT") is not None

    # 2. Create room with random code
    room1 = manager.create_room()
    assert len(room1.room_code) == 4
    assert room1.room_code.isupper()
    assert manager.get_room(room1.room_code) is not None

    # 3. Create room with custom code
    room2 = manager.create_room("TEST")
    assert room2.room_code == "TEST"

    # 4. Rooms have isolated registries
    dummy_vec1 = np.ones(192, dtype=np.float32)
    dummy_vec2 = np.ones(192, dtype=np.float32) * 2
    room1.registry.register_player("p_1", "Player One", dummy_vec1)
    room2.registry.register_player("p_2", "Player Two", dummy_vec2)

    assert room1.registry.count() == 1
    assert room2.registry.count() == 1
    assert room1.registry.get_player("p_1") is not None
    assert room1.registry.get_player("p_2") is None
    assert room2.registry.get_player("p_2") is not None
    assert room2.registry.get_player("p_1") is None

    # 5. Expiration / Cleanup
    room1.last_active_at = time.time() - 8000
    cleaned = manager.cleanup_expired_rooms(max_idle_seconds=7200)
    assert cleaned >= 1
    assert manager.get_room(room1.room_code) is None
    # Default room is never evicted
    assert manager.get_room("DEFAULT") is not None


def test_api_room_creation_and_status(client):
    # 1. Create a new room via API
    res = client.post("/api/rooms/create")
    assert res.status_code == 200
    data = res.json()
    assert "room_code" in data
    room_code = data["room_code"]
    assert len(room_code) == 4

    # 2. Check room status
    status_res = client.get(f"/api/rooms/{room_code}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["exists"] is True
    assert status_data["room_code"] == room_code
    assert status_data["player_count"] == 0

    # 3. Non-existent room status
    fake_res = client.get("/api/rooms/ZZZZ999/status")
    assert fake_res.status_code == 200
    assert fake_res.json()["exists"] is False


def test_cannot_join_uncreated_room(client):
    # Attempting to interact with an uncreated room must return 404
    uncreated_code = "NOPE"
    res = client.get("/api/players", headers={"X-Room-Code": uncreated_code})
    assert res.status_code == 404
    assert "has not been created yet" in res.json()["detail"]

    # Also via query parameter
    res2 = client.get(f"/api/players?room={uncreated_code}")
    assert res2.status_code == 404


def test_api_cross_room_isolation(client):
    room_a = "GRP1"
    room_b = "GRP2"

    # Explicitly create both rooms first
    create_a = client.post("/api/rooms/create", json={"room_code": room_a})
    assert create_a.status_code == 200
    create_b = client.post("/api/rooms/create", json={"room_code": room_b})
    assert create_b.status_code == 200

    wav_alice = make_wav_bytes(duration_sec=2.0, freq=220.0)
    wav_bob = make_wav_bytes(duration_sec=2.0, freq=140.0)

    # Register Alice in Room A via X-Room-Code header
    res_a = client.post(
        "/api/players/register",
        headers={"X-Room-Code": room_a},
        data={"player_name": "Alice"},
        files={"file": ("alice.wav", wav_alice, "audio/wav")},
    )
    assert res_a.status_code == 200
    assert "GRP1" in res_a.json()["message"]

    # Register Bob in Room B via query param ?room=
    res_b = client.post(
        "/api/players/register?room=GRP2",
        data={"player_name": "Bob"},
        files={"file": ("bob.wav", wav_bob, "audio/wav")},
    )
    assert res_b.status_code == 200
    assert "GRP2" in res_b.json()["message"]

    # Verify Room A only lists Alice
    list_a = client.get("/api/players", headers={"X-Room-Code": room_a})
    assert list_a.status_code == 200
    names_a = [p["name"] for p in list_a.json()["players"]]
    assert names_a == ["Alice"]

    # Verify Room B only lists Bob
    list_b = client.get("/api/players", headers={"X-Room-Code": room_b})
    assert list_b.status_code == 200
    names_b = [p["name"] for p in list_b.json()["players"]]
    assert names_b == ["Bob"]

    # Verify Resetting Room A does NOT clear Room B
    reset_res = client.post(
        "/api/game/reset",
        headers={"X-Room-Code": room_a},
        json={"reset_scores_only": False},
    )
    assert reset_res.status_code == 200

    # Room A is empty
    list_a_after = client.get("/api/players", headers={"X-Room-Code": room_a})
    assert list_a_after.json()["total"] == 0

    # Room B still has Bob intact!
    list_b_after = client.get("/api/players", headers={"X-Room-Code": room_b})
    assert list_b_after.json()["total"] == 1
    assert list_b_after.json()["players"][0]["name"] == "Bob"

