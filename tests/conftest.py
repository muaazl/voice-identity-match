import pytest
from app.core.engine.registry import VectorRegistry
from app.core.engine.game import GameEngine
from app.core.engine.room import GameRoom, RoomManager

@pytest.fixture
def mock_pipeline(mocker):
    pipeline = mocker.MagicMock()
    pipeline.process.return_value = [0.0] * 192
    pipeline.process_with_metadata.return_value = {
        "embedding": [0.0] * 192,
        "raw_duration_sec": 3.0,
        "speech_duration_sec": 2.5,
        "speech_ratio": 0.83,
        "embedding_dim": 192,
        "l2_norm": 1.0,
    }
    return pipeline

@pytest.fixture
def registry():
    return VectorRegistry(embedding_dim=192)

@pytest.fixture
def game(registry):
    return GameEngine(registry)

@pytest.fixture
def room_manager():
    return RoomManager(default_room_code="DEFAULT", embedding_dim=192)

@pytest.fixture
def default_room(room_manager):
    return room_manager.get_or_create_room("DEFAULT")
