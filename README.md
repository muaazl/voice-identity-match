# VoiceMimic

VoiceMimic is an interactive, real-time voice biometric party game powered by an optimized, PyTorch-free ONNX pipeline.

## What it does

Players enroll their voice by recording a short audio clip, which the system uses to create a unique biometric profile. In "Blind Identifier" mode, players take turns speaking a hidden phrase, and the system attempts to correctly guess who is speaking. In "Impostor Challenge" mode, players try to fool the biometric security by mimicking another enrolled player's voice.

## How it works

The engine is built for low-latency CPU execution and evaluates voice samples through a 4-stage pipeline:

1. **Noise Reduction:** A causal DTLN model removes background noise and enhances speech.
2. **Voice Activity Detection:** Silero VAD strips silence, pauses, and breath noises.
3. **Feature Extraction:** 80-dimensional log-mel filterbanks are generated from the clean speech.
4. **Speaker Embedding:** A CAM++ model extracts a 192-D unit-normalized biometric vector, which is then scored against the game's registry using vectorized cosine similarity.

## Quick Start

### Prerequisites
- Python 3.10+

### Setup and Run
```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required ONNX models
python scripts/download_models.py

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Open your browser to `http://localhost:7860`.

## Deploy (Render Example)

Deploy easily on Render or any Python cloud host:
- **Environment**: Python 3.10+
- **Build Command**: `pip install -r requirements.txt && python scripts/download_models.py`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Subsystem status and ONNX model readiness |
| `/api/rooms/create` | `POST` | Creates a new isolated game room |
| `/api/rooms/{room_code}/status`| `GET` | Retrieves the status of a specific room |
| `/api/rooms` | `GET` | Lists all currently active rooms |
| `/api/players/register` | `POST` | Registers a new player profile from an audio sample |
| `/api/players` | `GET` | Lists all currently enrolled players |
| `/api/players/{id}` | `DELETE` | Deletes a player from the registry |
| `/api/game/guess-who` | `POST` | Mode A: 1-of-N speaker classification |
| `/api/game/confirm-speaker` | `POST` | Mode A: Awards points and adaptively updates centroid via EMA |
| `/api/game/mimic-challenge` | `POST` | Mode B: Evaluates impersonation vs target centroid |
| `/api/game/scoreboard` | `GET` | Retrieves leaderboard sorted by points |
| `/api/game/reset` | `POST` | Resets scores or completely clears game roster |
| `/ws/stream-vad` | `WebSocket`| Real-time speech activity probability stream |
