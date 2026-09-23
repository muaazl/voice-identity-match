# VoiceMimic: Real-Time Voice Biometric Party Game

**VoiceMimic** is an interactive, real-time voice biometric party game. The application operates with **strictly zero PyTorch dependencies at runtime**, utilizing an optimized pipeline of DTLN speech enhancement, Silero VAD speech activity detection, and CAM++ acoustic feature extraction executed via `onnxruntime`.

---

## Game Modes & Concept

1. **Player Registration / Lobby**:
   - Players enter their name and record 10 seconds of 16kHz audio.
   - The audio pipeline cleans background noise, extracts active speech, and generates a baseline 192-D unit-normalized speaker centroid $c_i \in \mathbb{R}^{192}$.
2. **Mode A (Blind Identifier / "Guess Who")**:
   - A player speaks a hidden phrase.
   - The engine performs vectorized 1-of-N cosine similarity classification across all enrolled speaker centroids.
   - If the model guesses wrong, the true speaker claims the round, gains points, and their voice centroid is adaptively updated via **Exponential Moving Average (EMA)**:
     $$c_i^{(t)} = \text{Normalize}(\alpha c_i^{(t-1)} + (1 - \alpha) x)$$
3. **Mode B (Impostor Challenge)**:
   - Player $X$ attempts to mimic Player $Y$.
   - The engine evaluates acoustic proximity against Player $Y$'s centroid across 3 security tiers:
     - **System Fooled!** ($s \ge 0.82$): 100 points (Biometric security breached).
     - **Close Mimic!** ($0.60 \le s < 0.82$): 1–99 partial points scaled by proximity.
     - **Poor Attempt** ($s < 0.60$): 0 points.

---

## Audio Pipeline & Architecture

```
Raw Mic Input (16kHz Mono PCM/WAV)
       │
       ▼
1. DTLN ONNX Denoiser (512-sample frame / 128-sample hop STFT masking)
       │
       ▼
2. Silero VAD ONNX (Removes room silence, pauses, and breath noises)
       │
       ▼
3. 80-dim Log-Mel Filterbank Extractor (kaldi-native-fbank + CMN)
       │
       ▼
4. CAM++ ONNX Encoder (192-D L2-Normalized Speaker Embedding)
       │
       ▼
5. Vectorized Engine (BLAS GEMV 1-of-N Cosine Matching + EMA Updates)
```

---

## Local Development & Testing

### 1. Prerequisites
- Python 3.10+ (managed via `uv` or `venv`)
- Pre-downloaded ONNX checkpoints in `onnx_models/`

### 2. Setup
```bash
# Create virtual environment and install dependencies
uv venv .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe

# Fetch ONNX model checkpoints
python scripts/download_models.py
```

### 3. Run Unit & Integration Tests
```bash
python -m pytest tests/ -v
```

### 4. Run Locally
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Open your browser at `http://localhost:7860`.

---

## Docker Deployment (Hugging Face Spaces)

The application includes a production-ready, self-contained `Dockerfile` configured for non-root execution (UID 1000) with models pre-baked into the image layer:

### Build Image
```bash
docker build -t voice-mimic .
```

### Run Container
```bash
docker run -p 7860:7860 voice-mimic
```

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Minimalist Two-Tone Pastel Single-Page Application |
| `/api/health` | `GET` | Subsystem status and ONNX model readiness |
| `/api/players/register` | `POST` | Ingests WAV sample and registers new player profile |
| `/api/players` | `GET` | Lists all currently enrolled players |
| `/api/players/{id}` | `DELETE` | Deletes a player from the registry |
| `/api/game/guess-who` | `POST` | Mode A: 1-of-N speaker classification |
| `/api/game/confirm-speaker` | `POST` | Mode A: Awards points and adaptively updates centroid via EMA |
| `/api/game/mimic-challenge` | `POST` | Mode B: Evaluates impersonation vs target centroid |
| `/api/game/scoreboard` | `GET` | Retrieves leaderboard sorted by points |
| `/api/game/reset` | `POST` | Resets scores or completely clears game roster |
| `/ws/stream-vad` | `WebSocket` | Real-time speech activity probability stream |

---

## License
Apache 2.0 / MIT.
