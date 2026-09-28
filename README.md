# VoiceGate

A voice authentication API that uses biometric 1:1 or 1:N matching.

## Features

1.  **Enrollment:** Takes a short audio clip and a name, and generates a biometric profile (192-D centroid vector).
2.  **Verification:** Compares a voice sample against enrolled profiles to return a match verdict, confidence score, and ranking.

## Pipeline

The system evaluates audio through a 4-stage CPU-based pipeline:
1.  **Noise Reduction:** DTLN model removes background noise.
2.  **Voice Activity Detection:** Silero VAD strips silence.
3.  **Feature Extraction:** Generates 80-dimensional log-mel filterbanks.
4.  **Speaker Embedding:** CAM++ model extracts a 192-D vector, scored using cosine similarity.

## Setup and Run

Requires Python 3.10+.

```bash
python -m venv .venv
.\venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Open `http://localhost:7860` for the web UI.

## API Endpoints

The API relies on `X-Session-ID` headers to group identities into isolated sessions.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | System health status |
| `/api/sessions/create` | `POST` | Creates a new session |
| `/api/sessions/{session_id}/status`| `GET` | Gets session status |
| `/api/identities/enroll` | `POST` | Enrolls a new identity |
| `/api/identities` | `GET` | Lists enrolled identities |
| `/api/identities/{id}` | `DELETE` | Deletes an identity |
| `/api/verify` | `POST` | Verifies a speaker (1:1 with `identity_id` or 1:N) |
| `/api/audio/embed` | `POST` | Extracts a 192-D embedding from audio statelessly |
| `/ws/stream-vad` | `WebSocket`| VAD stream |
