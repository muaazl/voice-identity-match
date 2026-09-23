"""
WebSocket real-time streaming endpoint for live voice activity monitoring.
Streams 16kHz PCM chunks and returns instantaneous speech probability.
"""

import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/stream-vad")
async def websocket_vad_stream(websocket: WebSocket):
    """
    Live real-time VAD streaming socket.
    Receives 512-sample PCM chunks (either 16-bit int16 or 32-bit float32 bytes),
    evaluates speech probability, and returns live JSON feedback.
    """
    await websocket.accept()
    vad = websocket.app.state.pipeline.vad
    vad.reset_states()

    logger.info("Client connected to /ws/stream-vad")
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Convert bytes to float32 array
            if len(data) == 1024:  # 512 samples * 2 bytes (int16)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif len(data) == 2048:  # 512 samples * 4 bytes (float32)
                samples = np.frombuffer(data, dtype=np.float32)
            else:
                # Handle arbitrary length: truncate/pad to 512
                raw = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(raw) < 512:
                    samples = np.pad(raw, (0, 512 - len(raw)))
                else:
                    samples = raw[:512]

            prob = vad._infer_frame(samples)
            is_speech = prob >= vad.threshold

            await websocket.send_json({
                "prob": round(prob, 4),
                "is_speech": is_speech,
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/stream-vad")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}", exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass
