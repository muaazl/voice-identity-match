"""
Unit tests for VoiceMimic Audio Processing and Feature Extraction Pipeline.
Validates DTLN Denoiser, Silero VAD, and CAM++ Speaker Encoder.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from app.core.audio.denoiser import DTLNDenoiser
from app.core.audio.vad import SileroVAD
from app.core.audio.encoder import CAMPPEncoder
from app.core.audio.pipeline import AudioPipeline

MODELS_DIR = Path("onnx_models")


@pytest.fixture(scope="module")
def synthetic_speech_audio() -> np.ndarray:
    """
    Generate 2.5 seconds of synthetic speech-like harmonic signal
    (fundamental 150 Hz + harmonics) mixed with Gaussian background noise.
    """
    sample_rate = 16000
    duration_sec = 2.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    # Harmonics simulating voiced formant structure
    signal = (
        0.5 * np.sin(2 * np.pi * 150 * t)
        + 0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.2 * np.sin(2 * np.pi * 600 * t)
        + 0.1 * np.sin(2 * np.pi * 1200 * t)
    )

    # Add Gaussian noise (SNR ~ 15 dB)
    noise = np.random.normal(0, 0.05, len(signal))
    noisy_signal = signal + noise

    # Normalize to [-0.95, 0.95]
    noisy_signal = 0.95 * noisy_signal / np.max(np.abs(noisy_signal))
    return noisy_signal.astype(np.float32)


@pytest.fixture(scope="module")
def audio_pipeline() -> AudioPipeline:
    assert MODELS_DIR.exists(), "onnx_models directory must exist"
    return AudioPipeline(models_dir=MODELS_DIR, num_threads=2)


def test_models_exist():
    required_models = [
        "dtln_model_1.onnx",
        "dtln_model_2.onnx",
        "silero_vad.onnx",
        "campplus.onnx",
    ]
    for m in required_models:
        model_path = MODELS_DIR / m
        assert model_path.exists(), f"Missing model checkpoint: {model_path}"
        assert model_path.stat().st_size > 10000, f"Model file is too small/empty: {model_path}"


def test_dtln_denoiser(synthetic_speech_audio):
    denoiser = DTLNDenoiser(
        model_1_path=MODELS_DIR / "dtln_model_1.onnx",
        model_2_path=MODELS_DIR / "dtln_model_2.onnx",
    )
    cleaned = denoiser.process(synthetic_speech_audio)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == synthetic_speech_audio.shape
    assert cleaned.dtype == np.float32
    assert np.isfinite(cleaned).all()


def test_silero_vad(synthetic_speech_audio):
    vad = SileroVAD(model_path=MODELS_DIR / "silero_vad.onnx")
    speech = vad.extract_speech(synthetic_speech_audio)
    assert isinstance(speech, np.ndarray)
    assert len(speech) > 0
    assert speech.dtype == np.float32
    assert np.isfinite(speech).all()


def test_campp_encoder(synthetic_speech_audio):
    encoder = CAMPPEncoder(model_path=MODELS_DIR / "campplus.onnx")
    embedding = encoder.extract_embedding(synthetic_speech_audio)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (192,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    # Check L2 unit norm
    norm = np.linalg.norm(embedding)
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_full_pipeline_numpy_input(audio_pipeline, synthetic_speech_audio):
    embedding = audio_pipeline.process(synthetic_speech_audio)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape == (192,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1.0, rel=1e-4)


def test_full_pipeline_wav_bytes_input(audio_pipeline, synthetic_speech_audio):
    # Encode synthetic audio to WAV in-memory bytes
    buf = io.BytesIO()
    sf.write(buf, synthetic_speech_audio, 16000, format="WAV")
    wav_bytes = buf.getvalue()

    embedding = audio_pipeline.process(wav_bytes)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (192,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, rel=1e-4)


def test_pipeline_with_metadata(audio_pipeline, synthetic_speech_audio):
    meta = audio_pipeline.process_with_metadata(synthetic_speech_audio)
    assert "embedding" in meta
    assert meta["embedding"].shape == (192,)
    assert meta["embedding_dim"] == 192
    assert meta["raw_duration_sec"] > 2.0
    assert meta["speech_duration_sec"] > 0.0
    assert meta["l2_norm"] == pytest.approx(1.0, rel=1e-4)
