"""
Unified Audio Processing and Biometric Feature Extraction Pipeline.
Sequences: Raw Audio (16kHz PCM/WAV) -> DTLN Denoiser -> Silero VAD -> CAM++ 192-D Embedding.
Strictly CPU-optimized, zero-PyTorch at runtime.
"""

import io
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .denoiser import DTLNDenoiser
from .vad import SileroVAD
from .encoder import CamPPEncoder

logger = logging.getLogger(__name__)


class AudioPipeline:
    """End-to-end audio processing and speaker feature extraction pipeline."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        models_dir: Union[str, Path] = "onnx_models",
        num_threads: int = 2,
        vad_threshold: float = 0.5,
    ):
        models_path = Path(models_dir)
        self.denoiser = DTLNDenoiser(
            model_1_path=models_path / "dtln_model_1.onnx",
            model_2_path=models_path / "dtln_model_2.onnx",
            num_threads=num_threads,
        )
        self.vad = SileroVAD(
            model_path=models_path / "silero_vad.onnx",
            threshold=vad_threshold,
            num_threads=num_threads,
        )
        self.encoder = CamPPEncoder(
            model_path=models_path / "campplus.onnx",
            num_threads=num_threads,
        )
        logger.info("AudioPipeline successfully initialized with all ONNX components.")

    def load_audio(self, audio_input: Union[np.ndarray, bytes, str, Path]) -> np.ndarray:
        """
        Normalize and decode arbitrary audio inputs into 16kHz float32 mono waveform.

        Args:
            audio_input: Numpy array, raw bytes (WAV/FLAC/OGG/PCM), or file path.

        Returns:
            1D np.ndarray float32 normalized to [-1.0, 1.0] at 16000 Hz.
        """
        if isinstance(audio_input, (str, Path)):
            audio, sr = sf.read(str(audio_input), dtype="float32")
        elif isinstance(audio_input, bytes):
            # Try reading as encoded audio container (WAV, etc.)
            try:
                audio, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
            except Exception:
                # Fallback: Assume raw 16-bit PCM mono 16kHz
                audio = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32) / 32768.0
                sr = self.SAMPLE_RATE
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
            sr = self.SAMPLE_RATE
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        # Convert multi-channel to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if necessary
        if sr != self.SAMPLE_RATE:
            # Greatest Common Divisor resampling
            from math import gcd
            g = gcd(int(sr), self.SAMPLE_RATE)
            up = self.SAMPLE_RATE // g
            down = int(sr) // g
            audio = resample_poly(audio, up, down).astype(np.float32)

        # Remove DC offset & calibrate nominal signal peak
        audio = audio - np.mean(audio)
        max_val = float(np.max(np.abs(audio)))
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val > 1e-4:
            # Calibrate nominal peak level so quiet microphones are brought up to optimal range (~0.92)
            audio = audio * (0.92 / max_val)

        return audio.astype(np.float32)

    def extract_robust_embedding(
        self, speech_audio: np.ndarray, window_sec: float = 3.0, hop_sec: float = 1.5
    ) -> np.ndarray:
        """
        Extract multi-window ensemble embedding from speech audio.
        Combines the full-utterance embedding with overlapping sub-segment embeddings
        to construct a more invariant, noise-resilient speaker centroid.
        """
        win_samples = int(window_sec * self.SAMPLE_RATE)
        hop_samples = int(hop_sec * self.SAMPLE_RATE)

        if len(speech_audio) < int(3.5 * self.SAMPLE_RATE):
            return self.encoder.extract_embedding(speech_audio)

        embeddings = []
        for start in range(0, len(speech_audio) - win_samples + 1, hop_samples):
            chunk = speech_audio[start : start + win_samples]
            embeddings.append(self.encoder.extract_embedding(chunk))

        if not embeddings:
            return self.encoder.extract_embedding(speech_audio)

        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-12:
            return (avg_emb / norm).astype(np.float32)
        return avg_emb.astype(np.float32)

    def process(self, audio_input: Union[np.ndarray, bytes, str, Path]) -> np.ndarray:
        """
        Full pipeline execution returning normalized 192-D speaker embedding.

        Args:
            audio_input: Input audio (waveform, bytes, or path).

        Returns:
            1D np.ndarray of shape (192,), dtype float32, with unit norm.
        """
        raw_audio = self.load_audio(audio_input)

        if len(raw_audio) < 160:  # Less than 10ms
            raise ValueError(f"Input audio too short ({len(raw_audio)} samples). Minimum 160 samples required.")

        # Stage 1: DTLN Speech Denoising & Enhancement
        cleaned_audio = self.denoiser.process(raw_audio)

        # Stage 2: Silero VAD Active Speech Trimming
        speech_audio = self.vad.extract_speech(cleaned_audio, fallback_on_silence=True)

        # Stage 3: CAM++ Feature Extraction with Robust Multi-Window Ensemble
        embedding = self.extract_robust_embedding(speech_audio)

        return embedding

    def process_with_metadata(
        self, audio_input: Union[np.ndarray, bytes, str, Path]
    ) -> Dict[str, Any]:
        """
        Execute pipeline and return detailed stage metrics alongside the embedding.
        """
        raw_audio = self.load_audio(audio_input)
        raw_duration_sec = len(raw_audio) / self.SAMPLE_RATE

        cleaned_audio = self.denoiser.process(raw_audio)
        speech_audio = self.vad.extract_speech(cleaned_audio, fallback_on_silence=True)
        speech_duration_sec = len(speech_audio) / self.SAMPLE_RATE

        embedding = self.extract_robust_embedding(speech_audio)

        return {
            "embedding": embedding,
            "raw_duration_sec": round(raw_duration_sec, 3),
            "speech_duration_sec": round(speech_duration_sec, 3),
            "speech_ratio": round(speech_duration_sec / max(raw_duration_sec, 1e-6), 3),
            "embedding_dim": len(embedding),
            "l2_norm": float(np.linalg.norm(embedding)),
        }
