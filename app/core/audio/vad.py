"""
Silero VAD (Voice Activity Detection) ONNX wrapper.
Extracts active speech frames and strips non-speech / silence at 16kHz.
Supports both Silero v4 (h, c states) and v5 (single state) ONNX graphs dynamically.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class SileroVAD:
    """Voice Activity Detector wrapping Silero VAD ONNX."""

    WINDOW_SIZE = 512  # 32ms at 16kHz
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_path: Union[str, Path] = "onnx_models/silero_vad.onnx",
        threshold: float = 0.5,
        num_threads: int = 2,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {self.model_path}. "
                f"Please run scripts/download_models.py."
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options, providers=["CPUExecutionProvider"]
        )

        # Inspect model inputs to dynamically handle v4 vs v5
        input_names = [inp.name for inp in self.session.get_inputs()]
        self.input_names = input_names
        self.is_v5 = "state" in input_names

        logger.info(f"Initialized SileroVAD ({'v5' if self.is_v5 else 'v4'}) with {self.model_path.name}")
        self.reset_states()

    def reset_states(self, batch_size: int = 1):
        """Reset recurrent LSTM states."""
        if self.is_v5:
            self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, batch_size, 64), dtype=np.float32)
            self._c = np.zeros((2, batch_size, 64), dtype=np.float32)

    def _infer_frame(self, chunk: np.ndarray) -> float:
        """Run single 512-sample frame through ONNX VAD."""
        # Ensure chunk has shape (1, 512)
        chunk_in = chunk.reshape(1, -1).astype(np.float32)

        if self.is_v5:
            inputs = {
                "input": chunk_in,
                "state": self._state,
                "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
            }
            out, new_state = self.session.run(None, inputs)
            self._state = new_state
            return float(out[0][0])
        else:
            inputs = {
                "input": chunk_in,
                "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
                "h": self._h,
                "c": self._c,
            }
            out, new_h, new_c = self.session.run(None, inputs)
            self._h = new_h
            self._c = new_c
            return float(out[0][0])

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        threshold: Optional[float] = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        pad_ms: int = 100,
    ) -> List[Tuple[int, int]]:
        """
        Detect active speech intervals in samples.

        Returns:
            List of (start_sample, end_sample) tuples.
        """
        th = threshold or self.threshold
        self.reset_states()

        window_size = self.WINDOW_SIZE
        num_samples = len(audio)
        if num_samples < window_size:
            return [(0, num_samples)]

        pad_samples = int(pad_ms * self.SAMPLE_RATE / 1000)
        min_speech_samples = int(min_speech_duration_ms * self.SAMPLE_RATE / 1000)
        min_silence_samples = int(min_silence_duration_ms * self.SAMPLE_RATE / 1000)

        speech_probs = []
        for i in range(0, num_samples, window_size):
            chunk = audio[i : i + window_size]
            if len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)))
            prob = self._infer_frame(chunk)
            speech_probs.append(prob)

        # Convert chunk-level probabilities to speech segments
        triggered = False
        speech_segments: List[Tuple[int, int]] = []
        current_speech_start = 0

        for idx, prob in enumerate(speech_probs):
            sample_pos = idx * window_size

            if prob >= th and not triggered:
                triggered = True
                current_speech_start = max(0, sample_pos - pad_samples)

            elif prob < th and triggered:
                # Check how long silence has lasted
                triggered = False
                speech_end = min(num_samples, sample_pos + window_size + pad_samples)
                if (speech_end - current_speech_start) >= min_speech_samples:
                    speech_segments.append((current_speech_start, speech_end))

        if triggered:
            speech_segments.append((current_speech_start, num_samples))

        # Merge overlapping segments
        if not speech_segments:
            return []

        merged: List[Tuple[int, int]] = [speech_segments[0]]
        for start, end in speech_segments[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + min_silence_samples:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged

    def extract_speech(
        self,
        audio: np.ndarray,
        threshold: Optional[float] = None,
        fallback_on_silence: bool = True,
    ) -> np.ndarray:
        """
        Extract and concatenate active speech chunks from 16kHz audio.

        Args:
            audio: 1D np.ndarray float32 audio.
            threshold: Probability threshold.
            fallback_on_silence: If no speech is detected, return original audio rather than empty.

        Returns:
            1D np.ndarray containing trimmed speech audio.
        """
        if len(audio) == 0:
            return audio

        segments = self.get_speech_timestamps(audio, threshold=threshold)
        if not segments:
            if fallback_on_silence:
                logger.warning("No speech detected by VAD, falling back to full audio.")
                return audio
            return np.zeros(0, dtype=np.float32)

        speech_chunks = [audio[start:end] for start, end in segments]
        return np.concatenate(speech_chunks).astype(np.float32)
