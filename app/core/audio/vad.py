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

        from .utils import make_session_options
        sess_options = make_session_options(num_threads)

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

    def infer(self, chunk: np.ndarray) -> float:
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
        neg_threshold: Optional[float] = None,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 250,
        pad_ms: int = 150,
    ) -> List[Tuple[int, int]]:
        """
        Detect active speech intervals in samples using stateful hysteresis and hangover.

        Args:
            audio: 1D np.ndarray float32 waveform at 16kHz.
            threshold: Probability threshold to activate speech state (default self.threshold, 0.50).
            neg_threshold: Probability threshold to deactivate speech state (default threshold - 0.15).
            min_speech_duration_ms: Minimum duration of a speech segment in milliseconds.
            min_silence_duration_ms: Silence duration required to untrigger speech state.
            pad_ms: Padding added before and after speech segments.

        Returns:
            List of (start_sample, end_sample) tuples.
        """
        th = threshold or self.threshold
        neg_th = neg_threshold if neg_threshold is not None else max(0.15, th - 0.15)
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
            prob = self.infer(chunk)
            speech_probs.append(prob)

        # Stateful hysteresis: trigger on >= th, untrigger only after min_silence_samples < neg_th
        triggered = False
        speech_segments: List[Tuple[int, int]] = []
        current_speech_start = 0
        temp_end = 0

        for idx, prob in enumerate(speech_probs):
            sample_pos = idx * window_size

            if not triggered:
                if prob >= th:
                    triggered = True
                    current_speech_start = max(0, sample_pos - pad_samples)
                    temp_end = 0
            else:
                if prob < neg_th:
                    if temp_end == 0:
                        temp_end = sample_pos
                    if (sample_pos - temp_end) >= min_silence_samples:
                        speech_end = min(num_samples, temp_end + pad_samples)
                        if (speech_end - current_speech_start) >= min_speech_samples:
                            speech_segments.append((current_speech_start, speech_end))
                        triggered = False
                        temp_end = 0
                else:
                    # Still in speech or brief sub-threshold dip: keep speech alive
                    temp_end = 0

        if triggered:
            speech_end = num_samples if temp_end == 0 else min(num_samples, temp_end + pad_samples)
            if (speech_end - current_speech_start) >= min_speech_samples:
                speech_segments.append((current_speech_start, speech_end))

        if not speech_segments:
            return []

        # Merge overlapping or close segments (closer than min_silence_samples)
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
        crossfade_ms: int = 10,
    ) -> np.ndarray:
        """
        Extract and concatenate active speech chunks from 16kHz audio with Hann crossfade.

        Args:
            audio: 1D np.ndarray float32 audio.
            threshold: Probability threshold.
            fallback_on_silence: If no speech is detected, return original audio.
            crossfade_ms: Duration of smooth edge taper on spliced chunks.

        Returns:
            1D np.ndarray containing trimmed speech audio.
        """
        if len(audio) == 0:
            return audio

        segments = self.get_speech_timestamps(audio, threshold=threshold)
        raw_duration = len(audio) / self.SAMPLE_RATE

        # Energy check: check if audio actually contains voice energy
        audio_rms = float(np.sqrt(np.mean(audio ** 2)))

        if not segments:
            if fallback_on_silence:
                logger.warning("No speech detected by VAD, falling back to full audio.")
                return audio
            return np.zeros(0, dtype=np.float32)

        total_speech_samples = sum(end - start for start, end in segments)
        speech_ratio = total_speech_samples / max(len(audio), 1)

        # Fallback if VAD severely truncated audible audio (<15% speech on audible input)
        if speech_ratio < 0.15 and audio_rms > 0.015 and fallback_on_silence:
            logger.warning(
                f"VAD extracted suspiciously short speech ({total_speech_samples / self.SAMPLE_RATE:.2f}s "
                f"out of {raw_duration:.2f}s, RMS={audio_rms:.3f}). Falling back to full audio."
            )
            return audio

        # Extract segments with Hann window taper at chunk boundaries to eliminate clicks
        fade_samples = int(crossfade_ms * self.SAMPLE_RATE / 1000)
        chunks = []
        for start, end in segments:
            chunk = audio[start:end].copy()
            if len(chunk) > 2 * fade_samples and fade_samples > 0:
                ramp = np.sin(np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)) ** 2
                chunk[:fade_samples] *= ramp
                chunk[-fade_samples:] *= ramp[::-1]
            chunks.append(chunk)

        return np.concatenate(chunks).astype(np.float32)
