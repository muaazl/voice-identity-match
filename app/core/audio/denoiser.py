"""
DTLN (Dual-Signal Transformation LSTM Network) Speech Denoiser.
Runs 2-stage speech enhancement using ONNX Runtime (CPUExecutionProvider).
Reference: Nils L. Westhausen (DNS-Challenge).
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class DTLNDenoiser:
    """Real-time causal speech denoiser powered by DTLN ONNX models."""

    BLOCK_LEN = 512
    BLOCK_SHIFT = 128
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_1_path: Union[str, Path] = "onnx_models/dtln_model_1.onnx",
        model_2_path: Union[str, Path] = "onnx_models/dtln_model_2.onnx",
        num_threads: int = 2,
    ):
        self.model_1_path = Path(model_1_path)
        self.model_2_path = Path(model_2_path)

        if not self.model_1_path.exists() or not self.model_2_path.exists():
            raise FileNotFoundError(
                f"DTLN ONNX models not found at {self.model_1_path} or {self.model_2_path}. "
                f"Please run scripts/download_models.py."
            )

        from .utils import make_session_options
        sess_options = make_session_options(num_threads)

        self.session_1 = ort.InferenceSession(
            str(self.model_1_path), sess_options, providers=["CPUExecutionProvider"]
        )
        self.session_2 = ort.InferenceSession(
            str(self.model_2_path), sess_options, providers=["CPUExecutionProvider"]
        )

        # Inspect input tensor details
        self.input_names_1 = [inp.name for inp in self.session_1.get_inputs()]
        self.input_shapes_1 = [inp.shape for inp in self.session_1.get_inputs()]
        self.output_names_1 = [out.name for out in self.session_1.get_outputs()]

        self.input_names_2 = [inp.name for inp in self.session_2.get_inputs()]
        self.input_shapes_2 = [inp.shape for inp in self.session_2.get_inputs()]
        self.output_names_2 = [out.name for out in self.session_2.get_outputs()]

        logger.info(
            f"Initialized DTLNDenoiser with {self.model_1_path.name} and {self.model_2_path.name}"
        )

    def _init_states(self):
        """Allocate zero states matching the model signature."""
        states_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape],
                dtype=np.float32,
            )
            for inp in self.session_1.get_inputs()
            if inp.name != self.input_names_1[0]
        }
        states_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape],
                dtype=np.float32,
            )
            for inp in self.session_2.get_inputs()
            if inp.name != self.input_names_2[0]
        }
        return states_1, states_2

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance and denoise 16kHz mono audio waveform.

        Args:
            audio: 1D np.ndarray of float32 samples in range [-1.0, 1.0].

        Returns:
            1D np.ndarray containing the cleaned audio signal, matching input length.
        """
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1D mono, got shape {audio.shape}")

        orig_len = len(audio)
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32)

        # Pad if audio length is smaller than one block
        min_len = self.BLOCK_LEN
        if orig_len < min_len:
            padded_audio = np.pad(audio, (0, min_len - orig_len))
        else:
            padded_audio = audio

        # Calculate number of blocks
        num_blocks = (len(padded_audio) - (self.BLOCK_LEN - self.BLOCK_SHIFT)) // self.BLOCK_SHIFT
        if num_blocks <= 0:
            return audio.copy()

        # Buffers for overlap-add
        in_buffer = np.zeros(self.BLOCK_LEN, dtype=np.float32)
        out_buffer = np.zeros(self.BLOCK_LEN, dtype=np.float32)
        out_audio = np.zeros(num_blocks * self.BLOCK_SHIFT, dtype=np.float32)

        states_1, states_2 = self._init_states()

        for idx in range(num_blocks):
            # Shift in_buffer and insert new chunk
            in_buffer[:-self.BLOCK_SHIFT] = in_buffer[self.BLOCK_SHIFT:]
            in_buffer[-self.BLOCK_SHIFT:] = padded_audio[
                idx * self.BLOCK_SHIFT : (idx + 1) * self.BLOCK_SHIFT
            ]

            # 1. FFT
            fft_block = np.fft.rfft(in_buffer)
            mag = np.abs(fft_block).astype(np.float32)
            phase = np.angle(fft_block).astype(np.float32)

            # Reshape magnitude: (1, 1, 257)
            mag_in = np.reshape(mag, (1, 1, -1))

            # Run Stage 1
            feed_1 = {self.input_names_1[0]: mag_in, **states_1}
            out_1 = self.session_1.run(None, feed_1)
            out_mask = out_1[0]
            # Update stage 1 states
            for i, name in enumerate(self.input_names_1[1:], start=1):
                states_1[name] = out_1[i]

            # Inverse FFT with mask
            est_complex = mag_in * out_mask * np.exp(1j * phase)
            est_block = np.fft.irfft(est_complex)
            est_block = np.reshape(est_block, (1, 1, -1)).astype(np.float32)

            # Run Stage 2
            feed_2 = {self.input_names_2[0]: est_block, **states_2}
            out_2 = self.session_2.run(None, feed_2)
            out_frame = out_2[0]
            # Update stage 2 states
            for i, name in enumerate(self.input_names_2[1:], start=1):
                states_2[name] = out_2[i]

            # Overlap-add
            out_buffer[:-self.BLOCK_SHIFT] = out_buffer[self.BLOCK_SHIFT:]
            out_buffer[-self.BLOCK_SHIFT:] = 0.0
            out_buffer += np.squeeze(out_frame)

            out_audio[idx * self.BLOCK_SHIFT : (idx + 1) * self.BLOCK_SHIFT] = out_buffer[:self.BLOCK_SHIFT]

        # Truncate or pad back to original length
        if len(out_audio) >= orig_len:
            return out_audio[:orig_len].astype(np.float32)
        else:
            return np.pad(out_audio, (0, orig_len - len(out_audio))).astype(np.float32)
