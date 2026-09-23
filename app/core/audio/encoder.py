"""
CAM++ Speaker Verification Feature Extractor and Embedding Generator.
Extracts 80-dim log-mel fbank features and runs CAM++ ONNX model
to output an L2-normalized 192-D speaker embedding.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Try importing kaldi-native-fbank for bit-exact Kaldi features
try:
    import kaldi_native_fbank as knf
    HAS_KNF = True
except ImportError:
    HAS_KNF = False
    logger.warning("kaldi-native-fbank not found. Falling back to NumPy-based fbank extractor.")


def _compute_fbank_knf(audio: np.ndarray, sample_rate: int = 16000, num_mel_bins: int = 80) -> np.ndarray:
    """Compute 80-dimensional fbank features using kaldi-native-fbank."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.dither = 0.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.mel_opts.num_bins = num_mel_bins

    fbank = knf.OnlineFbank(opts)
    # kaldi-native-fbank expects 16-bit PCM amplitude scale [-32768, 32767]
    audio_scaled = audio * 32768.0
    fbank.accept_waveform(sample_rate, audio_scaled.tolist() if isinstance(audio_scaled, np.ndarray) else audio_scaled)
    fbank.input_finished()

    num_frames = fbank.num_frames_ready
    if num_frames == 0:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    frames = np.stack([fbank.get_frame(i) for i in range(num_frames)]).astype(np.float32)
    # Apply Cepstral Mean Normalization (CMN) across time
    frames -= frames.mean(axis=0, keepdims=True)
    return frames


def _compute_fbank_numpy(
    audio: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """NumPy/SciPy fallback for 80-dim log-mel fbank extraction."""
    from scipy.signal import get_window

    frame_length = int(frame_length_ms * sample_rate / 1000.0)  # 400
    frame_shift = int(frame_shift_ms * sample_rate / 1000.0)    # 160
    n_fft = 512

    # Pre-emphasis
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Framing
    num_samples = len(emphasized)
    if num_samples < frame_length:
        emphasized = np.pad(emphasized, (0, frame_length - num_samples))
        num_samples = frame_length

    num_frames = 1 + (num_samples - frame_length) // frame_shift
    if num_frames <= 0:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_length, 1)).T
    )
    frames = emphasized[indices]
    window = get_window("hamming", frame_length, fftbins=False)
    frames = frames * window

    # Magnitude FFT
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # Mel Filterbanks
    low_freq_mel = 0.0
    high_freq_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_mel_bins + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_mel_bins, int(n_fft / 2 + 1)))
    for m in range(1, num_mel_bins + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / max(f_m - f_m_minus, 1)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / max(f_m_plus - f_m, 1)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = np.log(filter_banks).astype(np.float32)

    # CMN
    filter_banks -= filter_banks.mean(axis=0, keepdims=True)
    return filter_banks


def extract_fbank(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Extract 80-dim log-mel fbank features with CMN."""
    if HAS_KNF:
        return _compute_fbank_knf(audio, sample_rate=sample_rate)
    return _compute_fbank_numpy(audio, sample_rate=sample_rate)


class CAMPPEncoder:
    """CAM++ 192-D Speaker Verification Embedding Extractor (ONNX)."""

    EMBEDDING_DIM = 192
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_path: Union[str, Path] = "onnx_models/campplus.onnx",
        num_threads: int = 2,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"CAM++ model not found at {self.model_path}. "
                f"Please run scripts/download_models.py."
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options, providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Initialized CAMPPEncoder with {self.model_path.name}")

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalized 192-D speaker embedding from 16kHz mono audio.

        Args:
            audio: 1D np.ndarray float32 waveform.

        Returns:
            1D np.ndarray of shape (192,), dtype np.float32, with unit norm.
        """
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1D, got shape {audio.shape}")

        # Compute 80-dim log-mel filterbanks
        fbank_features = extract_fbank(audio, sample_rate=self.SAMPLE_RATE)

        # Minimum required frames for pooling (typically >= 10 frames)
        if len(fbank_features) < 10:
            pad_len = 10 - len(fbank_features)
            fbank_features = np.pad(fbank_features, ((0, pad_len), (0, 0)), mode="edge")

        # CAM++ expects shape: (batch_size, num_frames, 80)
        input_tensor = np.expand_dims(fbank_features, axis=0).astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw_embedding = outputs[0][0]  # Shape: (192,)

        # L2 Normalization
        norm = np.linalg.norm(raw_embedding)
        if norm > 1e-12:
            normalized_embedding = raw_embedding / norm
        else:
            normalized_embedding = raw_embedding

        return normalized_embedding.astype(np.float32)
