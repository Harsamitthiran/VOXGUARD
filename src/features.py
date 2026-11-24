"""Feature extraction and data augmentation module for VOXGUARD.

Computes Log-Mel Spectrograms, MFCCs (+ deltas), Spectral Contrast,
Spectral Flatness, and Zero-Crossing Rate using Librosa, formatted for
CNN inference and diagnostic analysis.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import torch


# Fixed parameter defaults for 16 kHz audio windowed at 2.0 seconds
DEFAULT_SR = 16000
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 512
TARGET_FRAMES = 64  # Standardized time-step length for CNN input


def extract_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """Extract normalized Log-Mel Spectrogram from audio window.

    Args:
        y: 1D audio array (typically 2.0s window).
        sr: Sample rate in Hz.
        n_mels: Number of Mel frequency bands.
        n_fft: FFT window size.
        hop_length: FFT hop length.
        target_frames: Fixed number of time frames for CNN consistency.

    Returns:
        2D array of shape (n_mels, target_frames), normalized to [0, 1].
    """
    if len(y) == 0:
        return np.zeros((n_mels, target_frames), dtype=np.float32)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=20,
        fmax=sr // 2,
        power=2.0,
    )
    # Convert power to decibels
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Normalize from [-80, 0] dB to [0, 1] range
    norm_mel = (log_mel + 80.0) / 80.0
    norm_mel = np.clip(norm_mel, 0.0, 1.0)

    # Ensure fixed frame width via padding or truncating
    cur_frames = norm_mel.shape[1]
    if cur_frames < target_frames:
        pad_width = target_frames - cur_frames
        norm_mel = np.pad(norm_mel, ((0, 0), (0, pad_width)), mode="edge")
    elif cur_frames > target_frames:
        norm_mel = norm_mel[:, :target_frames]

    return norm_mel.astype(np.float32)


def extract_mfcc(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = 20,
) -> Dict[str, np.ndarray]:
    """Extract MFCCs along with delta and delta-delta features.

    Returns:
        Dictionary with keys 'mfcc', 'delta', 'delta2'.
    """
    if len(y) == 0:
        empty = np.zeros((n_mfcc, 1), dtype=np.float32)
        return {"mfcc": empty, "delta": empty, "delta2": empty}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return {
        "mfcc": mfcc.astype(np.float32),
        "delta": delta.astype(np.float32),
        "delta2": delta2.astype(np.float32),
    }


def extract_spectral_contrast(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_bands: int = 6,
) -> np.ndarray:
    """Extract spectral contrast across frequency sub-bands."""
    if len(y) == 0:
        return np.zeros((n_bands + 1, 1), dtype=np.float32)
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=n_bands, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH
    )
    return contrast.astype(np.float32)


def extract_spectral_flatness(y: np.ndarray) -> np.ndarray:
    """Extract spectral flatness (noisiness vs tonality)."""
    if len(y) == 0:
        return np.zeros((1, 1), dtype=np.float32)
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH
    )
    return flatness.astype(np.float32)


def extract_zero_crossing_rate(y: np.ndarray) -> np.ndarray:
    """Extract zero-crossing rate."""
    if len(y) == 0:
        return np.zeros((1, 1), dtype=np.float32)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=DEFAULT_HOP_LENGTH)
    return zcr.astype(np.float32)


def extract_acoustic_summary(y: np.ndarray, sr: int = DEFAULT_SR) -> Dict[str, Any]:
    """Compute summary acoustic metrics useful for diagnostic explainability.

    Returns:
        Dict with mean and std of MFCCs, Spectral Contrast, Spectral Flatness, and ZCR.
    """
    mfcc_dict = extract_mfcc(y, sr=sr)
    contrast = extract_spectral_contrast(y, sr=sr)
    flatness = extract_spectral_flatness(y)
    zcr = extract_zero_crossing_rate(y)

    return {
        "mfcc_mean": float(np.mean(mfcc_dict["mfcc"])),
        "mfcc_std": float(np.std(mfcc_dict["mfcc"])),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "energy_rms": float(np.sqrt(np.mean(y**2))) if len(y) > 0 else 0.0,
    }


def window_to_tensor(
    y_window: np.ndarray,
    sr: int = DEFAULT_SR,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert a single audio window into a PyTorch CNN input tensor (1, 1, 128, 64)."""
    spec = extract_log_mel_spectrogram(y_window, sr=sr)
    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mels, frames)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ==========================================
# Audio Data Augmentation for Robust Training
# ==========================================

def add_gaussian_noise(y: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
    """Add Gaussian noise at a specified or randomized Signal-to-Noise Ratio (SNR)."""
    if len(y) == 0:
        return y
    signal_power = np.mean(y**2)
    if signal_power < 1e-9:
        return y

    if snr_db is None:
        snr_db = random.uniform(15.0, 35.0)

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape).astype(np.float32)
    return y + noise


def time_shift(y: np.ndarray, max_shift_ratio: float = 0.2) -> np.ndarray:
    """Shift audio waveform in time randomly."""
    if len(y) == 0:
        return y
    shift = int(len(y) * random.uniform(-max_shift_ratio, max_shift_ratio))
    return np.roll(y, shift)


def pitch_shift(y: np.ndarray, sr: int = DEFAULT_SR, n_steps: Optional[float] = None) -> np.ndarray:
    """Apply slight pitch shift."""
    if len(y) == 0:
        return y
    if n_steps is None:
        n_steps = random.uniform(-2.0, 2.0)
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except Exception:
        return y


def apply_augmentation(y: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray:
    """Apply random combination of data augmentations for training."""
    augmented = y.copy()
    if random.random() < 0.5:
        augmented = add_gaussian_noise(augmented)
    if random.random() < 0.4:
        augmented = time_shift(augmented)
    if random.random() < 0.3:
        augmented = pitch_shift(augmented, sr=sr)
    return augmented
