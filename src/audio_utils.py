"""Audio preprocessing utilities for VOXGUARD.

Handles loading, resampling, channel down-mixing, silence trimming,
peak normalization, and windowed slicing for deepfake voice analysis.
"""

from __future__ import annotations

import io
import math
from typing import List, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    source: Union[str, bytes, io.BytesIO],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file or byte buffer, convert to mono, and resample.

    Args:
        source: File path, raw bytes, or BytesIO buffer (e.g. from Streamlit).
        target_sr: Target sampling rate in Hz (default 16000).

    Returns:
        Tuple of (audio_signal: np.ndarray, sample_rate: int).
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    # Load audio using soundfile first, falling back to librosa
    try:
        y, sr = sf.read(source, dtype="float32")
        # Handle multi-channel audio by averaging channels to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    except Exception:
        # Fallback to librosa if soundfile direct read encounters container issues
        if isinstance(source, io.BytesIO):
            source.seek(0)
        y, sr = librosa.load(source, sr=target_sr, mono=True)

    y = y.astype(np.float32)
    return y, sr


def trim_silence(y: np.ndarray, sr: int = 16000, top_db: float = 25.0) -> np.ndarray:
    """Trim leading and trailing silence from an audio signal.

    Args:
        y: 1D audio waveform array.
        sr: Audio sample rate.
        top_db: The threshold (in decibels) below reference to consider silence.

    Returns:
        Trimmed audio waveform.
    """
    if len(y) == 0:
        return y
    trimmed_y, _ = librosa.effects.trim(y, top_db=top_db)
    # If trimming removed virtually everything, keep original to avoid empty arrays
    if len(trimmed_y) < int(0.2 * sr):
        return y
    return trimmed_y


def normalize_audio(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio amplitude to a target peak value.

    Args:
        y: 1D audio waveform array.
        target_peak: Maximum absolute peak amplitude (default 0.95).

    Returns:
        Peak-normalized audio array.
    """
    if len(y) == 0:
        return y
    max_peak = np.max(np.abs(y))
    if max_peak > 1e-6:
        return (y / max_peak) * target_peak
    return y


def slice_into_windows(
    y: np.ndarray,
    sr: int = 16000,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Slice audio into overlapping fixed-duration windows.

    Args:
        y: 1D audio waveform array.
        sr: Audio sample rate (default 16000).
        window_sec: Window length in seconds (default 2.0).
        hop_sec: Hop size in seconds (default 1.0 -> 50% overlap).

    Returns:
        Tuple of:
          - List of windowed audio arrays (each of length window_samples).
          - List of (start_sec, end_sec) timestamps for each window.
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    total_len = len(y)
    windows: List[np.ndarray] = []
    timestamps: List[Tuple[float, float]] = []

    # If audio is shorter than a single window, pad it with reflected/zero values
    if total_len < window_samples:
        pad_len = window_samples - total_len
        padded = np.pad(y, (0, pad_len), mode="reflect" if total_len > 1 else "constant")
        windows.append(padded)
        timestamps.append((0.0, total_len / sr))
        return windows, timestamps

    start = 0
    while start + window_samples <= total_len:
        win = y[start : start + window_samples]
        win_start_sec = start / sr
        win_end_sec = (start + window_samples) / sr
        windows.append(win)
        timestamps.append((win_start_sec, win_end_sec))
        start += hop_samples

    # If there is an uncovered residual tail at the end
    if start < total_len and len(y) >= window_samples:
        last_end_sample = int(round(timestamps[-1][1] * sr)) if timestamps else 0
        if last_end_sample < total_len:
            win = y[-window_samples:]
            win_start_sec = (total_len - window_samples) / sr
            win_end_sec = total_len / sr
            windows.append(win)
            timestamps.append((win_start_sec, win_end_sec))

    return windows, timestamps


def preprocess_audio(
    source: Union[str, bytes, io.BytesIO],
    sr: int = 16000,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    trim: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[float, float]]]:
    """Complete preprocessing pipeline: load -> mono -> resample -> trim -> normalize -> window.

    Returns:
        Tuple of (full_normalized_audio, list_of_windows, list_of_timestamps).
    """
    y, sample_rate = load_audio(source, target_sr=sr)
    if trim:
        y = trim_silence(y, sr=sample_rate)
    y = normalize_audio(y)
    windows, timestamps = slice_into_windows(y, sr=sample_rate, window_sec=window_sec, hop_sec=hop_sec)
    return y, windows, timestamps


def save_audio(y: np.ndarray, file_path: str, sr: int = 16000) -> None:
    """Save audio waveform to a WAV file."""
    sf.write(file_path, y, sr, subtype="PCM_16")
