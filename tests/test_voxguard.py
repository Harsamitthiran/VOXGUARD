"""Unit tests for VOXGUARD components: audio utils, features, model, scam engine, and fusion.

Run with: pytest tests/
"""

import numpy as np
import pytest
import torch

from src.audio_utils import normalize_audio, slice_into_windows, trim_silence
from src.features import (
    DEFAULT_SR,
    TARGET_FRAMES,
    extract_acoustic_summary,
    extract_log_mel_spectrogram,
    extract_mfcc,
    extract_spectral_contrast,
    extract_spectral_flatness,
    extract_zero_crossing_rate,
    window_to_tensor,
)
from src.fusion import VoxGuardFusion
from src.infer import VoxGuardInference
from src.model import VoxGuardCNN
from src.scam_engine import ScamRiskEngine


# ==========================================
# Audio Utils Tests
# ==========================================

def test_normalize_audio():
    """Verify peak amplitude normalization."""
    # Test zero signal
    zeros = np.zeros(1000, dtype=np.float32)
    norm_zeros = normalize_audio(zeros)
    assert np.all(norm_zeros == 0.0)

    # Test peak normalization to target 0.95
    y = np.array([0.1, -0.5, 0.4, -0.2], dtype=np.float32)
    norm_y = normalize_audio(y, target_peak=0.95)
    assert np.isclose(np.max(np.abs(norm_y)), 0.95)


def test_trim_silence():
    """Verify silence trimming."""
    sr = 16000
    # Create 0.5s silence + 1.0s tone + 0.5s silence
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)).astype(np.float32)
    signal = np.concatenate([silence, tone, silence])

    trimmed = trim_silence(signal, sr=sr, top_db=20)
    assert len(trimmed) < len(signal)
    assert len(trimmed) >= int(0.8 * sr)


def test_slice_into_windows():
    """Verify window slicing and padding for short and long audio."""
    sr = 16000
    # Case 1: Audio shorter than 2s window (e.g. 1s)
    short_audio = np.random.randn(sr).astype(np.float32)
    windows, timestamps = slice_into_windows(short_audio, sr=sr, window_sec=2.0, hop_sec=1.0)
    assert len(windows) == 1
    assert len(windows[0]) == int(2.0 * sr)  # Padded to 2s
    assert len(timestamps) == 1

    # Case 2: 4-second audio with 2s window and 1s hop -> 3 windows
    audio_4s = np.random.randn(4 * sr).astype(np.float32)
    windows, timestamps = slice_into_windows(audio_4s, sr=sr, window_sec=2.0, hop_sec=1.0)
    assert len(windows) == 3
    assert timestamps[0] == (0.0, 2.0)
    assert timestamps[1] == (1.0, 3.0)
    assert timestamps[2] == (2.0, 4.0)


# ==========================================
# Feature Extraction Tests
# ==========================================

def test_extract_log_mel_spectrogram_shape():
    """Verify Log-Mel spectrogram output dimensions (n_mels=128, frames=64)."""
    sr = 16000
    y_window = np.random.randn(2 * sr).astype(np.float32)
    spec = extract_log_mel_spectrogram(y_window, sr=sr, n_mels=128, target_frames=TARGET_FRAMES)

    assert spec.shape == (128, TARGET_FRAMES)
    assert np.all(spec >= 0.0) and np.all(spec <= 1.0)  # Normalized to [0, 1]


def test_extract_acoustic_features():
    """Verify MFCC, spectral contrast, flatness, and ZCR extraction."""
    sr = 16000
    y = np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 2 * sr)).astype(np.float32)

    mfcc_dict = extract_mfcc(y, sr=sr, n_mfcc=20)
    assert mfcc_dict["mfcc"].shape[0] == 20
    assert "delta" in mfcc_dict and "delta2" in mfcc_dict

    contrast = extract_spectral_contrast(y, sr=sr)
    assert contrast.shape[0] == 7  # 6 bands + 1

    flatness = extract_spectral_flatness(y)
    assert flatness.ndim == 2

    zcr = extract_zero_crossing_rate(y)
    assert zcr.ndim == 2

    summary = extract_acoustic_summary(y, sr=sr)
    assert "mfcc_mean" in summary
    assert "spectral_contrast_mean" in summary
    assert "spectral_flatness_mean" in summary
    assert "zero_crossing_rate_mean" in summary


def test_window_to_tensor():
    """Verify tensor formatting for PyTorch model input."""
    sr = 16000
    y = np.random.randn(2 * sr).astype(np.float32)
    tensor = window_to_tensor(y, sr=sr)
    assert tensor.shape == (1, 1, 128, 64)
    assert isinstance(tensor, torch.Tensor)


# ==========================================
# Model Architecture Tests
# ==========================================

def test_model_forward():
    """Verify CNN forward pass, output shape, and probability bounds."""
    model = VoxGuardCNN(in_channels=1, num_classes=2)
    dummy_input = torch.randn(4, 1, 128, 64)  # Batch of 4 windows

    logits = model(dummy_input)
    assert logits.shape == (4, 2)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (4, 2)
    # Sum of probabilities across classes should equal 1.0
    row_sums = probs.sum(dim=1).numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    fake_probs = model.get_fake_prob(dummy_input)
    assert fake_probs.shape == (4,)
    assert (fake_probs >= 0.0).all() and (fake_probs <= 1.0).all()


# ==========================================
# Scam Risk Engine Tests
# ==========================================

def test_scam_engine_benign_text():
    """Verify that casual benign text receives zero scam score."""
    engine = ScamRiskEngine()
    benign_text = "Good morning! Are we still having our team meeting today at 2 PM?"
    result = engine.analyze_text(benign_text)

    assert result["overall_scam_score"] == 0.0
    assert result["matched_phrases_count"] == 0


def test_scam_engine_urgent_fraud_text():
    """Verify detection of financial demand, urgency, impersonation, and secrecy."""
    engine = ScamRiskEngine()
    scam_text = (
        "Dad, it is me, your son! I had a terrible accident and need you to transfer 50000 rupees "
        "immediately to this UPI ID before the police arrest me. Don't tell anyone!"
    )
    result = engine.analyze_text(scam_text)

    categories = result["categories"]
    assert categories["financial_requests"]["score"] > 0
    assert categories["urgency_and_pressure"]["score"] > 0
    assert categories["impersonation_claims"]["score"] > 0
    assert categories["secrecy_and_manipulation"]["score"] > 0

    # Multi-category triggers should yield a critical score and synergy bonus
    assert result["overall_scam_score"] >= 80.0
    assert result["compound_bonus"] > 0.0
    assert "UPI" in [m.upper() for m in categories["financial_requests"]["matches"]]


# ==========================================
# Multi-modal Fusion Tests
# ==========================================

def test_fusion_safe():
    """Verify SAFE tier when both voice and text indicators are minimal."""
    res = VoxGuardFusion.fuse(voice_fake_pct=10.0, context_scam_pct=5.0)
    assert res["risk_level"] == "SAFE"
    assert res["risk_score"] < 35.0
    assert "SAFE" in res["verdict_title"]


def test_fusion_suspicious():
    """Verify SUSPICIOUS tier for moderate indicators."""
    res = VoxGuardFusion.fuse(voice_fake_pct=50.0, context_scam_pct=40.0)
    assert res["risk_level"] == "SUSPICIOUS"
    assert 35.0 <= res["risk_score"] < 70.0


def test_fusion_high_risk_with_synergy():
    """Verify HIGH RISK tier and synergy boost when both voice and scam indicators are high."""
    res = VoxGuardFusion.fuse(voice_fake_pct=85.0, context_scam_pct=90.0)
    assert res["risk_level"] == "HIGH RISK"
    assert res["risk_score"] >= 70.0
    assert res["synergy_boost"] > 0.0
    assert len(res["advice"]) > 0


def test_fusion_engine_disabled():
    """Verify fusion output when scam engine is turned off (audio only)."""
    res = VoxGuardFusion.fuse(voice_fake_pct=72.0, context_scam_pct=90.0, scam_engine_enabled=False)
    assert res["risk_score"] == 72.0
    assert res["risk_level"] == "HIGH RISK"
