"""Windowed inference and score aggregation engine for VOXGUARD.

Processes full audio clips through 2-second overlapping windows, computes
CNN model predictions per window, identifies localized suspicious time spans,
and aggregates into an overall deepfake confidence score.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.audio_utils import preprocess_audio
from src.features import extract_log_mel_spectrogram
from src.model import VoxGuardCNN


class VoxGuardInference:
    """Inference engine for VOXGUARD CNN deepfake detection."""

    def __init__(
        self,
        model_path: Optional[str] = "models/voxguard_best.pt",
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = VoxGuardCNN(in_channels=1, num_classes=2).to(self.device)
        self.model_loaded = False
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.model_loaded = True
                print(f"VOXGUARD: Model weights loaded successfully from {model_path}")
            except Exception as e:
                print(f"VOXGUARD: Warning loading model weights ({e}). Running in uncalibrated baseline mode.")
                self.model.eval()
        else:
            print(f"VOXGUARD: No checkpoint found at {model_path}. Initialized with default architecture.")
            self.model.eval()

    def predict(
        self,
        audio_source: Union[str, bytes, io.BytesIO],
        threshold: float = 0.50,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
    ) -> Dict[str, Any]:
        """Perform windowed audio inference and suspicious segment localization.

        Args:
            audio_source: File path or raw audio bytes/buffer.
            threshold: Probability threshold above which a window is marked suspicious.
            window_sec: Analysis window duration in seconds.
            hop_sec: Stride in seconds (1.0 = 50% overlap).

        Returns:
            Dictionary with overall confidence, window breakdowns, and intervals.
        """
        # Load and preprocess full audio
        y, windows, timestamps = preprocess_audio(
            audio_source, sr=16000, window_sec=window_sec, hop_sec=hop_sec
        )

        duration = len(y) / 16000.0
        if not windows:
            return {
                "overall_fake_pct": 0.0,
                "overall_fake_prob": 0.0,
                "window_results": [],
                "suspicious_intervals": [],
                "audio_waveform": y,
                "sample_rate": 16000,
                "duration_sec": duration,
                "threshold": threshold,
            }

        # Process each window through the CNN
        window_results: List[Dict[str, Any]] = []
        fake_probs: List[float] = []

        self.model.eval()
        with torch.no_grad():
            for idx, (win, (start_t, end_t)) in enumerate(zip(windows, timestamps)):
                spec = extract_log_mel_spectrogram(win, sr=16000)
                tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device)
                probs = self.model.predict_proba(tensor)[0].cpu().numpy()
                fake_prob = float(probs[1])

                fake_probs.append(fake_prob)
                is_suspicious = fake_prob >= threshold

                window_results.append(
                    {
                        "window_index": idx,
                        "start_time": float(round(start_t, 2)),
                        "end_time": float(round(end_t, 2)),
                        "fake_prob": float(round(fake_prob, 4)),
                        "fake_pct": float(round(fake_prob * 100.0, 1)),
                        "is_suspicious": is_suspicious,
                    }
                )

        # Aggregate window probabilities
        # Weighted combination: 70% top-k most suspicious windows + 30% clip mean
        # This prevents localized spoofed inserts from being diluted by long genuine pauses
        sorted_probs = sorted(fake_probs, reverse=True)
        top_k_count = max(1, int(len(sorted_probs) * 0.35))
        top_k_mean = float(np.mean(sorted_probs[:top_k_count]))
        overall_mean = float(np.mean(fake_probs))

        aggregated_prob = (0.70 * top_k_mean) + (0.30 * overall_mean)
        aggregated_pct = float(round(aggregated_prob * 100.0, 1))

        # Merge contiguous suspicious windows into continuous suspicious intervals
        suspicious_intervals = self._merge_suspicious_windows(window_results, threshold)

        return {
            "overall_fake_pct": aggregated_pct,
            "overall_fake_prob": float(round(aggregated_prob, 4)),
            "window_results": window_results,
            "suspicious_intervals": suspicious_intervals,
            "audio_waveform": y,
            "sample_rate": 16000,
            "duration_sec": float(round(duration, 2)),
            "threshold": threshold,
            "total_windows": len(windows),
            "suspicious_windows_count": sum(1 for w in window_results if w["is_suspicious"]),
        }

    @staticmethod
    def _merge_suspicious_windows(
        windows: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Tuple[float, float, float]]:
        """Merge consecutive overlapping suspicious windows into cohesive time spans.

        Returns:
            List of (start_sec, end_sec, max_confidence) tuples.
        """
        intervals: List[Tuple[float, float, float]] = []
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        current_max_prob: float = 0.0

        for win in windows:
            if win["is_suspicious"]:
                if current_start is None:
                    current_start = win["start_time"]
                    current_end = win["end_time"]
                    current_max_prob = win["fake_prob"]
                else:
                    # If this window overlaps with current interval
                    if win["start_time"] <= (current_end + 0.1):
                        current_end = max(current_end, win["end_time"])
                        current_max_prob = max(current_max_prob, win["fake_prob"])
                    else:
                        intervals.append((current_start, current_end, current_max_prob))
                        current_start = win["start_time"]
                        current_end = win["end_time"]
                        current_max_prob = win["fake_prob"]
            else:
                if current_start is not None:
                    intervals.append((current_start, current_end, current_max_prob))
                    current_start = None
                    current_end = None
                    current_max_prob = 0.0

        if current_start is not None and current_end is not None:
            intervals.append((current_start, current_end, current_max_prob))

        return intervals
