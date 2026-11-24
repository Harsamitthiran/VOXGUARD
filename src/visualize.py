"""Visualization module for VOXGUARD.

Generates waveforms and spectrograms with highlighted suspicious time regions,
along with interactive risk gauges and radar/bar breakdown charts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_waveform_with_highlights(
    y: np.ndarray,
    sr: int = 16000,
    suspicious_intervals: Optional[List[Tuple[float, float, float]]] = None,
    threshold: float = 0.50,
) -> plt.Figure:
    """Generate audio waveform plot with suspicious AI segments highlighted in red.

    Args:
        y: Audio signal array.
        sr: Sample rate.
        suspicious_intervals: List of (start_sec, end_sec, fake_prob).
        threshold: Suspicious detection threshold.

    Returns:
        Matplotlib Figure object.
    """
    duration = len(y) / sr
    time_axis = np.linspace(0, duration, len(y))

    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    # Plot base waveform
    ax.plot(time_axis, y, color="#38bdf8", alpha=0.85, linewidth=0.8, label="Audio Waveform")

    # Shading suspicious regions
    if suspicious_intervals:
        added_label = False
        for start, end, prob in suspicious_intervals:
            alpha = min(0.65, 0.25 + (prob * 0.4))
            label = f"Suspicious Segment (Fake Prob >= {int(threshold*100)}%)" if not added_label else None
            ax.axvspan(start, end, color="#ef4444", alpha=alpha, label=label, edgecolor="#dc2626", linewidth=1.5)
            # Add small badge above waveform
            ax.text(
                (start + end) / 2.0,
                0.88,
                f"{prob*100:.0f}%",
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#dc2626", edgecolor="none", alpha=0.85),
            )
            added_label = True

    ax.set_xlim(0, max(0.1, duration))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, color="#334155", linestyle="--", alpha=0.5)

    if added_label if suspicious_intervals else False:
        leg = ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", fontsize=8)
        for text in leg.get_texts():
            text.set_color("#f8fafc")

    plt.tight_layout()
    return fig


def plot_spectrogram_with_highlights(
    y: np.ndarray,
    sr: int = 16000,
    suspicious_intervals: Optional[List[Tuple[float, float, float]]] = None,
    threshold: float = 0.50,
) -> plt.Figure:
    """Generate Mel-Spectrogram with red highlighted suspicious time segments."""
    duration = len(y) / sr
    n_fft = 1024
    hop_length = 512

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    img = librosa.display.specshow(
        log_mel,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        fmax=sr // 2,
        ax=ax,
        cmap="magma",
    )

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8", fontsize=8)
    cbar.outline.set_edgecolor("#334155")

    # Highlight suspicious time segments
    if suspicious_intervals:
        added_label = False
        for start, end, prob in suspicious_intervals:
            alpha = min(0.55, 0.2 + (prob * 0.35))
            label = "Suspicious Time Window" if not added_label else None
            ax.axvspan(start, end, color="#ef4444", alpha=alpha, edgecolor="#ff4444", linewidth=1.5, label=label)
            added_label = True

    ax.set_xlim(0, max(0.1, duration))
    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def create_risk_gauge_chart(score: float, title: str = "VOXGUARD Risk Score") -> go.Figure:
    """Create a sleek Plotly gauge indicator chart."""
    if score < 35.0:
        bar_color = "#16a34a"  # Green
    elif score < 70.0:
        bar_color = "#ea580c"  # Orange
    else:
        bar_color = "#dc2626"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": "#f8fafc"}},
            number={"suffix": "%", "font": {"size": 36, "color": bar_color, "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "#1e293b",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 35], "color": "rgba(22, 163, 74, 0.15)"},
                    {"range": [35, 70], "color": "rgba(234, 88, 12, 0.15)"},
                    {"range": [70, 100], "color": "rgba(220, 38, 38, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.85,
                    "value": 70,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=220,
    )
    return fig


def create_scam_categories_bar_chart(categories: Dict[str, Any]) -> go.Figure:
    """Create a horizontal bar chart displaying breakdown of scam sub-scores."""
    titles = [cat_data["title"] for cat_data in categories.values()]
    scores = [cat_data["score"] for cat_data in categories.values()]
    colors = [cat_data["color"] for cat_data in categories.values()]
    matches = [len(cat_data["matches"]) for cat_data in categories.values()]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=titles,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
            text=[f"{s:.0f}% ({m} matches)" if s > 0 else "0% (Clean)" for s, m in zip(scores, matches)],
            textposition="auto",
        )
    )

    fig.update_layout(
        title={"text": "<b>Scam Intent Dimension Breakdown</b>", "font": {"size": 14, "color": "#f8fafc"}},
        xaxis=dict(range=[0, 100], title="Risk Sub-Score (%)", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(color="#f8fafc", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=20),
        height=230,
    )
    return fig
