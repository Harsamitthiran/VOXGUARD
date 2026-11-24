"""VOXGUARD: Real-Time AI-Based Deepfake Voice Scam Detection System.

Streamlit Interactive Application providing:
1. Audio upload & microphone recording
2. Windowed CNN deepfake voice analysis & suspicious segment highlighting
3. Contextual speech transcription and 4-factor NLP scam scoring
4. Multi-modal risk fusion with SAFE / SUSPICIOUS / HIGH RISK tiers
5. Side-by-side comparative diagnostics (Genuine vs AI)
6. Downloadable audit report export
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Prevent permission issues on restricted sandbox environments
local_dir = os.path.dirname(os.path.abspath(__file__))
if "HOME" not in os.environ:
    os.environ["HOME"] = local_dir
if "STREAMLIT_CONFIG_DIR" not in os.environ:
    os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(local_dir, ".streamlit")

import numpy as np
import streamlit as st

# Internal VOXGUARD Modules
from src.audio_utils import preprocess_audio
from src.features import extract_acoustic_summary
from src.fusion import VoxGuardFusion
from src.infer import VoxGuardInference
from src.report_generator import generate_html_report
from src.scam_engine import ScamRiskEngine
from src.visualize import (
    create_risk_gauge_chart,
    create_scam_categories_bar_chart,
    plot_spectrogram_with_highlights,
    plot_waveform_with_highlights,
)

# Page Configuration
st.set_page_config(
    page_title="VOXGUARD | Deepfake Voice Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Cybersecurity Command Center Theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
    }
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .banner-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        border-left: 6px solid;
    }
    .advice-item {
        margin-bottom: 6px;
        color: #cbd5e1;
        font-size: 14px;
    }
    .pill-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system_engines():
    """Cache inference and NLP engines across user sessions."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "voxguard_best.pt")
    infer_engine = VoxGuardInference(model_path=model_path)
    scam_engine = ScamRiskEngine()
    return infer_engine, scam_engine


def main():
    infer_engine, scam_engine = load_system_engines()

    # Sidebar Controls
    with st.sidebar:
        st.markdown("## 🛡️ VOXGUARD Control")
        st.caption("AI-Powered Deepfake Voice & Scam Defense")
        st.divider()

        st.markdown("### ⚙️ Analysis Parameters")
        threshold = st.slider(
            "Suspicious Detection Threshold",
            min_value=0.20,
            max_value=0.85,
            value=0.50,
            step=0.05,
            help="Threshold above which a 2-second audio window is flagged as synthetic deepfake.",
        )

        enable_scam_engine = st.toggle(
            "Contextual Scam-Risk Engine",
            value=True,
            help="Enables transcription and multi-dimensional NLP rule scoring for financial demands and urgency.",
        )

        st.divider()
        st.markdown("### 🔍 Model Status")
        if infer_engine.model_loaded:
            st.success("PyTorch CNN: Trained Weights Active")
        else:
            st.warning("PyTorch CNN: Baseline Initialized (Run train.py to calibrate)")

        st.caption(f"Compute Device: `{infer_engine.device}`")

        st.divider()
        st.markdown("### ⚠️ Security Notice")
        st.info(
            "**VOXGUARD** runs 100% locally on your machine. Audio files and speech transcripts are never "
            "transmitted to external servers. Designed as a real-time decision-support system."
        )

    # Header
    col_head1, col_head2 = st.columns([0.8, 0.2])
    with col_head1:
        st.title("🛡️ VOXGUARD")
        st.markdown(
            "**Real-Time AI-Based Deepfake Voice Scam Detection System** &nbsp;•&nbsp; "
            "Multi-Modal Acoustic Artifact Localization & Contextual Coercion Analysis"
        )
    with col_head2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: right;"><span style="background: #1e293b; color: #38bdf8; '
            'padding: 6px 14px; border-radius: 20px; font-weight: bold; border: 1px solid #38bdf8;">'
            'v1.0.0 Active</span></div>',
            unsafe_allow_html=True,
        )

    # Main Tabs
    tab_analyze, tab_compare, tab_about = st.tabs(
        ["🔍 Analyze Audio", "⚖️ Compare Genuine vs AI Voice", "📘 How It Works & Architecture"]
    )

    # =========================================================================
    # TAB 1: ANALYZE AUDIO
    # =========================================================================
    with tab_analyze:
        st.markdown("### 🎙️ Audio Input")

        input_choice = st.radio(
            "Select Audio Input Source:",
            ["📁 Upload Audio File", "🎤 Record from Microphone", "⚡ Load Demo Scenario"],
            horizontal=True,
        )

        audio_bytes = None
        audio_name = "audio_sample.wav"
        preset_transcript = None

        if input_choice == "📁 Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload Voice Clip (WAV, MP3, FLAC)",
                type=["wav", "mp3", "flac"],
                help="Recommended: 3 to 30 seconds of clear speech.",
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                audio_name = uploaded_file.name

        elif input_choice == "🎤 Record from Microphone":
            st.info("Record speech using your browser microphone:")
            mic_audio = st.audio_input("Record audio clip")
            if mic_audio is not None:
                audio_bytes = mic_audio.read()
                audio_name = "mic_recording.wav"

        else:
            st.info("Select a pre-configured scenario to test VOXGUARD in seconds:")
            scenario = st.selectbox(
                "Pre-packaged Scenario:",
                [
                    "1. Genuine Voice - Casual Lunch Chat",
                    "2. AI Voice - Neutral Seminar Instruction",
                    "3. AI Voice - Urgent Hospital & UPI Scam Extortion",
                ],
            )
            samples_dir = os.path.join(os.path.dirname(__file__), "data", "samples")
            if "1." in scenario:
                sample_file = os.path.join(samples_dir, "genuine_casual.wav")
            elif "2." in scenario:
                sample_file = os.path.join(samples_dir, "ai_neutral.wav")
            else:
                sample_file = os.path.join(samples_dir, "ai_urgent_scam.wav")

            if os.path.exists(sample_file):
                with open(sample_file, "rb") as f:
                    audio_bytes = f.read()
                audio_name = os.path.basename(sample_file)
                transcript_file = sample_file.replace(".wav", "_transcript.txt")
                if os.path.exists(transcript_file):
                    with open(transcript_file, "r", encoding="utf-8") as tf:
                        preset_transcript = tf.read().strip()
            else:
                st.warning(
                    f"Sample file not found at {sample_file}. Run 'python generate_demo_data.py' to generate demo clips."
                )

        # Audio Playback
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            col_btn, _ = st.columns([0.3, 0.7])
            with col_btn:
                run_analysis = st.button("🚀 Run VOXGUARD Analysis", type="primary", use_container_width=True)

            if run_analysis:
                with st.spinner("Analyzing acoustic features and scanning conversational risk..."):
                    start_time = time.time()

                    # 1. Acoustic Inference
                    infer_result = infer_engine.predict(audio_bytes, threshold=threshold)

                    # 2. Contextual Scam Engine
                    transcript = scam_engine.transcribe_audio(
                        audio_bytes, provided_transcript=preset_transcript
                    )
                    scam_result = scam_engine.analyze_text(transcript)

                    # 3. Multi-modal Fusion
                    voice_fake_pct = infer_result["overall_fake_pct"]
                    context_scam_pct = scam_result["overall_scam_score"] if enable_scam_engine else 0.0
                    fusion_result = VoxGuardFusion.fuse(
                        voice_fake_pct=voice_fake_pct,
                        context_scam_pct=context_scam_pct,
                        scam_engine_enabled=enable_scam_engine,
                    )
                    elapsed = time.time() - start_time

                # Display Results
                st.divider()

                # Warning Banner
                bg = fusion_result["bg_color"]
                border = fusion_result["border_color"]
                st.markdown(
                    f"""
                    <div class="banner-card" style="background: {bg}; border-color: {border};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <h2 style="margin: 0; color: #fff;">{fusion_result['verdict_title']}</h2>
                            <span style="background: {border}; color: white; padding: 4px 12px; border-radius: 14px; font-weight: bold;">
                                Risk: {fusion_result['risk_score']:.1f}%
                            </span>
                        </div>
                        <p style="color: #e2e8f0; font-size: 15px; margin-bottom: 12px;">{fusion_result['verdict_summary']}</p>
                        <strong style="color: #f8fafc;">Recommended Action Plan:</strong>
                        <div style="margin-top: 6px;">
                            {''.join([f'<div class="advice-item">🛡️ {item}</div>' for item in fusion_result['advice']])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Core Metric Gauges
                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.plotly_chart(
                        create_risk_gauge_chart(fusion_result["risk_score"], "VOXGUARD Risk Score"),
                        use_container_width=True,
                    )
                    st.caption(f"**Formula:** {fusion_result['formula_description']}")
                with col_g2:
                    st.plotly_chart(
                        create_risk_gauge_chart(voice_fake_pct, "Acoustic Clone Score"),
                        use_container_width=True,
                    )
                    st.caption(
                        f"Analyzed {infer_result['total_windows']} windows (2.0s each). "
                        f"{infer_result['suspicious_windows_count']} window(s) above {int(threshold*100)}% threshold."
                    )
                with col_g3:
                    st.plotly_chart(
                        create_risk_gauge_chart(context_scam_pct, "Scam Intent Risk"),
                        use_container_width=True,
                    )
                    st.caption(
                        f"Matched {scam_result['matched_phrases_count']} scam indicator phrases. "
                        f"Compound Bonus: +{scam_result['compound_bonus']:.0f}%"
                    )

                # Acoustic Forensics Section
                st.markdown("### 🔬 Acoustic Artifact Analysis & Suspicious Segment Localization")
                col_spec, col_table = st.columns([0.65, 0.35])

                with col_spec:
                    st.markdown("**Waveform (Highlighted Suspicious Intervals in Red):**")
                    fig_wave = plot_waveform_with_highlights(
                        infer_result["audio_waveform"],
                        sr=infer_result["sample_rate"],
                        suspicious_intervals=infer_result["suspicious_intervals"],
                        threshold=threshold,
                    )
                    st.pyplot(fig_wave)

                    st.markdown("**Log-Mel Spectrogram (Time-Frequency Forensics):**")
                    fig_spec = plot_spectrogram_with_highlights(
                        infer_result["audio_waveform"],
                        sr=infer_result["sample_rate"],
                        suspicious_intervals=infer_result["suspicious_intervals"],
                        threshold=threshold,
                    )
                    st.pyplot(fig_spec)

                with col_table:
                    st.markdown("**Suspicious Time Windows:**")
                    intervals = infer_result["suspicious_intervals"]
                    if intervals:
                        table_data = [
                            {
                                "Interval": f"{s:.2f}s - {e:.2f}s",
                                "Duration": f"{(e - s):.2f}s",
                                "Confidence": f"{p*100:.1f}%",
                            }
                            for s, e, p in intervals
                        ]
                        st.dataframe(table_data, use_container_width=True)
                    else:
                        st.success("✅ No suspicious synthetic segments detected across all audio windows.")

                    st.markdown("**Acoustic Diagnostic Metrics:**")
                    diag = extract_acoustic_summary(infer_result["audio_waveform"], infer_result["sample_rate"])
                    st.metric("Spectral Flatness", f"{diag['spectral_flatness_mean']:.5f}")
                    st.metric("Spectral Contrast (dB)", f"{diag['spectral_contrast_mean']:.2f}")
                    st.metric("Zero-Crossing Rate", f"{diag['zero_crossing_rate_mean']:.4f}")

                # Contextual Scam Forensics Section
                if enable_scam_engine:
                    st.divider()
                    st.markdown("### 💬 Conversational Scam Intent Analysis")
                    col_nlp1, col_nlp2 = st.columns([0.5, 0.5])

                    with col_nlp1:
                        st.markdown("**Transcript Forensics (Color-Coded Scam Markers):**")
                        st.markdown(
                            f"""
                            <div style="background: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155; line-height: 1.8;">
                                {scam_result['highlighted_html']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("<br>**Matched Keyword Triggers by Category:**", unsafe_allow_html=True)
                        for cat_id, cat_info in scam_result["categories"].items():
                            if cat_info["matches"]:
                                pills = " ".join(
                                    [
                                        f'<span class="pill-badge" style="background: {cat_info["color"]}33; color: {cat_info["color"]}; border: 1px solid {cat_info["color"]};">{m}</span>'
                                        for m in cat_info["matches"]
                                    ]
                                )
                                st.markdown(f"**{cat_info['title']}:** {pills}", unsafe_allow_html=True)

                    with col_nlp2:
                        fig_bars = create_scam_categories_bar_chart(scam_result["categories"])
                        st.plotly_chart(fig_bars, use_container_width=True)

                # Export Report Button
                st.divider()
                html_report = generate_html_report(
                    audio_filename=audio_name,
                    inference_result=infer_result,
                    scam_result=scam_result,
                    fusion_result=fusion_result,
                )

                col_down, _ = st.columns([0.4, 0.6])
                with col_down:
                    st.download_button(
                        label="📥 Download Forensic Audit Report (HTML)",
                        data=html_report,
                        file_name=f"voxguard_audit_{int(time.time())}.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                st.caption("The downloaded report is self-contained and formatted for security reviews and printing.")

    # =========================================================================
    # TAB 2: COMPARE GENUINE VS AI VOICE
    # =========================================================================
    with tab_compare:
        st.markdown("### ⚖️ Side-by-Side Acoustic Comparison: Genuine vs AI Synthetic Voice")
        st.write(
            "Inspect acoustic differences between human vocal tract mechanics and synthetic neural vocoder generation."
        )

        samples_dir = os.path.join(os.path.dirname(__file__), "data", "samples")
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith(".wav")] if os.path.exists(samples_dir) else []

        if not sample_files:
            st.info("Run 'python generate_demo_data.py' to generate sample comparison audio files.")
        else:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                real_choice = st.selectbox(
                    "Select Genuine Reference Audio:",
                    [s for s in sample_files if "genuine" in s] or sample_files,
                )
            with col_sel2:
                fake_choice = st.selectbox(
                    "Select AI Synthetic Audio:",
                    [s for s in sample_files if "ai" in s or "fake" in s] or sample_files,
                )

            if real_choice and fake_choice:
                real_path = os.path.join(samples_dir, real_choice)
                fake_path = os.path.join(samples_dir, fake_choice)

                col_r, col_f = st.columns(2)

                # Process Real
                with col_r:
                    st.markdown("#### 🟢 Genuine Audio")
                    with open(real_path, "rb") as rf:
                        r_bytes = rf.read()
                    st.audio(r_bytes, format="audio/wav")

                    r_infer = infer_engine.predict(r_bytes, threshold=threshold)
                    st.metric("Acoustic Clone Score", f"{r_infer['overall_fake_pct']:.1f}%")

                    fig_r = plot_spectrogram_with_highlights(
                        r_infer["audio_waveform"],
                        sr=r_infer["sample_rate"],
                        suspicious_intervals=r_infer["suspicious_intervals"],
                        threshold=threshold,
                    )
                    st.pyplot(fig_r)

                    r_diag = extract_acoustic_summary(r_infer["audio_waveform"], r_infer["sample_rate"])
                    st.markdown(
                        f"""
                        - **Spectral Flatness:** `{r_diag['spectral_flatness_mean']:.5f}` (Natural harmonic resonances)
                        - **Spectral Contrast:** `{r_diag['spectral_contrast_mean']:.2f} dB`
                        - **ZCR:** `{r_diag['zero_crossing_rate_mean']:.4f}`
                        """
                    )

                # Process Fake
                with col_f:
                    st.markdown("#### 🔴 AI Synthetic Audio")
                    with open(fake_path, "rb") as ff:
                        f_bytes = ff.read()
                    st.audio(f_bytes, format="audio/wav")

                    f_infer = infer_engine.predict(f_bytes, threshold=threshold)
                    st.metric("Acoustic Clone Score", f"{f_infer['overall_fake_pct']:.1f}%")

                    fig_f = plot_spectrogram_with_highlights(
                        f_infer["audio_waveform"],
                        sr=f_infer["sample_rate"],
                        suspicious_intervals=f_infer["suspicious_intervals"],
                        threshold=threshold,
                    )
                    st.pyplot(fig_f)

                    f_diag = extract_acoustic_summary(f_infer["audio_waveform"], f_infer["sample_rate"])
                    st.markdown(
                        f"""
                        - **Spectral Flatness:** `{f_diag['spectral_flatness_mean']:.5f}` (Flattened vocoder artifacts)
                        - **Spectral Contrast:** `{f_diag['spectral_contrast_mean']:.2f} dB`
                        - **ZCR:** `{f_diag['zero_crossing_rate_mean']:.4f}`
                        """
                    )

                st.divider()
                st.markdown("#### 🔍 Forensic Explanation of Differences")
                st.markdown(
                    """
                    1. **Harmonic Consistency & Phase Structure:** Human speech exhibits continuous pitch contour micro-jitter and dynamic formant filtering through the oral cavity. AI TTS and neural vocoders (e.g. HiFi-GAN) often introduce subtle periodic phase discontinuities and unnatural energy distributions above 4 kHz.
                    2. **Spectral Flatness:** Synthetic speech frequently displays higher spectral flatness in unvoiced regions due to vocoder noise and oversmoothed Mel-spectrogram reconstruction.
                    3. **Prosodic Naturalness:** AI voice clones often suffer from rigid syllable duration patterns and pitch quantization, especially under stressed emotional contexts.
                    """
                )

    # =========================================================================
    # TAB 3: HOW IT WORKS & ARCHITECTURE
    # =========================================================================
    with tab_about:
        st.markdown("### 📘 System Architecture & Detection Pipeline")

        st.markdown(
            """
            VOXGUARD addresses the surging threat of **voice cloning fraud** by fusing acoustic deepfake detection
            with conversational scam-intent analysis.

            #### 1. Audio Preprocessing & Windowing
            - **Resampling:** Uniform 16,000 Hz single-channel (mono) representation.
            - **Silence Trimming:** Removal of silent leading/trailing regions via top-dB thresholding.
            - **Sliding Window:** Audio is partitioned into **2.0-second overlapping windows (1.0-second stride)**. This preserves fine-grained temporal resolution to flag brief synthetic voice insertions.

            #### 2. Feature Extraction & CNN Model
            - **Log-Mel Spectrograms:** 128 Mel frequency bands covering 20 Hz to 8,000 Hz, converted to decibels and normalized to [0, 1].
            - **Convolutional Neural Network:** 4 Conv2d blocks with Batch Normalization, ReLU activations, and Max Pooling, followed by Global Average Pooling (GAP) and a dense classification head with Dropout (0.35).
            - **Aggregation:** Window predictions are combined via a top-k weighted scheme (70% top 35% most suspicious windows + 30% overall mean) to detect localized synthetic splices.

            #### 3. Contextual Scam-Risk Engine
            - Transcribes speech via Whisper and scans across four psychological fraud pillars:
              1. **Financial Requests:** Transfer, UPI, OTP, gift card, wire, crypto.
              2. **Urgency & Intimidation:** Immediately, right now, arrest, frozen account, hurry.
              3. **Impersonation Claims:** "I am your son/boss/bank officer/police".
              4. **Secrecy & Coercion:** "Don't tell anyone", accident, kidnapped, hospital.

            #### 4. Multi-Modal Risk Fusion Formula
            $$Risk = 0.60 \\times VoiceFake\\% + 0.40 \\times ScamRisk\\% + SynergyBoost$$
            - A **synergy boost** (up to +18%) triggers when both acoustic cloning indicators and fraud coercion occur simultaneously.

            #### 5. Responsible AI & Limitations
            - **Decision-Support Tool:** VOXGUARD provides risk assessments to assist users and fraud analysts; it is not an absolute mathematical guarantee.
            - **Noise & Compression:** Extreme telephone line compression (e.g., 8 kHz AMR-NB) can degrade acoustic features. Out-of-band verification remains mandatory.
            """
        )


if __name__ == "__main__":
    main()
