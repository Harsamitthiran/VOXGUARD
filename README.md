# 🛡️ VOXGUARD: Real-Time AI-Based Deepfake Voice Scam Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

VOXGUARD is an end-to-end, multi-modal cybersecurity defense platform engineered to protect individuals, families, students, and financial institutions from **voice-cloning deepfake fraud** and extortion scams. By coupling **PyTorch CNN acoustic artifact detection** with a novel **contextual NLP scam-intent engine**, VOXGUARD pinpoints synthetic voice fragments and generates actionable security warnings before financial transactions occur.

---

## 📌 Problem Overview

Modern generative voice cloning architectures (e.g. VALL-E, XTTS, ElevenLabs, OpenVoice) require as little as 3 seconds of reference audio to create high-fidelity vocal replicas. Malicious actors leverage cloned voices in phone and voice message scams targeting:
- **Family Emergencies ("Grandparent Scam"):** Impersonating a child claiming a car crash or arrest.
- **CEO & Executive Fraud:** Ordering immediate wire transfers or gift cards.
- **Banking / Law Enforcement Impersonation:** Demanding OTPs, UPI payments, or crypto deposits under threat of frozen accounts.

Traditional acoustic-only detectors suffer from high false-alarm rates on compressed telephony audio. VOXGUARD introduces **Multi-Modal Decision Fusion**: fusing acoustic artifact probabilities with NLP coercion indicators to isolate malicious intent.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Input Audio: WAV / MP3 / Mic] --> B[Audio Preprocessing & Resampling 16kHz]
    B --> C[Silence Trimming & Peak Normalization]
    C --> D[Sliding Window Slicing: 2.0s Windows, 50% Overlap]

    subgraph Acoustic Stream
        D --> E[Librosa Log-Mel Spectrogram Extraction 128 Mels]
        E --> F[PyTorch 4-Block 2D CNN]
        F --> G[Window Probability Aggregation: Top-k + Clip Mean]
        G --> H[Suspicious Segment Time Highlights]
    end

    subgraph Contextual Scam Stream
        B --> I[Speech Transcription: Whisper / Sidecar]
        I --> J[4-Factor Rule NLP Engine]
        J --> K[Financial Requests]
        J --> L[Urgency & Pressure]
        J --> M[Identity Impersonation]
        J --> N[Secrecy & Coercion]
        K & L & M & N --> O[Contextual Scam-Risk Score 0-100%]
    end

    G --> P[Multi-Modal Fusion Engine]
    O --> P
    P --> Q{Risk Assessment Tier}
    Q -->|Score < 35%| R[🟢 SAFE / LOW RISK]
    Q -->|35% <= Score < 70%| S[🟠 SUSPICIOUS / ELEVATED]
    Q -->|Score >= 70%| T[🔴 HIGH RISK / DEEPFAKE SCAM]
    T --> U[Plain-Language Security Advice & Audit Report]
```

---

## 🌟 Key Features

1. **Multi-Source Audio Processing:** Upload audio (`.wav`, `.mp3`, `.flac`) or record live speech directly via the in-browser microphone.
2. **Standardized Preprocessing:** Resamples to 16 kHz mono, trims leading/trailing silence, normalizes peak amplitude, and segments audio into 2.0-second sliding windows with 1.0-second overlap (50% stride).
3. **Acoustic Deepfake CNN:** 4-Block Conv2D neural network with Batch Normalization, ReLU, Max Pooling, and Global Average Pooling that classifies Log-Mel spectrograms for acoustic vocoder artifacts.
4. **Suspicious Segment Highlighting:** Shaded red intervals on both raw waveform and Mel-spectrogram timelines pinpointing synthetic spliced speech fragments with exact timestamps.
5. **Contextual Scam-Risk Engine (Novelty):** Multi-pillar NLP rule engine categorizing speech into:
   - *Financial Demands:* UPI, transfer, OTP, gift cards, crypto, wire, bank accounts.
   - *Urgency & Pressure:* Immediately, right now, arrest, police, frozen account, hurry.
   - *Impersonation Claims:* "I am your son/daughter/boss/bank officer/police".
   - *Secrecy & Manipulation:* "Don't tell anyone", accident, kidnapped, hospital.
6. **Synergy-Boosted Risk Fusion:** Computes a composite VOXGUARD Risk Score:
   $$\text{Risk}_{\text{base}} = 0.60 \times \text{VoiceFake}\% + 0.40 \times \text{ScamRisk}\%$$
   Applies a non-linear synergy boost (up to +18%) when high acoustic spoofing probability coincides with scam coercion patterns.
7. **Side-by-Side Comparative Diagnostics:** Inspect genuine vs. AI synthetic audio side-by-side with spectral flatness, contrast, and zero-crossing rates.
8. **Forensic Audit Report Export:** One-click download of a standalone, self-contained HTML forensic security report ready for fraud teams or law enforcement.

---

## 📂 Project Structure

```
voxguard/
├── app.py                     # Interactive Streamlit Web Application
├── generate_demo_data.py      # Demo dataset & evaluation scenario generator
├── requirements.txt           # Project dependencies
├── README.md                  # System documentation & benchmark guide
├── DEMO_SCRIPT.md             # 3-5 minute faculty presentation demo script
├── src/
│   ├── __init__.py
│   ├── audio_utils.py         # Audio loading, resampling, normalization, windowing
│   ├── features.py            # Log-Mel, MFCC, spectral contrast/flatness, ZCR, augmentations
│   ├── model.py               # 4-Block PyTorch CNN architecture
│   ├── train.py               # File-partitioned training loop with class weighting
│   ├── evaluate.py            # Evaluation metrics (Accuracy, F1, ROC-AUC, EER)
│   ├── infer.py               # Windowed inference & suspicious interval aggregation
│   ├── scam_engine.py         # Speech transcription & 4-factor NLP scam scoring
│   ├── fusion.py              # Multi-modal fusion & actionable incident response
│   ├── visualize.py           # Waveform/spectrogram highlighting & Plotly gauges
│   └── report_generator.py    # Standalone HTML audit report generator
├── data/
│   ├── real/                  # Real / genuine training audio files
│   ├── fake/                  # Synthetic / fake training audio files
│   └── samples/               # Packaged test scenarios (genuine, AI neutral, AI scam)
├── models/
│   ├── voxguard_best.pt       # Saved PyTorch CNN model weights
│   └── test_split.npz         # Held-out test split for reproducibility
├── reports/
│   ├── loss_acc_curves.png    # Training & validation loss/accuracy curves
│   ├── confusion_matrix.png   # Test confusion matrix plot
│   ├── roc_curve.png          # ROC curve annotated with EER
│   └── metrics.json           # Serialized evaluation metrics
└── tests/
    └── test_voxguard.py       # Pytest unit test suite
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
Ensure Python 3.10+ is installed. Clone or navigate to the `voxguard/` directory:

```bash
cd voxguard
pip install -r requirements.txt
```

### 2. Generate Demo Data
Generate baseline training files and 3 realistic test scenarios:

```bash
python generate_demo_data.py
```

> **Notice:** The demo set generated by `generate_demo_data.py` is intended for local validation and functional demonstrations. For academic or production benchmarking, integrate public benchmark datasets described below.

### 3. Train the Model
Train the PyTorch CNN on the dataset, partition by file to prevent data leakage, and save evaluation artifacts to `reports/`:

```bash
python -m src.train --epochs 12 --batch-size 16
```

### 4. Evaluate Test Split
Evaluate the model on held-out test audio to produce confusion matrix and ROC curves:

```bash
python -m src.evaluate
```

### 5. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Unit Testing

Run the automated test suite covering audio preprocessing, feature dimensions, CNN inference, scam NLP rules, and risk fusion:

```bash
pytest tests/ -v
```

---

## 📊 Evaluation & Benchmarking

When trained on the synthesized acoustic baseline dataset, the system yields the following performance on held-out audio windows:

| Metric | Baseline Value | Description |
| :--- | :---: | :--- |
| **Accuracy** | **95.0% - 100%** | Window classification accuracy |
| **Precision** | **94.5% - 100%** | Positive predictive value for synthetic voice |
| **Recall** | **95.0% - 100%** | True synthetic voice detection rate |
| **F1 Score** | **0.95 - 1.00** | Harmonic mean of precision and recall |
| **ROC AUC** | **0.98 - 1.00** | Area Under the Receiver Operating Characteristic Curve |
| **EER (Equal Error Rate)** | **< 3.5%** | Operating point where FAR equals FRR |

> **Transparency Note:** These metrics reflect performance on the controlled demo dataset. Real-world accuracy varies depending on microphone quality, background noise, and telephony compression codecs.

---

## 🌐 Plugging in Public Benchmark Datasets

VOXGUARD's `src/train.py` is architected for drop-in compatibility with academic audio deepfake datasets:

### 1. ASVspoof 2019 LA (Logical Access)
- **Source:** [ASVspoof Consortium](https://www.asvspoof.org/)
- **Setup:**
  1. Download `LA.zip` and extract audio files.
  2. Map genuine audio files (`bonafide`) into `data/real/`.
  3. Map spoofed audio files (`spoof` from neural vocoders / TTS) into `data/fake/`.
  4. Run `python -m src.train --data-dir data --epochs 25`.

### 2. WaveFake Dataset
- **Source:** [WaveFake (Frank & Schönherr, 2021)](https://github.com/philschmid/wavefake)
- Contains 104,885 audio clips across 6 neural vocoders (MelGAN, HiFi-GAN, PWG, WaveGlow, etc.).
- Place LJSpeech genuine clips in `data/real/` and generated architecture folders in `data/fake/`.

### 3. FakeAVCeleb
- **Source:** [FakeAVCeleb Dataset](https://github.com/DASH-Lab/FakeAVCeleb)
- Extract the audio track from multimodal deepfakes into `data/fake/` and genuine celebrity tracks into `data/real/`.

---

## 🔒 Privacy, Ethics & Limitations

- **100% Local Processing:** VOXGUARD processes all audio waveforms and transcripts entirely in local memory. No audio is dispatched to third-party cloud APIs.
- **Decision Support, Not Absolute Guarantee:** VOXGUARD is engineered to assist humans in detecting fraudulent calls. It must not be used as the sole deciding factor in criminal accusations or automated account lockouts.
- **Telephony Codec Degradation:** Severe audio downsampling (e.g. 8 kHz PSTN / cellular AMR codecs) can compress upper-frequency vocoder harmonics. In such environments, the Contextual Scam Engine acts as a vital secondary line of defense.
