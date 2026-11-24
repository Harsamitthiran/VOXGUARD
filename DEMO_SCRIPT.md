# 🎙️ VOXGUARD Faculty Presentation & Live Demo Script

**Project:** VOXGUARD: Real-Time AI-Based Deepfake Voice Scam Detection System  
**Presentation Duration:** 3 to 5 Minutes  
**Audience:** Faculty Reviewers, Cybersecurity Analysts, and Engineering Peers  

---

## ⏱️ Timeline & Presentation Flow

```
[00:00 - 00:45] The Threat: Generative Voice Cloning Fraud
[00:45 - 01:30] System Architecture & Novelty (Acoustic + NLP Context)
[01:30 - 02:45] Live Interactive Demonstration (3 Scenarios)
[02:45 - 03:45] Results, Diagnostic Forensics & Report Download
[03:45 - 04:30] Limitations, Telephony Challenges & Future Roadmap
```

---

## 🎬 Step-by-Step Script & Talking Points

### 1. The Threat: Generative Voice Cloning Fraud (0:00 - 0:45)
- **Speaker:** 
  > *"Good morning, professors and evaluators. Today, generative AI allows an attacker to clone a person's voice from just a 3-second WhatsApp voice note or Instagram clip. Fraudsters use these cloned voices to target families, elderly parents, and corporate employees—demanding emergency UPI transfers, claiming car crashes, or posing as bank officers.*
  >
  > *Acoustic-only detectors often fail in the wild because background noise and compression cause false positives. To solve this, we created **VOXGUARD**: a multi-modal defense system that combines PyTorch CNN acoustic forensics with a contextual scam-intent engine to warn victims before money is transferred."*

---

### 2. System Architecture & The Multi-Modal Novelty (0:45 - 1:30)
- **Action:** Open the **"📘 How It Works & Architecture"** tab in the Streamlit UI.
- **Speaker:**
  > *"VOXGUARD operates in two synchronized pipelines:*
  > 1. ***Acoustic Stream:*** *The audio is resampled to 16 kHz mono, normalized, and sliced into 2.0-second sliding windows with 50% overlap. We extract 128-band Log-Mel Spectrograms and feed them into a 4-block 2D Convolutional Neural Network trained to spot neural vocoder artifacts.*
  > 2. ***Contextual Scam Stream (Our Core Innovation):*** *We transcribe the speech and scan it across four behavioral fraud pillars: financial requests (UPI, wire, OTP), urgency/pressure ('immediately', 'police'), impersonation ('I am your son/boss'), and secrecy ('don't tell anyone').*
  > 3. ***Synergy Fusion:*** *We fuse these streams into a composite VOXGUARD Risk Score (0-100%) with a non-linear synergy boost when high acoustic fake probability coincides with urgent extortion."*

---

### 3. Live Interactive Demonstration (1:30 - 2:45)
- **Action:** Navigate to the **"🔍 Analyze Audio"** tab.

#### Scenario A: Genuine Voice (Casual Lunch Chat)
- **Action:** Under *Select Audio Input Source*, select **"⚡ Load Demo Scenario"** -> Choose **"1. Genuine Voice - Casual Lunch Chat"**. Click **"🚀 Run VOXGUARD Analysis"**.
- **Result:** 
  - Risk Score: `< 20%` (Tier: `🟢 SAFE / LOW RISK`).
  - Spectrogram: Natural formant transitions, zero red shaded windows.
  - Scam Engine: `0%` Scam Intent (Clean).
- **Speaker:** 
  > *"Notice how genuine human speech demonstrates natural continuous prosody and absence of fraud triggers. The verdict clearly displays SAFE."*

#### Scenario B: AI Voice (Neutral Seminar Announcement)
- **Action:** Select **"2. AI Voice - Neutral Seminar Instruction"**. Click **"🚀 Run VOXGUARD Analysis"**.
- **Result:**
  - Acoustic Clone Score: `> 85%` (Acoustic CNN detects synthetic vocoder harmonics).
  - Scam Intent Risk: `0%` (No financial or pressure phrases).
  - Composite Risk Score: `~ 51%` (Tier: `🟠 SUSPICIOUS / ELEVATED`).
- **Speaker:** 
  > *"Here is the power of our multi-modal approach! The voice is synthesized by an AI engine, so the acoustic CNN correctly flags it. However, because it is merely reading a neutral conference schedule, the Scam Engine remains zero. The system avoids an alarming false alarm, ranking it as SUSPICIOUS rather than a critical extortion attack."*

#### Scenario C: AI Voice (Urgent Hospital & UPI Scam Extortion)
- **Action:** Select **"3. AI Voice - Urgent Hospital & UPI Scam Extortion"**. Click **"🚀 Run VOXGUARD Analysis"**.
- **Result:**
  - Acoustic Clone Score: `> 90%`.
  - Scam Intent Risk: `> 90%` (Triggers: 'Dad, it is me', 'accident', 'wire 50,000 rupees', 'immediately', 'UPI', 'police', 'don't tell anyone').
  - Composite Risk Score: `95% - 100%` (Tier: `🔴 HIGH RISK / CRITICAL DEEPFAKE SCAM`).
  - Waveform & Spectrogram: Marked with vivid red highlights over the synthetic windows.
- **Speaker:** 
  > *"Now observe the critical attack scenario: an AI voice demanding urgent money for a fake accident. Both the acoustic CNN and the NLP scam engine trigger simultaneously. The synergy boost activates, pushing the score to CRITICAL HIGH RISK. VOXGUARD displays emergency advice: 'DO NOT SEND MONEY. HANG UP IMMEDIATELY. CALL BACK ON A KNOWN NUMBER.'*
  > *Notice also the time-frequency localization: the exact suspicious audio intervals are shaded in red with timestamps."*

---

### 4. Side-by-Side Comparison & Audit Report (2:45 - 3:45)
- **Action 1:** Switch to the **"⚖️ Compare Genuine vs AI Voice"** tab.
- **Speaker:** 
  > *"In this comparison view, examiners can inspect the raw acoustics side-by-side: human vocal micro-tremors versus the unnaturally flattened spectral harmonics and vocoder phase jitter characteristic of synthetic speech."*
- **Action 2:** Switch back to Analyze tab and click **"📥 Download Forensic Audit Report (HTML)"**. Open the downloaded HTML file.
- **Speaker:** 
  > *"VOXGUARD generates an exportable, self-contained forensic audit report with timestamped intervals, category breakdowns, and security recommendations, suitable for compliance records or cybercrime complaints."*

---

### 5. Limitations, Privacy & Future Roadmap (3:45 - 4:30)
- **Speaker:**
  > *"To conclude, we emphasize responsible engineering:*
  > 1. ***100% Local Privacy:*** *Audio is processed in memory on-device; no private speech data ever leaves the user's computer.*
  > 2. ***Current Limitations:*** *Heavy cellular compression (8 kHz AMR codecs) can degrade high-frequency vocoder artifacts. VOXGUARD compensates through its NLP scam stream, but physical out-of-band callback verification remains essential.*
  > 3. ***Future Work:*** *We plan to integrate real-time WebRTC audio streaming for live telephony call interception, expand to low-resource regional Indian languages, and build a mobile SIP dialer integration.*
  >
  > *Thank you. We welcome your questions."*
