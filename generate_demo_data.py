"""Demo dataset and sample scenario generation for VOXGUARD.

Generates:
1. data/real/ and data/fake/ audio clips for training and unit testing.
2. data/samples/ showcasing 3 realistic evaluation scenarios:
   - genuine_casual.wav (Natural casual tone)
   - ai_neutral.wav (Synthetic voice reading neutral instructions)
   - ai_urgent_scam.wav (Synthetic clone voice demanding urgent UPI transfer)

NOTE: This synthetic demo set is strictly for system functional demonstration
and local validation. It is NOT a substitute for academic benchmark datasets
such as ASVspoof 2019 LA or WaveFake.
"""

from __future__ import annotations

import math
import os
import random
from typing import List, Tuple

import numpy as np
import soundfile as sf

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


SAMPLE_RATE = 16000


def synthesize_fallback_speech(
    text: str,
    is_fake: bool = False,
    duration_sec: float = 4.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize acoustic waveforms simulating voice characteristics.

    Genuine audio simulates natural human vocal tract formant resonance, micro-prosody,
    subtle breathing noise, and continuous pitch variation.
    Fake audio simulates acoustic vocoder artifacts, spectral flattening, harmonic phase
    jitter, and robotic pitch quantization typical of neural vocoder artifacts.
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    if not is_fake:
        # Genuine: Natural fundamental frequency (F0) with expressive prosodic modulation (120-160 Hz)
        f0_base = 135.0 + 15.0 * np.sin(2 * np.pi * 0.4 * t) + 8.0 * np.cos(2 * np.pi * 0.9 * t)
        phase = 2 * np.pi * np.cumsum(f0_base) / sr

        # Human formant resonances (F1 ~ 500Hz, F2 ~ 1500Hz, F3 ~ 2500Hz)
        formant1 = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
        f1_mod = np.sin(2 * np.pi * 500 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t))
        f2_mod = np.sin(2 * np.pi * 1500 * t) * (0.4 + 0.3 * np.cos(2 * np.pi * 3.0 * t))
        f3_mod = np.sin(2 * np.pi * 2500 * t) * 0.15

        # Breath and natural acoustic warmth
        breath = np.random.normal(0, 0.02, num_samples)
        signal = 0.5 * formant1 + 0.25 * f1_mod + 0.15 * f2_mod + 0.05 * f3_mod + breath

        # Speech envelope (simulating syllabic cadence: vowels and consonants)
        envelope = np.clip(np.sin(2 * np.pi * 3.2 * t) ** 2, 0.05, 1.0)
        signal = signal * envelope
    else:
        # Fake / AI: Robotic fixed pitch, quantized harmonic grid, phase discontinuities & vocoder buzz
        f0_quantized = 145.0  # Unnatural pitch rigidity
        phase = 2 * np.pi * f0_quantized * t

        # High harmonic buzzing (vocoder artifact)
        harmonics = sum(
            (0.8 ** k) * np.sin((k + 1) * phase + (0.3 if k % 2 == 0 else -0.3))
            for k in range(12)
        )
        # Synthetic metallic artifact & high frequency ripple
        high_freq_buzz = 0.12 * np.sin(2 * np.pi * 3800 * t) + 0.08 * np.sin(2 * np.pi * 7200 * t)
        glitch = (np.sin(2 * np.pi * 10 * t) > 0.98).astype(np.float32) * 0.1

        signal = 0.4 * harmonics + high_freq_buzz + glitch
        # Stiff, mechanical envelope
        envelope = np.where(np.sin(2 * np.pi * 3.5 * t) > -0.2, 0.85, 0.1)
        signal = signal * envelope

    # Peak normalize
    max_amp = np.max(np.abs(signal))
    if max_amp > 1e-5:
        signal = (signal / max_amp) * 0.9

    return signal.astype(np.float32)


def generate_tts_file(text: str, output_path: str, rate: int = 150, pitch_shift_factor: float = 1.0) -> bool:
    """Generate speech using pyttsx3 if available."""
    if not HAS_PYTTSX3:
        return False

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"pyttsx3 error ({e}), falling back to acoustic synthesis.")
        return False


def create_scenario_sample(
    text: str,
    output_path: str,
    is_fake: bool,
    scenario_type: str = "casual",
) -> None:
    """Create a sample file with TTS or acoustic synthesizer, ensuring valid 16kHz WAV format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = False

    if HAS_PYTTSX3:
        rate = 135 if not is_fake else (180 if "scam" in scenario_type else 150)
        success = generate_tts_file(text, output_path, rate=rate)

    if not success:
        # Use acoustic wave synthesizer
        duration = 5.0 if "scam" in scenario_type else 4.0
        audio = synthesize_fallback_speech(text, is_fake=is_fake, duration_sec=duration, sr=SAMPLE_RATE)
        sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
    else:
        # Re-save to guarantee 16kHz mono
        try:
            y, sr = sf.read(output_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sf.write(output_path, y.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        except Exception:
            pass


def generate_dataset(
    base_dir: str = "data",
    num_real: int = 15,
    num_fake: int = 15,
) -> None:
    """Generate training/testing dataset and demo scenarios."""
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    samples_dir = os.path.join(base_dir, "samples")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Generating {num_real} genuine and {num_fake} fake clips for training...")

    # Real training clips
    for i in range(num_real):
        path = os.path.join(real_dir, f"real_spk_{(i % 3) + 1}_{i+1:03d}.wav")
        duration = random.uniform(3.0, 5.0)
        audio = synthesize_fallback_speech(f"Real speech clip {i}", is_fake=False, duration_sec=duration)
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")

    # Fake training clips
    for i in range(num_fake):
        path = os.path.join(fake_dir, f"fake_model_{(i % 3) + 1}_{i+1:03d}.wav")
        duration = random.uniform(3.0, 5.0)
        audio = synthesize_fallback_speech(f"Fake synthetic speech {i}", is_fake=True, duration_sec=duration)
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")

    print("Generating standard demo scenarios in data/samples/...")

    # 1. Genuine casual scenario
    sample_genuine_path = os.path.join(samples_dir, "genuine_casual.wav")
    text_genuine = "Hey, just checking in to see if we are still meeting for lunch tomorrow. Let me know when you are free."
    create_scenario_sample(text_genuine, sample_genuine_path, is_fake=False, scenario_type="casual")
    with open(sample_genuine_path.replace(".wav", "_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(text_genuine)

    # 2. AI Neutral scenario
    sample_ai_neutral_path = os.path.join(samples_dir, "ai_neutral.wav")
    text_neutral = "Welcome to today's audio technology seminar. Please silence your mobile devices and take your seats."
    create_scenario_sample(text_neutral, sample_ai_neutral_path, is_fake=True, scenario_type="neutral")
    with open(sample_ai_neutral_path.replace(".wav", "_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(text_neutral)

    # 3. AI Urgent Scam scenario
    sample_ai_scam_path = os.path.join(samples_dir, "ai_urgent_scam.wav")
    text_scam = (
        "Dad, it is me! I had a terrible car accident and my phone broke. "
        "I need you to wire 50,000 rupees immediately to this UPI ID before the police arrest me. "
        "Please hurry, don't tell anyone and don't call mom!"
    )
    create_scenario_sample(text_scam, sample_ai_scam_path, is_fake=True, scenario_type="scam")
    with open(sample_ai_scam_path.replace(".wav", "_transcript.txt"), "w", encoding="utf-8") as f:
        f.write(text_scam)

    print("Demo dataset generation complete.")


if __name__ == "__main__":
    generate_dataset()
