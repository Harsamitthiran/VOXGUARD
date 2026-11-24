"""Multi-modal fusion engine for VOXGUARD.

Integrates acoustic deepfake probability and contextual scam-risk scoring into
a unified VOXGUARD Risk Score, risk tier (SAFE, SUSPICIOUS, HIGH RISK), and
actionable security advice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class VoxGuardFusion:
    """Fuses acoustic model predictions and NLP scam indicators."""

    # Risk tier thresholds
    SAFE_THRESHOLD = 35.0
    SUSPICIOUS_THRESHOLD = 70.0

    @classmethod
    def fuse(
        cls,
        voice_fake_pct: float,
        context_scam_pct: float = 0.0,
        scam_engine_enabled: bool = True,
        voice_weight: float = 0.60,
        scam_weight: float = 0.40,
    ) -> Dict[str, Any]:
        """Compute unified VOXGUARD Risk Score and decision recommendations.

        Formula:
          If scam_engine_enabled:
            Base = (0.60 * VoiceScore) + (0.40 * ScamScore)
            Synergy Boost = added if both VoiceScore >= 55% and ScamScore >= 40%
            Final = min(100.0, Base + Synergy Boost)
          Else:
            Final = VoiceScore

        Args:
            voice_fake_pct: Acoustic CNN deepfake confidence percentage (0-100).
            context_scam_pct: Text scam NLP score (0-100).
            scam_engine_enabled: Whether contextual scam scoring is factored in.
            voice_weight: Weight given to acoustic classification (default 0.60).
            scam_weight: Weight given to conversational NLP risk (default 0.40).

        Returns:
            Dictionary containing final risk score, level, banner details, and advice.
        """
        voice_fake_pct = float(max(0.0, min(100.0, voice_fake_pct)))
        context_scam_pct = float(max(0.0, min(100.0, context_scam_pct)))

        synergy_boost = 0.0
        if not scam_engine_enabled:
            final_score = voice_fake_pct
            formula_desc = "Voice Acoustic Score only (Context Engine disabled)"
        else:
            base_score = (voice_weight * voice_fake_pct) + (scam_weight * context_scam_pct)

            # Synergy boost: simultaneous acoustic cloning artifact + scam extortion
            if voice_fake_pct >= 55.0 and context_scam_pct >= 40.0:
                synergy_boost = min(18.0, ((voice_fake_pct - 55.0) * (context_scam_pct - 40.0)) / 120.0)
                final_score = min(100.0, base_score + synergy_boost)
                formula_desc = (
                    f"Weighted Fusion: ({voice_weight:.1f} × {voice_fake_pct:.1f}%) + "
                    f"({scam_weight:.1f} × {context_scam_pct:.1f}%) + Synergy Boost (+{synergy_boost:.1f}%)"
                )
            elif voice_fake_pct < 20.0 and context_scam_pct < 20.0:
                final_score = base_score * 0.85  # Safe discount
                formula_desc = (
                    f"Weighted Fusion (Discounted for dual-low indicators): "
                    f"({voice_weight:.1f} × {voice_fake_pct:.1f}%) + ({scam_weight:.1f} × {context_scam_pct:.1f}%)"
                )
            else:
                final_score = base_score
                formula_desc = (
                    f"Weighted Fusion: ({voice_weight:.1f} × {voice_fake_pct:.1f}%) + "
                    f"({scam_weight:.1f} × {context_scam_pct:.1f}%)"
                )

        final_score = float(round(final_score, 1))

        # Categorize into risk tiers
        if final_score < cls.SAFE_THRESHOLD:
            level = "SAFE"
            badge_color = "#16a34a"  # Green
            bg_color = "rgba(22, 163, 74, 0.12)"
            border_color = "#16a34a"
            title = "VERDICT: SAFE / LOW RISK"
            summary = (
                "No strong acoustic cloning artifacts or urgent scam coercion detected. "
                "The audio demonstrates natural human vocal characteristics."
            )
            advice = [
                "Standard vigilance is advised for unfamiliar callers.",
                "No immediate indicators of AI voice impersonation.",
            ]
        elif final_score < cls.SUSPICIOUS_THRESHOLD:
            level = "SUSPICIOUS"
            badge_color = "#ea580c"  # Orange/Amber
            bg_color = "rgba(234, 88, 12, 0.12)"
            border_color = "#ea580c"
            title = "VERDICT: SUSPICIOUS / ELEVATED RISK"
            summary = (
                "Elevated synthetic acoustic artifacts or conversational pressure phrases were detected. "
                "The identity and intent of this speaker should be questioned."
            )
            advice = [
                "Do NOT transfer funds, share verification codes, or make urgent decisions.",
                "Verify identity: disconnect and call the person back on their verified saved number.",
                "Ask a private security question that only your real acquaintance would know.",
            ]
        else:
            level = "HIGH RISK"
            badge_color = "#dc2626"  # Red
            bg_color = "rgba(220, 38, 38, 0.14)"
            border_color = "#dc2626"
            title = "VERDICT: HIGH RISK / POTENTIAL DEEPFAKE SCAM"
            summary = (
                "CRITICAL WARNING: Strong indicators of an AI-cloned voice combined with "
                "predatory scam patterns (urgency, monetary demands, or impersonation)."
            )
            advice = [
                "DO NOT SEND MONEY: Under no circumstances send UPI, crypto, wire, or gift card codes.",
                "DO NOT SHARE OTPS OR PINS: Banks and police never demand OTPs over the phone.",
                "DISCONNECT THE CALL IMMEDIATELY.",
                "CALL DIRECTLY ON A KNOWN NUMBER: Do not use the number this caller dialed from.",
                "ALERT FAMILY OR IT SECURITY: If they claimed an emergency, verify with other relatives first.",
            ]

        return {
            "risk_score": final_score,
            "risk_level": level,
            "voice_fake_pct": voice_fake_pct,
            "context_scam_pct": context_scam_pct,
            "synergy_boost": float(round(synergy_boost, 1)),
            "formula_description": formula_desc,
            "badge_color": badge_color,
            "bg_color": bg_color,
            "border_color": border_color,
            "verdict_title": title,
            "verdict_summary": summary,
            "advice": advice,
        }
