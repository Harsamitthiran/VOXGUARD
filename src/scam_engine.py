"""Contextual Scam-Risk Engine for VOXGUARD.

Transcribes audio using Whisper / faster-whisper (with robust graceful fallbacks)
and executes rule-based NLP scoring across four critical scam dimensions:
1. Financial requests
2. Urgency and pressure
3. Impersonation claims
4. Secrecy and emotional manipulation
"""

from __future__ import annotations

import html
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# Define regex patterns and weights for the 4 scam categories
SCAM_CATEGORIES_CONFIG = {
    "financial_requests": {
        "title": "Financial Demand / Transfer",
        "weight": 0.30,
        "color": "#ef4444",  # Red
        "patterns": [
            r"\b(transfer|wire|upi|otp|pin|cvv)\b",
            r"\b(bank\s+account|gift\s+card|crypto|bitcoin|ethereum|usdt)\b",
            r"\b(deposit|send\s+money|send\s+cash|credit\s+card|debit\s+card)\b",
            r"\b(google\s*pay|phonepe|paytm|cash\s*app|venmo|zelle|paypal)\b",
            r"\b(\d+[\d,]*\s*(rupees|dollars|inr|usd|bucks|lakhs|thousand))\b",
            r"\b(wire\s+\d+|pay\s+\d+|send\s+\d+)\b",
        ],
    },
    "urgency_and_pressure": {
        "title": "Urgency & Intimidation",
        "weight": 0.25,
        "color": "#f97316",  # Orange
        "patterns": [
            r"\b(immediately|right\s+now|right\s+away|urgent|urgently|hurry)\b",
            r"\b(within\s+\d+\s*(minutes|hours)|fast|asap|no\s+time)\b",
            r"\b(police|arrest|jail|custody|warrant|frozen|blocked|suspended)\b",
            r"\b(without\s+delay|last\s+chance|matter\s+of\s+life|emergency)\b",
            r"\b(before\s+it['’]?s\s+too\s+late)\b",
        ],
    },
    "impersonation_claims": {
        "title": "Identity Impersonation",
        "weight": 0.25,
        "color": "#8b5cf6",  # Purple
        "patterns": [
            r"\b(i\s+am\s+(your\s+)?(son|daughter|grandson|granddaughter|father|mother|dad|mom|nephew|niece|cousin))\b",
            r"\b(it\s+is\s+me|it['’]s\s+me)\b",
            r"\b(your\s+(boss|manager|director|ceo|colleague|supervisor))\b",
            r"\b(bank\s+officer|branch\s+manager|customer\s+care|fraud\s+department)\b",
            r"\b(police\s+officer|customs\s+officer|tax\s+officer|cbi\s+officer|fbi|irs|detective)\b",
            r"\b(it\s+support|tech\s+support|helpdesk|microsoft\s+support|apple\s+support)\b",
        ],
    },
    "secrecy_and_manipulation": {
        "title": "Secrecy & Coercion",
        "weight": 0.20,
        "color": "#ec4899",  # Pink
        "patterns": [
            r"\b(don['’]?t\s+tell\s+(anyone|your\s+mom|your\s+dad|family|police|bank))\b",
            r"\b(keep\s+(this\s+)?(secret|confidential|quiet)|between\s+us)\b",
            r"\b(stay\s+on\s+the\s+line|do\s+not\s+hang\s+up|don['’]?t\s+disconnect)\b",
            r"\b(trust\s+me|believe\s+me|please\s+help\s+me|desperate)\b",
            r"\b(in\s+danger|terrible\s+accident|accident|hospitalized|kidnapped|trouble)\b",
            r"\b(phone\s+broke|lost\s+my\s+phone|calling\s+from\s+a\s+stranger['’]?s\s+phone)\b",
        ],
    },
}


class ScamRiskEngine:
    """Performs speech transcription and multi-dimensional scam keyword risk analysis."""

    def __init__(self, whisper_model_size: str = "base"):
        self.whisper_model_size = whisper_model_size
        self._whisper_model = None
        self._whisper_initialized = False

    def _get_whisper_model(self):
        """Lazy loader for Whisper model with graceful fallback."""
        if self._whisper_initialized:
            return self._whisper_model

        try:
            import whisper
            print(f"Loading Whisper '{self.whisper_model_size}' model for speech transcription...")
            self._whisper_model = whisper.load_model(self.whisper_model_size)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Whisper initialization note: {e}. Falling back to accompanying transcript or rule engine.")
            self._whisper_model = None

        self._whisper_initialized = True
        return self._whisper_model

    def transcribe_audio(
        self,
        audio_path_or_bytes: Any,
        provided_transcript: Optional[str] = None,
    ) -> str:
        """Transcribe audio using Whisper, reading sidecar file, or using provided text."""
        if provided_transcript and provided_transcript.strip():
            return provided_transcript.strip()

        # Check for sidecar transcript file if path is string
        if isinstance(audio_path_or_bytes, str):
            sidecar = os.path.splitext(audio_path_or_bytes)[0] + "_transcript.txt"
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass

        # Attempt Whisper transcription
        whisper_model = self._get_whisper_model()
        if whisper_model is not None and isinstance(audio_path_or_bytes, str) and os.path.exists(audio_path_or_bytes):
            try:
                result = whisper_model.transcribe(audio_path_or_bytes, fp16=False)
                return result.get("text", "").strip()
            except Exception as e:
                print(f"Whisper inference error: {e}")

        # Fallback if no transcript is detected
        return "[Audio loaded. No automatic speech transcript generated. You may enter transcript text manually below.]"

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text transcript for scam risk indicators across 4 key dimensions."""
        if not text or text.startswith("[Audio loaded"):
            return {
                "transcript": text,
                "overall_scam_score": 0.0,
                "categories": {
                    k: {
                        "title": v["title"],
                        "score": 0.0,
                        "weight": v["weight"],
                        "color": v["color"],
                        "matches": [],
                    }
                    for k, v in SCAM_CATEGORIES_CONFIG.items()
                },
                "matched_phrases_count": 0,
                "compound_bonus": 0.0,
                "highlighted_html": html.escape(text),
            }

        categories_result: Dict[str, Any] = {}
        total_weighted_score = 0.0
        active_categories = 0
        all_matches_flat: List[str] = []
        highlight_spans: List[Tuple[int, int, str, str]] = []  # (start, end, category, color)

        text_lower = text.lower()

        for cat_id, cfg in SCAM_CATEGORIES_CONFIG.items():
            cat_matches: List[str] = []

            for pattern in cfg["patterns"]:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matched_str = match.group(0)
                    if matched_str.lower() not in [m.lower() for m in cat_matches]:
                        cat_matches.append(matched_str)
                    highlight_spans.append((match.start(), match.end(), cat_id, cfg["color"]))

            # Scoring: base score calculated from distinct matches + frequency
            num_unique = len(cat_matches)
            if num_unique > 0:
                active_categories += 1
                cat_score = min(100.0, num_unique * 35.0 + (num_unique - 1) * 15.0)
            else:
                cat_score = 0.0

            total_weighted_score += cat_score * cfg["weight"]
            all_matches_flat.extend(cat_matches)

            categories_result[cat_id] = {
                "title": cfg["title"],
                "score": float(cat_score),
                "weight": cfg["weight"],
                "color": cfg["color"],
                "matches": cat_matches,
            }

        # Compound synergy bonus for simultaneous triggers
        # (e.g. Impersonation + Urgency + Financial is the classic signature of an emergency clone scam)
        compound_bonus = 0.0
        fin_score = categories_result["financial_requests"]["score"]
        urg_score = categories_result["urgency_and_pressure"]["score"]
        imp_score = categories_result["impersonation_claims"]["score"]
        sec_score = categories_result["secrecy_and_manipulation"]["score"]

        if fin_score > 0 and urg_score > 0:
            compound_bonus += 15.0
        if imp_score > 0 and (fin_score > 0 or urg_score > 0):
            compound_bonus += 10.0
        if active_categories >= 3:
            compound_bonus += 15.0

        overall_score = min(100.0, total_weighted_score + compound_bonus)

        # Generate HTML with highlighted text
        highlighted_html = self._generate_highlighted_html(text, highlight_spans)

        return {
            "transcript": text,
            "overall_scam_score": float(round(overall_score, 1)),
            "categories": categories_result,
            "matched_phrases_count": len(all_matches_flat),
            "compound_bonus": float(round(compound_bonus, 1)),
            "highlighted_html": highlighted_html,
        }

    def _generate_highlighted_html(
        self,
        text: str,
        spans: List[Tuple[int, int, str, str]],
    ) -> str:
        """Generate safe HTML with styled highlight spans for matched scam keywords."""
        if not spans:
            return html.escape(text)

        # Sort spans by start offset, remove overlapping spans
        spans.sort(key=lambda s: (s[0], -s[1]))
        non_overlapping = []
        cur_end = -1
        for start, end, cat, color in spans:
            if start >= cur_end:
                non_overlapping.append((start, end, cat, color))
                cur_end = end

        out: List[str] = []
        last_idx = 0
        for start, end, cat, color in non_overlapping:
            out.append(html.escape(text[last_idx:start]))
            fragment = html.escape(text[start:end])
            out.append(
                f'<mark style="background-color: {color}33; color: {color}; '
                f'font-weight: 600; padding: 2px 4px; border-radius: 4px; border: 1px solid {color}88;" '
                f'title="{cat}">{fragment}</mark>'
            )
            last_idx = end

        out.append(html.escape(text[last_idx:]))
        return "".join(out)
