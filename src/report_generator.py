"""Diagnostic Report Generator for VOXGUARD.

Generates standalone, self-contained HTML audit reports of audio deepfake
and scam forensic analyses for security teams, bank investigations, and users.
"""

from __future__ import annotations

import datetime
import html
from typing import Any, Dict, List, Optional


def generate_html_report(
    audio_filename: str,
    inference_result: Dict[str, Any],
    scam_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> str:
    """Generate self-contained HTML report with styling and security recommendations."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = fusion_result.get("verdict_title", "ANALYSIS COMPLETE")
    level = fusion_result.get("risk_level", "SAFE")
    badge_color = fusion_result.get("badge_color", "#16a34a")
    risk_score = fusion_result.get("risk_score", 0.0)
    voice_fake_pct = fusion_result.get("voice_fake_pct", 0.0)
    scam_score = scam_result.get("overall_scam_score", 0.0)
    duration = inference_result.get("duration_sec", 0.0)
    intervals = inference_result.get("suspicious_intervals", [])
    transcript = scam_result.get("transcript", "No transcript available.")
    advice_list = fusion_result.get("advice", [])

    interval_rows = ""
    if intervals:
        for s, e, p in intervals:
            interval_rows += f"""
            <tr>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{s:.2f}s - {e:.2f}s</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{(e - s):.2f}s</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #ef4444; font-weight: bold;">{p*100:.1f}%</td>
            </tr>
            """
    else:
        interval_rows = """
        <tr>
            <td colspan="3" style="padding: 12px; text-align: center; color: #10b981;">No suspicious synthetic segments detected</td>
        </tr>
        """

    advice_items = "".join([f"<li style='margin-bottom: 6px;'>{html.escape(item)}</li>" for item in advice_list])

    categories = scam_result.get("categories", {})
    category_cards = ""
    for cat_id, cat_info in categories.items():
        matches = cat_info.get("matches", [])
        match_pills = " ".join([f"<span class='pill'>{html.escape(m)}</span>" for m in matches]) if matches else "<span style='color: #64748b;'>None</span>"
        category_cards += f"""
        <div style="background: #1e293b; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {cat_info['color']};">
            <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 4px;">
                <span>{html.escape(cat_info['title'])}</span>
                <span style="color: {cat_info['color']};">{cat_info['score']:.0f}%</span>
            </div>
            <div style="font-size: 13px; color: #94a3b8;">Matched Phrases: {match_pills}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VOXGUARD Security Audit Report - {html.escape(audio_filename)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            line-height: 1.5;
            padding: 30px;
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 2px solid #334155;
            padding-bottom: 20px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            color: #fff;
            background: {badge_color};
        }}
        .metric-card {{
            background: #1e293b;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 25px;
        }}
        .section {{
            background: #1e293b;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }}
        .pill {{
            background: #334155;
            color: #f1f5f9;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #0f172a;
            padding: 10px 12px;
            text-align: left;
            color: #94a3b8;
            font-size: 13px;
        }}
        .disclaimer {{
            font-size: 12px;
            color: #64748b;
            border-top: 1px solid #334155;
            padding-top: 15px;
            margin-top: 30px;
            text-align: center;
        }}
        @media print {{
            body {{ background: #fff; color: #000; }}
            .section, .metric-card {{ border: 1px solid #ccc; background: #fafafa; color: #000; }}
            th {{ background: #eee; color: #000; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin: 0; font-size: 26px; color: #38bdf8;">VOXGUARD Security Audit</h1>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">Real-Time AI Voice Deepfake & Scam Risk Analysis</div>
        </div>
        <div>
            <span class="badge">{level}</span>
        </div>
    </div>

    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 20px;">
        <strong>Target File:</strong> {html.escape(audio_filename)} &nbsp;|&nbsp;
        <strong>Duration:</strong> {duration:.2f}s &nbsp;|&nbsp;
        <strong>Audit Generated:</strong> {now_str}
    </div>

    <div class="grid-3">
        <div class="metric-card">
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Composite Risk Score</div>
            <div style="font-size: 38px; font-weight: bold; color: {badge_color}; margin: 8px 0;">{risk_score:.1f}%</div>
            <div style="font-size: 12px; color: #94a3b8;">{level} TIER</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Acoustic Clone Score</div>
            <div style="font-size: 38px; font-weight: bold; color: #38bdf8; margin: 8px 0;">{voice_fake_pct:.1f}%</div>
            <div style="font-size: 12px; color: #94a3b8;">PyTorch CNN Artifacts</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Scam Intent Risk</div>
            <div style="font-size: 38px; font-weight: bold; color: #ec4899; margin: 8px 0;">{scam_score:.1f}%</div>
            <div style="font-size: 12px; color: #94a3b8;">Multi-Factor NLP Scoring</div>
        </div>
    </div>

    <div class="section" style="border-left: 5px solid {badge_color};">
        <h3 style="margin-top: 0; color: #f8fafc;">{html.escape(verdict)}</h3>
        <p style="color: #cbd5e1; margin-bottom: 12px;">{html.escape(fusion_result.get('verdict_summary', ''))}</p>
        <h4 style="margin-bottom: 8px; color: #f8fafc;">Incident Response Actions:</h4>
        <ul style="color: #cbd5e1; padding-left: 20px;">
            {advice_items}
        </ul>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #38bdf8;">Suspicious Audio Time Windows</h3>
        <table>
            <thead>
                <tr>
                    <th>Timestamp Range</th>
                    <th>Duration</th>
                    <th>Synthetic Fake Probability</th>
                </tr>
            </thead>
            <tbody>
                {interval_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h3 style="margin-top: 0; color: #ec4899;">Conversational Scam Risk Dimensions</h3>
        {category_cards}
        <div style="margin-top: 15px; background: #0f172a; padding: 12px; border-radius: 6px; font-size: 13px;">
            <strong>Analyzed Transcript:</strong><br>
            <div style="color: #cbd5e1; margin-top: 6px; font-style: italic;">"{html.escape(transcript)}"</div>
        </div>
    </div>

    <div class="disclaimer">
        <strong>IMPORTANT COMPLIANCE NOTICE:</strong> VOXGUARD is an AI-assisted decision-support system designed to augment human judgment and fraud response protocols. Acoustic artifacts and conversational signals may vary across recording devices and background conditions. Always perform independent out-of-band verification prior to financial transfers.
    </div>
</body>
</html>
"""
    return html_content
