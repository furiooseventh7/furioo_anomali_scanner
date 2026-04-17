"""
Derivatives Engine — analisis Funding Rate, Open Interest,
Long/Short Ratio untuk mendeteksi potensi squeeze.
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from engine.data_fetcher import (
    get_funding_rate_history, get_open_interest_hist, get_long_short_ratio
)

logger = logging.getLogger(__name__)

@dataclass
class DerivativesResult:
    is_in_futures        : bool  = False
    latest_funding_rate  : float = 0.0
    funding_trend        : str   = "NEUTRAL"   # RISING / FALLING / NEUTRAL
    funding_reversal     : bool  = False        # negatif → positif (bullish signal)
    oi_change_24h_pct    : float = 0.0
    oi_change_trend      : str   = "FLAT"       # RISING / FALLING / FLAT
    long_short_ratio     : float = 1.0
    ls_trend             : str   = "NEUTRAL"    # SHORT_DOMINATED / LONG_DOMINATED
    short_squeeze_risk   : str   = "LOW"        # LOW / MEDIUM / HIGH / EXTREME
    long_squeeze_risk    : str   = "LOW"
    signals              : list  = field(default_factory=list)
    score                : float = 0.0          # kontribusi 0–30

def analyze_derivatives(symbol: str, futures_symbols: set) -> DerivativesResult:
    res = DerivativesResult()
    if symbol not in futures_symbols:
        return res

    res.is_in_futures = True

    # ── 1. Funding Rate Analysis ──────────────────────────
    fr_history = get_funding_rate_history(symbol, limit=10)
    if fr_history:
        res.latest_funding_rate = fr_history[-1]
        prev_fr = fr_history[-2] if len(fr_history) > 1 else fr_history[-1]

        # Trend funding rate
        if res.latest_funding_rate > prev_fr * 1.05:
            res.funding_trend = "RISING"
        elif res.latest_funding_rate < prev_fr * 0.95:
            res.funding_trend = "FALLING"

        # Reversal: funding negatif → positif (katalis squeeze)
        if prev_fr < 0 and res.latest_funding_rate > 0:
            res.funding_reversal = True

        # Rata-rata 5 funding terakhir
        avg_5fr = np.mean(fr_history[-5:]) if len(fr_history) >= 5 else res.latest_funding_rate

        # Short squeeze: funding sangat negatif = banyak yang short, squeeze potential
        if avg_5fr <= -0.002:
            res.short_squeeze_risk = "EXTREME"
        elif avg_5fr <= -0.001:
            res.short_squeeze_risk = "HIGH"
        elif avg_5fr <= -0.0003:
            res.short_squeeze_risk = "MEDIUM"

        # Long squeeze: funding sangat positif = banyak yang long, danger
        if avg_5fr >= 0.003:
            res.long_squeeze_risk = "HIGH"
        elif avg_5fr >= 0.001:
            res.long_squeeze_risk = "MEDIUM"

    # ── 2. Open Interest Analysis ─────────────────────────
    df_oi = get_open_interest_hist(symbol, period="1h", limit=24)
    if not df_oi.empty:
        oi_now = df_oi["sumOpenInterestValue"].iloc[-1]
        oi_24h = df_oi["sumOpenInterestValue"].iloc[0]
        if oi_24h > 0:
            res.oi_change_24h_pct = (oi_now - oi_24h) / oi_24h * 100

        # Trend OI 4 jam terakhir
        oi_4h_ago = df_oi["sumOpenInterestValue"].iloc[-4]
        if oi_now > oi_4h_ago * 1.05:
            res.oi_change_trend = "RISING"
        elif oi_now < oi_4h_ago * 0.95:
            res.oi_change_trend = "FALLING"

    # ── 3. Long/Short Ratio ───────────────────────────────
    df_ls = get_long_short_ratio(symbol, period="1h", limit=5)
    if not df_ls.empty:
        res.long_short_ratio = df_ls["longShortRatio"].iloc[-1]
        short_pct = df_ls["shortAccount"].iloc[-1]

        if short_pct > 0.60:
            res.ls_trend = "SHORT_DOMINATED"
        elif short_pct < 0.40:
            res.ls_trend = "LONG_DOMINATED"

    # ── 4. Generate Signals ───────────────────────────────
    score = 0.0
    fr = res.latest_funding_rate

    # Funding rate signals
    if fr <= -0.002:
        res.signals.append(f"💥 Funding Rate EKSTREM NEGATIF {fr*100:.4f}% → SHORT SQUEEZE IMMINENT!")
        score += 30
    elif fr <= -0.001:
        res.signals.append(f"🔥 Funding Rate sangat negatif {fr*100:.4f}% → SHORT SQUEEZE HIGH")
        score += 22
    elif fr <= -0.0003:
        res.signals.append(f"⚡ Funding Rate negatif {fr*100:.4f}% → Potensi squeeze")
        score += 15
    elif -0.0003 < fr < 0.0003:
        res.signals.append(f"🔵 Funding Rate netral {fr*100:.4f}% (fresh position)")
        score += 5
    elif fr >= 0.003:
        res.signals.append(f"⚠️ Funding Rate sangat tinggi {fr*100:.4f}% (Overbought, HATI-HATI)")
        score -= 5

    if res.funding_reversal:
        res.signals.append("🔄 FUNDING REVERSAL: Negatif → Positif (BULLISH catalyst!)")
        score += 10

    # OI signals
    if res.oi_change_24h_pct >= 60:
        res.signals.append(f"🐋 OI MELEDAK {res.oi_change_24h_pct:.1f}% dalam 24H! Smart money masuk!")
        score += 20
    elif res.oi_change_24h_pct >= 30:
        res.signals.append(f"📈 OI naik signifikan {res.oi_change_24h_pct:.1f}% dalam 24H")
        score += 12
    elif res.oi_change_24h_pct >= 15:
        res.signals.append(f"📊 OI naik moderat {res.oi_change_24h_pct:.1f}%")
        score += 6
    elif res.oi_change_24h_pct <= -30:
        res.signals.append(f"📉 OI turun drastis {res.oi_change_24h_pct:.1f}% (mass liquidation)")
        score -= 8

    # Long/Short ratio
    if res.ls_trend == "SHORT_DOMINATED":
        res.signals.append(f"🎯 Short dominated {res.long_short_ratio:.2f} L/S ratio (fertile for squeeze)")
        score += 8
    elif res.ls_trend == "LONG_DOMINATED":
        res.signals.append(f"⚠️ Long dominated {res.long_short_ratio:.2f} L/S ratio (long squeeze risk)")
        score -= 3

    res.score = max(0.0, min(score, 30.0))
    return res
