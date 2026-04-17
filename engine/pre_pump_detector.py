"""
Pre-Pump Detector — identifikasi kondisi 'matang' sebelum pump besar.
Berdasarkan pattern historis $RAVE dan coin-coin serupa.
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from engine.data_fetcher import get_klines

logger = logging.getLogger(__name__)

@dataclass
class PrePumpResult:
    volume_spike_ratio    : float = 1.0    # vs avg 7D
    volume_zscore         : float = 0.0
    is_consolidating      : bool  = False  # range sempit 7 candle
    consolidation_range   : float = 0.0   # % range
    is_breakout           : bool  = False  # breakout dari konsolidasi
    breakout_pct          : float = 0.0
    is_low_vol_accum      : bool  = False  # volume rendah + harga stabil (stealth accum)
    price_momentum_1h     : float = 0.0   # % change 1H terakhir
    price_momentum_4h     : float = 0.0   # % change 4H terakhir
    volatility_contraction: bool  = False  # ATR mengecil (spring-loaded)
    signals               : list  = field(default_factory=list)
    score                 : float = 0.0    # kontribusi 0–25

def analyze_pre_pump(symbol: str) -> PrePumpResult:
    res = PrePumpResult()

    # ── Data ──────────────────────────────────────────────
    df_1h = get_klines(symbol, interval="1h", limit=168)   # 7 hari
    df_4h = get_klines(symbol, interval="4h", limit=42)    # 7 hari
    df_15m = get_klines(symbol, interval="15m", limit=48)  # 12 jam

    if df_1h.empty or len(df_1h) < 48:
        return res

    # ── 1. Volume Spike ───────────────────────────────────
    recent_vol  = df_1h["volume"].iloc[-1]
    avg_vol_7d  = df_1h["volume"].iloc[:-1].mean()
    std_vol_7d  = df_1h["volume"].iloc[:-1].std()

    res.volume_spike_ratio = recent_vol / avg_vol_7d if avg_vol_7d > 0 else 1.0
    res.volume_zscore = (recent_vol - avg_vol_7d) / std_vol_7d if std_vol_7d > 0 else 0

    # ── 2. Konsolidasi (4H timeframe) ────────────────────
    if not df_4h.empty and len(df_4h) >= 8:
        recent_4h = df_4h.iloc[-8:-1]   # 7 candle terakhir (ex current)
        high_max  = recent_4h["high"].max()
        low_min   = recent_4h["low"].min()
        mid_price = recent_4h["close"].mean()

        res.consolidation_range = (high_max - low_min) / mid_price if mid_price > 0 else 0

        if res.consolidation_range < 0.08:   # range < 8%
            res.is_consolidating = True

        # Breakout dari konsolidasi
        current_close = df_4h["close"].iloc[-1]
        if current_close > high_max * 1.02:
            res.is_breakout = True
            res.breakout_pct = (current_close / high_max - 1) * 100

    # ── 3. Stealth Accumulation (volume rendah + harga stabil) ─
    if not df_1h.empty and len(df_1h) >= 24:
        last_24h = df_1h.iloc[-25:-1]
        avg_24h_vol  = last_24h["volume"].mean()
        prev_7d_vol  = df_1h["volume"].iloc[:-25].mean()
        price_std_24h = last_24h["close"].std() / last_24h["close"].mean()

        if avg_24h_vol < prev_7d_vol * 0.6 and price_std_24h < 0.02:
            res.is_low_vol_accum = True

    # ── 4. Price Momentum ─────────────────────────────────
    if not df_1h.empty and len(df_1h) >= 5:
        close_now = df_1h["close"].iloc[-1]
        close_1h  = df_1h["close"].iloc[-2]
        close_4h  = df_1h["close"].iloc[-5]
        res.price_momentum_1h = (close_now / close_1h - 1) * 100
        res.price_momentum_4h = (close_now / close_4h - 1) * 100

    # ── 5. Volatility Contraction (ATR mengecil) ──────────
    if not df_4h.empty and len(df_4h) >= 14:
        atr_recent = (df_4h["high"] - df_4h["low"]).iloc[-5:].mean()
        atr_prev   = (df_4h["high"] - df_4h["low"]).iloc[-14:-5].mean()
        if atr_prev > 0 and atr_recent < atr_prev * 0.60:
            res.volatility_contraction = True

    # ── 6. Generate Signals ───────────────────────────────
    score = 0.0

    if res.volume_spike_ratio >= 10:
        res.signals.append(f"🔥🔥 VOLUME SPIKE {res.volume_spike_ratio:.1f}x vs avg 7D (Z={res.volume_zscore:.1f})")
        score += 25
    elif res.volume_spike_ratio >= 5:
        res.signals.append(f"🔥 Volume spike {res.volume_spike_ratio:.1f}x vs avg 7D")
        score += 18
    elif res.volume_spike_ratio >= 3:
        res.signals.append(f"⚡ Volume naik {res.volume_spike_ratio:.1f}x vs avg 7D")
        score += 12
    elif res.volume_spike_ratio >= 2:
        res.signals.append(f"📊 Volume meningkat {res.volume_spike_ratio:.1f}x")
        score += 6

    if res.is_consolidating:
        res.signals.append(f"📦 Konsolidasi ketat {res.consolidation_range*100:.1f}% selama 7 candle 4H")
        score += 5

    if res.is_breakout:
        res.signals.append(f"🚀 BREAKOUT +{res.breakout_pct:.1f}% dari konsolidasi!")
        score += 8

    if res.is_low_vol_accum:
        res.signals.append("🔕 STEALTH ACCUMULATION: Volume rendah + harga stabil (smart money diam-diam masuk)")
        score += 10

    if res.volatility_contraction:
        res.signals.append("🌀 Volatility contraction (ATR mengecil) → Spring loaded untuk breakout!")
        score += 7

    if res.price_momentum_1h > 3:
        res.signals.append(f"📈 Momentum 1H: +{res.price_momentum_1h:.1f}% (akselerasi)")
        score += 3

    res.score = max(0.0, min(score, 25.0))
    return res
