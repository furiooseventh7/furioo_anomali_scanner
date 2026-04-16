"""
Decision Engine — sintesis semua faktor → keputusan LONG/SHORT/BUY SPOT
beserta Entry, SL, TP dan Reasoning Log lengkap.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from config import DEFAULT_SL_PCT, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT, DEFAULT_TP3_PCT

logger = logging.getLogger(__name__)

SignalType = Literal["LONG", "SHORT", "BUY SPOT", "WATCH", "NEUTRAL"]

@dataclass
class FinalSignal:
    symbol              : str
    signal_type         : SignalType
    confidence_score    : float       # 0–100
    alert_level         : str         # LOW / MEDIUM / HIGH / CRITICAL
    price               : float
    entry_zone_low      : float
    entry_zone_high     : float
    stop_loss           : float
    tp1                 : float
    tp2                 : float
    tp3                 : float
    risk_reward         : float
    reasoning_log       : list        # Daftar alasan terstruktur
    whale_score         : float
    derivatives_score   : float
    supply_score        : float
    pre_pump_score      : float
    price_change_24h    : float
    volume_24h          : float
    volume_spike        : float
    funding_rate        : Optional[float]
    oi_change           : Optional[float]
    market_cap          : Optional[float]
    supply_category     : str
    selling_pressure    : str
    short_squeeze_risk  : str
    is_accumulating     : bool
    extra_context       : dict = field(default_factory=dict)

def make_decision(
    symbol        : str,
    price         : float,
    price_chg_24h : float,
    volume_24h    : float,
    whale_res,
    deriv_res,
    supply_res,
    prepump_res,
    fear_greed    : dict,
) -> FinalSignal:
    """
    Sintesis semua analisis → FinalSignal dengan reasoning log lengkap.
    """

    # ── Hitung Total Confluence Score ────────────────────
    raw_score = (
        whale_res.score    +   # max 25
        deriv_res.score    +   # max 30
        supply_res.score   +   # max 20
        prepump_res.score      # max 25
    )   # Total max = 100

    # Market sentiment modifier
    fg_val = fear_greed.get("value", 50)
    if fg_val <= 25:      # Extreme Fear → potensi bottom (good for longs)
        raw_score += 5
    elif fg_val >= 75:    # Extreme Greed → pasar terlalu panas
        raw_score -= 3

    score = min(raw_score, 100.0)

    # ── Tentukan Signal Type ──────────────────────────────
    fr   = deriv_res.latest_funding_rate
    oi_c = deriv_res.oi_change_24h_pct

    # SHORT signal: funding sangat positif + OI turun + sell pressure tinggi
    if (fr >= 0.002 and oi_c < -20 and whale_res.sell_pressure > 0.65
            and price_chg_24h > 20):
        signal_type = "SHORT"
    # BUY SPOT: tidak ada di futures atau micro cap
    elif not deriv_res.is_in_futures or supply_res.category == "MICRO":
        signal_type = "BUY SPOT"
    # LONG: ada di futures, kondisi bullish
    elif score >= 40 and whale_res.buy_pressure >= 0.55:
        signal_type = "LONG"
    elif score >= 25:
        signal_type = "WATCH"
    else:
        signal_type = "NEUTRAL"

    # ── Alert Level ───────────────────────────────────────
    if score >= 78:
        alert_level = "CRITICAL"
    elif score >= 58:
        alert_level = "HIGH"
    elif score >= 38:
        alert_level = "MEDIUM"
    else:
        alert_level = "LOW"

    # ── Risk Management Calculation ───────────────────────
    sl_pct  = DEFAULT_SL_PCT
    tp1_pct = DEFAULT_TP1_PCT
    tp2_pct = DEFAULT_TP2_PCT
    tp3_pct = DEFAULT_TP3_PCT

    # Adjust SL berdasarkan volatility
    if prepump_res.volatility_contraction:
        sl_pct = 0.04    # tight SL saat volatility rendah
    if supply_res.category in ["MICRO", "SMALL"]:
        sl_pct  = 0.07   # wider SL untuk small cap
        tp3_pct = 0.80   # target lebih tinggi untuk micro/small cap

    if signal_type == "LONG":
        stop_loss = price * (1 - sl_pct)
        tp1 = price * (1 + tp1_pct)
        tp2 = price * (1 + tp2_pct)
        tp3 = price * (1 + tp3_pct)
        entry_low  = price * 0.995
        entry_high = price * 1.005
    elif signal_type == "SHORT":
        stop_loss = price * (1 + sl_pct)
        tp1 = price * (1 - tp1_pct)
        tp2 = price * (1 - tp2_pct)
        tp3 = price * (1 - tp3_pct)
        entry_low  = price * 0.995
        entry_high = price * 1.005
    else:  # BUY SPOT / WATCH
        stop_loss = price * (1 - sl_pct)
        tp1 = price * (1 + tp1_pct)
        tp2 = price * (1 + tp2_pct)
        tp3 = price * (1 + tp3_pct)
        entry_low  = price * 0.98   # sedikit lebih lebar untuk spot
        entry_high = price * 1.01

    risk   = abs(price - stop_loss)
    reward = abs(tp2 - price)
    rr     = reward / risk if risk > 0 else 0

    # ── Bangun Reasoning Log ─────────────────────────────
    reasoning = []

    # Whale sonar
    if whale_res.signals:
        reasoning.extend(whale_res.signals)

    # Derivatives
    if deriv_res.signals:
        reasoning.extend(deriv_res.signals)

    # Supply
    if supply_res.signals:
        reasoning.extend(supply_res.signals)

    # Pre-pump
    if prepump_res.signals:
        reasoning.extend(prepump_res.signals)

    # Market context
    fg_label = fear_greed.get("label", "Neutral")
    reasoning.append(f"😱 Fear & Greed: {fg_val}/100 ({fg_label})")

    if price_chg_24h > 15:
        reasoning.append(f"⚠️ Harga sudah naik {price_chg_24h:.1f}% 24H — jangan FOMO!")
    elif price_chg_24h < -10:
        reasoning.append(f"📉 Koreksi {price_chg_24h:.1f}% 24H — potensi entry di area diskon")

    return FinalSignal(
        symbol              = symbol,
        signal_type         = signal_type,
        confidence_score    = score,
        alert_level         = alert_level,
        price               = price,
        entry_zone_low      = entry_low,
        entry_zone_high     = entry_high,
        stop_loss           = stop_loss,
        tp1                 = tp1,
        tp2                 = tp2,
        tp3                 = tp3,
        risk_reward         = rr,
        reasoning_log       = reasoning,
        whale_score         = whale_res.score,
        derivatives_score   = deriv_res.score,
        supply_score        = supply_res.score,
        pre_pump_score      = prepump_res.score,
        price_change_24h    = price_chg_24h,
        volume_24h          = volume_24h,
        volume_spike        = prepump_res.volume_spike_ratio,
        funding_rate        = deriv_res.latest_funding_rate if deriv_res.is_in_futures else None,
        oi_change           = deriv_res.oi_change_24h_pct  if deriv_res.is_in_futures else None,
        market_cap          = supply_res.market_cap,
        supply_category     = supply_res.category,
        selling_pressure    = supply_res.selling_pressure,
        short_squeeze_risk  = deriv_res.short_squeeze_risk,
        is_accumulating     = whale_res.is_accumulating,
    )
