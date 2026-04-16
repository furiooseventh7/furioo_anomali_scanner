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
    confidence_score    : float
    alert_level         : str
    price               : float
    entry_zone_low      : float
    entry_zone_high     : float
    stop_loss           : float
    tp1                 : float
    tp2                 : float
    tp3                 : float
    risk_reward         : float
    reasoning_log       : list
    whale_score         : float
    derivatives_score   : float
    supply_score        : float
    pre_pump_score      : float
    technical_score     : float
    structure_score     : float
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
    rsi_1h              : float = 50.0
    rsi_4h              : float = 50.0
    macd_signal         : str   = "NEUTRAL"
    bb_position         : str   = "MID"
    ema_trend           : str   = "NEUTRAL"
    support_level       : float = 0.0
    resistance_level    : float = 0.0
    bullish_fvg         : bool  = False
    bullish_ob          : bool  = False
    ob_level            : float = 0.0
    double_bottom       : bool  = False
    bull_flag           : bool  = False
    extra_context       : dict  = field(default_factory=dict)


def make_decision(
    symbol        : str,
    price         : float,
    price_chg_24h : float,
    volume_24h    : float,
    whale_res,
    deriv_res,
    supply_res,
    prepump_res,
    tech_res,
    struct_res,
    fear_greed    : dict,
) -> FinalSignal:

    # ── Hitung Total Confluence Score ────────────────────
    # Semua engine tetap ada, ditambah 2 engine baru
    raw_score = (
        whale_res.score    +   # max 25
        deriv_res.score    +   # max 30
        supply_res.score   +   # max 20
        prepump_res.score  +   # max 25
        tech_res.score     +   # max 30 (BARU)
        struct_res.score       # max 20 (BARU)
    )   # Total max = 150 → normalize ke 100

    raw_score = raw_score / 150 * 100

    # Market sentiment modifier
    fg_val = fear_greed.get("value", 50)
    if fg_val <= 25:
        raw_score += 3
    elif fg_val >= 75:
        raw_score -= 2

    score = min(raw_score, 100.0)

    # ── Tentukan Signal Type ──────────────────────────────
    fr   = deriv_res.latest_funding_rate
    oi_c = deriv_res.oi_change_24h_pct

    # SHORT: funding ekstrem + OI turun + sell pressure + RSI overbought
    if (fr >= 0.002 and oi_c < -20 and whale_res.sell_pressure > 0.65
            and price_chg_24h > 20 and tech_res.rsi_1h >= 70):
        signal_type = "SHORT"
    # BUY SPOT: tidak ada di futures atau micro cap
    elif not deriv_[res.is](http://res.is)_in_futures or supply_res.category == "MICRO":
        if score >= 40:
            signal_type = "BUY SPOT"
        else:
            signal_type = "NEUTRAL"
    # LONG: ada di futures, kondisi bullish, RSI tidak overbought
    elif score >= 40 and whale_[res.buy](http://res.buy)_pressure >= 0.55 and tech_res.rsi_1h < 70:
        signal_type = "LONG"
    elif score >= 28:
        signal_type = "WATCH"
    else:
        signal_type = "NEUTRAL"

    # Konfirmasi teknikal untuk LONG
    if signal_type == "LONG":
        tech_confirms = sum([
            tech_res.rsi_1h <= 45,
            tech_res.macd_signal in ["BULLISH_CROSS", "BULLISH"],
            tech_[res.bb](http://res.bb)_position in ["LOWER", "SQUEEZE"],
            tech_res.ema_trend == "BULLISH",
            struct_res.bullish_fvg,
            struct_res.bullish_ob,
            struct_res.double_bottom,
        ])
        if tech_confirms == 0 and score < 60:
            signal_type = "WATCH"

    # ── Alert Level ───────────────────────────────────────
    if score >= 78:
        alert_level = "CRITICAL"
    elif score >= 58:
        alert_level = "HIGH"
    elif score >= 38:
        alert_level = "MEDIUM"
    else:
        alert_level = "LOW"

    # ── Risk Management ───────────────────────────────────
    sl_pct  = DEFAULT_SL_PCT
    tp1_pct = DEFAULT_TP1_PCT
    tp2_pct = DEFAULT_TP2_PCT
    tp3_pct = DEFAULT_TP3_PCT

    if prep
