"""
Decision Engine — CMI-ASS (GOD-TIER SMC OPTIMIZED)
Sintesis semua faktor (Whale + Derivatives + Supply + PrePump + Technical Analysis)
→ Keputusan LONG / SHORT / BUY SPOT dengan area Entry Limit (FVG/OB) & Reasoning Log.
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
    # ── Sub-scores ────────────────────────────────────────
    whale_score         : float
    derivatives_score   : float
    supply_score        : float
    pre_pump_score      : float
    ta_score            : float        
    # ── Market Info ───────────────────────────────────────
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
    ta_bias             : str
    rsi_14              : float
    macd_signal_type    : str


def make_decision(
    symbol: str,
    price: float,
    price_chg_24h: float,
    volume_24h: float,
    whale_res,
    deriv_res,
    supply_res,
    prepump_res,
    ta_res
) -> Optional[FinalSignal]:
    
    if price <= 0:
        return None

    # ── 1. Kalkulasi Total Score ──────────────────────────────
    score = 0.0
    reasoning = []

    score += getattr(whale_res, 'score', 0)
    score += getattr(deriv_res, 'score', 0)
    score += getattr(supply_res, 'score', 0)
    score += getattr(prepump_res, 'score', 0)

    ta_score_val = 0.0
    ta_bias_val = "UNKNOWN"
    rsi_val = 50.0
    macd_sig_val = "NONE"

    is_in_golden_zone = False

    if ta_res is not None:
        score += getattr(ta_res, 'score', 0)
        ta_score_val = getattr(ta_res, 'score', 0)
        ta_bias_val = getattr(ta_res, 'ta_bias', 'UNKNOWN')
        rsi_val = getattr(ta_res, 'rsi_14', 50.0)
        macd_sig_val = getattr(ta_res, 'macd_signal_type', 'NONE')
        
        # Deteksi apakah harga ada di area Smart Money
        if (getattr(ta_res, 'price_in_fvg', False) or 
            getattr(ta_res, 'price_in_ob', False) or 
            getattr(ta_res, 'price_at_support', False)):
            is_in_golden_zone = True

    # ── 2. Tentukan Signal Type (Dewa Trader Mode) ───────────
    fr   = getattr(deriv_res, 'latest_funding_rate', 0.0)
    oi_c = getattr(deriv_res, 'oi_change_24h_pct', 0.0)

    ta_bullish = ta_res is not None and ta_bias_val in ("STRONG_BULL", "BULL")
    ta_bearish = ta_res is not None and ta_bias_val in ("STRONG_BEAR", "BEAR")
    
    signal_type = "NEUTRAL"

    # ==============================================================
    # 🏆 GOD-TIER ENTRY (Smart Money + OnChain + FVG/OB Confluence)
    # ==============================================================
    if (getattr(whale_res, 'buy_pressure', 0) >= 0.60 and is_in_golden_zone and oi_c > 10):
        signal_type = "LONG"
        score += 15 # Extra confidence
        reasoning.insert(0, "🔥 GOD-TIER ENTRY: Harga masuk Zona Liquidity (FVG/OB) didukung Whale Buy & OI Spike!")
        
    # SHORT signal
    elif (fr >= 0.002 and oi_c < -20 and getattr(whale_res, 'sell_pressure', 0) > 0.65
            and price_chg_24h > 20 and (ta_bearish or ta_res is None)):
        signal_type = "SHORT"

    # BUY SPOT: Tidak ada di futures atau market cap kecil
    elif (not getattr(deriv_res, 'is_in_futures', False) or getattr(supply_res, 'category', '') == "MICRO"):
        if score >= 35 and (ta_bullish or is_in_golden_zone):
            signal_type = "BUY SPOT"

    # LONG Standard
    elif score >= 40 and getattr(whale_res, 'buy_pressure', 0) >= 0.55 and (ta_bullish or ta_res is None):
        signal_type = "LONG"

    # WATCH: Pantau dulu
    elif score >= 25:
        signal_type = "WATCH"

    # ── 3. Risk Management (SMC Optimized - LIMIT ORDER) ─────
    sl_pct  = DEFAULT_SL_PCT
    tp1_pct = DEFAULT_TP1_PCT
    tp2_pct = DEFAULT_TP2_PCT
    tp3_pct = DEFAULT_TP3_PCT

    # Fallback Default (Jika TA tidak ada, entry di area harga sekarang)
    entry_low  = price * 0.985
    entry_high = price * 1.005

    if ta_res is not None:
        if signal_type in ("LONG", "BUY SPOT", "WATCH"):
            
            # 🎯 1. Prioritaskan Bullish Order Block
            bull_ob = getattr(ta_res, 'nearest_bull_ob', None)
            if bull_ob is not None and getattr(bull_ob, 'top', 0) <= price:
                entry_high = bull_ob.top
                entry_low  = bull_ob.bottom
                sl_pct = abs(price - (entry_low * 0.99)) / price
                reasoning.insert(0, f"🎯 LIMIT ENTRY di Bullish Order Block: ${entry_low:.5f} - ${entry_high:.5f}")
                
            # 🎯 2. Jika tidak ada OB, antri di Bullish FVG
            elif getattr(ta_res, 'nearest_bull_fvg', None) is not None and ta_res.nearest_bull_fvg.top <= price:
                entry_high = ta_res.nearest_bull_fvg.top
                entry_low  = ta_res.nearest_bull_fvg.bottom
                sl_pct = abs(price - (entry_low * 0.99)) / price
                reasoning.insert(0, f"🎯 LIMIT ENTRY di Bullish FVG: ${entry_low:.5f} - ${entry_high:.5f}")
                
            # 🎯 3. Jika tidak ada FVG/OB, gunakan nearest support
            elif getattr(ta_res, 'nearest_support', 0) > 0:
                support_based_sl = (price - ta_res.nearest_support) / price
                if 0.01 < support_based_sl < 0.08:
                    sl_pct = support_based_sl + 0.01
                    entry_low = ta_res.nearest_support
                    entry_high = ta_res.nearest_support * 1.015

            # TP presisi menggunakan nearest resistance
            if getattr(ta_res, 'nearest_resist', 0) > 0:
                resist_based_tp1 = (ta_res.nearest_resist - price) / price
                if 0.03 < resist_based_tp1 < 0.20:
                    tp1_pct = resist_based_tp1 * 0.95

        elif signal_type == "SHORT":
            # 🎯 1. Prioritaskan Bearish Order Block
            bear_ob = getattr(ta_res, 'nearest_bear_ob', None)
            if bear_ob is not None and getattr(bear_ob, 'bottom', float('inf')) >= price:
                entry_low  = bear_ob.bottom
                entry_high = bear_ob.top
                sl_pct = abs((entry_high * 1.01) - price) / price
                reasoning.insert(0, f"🎯 LIMIT SHORT di Bearish Order Block: ${entry_low:.5f} - ${entry_high:.5f}")

            # 🎯 2. Jika tidak ada OB, antri di Bearish FVG
            elif getattr(ta_res, 'nearest_bear_fvg', None) is not None and ta_res.nearest_bear_fvg.bottom >= price:
                entry_low  = ta_res.nearest_bear_fvg.bottom
                entry_high = ta_res.nearest_bear_fvg.top
                sl_pct = abs((entry_high * 1.01) - price) / price
                reasoning.insert(0, f"🎯 LIMIT SHORT di Bearish FVG: ${entry_low:.5f} - ${entry_high:.5f}")

    # 🔒 Keamanan: Batasi Max SL 15% dan Min SL 1.5%
    sl_pct = max(min(sl_pct, 0.15), 0.015)

    # Modifier tambahan
    if getattr(prepump_res, 'volatility_contraction', False):
        sl_pct = min(sl_pct, 0.05)
    if getattr(supply_res, 'category', '') in ("MICRO", "SMALL"):
        sl_pct  = max(sl_pct, 0.08)
        tp3_pct = 0.80

    # Kalkulasi nilai uang/harga (Absolut)
    if signal_type in ("LONG", "BUY SPOT", "WATCH", "NEUTRAL"):
        stop_loss  = price * (1 - sl_pct)
        tp1        = price * (1 + tp1_pct)
        tp2        = price * (1 + tp2_pct)
        tp3        = price * (1 + tp3_pct)
    elif signal_type == "SHORT":
        stop_loss  = price * (1 + sl_pct)
        tp1        = price * (1 - tp1_pct)
        tp2        = price * (1 - tp2_pct)
        tp3        = price * (1 - tp3_pct)

    risk   = abs(price - stop_loss)
    avg_entry = (entry_low + entry_high) / 2
    reward = abs(tp2 - avg_entry) 
    rr     = reward / risk if risk > 0 else 0

    # ── 4. Bangun Reasoning Log ─────────────────────────────
    # Gabungkan semua sinyal dari masing-masing engine
    if getattr(whale_res, 'signals', None): reasoning.extend(whale_res.signals)
    if getattr(deriv_res, 'signals', None): reasoning.extend(deriv_res.signals)
    if getattr(supply_res, 'signals', None): reasoning.extend(supply_res.signals)
    if getattr(prepump_res, 'signals', None): reasoning.extend(prepump_res.signals)
    if getattr(ta_res, 'signals', None): reasoning.extend(ta_res.signals)

    # ── 5. Tentukan Alert Level ─────────────────────────────
    alert_level = "INFO"
    if score >= 50 and signal_type in ("LONG", "SHORT", "BUY SPOT"):
        alert_level = "URGENT"
    elif score >= 35 and signal_type != "NEUTRAL":
        alert_level = "WARNING"

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
        whale_score         = getattr(whale_res, 'score', 0),
        derivatives_score   = getattr(deriv_res, 'score', 0),
        supply_score        = getattr(supply_res, 'score', 0),
        pre_pump_score      = getattr(prepump_res, 'score', 0),
        ta_score            = ta_score_val,
        price_change_24h    = price_chg_24h,
        volume_24h          = volume_24h,
        volume_spike        = getattr(prepump_res, 'volume_spike_ratio', 0),
        funding_rate        = fr if getattr(deriv_res, 'is_in_futures', False) else None,
        oi_change           = oi_c if getattr(deriv_res, 'is_in_futures', False) else None,
        market_cap          = getattr(supply_res, 'market_cap', 0),
        supply_category     = getattr(supply_res, 'category', 'UNKNOWN'),
        selling_pressure    = getattr(supply_res, 'selling_pressure', 'UNKNOWN'),
        short_squeeze_risk  = getattr(deriv_res, 'short_squeeze_risk', 'LOW'),
        is_accumulating     = getattr(whale_res, 'is_accumulating', False),
        ta_bias             = ta_bias_val,
        rsi_14              = rsi_val,
        macd_signal_type    = macd_sig_val,
    )
