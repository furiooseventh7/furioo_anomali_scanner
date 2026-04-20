"""
Decision Engine — CMI-ASS
Sintesis semua faktor (Whale + Derivatives + Supply + PrePump + TechnicalAnalysis)
→ Keputusan LONG / SHORT / BUY SPOT dengan Reasoning Log lengkap.
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
    ta_score            : float        # ← BARU: Technical Analysis score
    quant_score         : float = 0.0  # ← Quant Engine (151 Strategies)
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
    # ── TA Summary ────────────────────────────────────────
    ta_bias             : str          # ← BARU: STRONG_BULL/BULL/NEUTRAL/BEAR/STRONG_BEAR
    rsi_14              : float = 50.0 # ← BARU
    macd_signal_type    : str   = "NEUTRAL"  # ← BARU
    trend_alignment     : str   = "SIDEWAYS" # ← BARU
    bb_squeeze          : bool  = False      # ← BARU
    dominant_pattern    : str   = ""         # ← BARU
    nearest_support     : float = 0.0        # ← BARU
    nearest_resist      : float = 0.0        # ← BARU
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
    fear_greed    : dict,
    ta_res        = None,
    narr_res      = None,   # ← NarrativeResult
    quant_res     = None,  # ← QuantResult dari quant_engine (151 Strategies) (optional)
) -> FinalSignal:
    """
    Sintesis semua analisis → FinalSignal dengan reasoning log lengkap.
    ta_res adalah TechnicalResult dari engine/technical_engine.py
    """

    # ── 1. Hitung Total Confluence Score ─────────────────
    raw_score = (
        whale_res.score    +   # max 25
        deriv_res.score    +   # max 30
        supply_res.score   +   # max 20
        prepump_res.score      # max 25
    )   # Subtotal max = 100

    # Tambahkan TA score jika ada (max 30)
    ta_score_val = 0.0
    if ta_res is not None:
        ta_score_val = ta_res.score
        raw_score += ta_score_val   # Total max = 130 → normalize ke 100

    # Tambahkan Quant score jika ada (max 40, dari 151 Trading Strategies)
    quant_score_val = 0.0
    if quant_res is not None:
        quant_score_val = quant_res.score
        raw_score += quant_score_val  # Total max = 170 jika semua ada

    # Normalize ke 100
    # Base: 100, + TA(30) + Quant(40) = 170 max
    denom = 100.0
    if ta_res is not None:   denom += 30.0
    if quant_res is not None: denom += 40.0
    raw_score = raw_score * (100.0 / denom)

    # Market sentiment modifier (Fear & Greed)
    fg_val = fear_greed.get("value", 50)
    if fg_val <= 25:
        raw_score += 5
    elif fg_val >= 75:
        raw_score -= 3

    # Narrative Hype modifier — narasi sedang hype = coin lebih relevan
    narr_boost = 0.0
    if narr_res is not None:
        hs = narr_res.hype_score
        if hs >= 75:   narr_boost =  6.0
        elif hs >= 55: narr_boost =  3.0
        elif hs >= 40: narr_boost =  1.0
        elif hs <= 15: narr_boost = -2.0
        raw_score += narr_boost

    # TA Bias modifier — jika TA dan on-chain/derivatives sinkron, boost score
    if ta_res is not None:
        if ta_res.ta_bias in ("STRONG_BULL", "BULL") and whale_res.is_accumulating:
            raw_score += 5    # Sinkronisasi TA + smart money = sinyal kuat
        elif ta_res.ta_bias in ("STRONG_BEAR", "BEAR") and whale_res.sell_pressure > 0.65:
            raw_score -= 5

    score = min(max(raw_score, 0.0), 100.0)

    # ── 2. Tentukan Signal Type ───────────────────────────
    fr   = deriv_res.latest_funding_rate
    oi_c = deriv_res.oi_change_24h_pct

    # TA bias + is_gated (v3 confluence gate)
    ta_bullish    = ta_res is not None and ta_res.ta_bias in ("STRONG_BULL", "BULL")
    ta_bearish    = ta_res is not None and ta_res.ta_bias in ("STRONG_BEAR", "BEAR")
    ta_gated      = ta_res is not None and getattr(ta_res, "is_gated", False)
    ta_mtf_strong = ta_res is not None and getattr(ta_res, "mtf_bias", "") in ("STRONG_BULLISH", "BULLISH")

    # Quant confluence signals (dari 151 Trading Strategies)
    quant_bullish     = quant_res is not None and quant_res.score >= 18.0
    quant_strong_bull = quant_res is not None and quant_res.score >= 28.0
    quant_dual_pass   = quant_res is not None and quant_res.dual_mom_pass
    quant_trend_up    = quant_res is not None and quant_res.trend_signal > 0.2
    quant_ma3_bull    = quant_res is not None and quant_res.ma3_bias == "BULL"
    quant_ibs_cheap   = quant_res is not None and quant_res.ibs_value < 0.30
    quant_channel_buy = quant_res is not None and quant_res.channel_score >= 3.0
    quant_pivot_bull  = quant_res is not None and quant_res.pivot_score >= 2.0
    quant_bearish     = quant_res is not None and quant_res.score < 5.0 and quant_res.trend_signal < -0.2

    # Composite bullish check: TA + Quant saling mendukung
    tech_quant_aligned = (ta_bullish and quant_bullish) or (ta_bullish and quant_dual_pass) or \
                         (quant_strong_bull and quant_dual_pass)

    # SHORT signal
    if (fr >= 0.002 and oi_c < -20 and whale_res.sell_pressure > 0.65
            and price_chg_24h > 20 and (ta_bearish or ta_res is None)
            and not quant_bullish):  # quant bullish mencegah false short
        signal_type = "SHORT"

    # BUY SPOT: tidak ada di futures atau micro cap
    elif (not deriv_res.is_in_futures or supply_res.category == "MICRO"):
        if ta_bullish or ta_gated or score >= 40 or (quant_bullish and quant_dual_pass):
            signal_type = "BUY SPOT"
        elif quant_ibs_cheap and quant_channel_buy and score >= 32:
            # IBS oversold + Donchian floor = high-prob reversal entry
            signal_type = "BUY SPOT"
        else:
            signal_type = "WATCH"

    # LONG: TA + Quant alignment (tertinggi winrate)
    elif score >= 40 and whale_res.buy_pressure >= 0.50 and tech_quant_aligned:
        signal_type = "LONG"

    # LONG: TA bullish tanpa quant (sama seperti sebelumnya)
    elif score >= 40 and whale_res.buy_pressure >= 0.50 and ta_bullish:
        signal_type = "LONG"

    # LONG: Quant sangat kuat (dual momentum + trend) meski TA netral
    elif score >= 45 and whale_res.buy_pressure >= 0.50 and quant_strong_bull and quant_dual_pass:
        signal_type = "LONG"

    # LONG: score tinggi
    elif score >= 60 and whale_res.buy_pressure >= 0.60 and (ta_bullish or ta_res is None):
        signal_type = "LONG"

    elif score >= 25:
        signal_type = "WATCH"
    else:
        signal_type = "NEUTRAL"

    # ── 3. Alert Level ────────────────────────────────────
    if score >= 78:
        alert_level = "CRITICAL"
    elif score >= 58:
        alert_level = "HIGH"
    elif score >= 38:
        alert_level = "MEDIUM"
    else:
        alert_level = "LOW"

    # Upgrade alert level jika TA + Quant sangat selaras
    if ta_res is not None and ta_res.ta_bias == "STRONG_BULL" and score >= 55:
        if alert_level == "MEDIUM":
            alert_level = "HIGH"
    if tech_quant_aligned and quant_strong_bull and score >= 60:
        if alert_level == "HIGH":
            alert_level = "CRITICAL"
        elif alert_level == "MEDIUM":
            alert_level = "HIGH"

    # ── 4. Risk Management ────────────────────────────────
    sl_pct  = DEFAULT_SL_PCT
    tp1_pct = DEFAULT_TP1_PCT
    tp2_pct = DEFAULT_TP2_PCT
    tp3_pct = DEFAULT_TP3_PCT

    # v3: Gunakan structure SL dan optimal entry dari technical_engine v3
    has_precise = ta_res is not None and getattr(ta_res, "has_precise_entry", False)
    struct_sl   = ta_res.structure_sl if ta_res is not None else 0.0
    opt_low     = ta_res.optimal_entry_low  if ta_res is not None else 0.0
    opt_high    = ta_res.optimal_entry_high if ta_res is not None else 0.0

    # SL berdasarkan swing structure (lebih presisi dari % flat)
    if struct_sl > 0 and signal_type in ("LONG", "BUY SPOT"):
        struct_sl_pct = (price - struct_sl) / price
        if 0.005 < struct_sl_pct < 0.12:
            sl_pct = struct_sl_pct

    # Fallback: gunakan support level dari TA
    elif ta_res is not None and ta_res.nearest_support > 0 and signal_type in ("LONG", "BUY SPOT"):
        support_based_sl = (price - ta_res.nearest_support) / price
        if 0.01 < support_based_sl < 0.08:
            sl_pct = support_based_sl + 0.005

    # Quant: gunakan Pivot S1 sebagai SL jika lebih dekat dan valid (§3.14)
    if quant_res is not None and quant_res.pivot_s1 > 0 and signal_type in ("LONG", "BUY SPOT"):
        pivot_sl_pct = (price - quant_res.pivot_s1) / price
        if 0.005 < pivot_sl_pct < 0.08:
            # Pilih SL yang lebih ketat (lebih kecil % = lebih dekat ke harga)
            if pivot_sl_pct < sl_pct:
                sl_pct = pivot_sl_pct

    # Quant: gunakan Donchian lower sebagai SL fallback (§3.15)
    if quant_res is not None and quant_res.donchian_lower > 0 and signal_type in ("LONG", "BUY SPOT") \
            and struct_sl <= 0:
        don_sl_pct = (price - quant_res.donchian_lower) / price
        if 0.005 < don_sl_pct < 0.10:
            if don_sl_pct < sl_pct:
                sl_pct = don_sl_pct

    # TP1 berdasarkan nearest resistance dari TA
    if ta_res is not None and ta_res.nearest_resist > 0 and signal_type in ("LONG", "BUY SPOT"):
        resist_based_tp1 = (ta_res.nearest_resist - price) / price
        if 0.03 < resist_based_tp1 < 0.25:
            tp1_pct = resist_based_tp1 * 0.95

    # Quant: gunakan Pivot R1 sebagai TP1 jika lebih dekat dan valid (§3.14)
    if quant_res is not None and quant_res.pivot_r1 > 0 and signal_type in ("LONG", "BUY SPOT"):
        pivot_tp1_pct = (quant_res.pivot_r1 - price) / price
        if 0.01 < pivot_tp1_pct < 0.20:
            # Gunakan Pivot R1 jika memberikan TP1 yang lebih konservatif (lebih aman)
            if pivot_tp1_pct < tp1_pct or tp1_pct < 0.02:
                tp1_pct = pivot_tp1_pct * 0.95

    # Quant: gunakan Donchian upper sebagai TP2 (§3.15)
    if quant_res is not None and quant_res.donchian_upper > 0 and signal_type in ("LONG", "BUY SPOT"):
        don_tp2_pct = (quant_res.donchian_upper - price) / price
        if 0.05 < don_tp2_pct < 0.40:
            tp2_pct = don_tp2_pct * 0.95

    # Kualitas sinyal: jika quant kuat, tightkan SL & perlebar TP (confidence tinggi)
    if quant_strong_bull and quant_dual_pass and signal_type in ("LONG", "BUY SPOT"):
        sl_pct  = sl_pct * 0.90   # tightkan SL 10% — lebih percaya diri
        tp3_pct = tp3_pct * 1.20  # perlebar TP3 20% — potensi lebih besar

    # Kualitas sinyal rendah: quant bearish = perlebar SL sebagai proteksi
    if quant_bearish and signal_type in ("LONG", "BUY SPOT"):
        sl_pct = sl_pct * 1.15    # lebarkan SL 15%

    if prepump_res.volatility_contraction:
        sl_pct = min(sl_pct, 0.04)
    if supply_res.category in ("MICRO", "SMALL"):
        sl_pct  = max(sl_pct, 0.07)
        tp3_pct = 0.80

    if signal_type in ("LONG", "BUY SPOT", "WATCH"):
        stop_loss  = price * (1 - sl_pct) if struct_sl <= 0 else struct_sl
        tp1        = price * (1 + tp1_pct)
        tp2        = price * (1 + tp2_pct)
        tp3        = price * (1 + tp3_pct)
        # Entry zone: prioritas SMC (FVG/OB) → Quant Pivot S/R → default
        if has_precise and opt_low > 0 and opt_high > 0:
            entry_low  = opt_low
            entry_high = opt_high
        elif quant_res is not None and quant_res.pivot_s1 > 0 and quant_res.pivot_center > 0 \
                and quant_res.pivot_s1 < price < quant_res.pivot_center:
            # Harga di antara S1 dan pivot center = ideal entry zone (§3.14)
            entry_low  = quant_res.pivot_s1 * 1.002
            entry_high = quant_res.pivot_center * 0.998
        else:
            entry_low  = price * 0.995
            entry_high = price * 1.005
    elif signal_type == "SHORT":
        stop_loss  = price * (1 + sl_pct)
        tp1        = price * (1 - tp1_pct)
        tp2        = price * (1 - tp2_pct)
        tp3        = price * (1 - tp3_pct)
        entry_low  = price * 0.995
        entry_high = price * 1.005
    else:
        stop_loss  = price * (1 - sl_pct)
        tp1        = price * (1 + tp1_pct)
        tp2        = price * (1 + tp2_pct)
        tp3        = price * (1 + tp3_pct)
        entry_low  = price * 0.98
        entry_high = price * 1.01

    risk   = abs(price - stop_loss)
    reward = abs(tp2 - price)
    rr     = reward / risk if risk > 0 else 0

    # ── 5. Bangun Reasoning Log ───────────────────────────
    reasoning = []

    # Whale signals
    if whale_res.signals:
        reasoning.extend(whale_res.signals)

    # Derivatives signals
    if deriv_res.signals:
        reasoning.extend(deriv_res.signals)

    # Supply signals
    if supply_res.signals:
        reasoning.extend(supply_res.signals)

    # Pre-pump signals
    if prepump_res.signals:
        reasoning.extend(prepump_res.signals)

    # TA signals — BARU, dari technical_engine
    if ta_res is not None and ta_res.signals:
        reasoning.extend(ta_res.signals[:6])  # max 6 sinyal TA

    # Quant signals — dari quant_engine (151 Trading Strategies)
    if quant_res is not None and quant_res.signals:
        reasoning.extend(quant_res.signals[:5])  # max 5 sinyal quant

    # TA Invalidation warnings
    if ta_res is not None and getattr(ta_res, "invalidation", []):
        for inv_msg in ta_res.invalidation[:2]:
            reasoning.append(inv_msg)

    # MTF consensus
    if ta_res is not None:
        mtf = getattr(ta_res, "mtf_bias", "NEUTRAL")
        agree = getattr(ta_res, "mtf_agree_count", 0)
        if agree >= 2:
            reasoning.append(f"🔭 MTF Consensus: {mtf} ({agree}/4 TF setuju)")

    # Market context
    fg_label = fear_greed.get("label", "Neutral")
    reasoning.append(f"😱 Fear & Greed: {fg_val}/100 ({fg_label})")

    if price_chg_24h > 15:
        reasoning.append(f"⚠️ Harga sudah naik {price_chg_24h:.1f}% 24H — jangan FOMO!")
    elif price_chg_24h < -10:
        reasoning.append(f"📉 Koreksi {price_chg_24h:.1f}% 24H — potensi entry di area diskon")

    # Resolusi nilai TA untuk FinalSignal
    ta_bias_val        = ta_res.ta_bias        if ta_res else "NEUTRAL"
    rsi_val            = ta_res.rsi_14         if ta_res else 50.0
    macd_sig_val       = ta_res.macd_signal_type if ta_res else "NEUTRAL"
    trend_align_val    = ta_res.trend_alignment  if ta_res else "SIDEWAYS"
    bb_squeeze_val     = ta_res.bb_squeeze       if ta_res else False
    nearest_sup_val    = ta_res.nearest_support  if ta_res else 0.0
    nearest_res_val    = ta_res.nearest_resist   if ta_res else 0.0
    dominant_pat_val   = ""
    if ta_res and ta_res.dominant_pattern:
        dominant_pat_val = f"{ta_res.dominant_pattern.name} ({ta_res.dominant_pattern.direction})"

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
        ta_score            = ta_score_val,
        quant_score         = quant_score_val,
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
        ta_bias             = ta_bias_val,
        rsi_14              = rsi_val,
        macd_signal_type    = macd_sig_val,
        trend_alignment     = trend_align_val,
        bb_squeeze          = bb_squeeze_val,
        dominant_pattern    = dominant_pat_val,
        nearest_support     = nearest_sup_val,
        nearest_resist      = nearest_res_val,
    )
