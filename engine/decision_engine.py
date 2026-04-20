"""
Decision Engine — CMI-ASS v4
==============================
Sintesis semua faktor:
  Whale + Derivatives + Supply + PrePump  (base 100)
  + Technical Analysis SMC v3             (+max 30)
  + Quant Engine 151 Trading Strategies   (+max 40)
  Normalized → confidence score 0-100

Entry/SL/TP hierarchy (best-first):
  Entry: SMC FVG/OB > Quant Pivot S1-C > Donchian floor > EMA21 > default
  SL:    Structure swing low > Quant S2/DonchianLow > TA support > default%
  TP1:   Quant Pivot R1 > TA resistance > default%
  TP2:   Quant Donchian upper > Quant R2 > default%
  TP3:   Extended R2 projection > default%
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from config import DEFAULT_SL_PCT, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT, DEFAULT_TP3_PCT

logger = logging.getLogger(__name__)
SignalType = Literal["LONG","SHORT","BUY SPOT","WATCH","NEUTRAL"]


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
    ta_score            : float
    quant_score         : float = 0.0
    price_change_24h    : float = 0.0
    volume_24h          : float = 0.0
    volume_spike        : float = 0.0
    funding_rate        : Optional[float] = None
    oi_change           : Optional[float] = None
    market_cap          : Optional[float] = None
    supply_category     : str   = "UNKNOWN"
    selling_pressure    : str   = "UNKNOWN"
    short_squeeze_risk  : str   = "UNKNOWN"
    is_accumulating     : bool  = False
    ta_bias             : str   = "NEUTRAL"
    rsi_14              : float = 50.0
    macd_signal_type    : str   = "NEUTRAL"
    trend_alignment     : str   = "SIDEWAYS"
    bb_squeeze          : bool  = False
    dominant_pattern    : str   = ""
    nearest_support     : float = 0.0
    nearest_resist      : float = 0.0
    extra_context       : dict  = field(default_factory=dict)


def _g(obj, attr, default=0.0):
    """Safe getattr with default."""
    v = getattr(obj, attr, default)
    return v if v is not None else default


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
    quant_res     = None,
    narr_res      = None,
) -> FinalSignal:

    # ── 1. SCORE ─────────────────────────────────────────────────────────────
    raw = whale_res.score + deriv_res.score + supply_res.score + prepump_res.score

    ta_score_val    = _g(ta_res,    "score") if ta_res    is not None else 0.0
    quant_score_val = _g(quant_res, "score") if quant_res is not None else 0.0

    if ta_res    is not None: raw += ta_score_val
    if quant_res is not None: raw += quant_score_val

    denom = 100.0
    if ta_res    is not None: denom += 30.0
    if quant_res is not None: denom += 40.0
    raw = raw * (100.0 / denom)

    fg_val = fear_greed.get("value", 50)
    if fg_val <= 20:   raw += 6
    elif fg_val <= 30: raw += 3
    elif fg_val >= 80: raw -= 4
    elif fg_val >= 70: raw -= 2

    if narr_res is not None:
        hs = narr_res.hype_score
        if hs >= 75:   raw += 6
        elif hs >= 55: raw += 3
        elif hs >= 40: raw += 1
        elif hs <= 15: raw -= 2

    if ta_res is not None:
        ta_bias = _g(ta_res, "ta_bias", "NEUTRAL")
        if ta_bias in ("STRONG_BULL","BULL") and _g(whale_res,"is_accumulating",False): raw += 5
        elif ta_bias in ("STRONG_BEAR","BEAR") and _g(whale_res,"sell_pressure",0) > 0.65: raw -= 5

    if quant_res is not None:
        if _g(quant_res,"dual_mom_pass",False) and _g(quant_res,"trend_eta",0) > 0.3: raw += 4
        if _g(quant_res,"triple_ma_bias","") == "BULL" and ta_res and _g(ta_res,"ta_bias","") in ("STRONG_BULL","BULL"): raw += 3

    score = min(max(raw, 0.0), 100.0)

    # ── 2. SIGNAL TYPE ───────────────────────────────────────────────────────
    fr   = _g(deriv_res, "latest_funding_rate", 0.0)
    oi_c = _g(deriv_res, "oi_change_24h_pct",   0.0)

    ta_bullish  = ta_res is not None and _g(ta_res,"ta_bias","") in ("STRONG_BULL","BULL")
    ta_bearish  = ta_res is not None and _g(ta_res,"ta_bias","") in ("STRONG_BEAR","BEAR")
    ta_gated    = ta_res is not None and _g(ta_res,"is_gated",False)

    q_ok        = quant_res is not None and quant_score_val >= 15.0
    q_strong    = quant_res is not None and quant_score_val >= 25.0
    q_pass      = quant_res is not None and _g(quant_res,"dual_mom_pass",False)
    q_ma_bull   = quant_res is not None and _g(quant_res,"triple_ma_bias","") == "BULL"
    q_bear      = quant_res is not None and quant_score_val < 4.0 and _g(quant_res,"trend_eta",0) < -0.3
    q_ibs_low   = quant_res is not None and _g(quant_res,"ibs_value",0.5) < 0.25
    q_ch_buy    = quant_res is not None and _g(quant_res,"channel_score",0) >= 3.0

    tech_q_ok   = (ta_bullish and q_ok) or (ta_bullish and q_pass) or (q_strong and q_pass and q_ma_bull)

    buy_pressure = _g(whale_res,"buy_pressure",0)
    sell_pressure= _g(whale_res,"sell_pressure",0)
    in_futures   = _g(deriv_res,"is_in_futures",True)
    cat          = _g(supply_res,"category","")

    if (fr >= 0.002 and oi_c < -20 and sell_pressure > 0.65
            and price_chg_24h > 20 and (ta_bearish or ta_res is None) and not q_ok):
        signal_type = "SHORT"

    elif not in_futures or cat == "MICRO":
        if ta_bullish or ta_gated or score >= 40 or (q_ok and q_pass):
            signal_type = "BUY SPOT"
        elif q_ibs_low and q_ch_buy and score >= 32:
            signal_type = "BUY SPOT"
        else:
            signal_type = "WATCH"

    elif score >= 40 and buy_pressure >= 0.50 and tech_q_ok:
        signal_type = "LONG"
    elif score >= 40 and buy_pressure >= 0.50 and ta_bullish:
        signal_type = "LONG"
    elif score >= 45 and buy_pressure >= 0.48 and q_strong and q_pass:
        signal_type = "LONG"
    elif score >= 62 and buy_pressure >= 0.60:
        signal_type = "LONG"
    elif score >= 25:
        signal_type = "WATCH"
    else:
        signal_type = "NEUTRAL"

    # ── 3. ALERT LEVEL ───────────────────────────────────────────────────────
    if score >= 80:   alert_level = "CRITICAL"
    elif score >= 60: alert_level = "HIGH"
    elif score >= 40: alert_level = "MEDIUM"
    else:             alert_level = "LOW"

    if ta_res and _g(ta_res,"ta_bias","") == "STRONG_BULL" and score >= 55:
        if alert_level == "MEDIUM": alert_level = "HIGH"
    if tech_q_ok and q_strong and score >= 62:
        if alert_level == "HIGH":   alert_level = "CRITICAL"
        elif alert_level == "MEDIUM": alert_level = "HIGH"

    # ── 4. RISK MANAGEMENT ───────────────────────────────────────────────────
    sl_pct  = DEFAULT_SL_PCT
    tp1_pct = DEFAULT_TP1_PCT
    tp2_pct = DEFAULT_TP2_PCT
    tp3_pct = DEFAULT_TP3_PCT

    has_smc  = ta_res is not None and _g(ta_res,"has_precise_entry",False)
    struct_sl= _g(ta_res,"structure_sl",0.0) if ta_res else 0.0
    opt_low  = _g(ta_res,"optimal_entry_low",0.0)  if ta_res else 0.0
    opt_high = _g(ta_res,"optimal_entry_high",0.0) if ta_res else 0.0

    q_el  = _g(quant_res,"quant_entry_low",0.0)  if quant_res else 0.0
    q_eh  = _g(quant_res,"quant_entry_high",0.0) if quant_res else 0.0
    q_sl  = _g(quant_res,"quant_sl",0.0)         if quant_res else 0.0
    q_tp1 = _g(quant_res,"quant_tp1",0.0)        if quant_res else 0.0
    q_tp2 = _g(quant_res,"quant_tp2",0.0)        if quant_res else 0.0
    q_tp3 = _g(quant_res,"quant_tp3",0.0)        if quant_res else 0.0
    q_has = _g(quant_res,"has_quant_levels",False)if quant_res else False

    if signal_type in ("LONG","BUY SPOT"):
        # SL — hierarchy
        if struct_sl > 0:
            p = (price - struct_sl) / price
            if 0.005 < p < 0.12: sl_pct = p
        elif ta_res and _g(ta_res,"nearest_support",0) > 0:
            p = (price - ta_res.nearest_support) / price
            if 0.008 < p < 0.10: sl_pct = p + 0.005

        if q_sl > 0 and q_sl < price:
            p = (price - q_sl) / price
            if 0.005 < p < 0.12 and p < sl_pct: sl_pct = p

        if q_strong and q_pass: sl_pct *= 0.88   # tighter SL on high conviction
        elif q_bear:             sl_pct *= 1.18   # wider on signal conflict

        # TP — hierarchy
        if q_tp1 > price:
            p = (q_tp1 - price) / price
            if 0.01 < p < 0.30: tp1_pct = p * 0.98
        elif ta_res and _g(ta_res,"nearest_resist",0) > price:
            p = (ta_res.nearest_resist - price) / price
            if 0.025 < p < 0.28: tp1_pct = p * 0.95

        if q_tp2 > price:
            p = (q_tp2 - price) / price
            if 0.04 < p < 0.45: tp2_pct = p * 0.97

        if q_tp3 > price:
            p = (q_tp3 - price) / price
            if 0.08 < p < 0.80: tp3_pct = p

        if q_strong and q_pass and _g(quant_res,"confluence_count",0) >= 7:
            tp3_pct *= 1.25

        if _g(prepump_res,"volatility_contraction",False): sl_pct = min(sl_pct,0.04)
        if cat in ("MICRO","SMALL"): sl_pct = max(sl_pct,0.07); tp3_pct = max(tp3_pct,0.80)

        stop_loss = price*(1-sl_pct) if struct_sl<=0 else struct_sl
        tp1 = price*(1+tp1_pct); tp2 = price*(1+tp2_pct); tp3 = price*(1+tp3_pct)

        # Entry — hierarchy
        if has_smc and opt_low>0 and opt_high>0:
            entry_low,entry_high = opt_low, opt_high
        elif q_has and q_el>0 and q_eh>0:
            entry_low,entry_high = q_el, q_eh
        else:
            e21 = _g(quant_res,"ema21",0.0) if quant_res else 0.0
            if e21>0 and abs(price-e21)/price<0.04:
                entry_low,entry_high = e21*0.997, e21*1.010
            else:
                entry_low,entry_high = price*0.995, price*1.005

    elif signal_type == "SHORT":
        stop_loss = price*(1+sl_pct)
        tp1=price*(1-tp1_pct); tp2=price*(1-tp2_pct); tp3=price*(1-tp3_pct)
        q_r2=_g(quant_res,"pivot_r2",0.0) if quant_res else 0.0
        q_s1=_g(quant_res,"pivot_s1",0.0) if quant_res else 0.0
        q_s2=_g(quant_res,"pivot_s2",0.0) if quant_res else 0.0
        if q_r2>price: stop_loss=q_r2*1.003
        if q_s1>0 and q_s1<price: tp1=q_s1*1.002
        if q_s2>0 and q_s2<price: tp2=q_s2*1.002
        entry_low=price*0.995; entry_high=price*1.005
    else:
        stop_loss=price*(1-sl_pct)
        tp1=price*(1+tp1_pct); tp2=price*(1+tp2_pct); tp3=price*(1+tp3_pct)
        entry_low=price*0.98; entry_high=price*1.01

    risk=abs(price-stop_loss); reward=abs(tp2-price)
    rr=reward/risk if risk>1e-10 else 0.0

    # ── 5. REASONING LOG ─────────────────────────────────────────────────────
    reasoning = []
    for src in [whale_res,deriv_res,supply_res,prepump_res]:
        if getattr(src,"signals",None): reasoning.extend(src.signals)
    if ta_res and getattr(ta_res,"signals",None):   reasoning.extend(ta_res.signals[:6])
    if ta_res and getattr(ta_res,"invalidation",None): reasoning.extend(ta_res.invalidation[:2])
    if ta_res:
        mtf=_g(ta_res,"mtf_bias","NEUTRAL"); agree=int(_g(ta_res,"mtf_agree_count",0))
        if agree>=2: reasoning.append(f"🔭 MTF Consensus: {mtf} ({agree}/4 TF setuju)")
    if quant_res and quant_res.signals: reasoning.extend(quant_res.signals[:6])
    reasoning.append(f"😱 Fear & Greed: {fg_val}/100 ({fear_greed.get('label','Neutral')})")
    if price_chg_24h>15: reasoning.append(f"⚠️ Naik {price_chg_24h:.1f}% 24H — jangan FOMO!")
    elif price_chg_24h<-10: reasoning.append(f"📉 Koreksi {price_chg_24h:.1f}% 24H — area diskon")

    ta_bias_val = _g(ta_res,"ta_bias","NEUTRAL")     if ta_res else "NEUTRAL"
    dom_pat = ""
    if ta_res and getattr(ta_res,"dominant_pattern",None):
        dom_pat = f"{ta_res.dominant_pattern.name} ({ta_res.dominant_pattern.direction})"

    return FinalSignal(
        symbol=symbol, signal_type=signal_type, confidence_score=round(score,2),
        alert_level=alert_level, price=price,
        entry_zone_low=entry_low, entry_zone_high=entry_high,
        stop_loss=stop_loss, tp1=tp1, tp2=tp2, tp3=tp3, risk_reward=round(rr,2),
        reasoning_log=reasoning,
        whale_score=whale_res.score, derivatives_score=deriv_res.score,
        supply_score=supply_res.score, pre_pump_score=prepump_res.score,
        ta_score=ta_score_val, quant_score=quant_score_val,
        price_change_24h=price_chg_24h, volume_24h=volume_24h,
        volume_spike=_g(prepump_res,"volume_spike_ratio",0.0),
        funding_rate=_g(deriv_res,"latest_funding_rate") if _g(deriv_res,"is_in_futures",False) else None,
        oi_change=_g(deriv_res,"oi_change_24h_pct")      if _g(deriv_res,"is_in_futures",False) else None,
        market_cap=_g(supply_res,"market_cap"),
        supply_category=_g(supply_res,"category","UNKNOWN"),
        selling_pressure=_g(supply_res,"selling_pressure","UNKNOWN"),
        short_squeeze_risk=_g(deriv_res,"short_squeeze_risk","UNKNOWN"),
        is_accumulating=bool(_g(whale_res,"is_accumulating",False)),
        ta_bias=ta_bias_val, rsi_14=_g(ta_res,"rsi_14",50.0) if ta_res else 50.0,
        macd_signal_type=_g(ta_res,"macd_signal_type","NEUTRAL") if ta_res else "NEUTRAL",
        trend_alignment=_g(ta_res,"trend_alignment","SIDEWAYS")  if ta_res else "SIDEWAYS",
        bb_squeeze=bool(_g(ta_res,"bb_squeeze",False))           if ta_res else False,
        dominant_pattern=dom_pat,
        nearest_support=_g(ta_res,"nearest_support",0.0) if ta_res else 0.0,
        nearest_resist=_g(ta_res,"nearest_resist",0.0)   if ta_res else 0.0,
    )
