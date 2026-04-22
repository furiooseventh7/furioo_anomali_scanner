"""
Quant Engine — CMI-ASS v4
=========================
Implementasi 10 strategi kuantitatif dari:
  "151 Trading Strategies" — Kakushadze & Serur (2018, SSRN-3247865)

Strategi yang diimplementasikan (dipilih untuk relevansi crypto spot/futures):

  §3.1  Price Momentum         — Risk-adjusted cumulative return (Eq.266-270)
  §3.4  Low Volatility Anomaly — Low-vol coins outperform high-vol
  §3.7  Residual Momentum      — Alpha murni (strip market beta via OLS)
  §3.13 Three Moving Averages  — Triple EMA confluence filter (Eq.324)
  §3.14 Pivot Point S/R        — C=(H+L+C)/3, R1=2C−L, S1=2C−H (Eq.325-328)
  §3.15 Donchian Channel       — Breakout/reversal via N-period price channel (Eq.329-331)
  §4.4  IBS Mean Reversion     — Internal Bar Strength reversal (Eq.370)
  §4.1.2 Dual Momentum         — Relative + absolute momentum filter (Antonacci)
  §10.4 Trend Following+tanh   — η=tanh(R/κ)/σ smoothed momentum signal (Eq.474-475)
  §10.3.1 Contrarian+Volume    — Mean-reversion weighted by volume change (Eq.470-473)

Output: QuantResult dengan:
  - score 0–40 (ditambahkan ke total confluence score)
  - level_entry, sl, tp1, tp2, tp3 berbasis formula buku
  - signals list untuk reasoning log
"""

import numpy as np
import pandas as pd
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DATACLASS OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantResult:
    # ── Score total ──────────────────────────────────────────────────────────
    score            : float = 0.0   # 0–40

    # ── Sub-scores per strategy ──────────────────────────────────────────────
    momentum_score   : float = 0.0   # §3.1 Price Momentum
    lowvol_score     : float = 0.0   # §3.4 Low Volatility Anomaly
    residual_score   : float = 0.0   # §3.7 Residual Momentum
    triple_ma_score  : float = 0.0   # §3.13 Three Moving Averages
    pivot_score      : float = 0.0   # §3.14 Pivot Point
    channel_score    : float = 0.0   # §3.15 Donchian Channel
    ibs_score        : float = 0.0   # §4.4 IBS Mean Reversion
    dual_mom_score   : float = 0.0   # §4.1.2 Dual Momentum
    trend_score      : float = 0.0   # §10.4 Trend Following
    contrarian_score : float = 0.0   # §10.3.1 Contrarian+Volume

    # ── Key computed values untuk entry/SL/TP ───────────────────────────────
    # Pivot Point (§3.14)
    pivot_c          : float = 0.0
    pivot_r1         : float = 0.0
    pivot_r2         : float = 0.0
    pivot_s1         : float = 0.0
    pivot_s2         : float = 0.0

    # Donchian Channel (§3.15)
    donchian_upper   : float = 0.0
    donchian_lower   : float = 0.0
    donchian_mid     : float = 0.0

    # MA levels
    ema9             : float = 0.0
    ema21            : float = 0.0
    ema50            : float = 0.0
    triple_ma_bias   : str   = "NEUTRAL"

    # Momentum metrics
    risk_adj_return  : float = 0.0   # Sharpe-like (§3.1 Eq.270)
    residual_mom     : float = 0.0   # Alpha strip (§3.7)
    trend_eta        : float = 0.0   # tanh signal (§10.4)
    ibs_value        : float = 0.5   # IBS 0-1 (§4.4)
    dual_mom_pass    : bool  = False

    # ── Entry zone dari Quant ────────────────────────────────────────────────
    quant_entry_low  : float = 0.0
    quant_entry_high : float = 0.0
    quant_sl         : float = 0.0
    quant_tp1        : float = 0.0
    quant_tp2        : float = 0.0
    quant_tp3        : float = 0.0
    has_quant_levels : bool  = False

    # ── Signals ──────────────────────────────────────────────────────────────
    signals          : list  = field(default_factory=list)
    confluence_count : int   = 0   # jumlah strategi yang bullish


# ─────────────────────────────────────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _last(arr: np.ndarray) -> float:
    if arr is None or len(arr) == 0: return 0.0
    v = arr[~np.isnan(arr)]
    return float(v[-1]) if len(v) > 0 else 0.0


def _ema(close: np.ndarray, n: int) -> np.ndarray:
    """Exponential Moving Average — λ=2/(n+1)  (§3.11 Eq.320)"""
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    k = 2.0 / (n + 1)
    out[n - 1] = float(np.mean(close[:n]))
    for i in range(n, len(close)):
        out[i] = close[i] * k + out[i - 1] * (1 - k)
    return out


def _sma(close: np.ndarray, n: int) -> np.ndarray:
    """Simple Moving Average  (§3.11 Eq.319)"""
    out = np.full(len(close), np.nan)
    if len(close) < n: return out
    cs = np.cumsum(close)
    out[n - 1:] = (cs[n - 1:] - np.concatenate([[0], cs[:len(close) - n]])) / n
    return out


def _returns(close: np.ndarray) -> np.ndarray:
    """Log returns"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(close[:-1] > 0, np.log(close[1:] / close[:-1]), 0.0)


def _cum_return(close: np.ndarray, lookback: int) -> float:
    """R_cum = P(t)/P(t+T) - 1  (§3.1 Eq.267)"""
    if len(close) <= lookback or close[-lookback - 1] <= 0: return 0.0
    return float(close[-1] / close[-lookback - 1] - 1.0)


def _risk_adj_return(close: np.ndarray, lookback: int) -> float:
    """R_risk_adj = mean(returns) / std(returns)  (§3.1 Eq.268-270)"""
    n = min(lookback, len(close) - 1)
    if n < 5: return 0.0
    rets = _returns(close[-n - 1:])
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))
    return mu / sigma if sigma > 1e-10 else 0.0


def _rolling_vol(close: np.ndarray, n: int) -> float:
    """Realized volatility over last n bars"""
    if len(close) <= n: return 0.0
    rets = _returns(close[-n - 1:])
    return float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §3.1 — PRICE MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def _strat_momentum(close: np.ndarray) -> Tuple[float, List[str], float]:
    """
    §3.1 Price Momentum (Eq.266-270)
    Risk-adjusted return = mean_return / volatility over formation period.
    Positive → coin has inertia/momentum → score up.

    Returns: (score 0-8, signals, risk_adj_return)
    """
    score, sigs = 0.0, []
    lb = min(48, len(close) - 2)  # 48 candles = ~2 hari H1 formation period
    if lb < 5: return 0.0, sigs, 0.0

    r_adj = _risk_adj_return(close, lb)
    r_cum = _cum_return(close, lb)

    if r_adj > 1.0:
        score = 8.0
        sigs.append(f"📈 [§3.1 Momentum] Risk-adj return sangat kuat ({r_adj:.2f}) — strong inertia")
    elif r_adj > 0.5:
        score = 5.0
        sigs.append(f"📈 [§3.1 Momentum] Risk-adj return positif ({r_adj:.2f})")
    elif r_adj > 0.2:
        score = 2.0
    elif r_adj < -0.8:
        score = -4.0
        sigs.append(f"📉 [§3.1 Momentum] Negative momentum ({r_adj:.2f}) — trend turun")
    elif r_adj < -0.3:
        score = -2.0

    return score, sigs, r_adj


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §3.4 — LOW VOLATILITY ANOMALY
# ─────────────────────────────────────────────────────────────────────────────

def _strat_low_vol(close: np.ndarray) -> Tuple[float, List[str]]:
    """
    §3.4 Low Volatility Anomaly
    Empirically low-vol assets outperform high-vol — counter to naive CAPM.
    Low vol → bullish quality score boost.
    """
    score, sigs = 0.0, []
    if len(close) < 20: return 0.0, sigs

    vol_pct = _rolling_vol(close, 20) * 100  # % per candle
    if vol_pct <= 0: return 0.0, sigs

    if vol_pct < 1.5:
        score = 5.0
        sigs.append(f"🧘 [§3.4 LowVol] Volatilitas rendah ({vol_pct:.1f}%) — low-vol anomaly, quality coin")
    elif vol_pct < 3.0:
        score = 2.5
    elif vol_pct < 5.0:
        score = 0.0
    elif vol_pct > 12.0:
        score = -4.0
        sigs.append(f"⚡ [§3.4 LowVol] Volatilitas ekstrem ({vol_pct:.1f}%) — risiko tinggi")
    elif vol_pct > 7.0:
        score = -1.5

    return score, sigs


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §3.7 — RESIDUAL MOMENTUM (ALPHA)
# ─────────────────────────────────────────────────────────────────────────────

def _strat_residual_momentum(close: np.ndarray, btc_returns: np.ndarray) -> Tuple[float, List[str], float]:
    """
    §3.7 Residual Momentum (Eq.278)
    R_i(t) = α_i + β_i * MKT(t) + ε_i(t)
    Residual ε_i = coin return not explained by BTC (market proxy).
    Positive ε = coin has genuine alpha beyond market.

    BTC returns used as MKT proxy.
    """
    score, sigs = 0.0, []
    n = min(len(close) - 1, len(btc_returns), 24)
    if n < 8: return 0.0, sigs, 0.0

    coin_rets = _returns(close[-n - 1:])
    mkt_rets  = btc_returns[-n:] if len(btc_returns) >= n else btc_returns

    min_len = min(len(coin_rets), len(mkt_rets))
    if min_len < 5: return 0.0, sigs, 0.0

    coin_rets = coin_rets[-min_len:]
    mkt_rets  = mkt_rets[-min_len:]

    # OLS beta: β = cov(coin, mkt) / var(mkt)
    with np.errstate(divide='ignore', invalid='ignore'):
        cov_matrix = np.cov(coin_rets, mkt_rets)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 1e-12 else 1.0
        alpha = float(np.mean(coin_rets)) - beta * float(np.mean(mkt_rets))
        residuals = coin_rets - (alpha + beta * mkt_rets)
        residual_mom = float(np.mean(residuals))

    if residual_mom > 0.003:
        score = 6.0
        sigs.append(f"⚡ [§3.7 ResidualMom] Alpha positif ({residual_mom:.4f}) — genuine outperformance vs BTC")
    elif residual_mom > 0.001:
        score = 3.0
        sigs.append(f"✅ [§3.7 ResidualMom] Residual positif ({residual_mom:.4f})")
    elif residual_mom < -0.003:
        score = -3.0
        sigs.append(f"📉 [§3.7 ResidualMom] Alpha negatif ({residual_mom:.4f}) — underperform market")

    return score, sigs, residual_mom


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §3.13 — THREE MOVING AVERAGES
# ─────────────────────────────────────────────────────────────────────────────

def _strat_triple_ma(close: np.ndarray) -> Tuple[float, List[str], str, float, float, float]:
    """
    §3.13 Three Moving Averages (Eq.324)
    LONG  if MA(T1) > MA(T2) > MA(T3)  — T1=9, T2=21, T3=50
    SHORT if MA(T1) < MA(T2) < MA(T3)

    Returns: (score, signals, bias, ema9, ema21, ema50)
    """
    score, sigs = 0.0, []
    bias = "NEUTRAL"
    T1, T2, T3 = 9, 21, 50

    if len(close) < T3:
        return 0.0, sigs, bias, 0.0, 0.0, 0.0

    e9  = _last(_ema(close, T1))
    e21 = _last(_ema(close, T2))
    e50 = _last(_ema(close, T3))

    if e9 <= 0 or e21 <= 0 or e50 <= 0:
        return 0.0, sigs, bias, e9, e21, e50

    # Perfect bullish alignment (§3.13 Eq.324)
    if e9 > e21 > e50:
        score = 7.0
        bias  = "BULL"
        sigs.append(f"🔺 [§3.13 TripleMA] EMA9({e9:.4f}) > EMA21({e21:.4f}) > EMA50({e50:.4f}) — perfect alignment")
    elif e9 > e21 and close[-1] > e50:
        score = 3.5
        bias  = "BULL"
        sigs.append(f"📈 [§3.13 TripleMA] EMA9>EMA21 & harga>EMA50 — partial bullish")
    elif e9 < e21 < e50:
        score = -5.0
        bias  = "BEAR"
        sigs.append(f"🔻 [§3.13 TripleMA] EMA9 < EMA21 < EMA50 — bearish alignment (§3.13)")
    elif e9 < e21 and close[-1] < e50:
        score = -2.5
        bias  = "BEAR"

    return score, sigs, bias, e9, e21, e50


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §3.14 — PIVOT POINT SUPPORT & RESISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def _strat_pivot(
    price: float, prev_high: float, prev_low: float, prev_close: float
) -> Tuple[float, List[str], float, float, float, float, float]:
    """
    §3.14 Pivot Point (Eq.325-328)
    C  = (PH + PL + PC) / 3
    R1 = 2*C - PL     R2 = C + (PH - PL)
    S1 = 2*C - PH     S2 = C - (PH - PL)

    Signal: Long if P>C, Liquidate at R; Short if P<C, Liquidate at S

    Returns: (score, signals, C, R1, R2, S1, S2)
    """
    score, sigs = 0.0, []
    if prev_high <= 0 or prev_low <= 0 or prev_close <= 0 or price <= 0:
        return 0.0, sigs, 0.0, 0.0, 0.0, 0.0, 0.0

    C  = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2.0 * C - prev_low
    R2 = C + (prev_high - prev_low)
    S1 = 2.0 * C - prev_high
    S2 = C - (prev_high - prev_low)

    dist_to_s1 = (price - S1) / price * 100  # % above S1
    dist_to_r1 = (R1 - price) / price * 100  # % below R1

    if price > C:
        # Above pivot = bullish (§3.14 Eq.328)
        if 0 < dist_to_r1 <= 3.0:
            # Near R1 = approaching resistance
            score = 3.0
            sigs.append(f"🎯 [§3.14 Pivot] Harga di atas C={C:.5f}, mendekati R1={R1:.5f} ({dist_to_r1:.1f}%)")
        elif 0 < dist_to_r1 <= 8.0:
            score = 5.0
            sigs.append(f"✅ [§3.14 Pivot] Harga di atas Pivot Center ({C:.5f}) — bullish bias, R1={R1:.5f}")
        else:
            score = 2.0
    else:
        # Below pivot = bearish
        if abs(dist_to_s1) <= 2.0:
            # Very near S1 = potential support bounce
            score = 2.0
            sigs.append(f"🟡 [§3.14 Pivot] Harga dekat S1={S1:.5f} — watch for bounce")
        else:
            score = -2.0
            sigs.append(f"📉 [§3.14 Pivot] Harga di bawah Pivot Center ({C:.5f}) — bearish bias")

    return score, sigs, C, R1, R2, S1, S2


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §3.15 — DONCHIAN CHANNEL
# ─────────────────────────────────────────────────────────────────────────────

def _strat_donchian(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    price: float, period: int = 20
) -> Tuple[float, List[str], float, float, float]:
    """
    §3.15 Donchian Channel (Eq.329-331)
    B_up   = max(High[1..T])
    B_down = min(Low[1..T])

    Long at floor (mean reversion), Breakout above ceiling = strong trend.

    Returns: (score, signals, upper, lower, mid)
    """
    score, sigs = 0.0, []
    if len(close) < period + 1:
        return 0.0, sigs, 0.0, 0.0, 0.0

    b_up   = float(np.max(high[-period:]))
    b_down = float(np.min(low[-period:]))
    b_mid  = (b_up + b_down) / 2.0

    if b_up <= b_down or b_up <= 0:
        return 0.0, sigs, b_up, b_down, b_mid

    pos = (price - b_down) / (b_up - b_down)  # 0=floor, 1=ceiling
    width_pct = (b_up - b_down) / b_up * 100

    if price > b_up:
        # Breakout — strong bullish trend signal (§3.15)
        score = 8.0
        sigs.append(f"🚀 [§3.15 Donchian] BREAKOUT di atas ceiling {b_up:.5f} — trend momentum kuat!")
    elif pos <= 0.15:
        # Near floor — mean-reversion long setup (§3.15)
        score = 6.0
        sigs.append(f"🟢 [§3.15 Donchian] Harga dekat floor ({b_down:.5f}) — ideal mean-reversion buy zone")
    elif pos <= 0.30:
        score = 3.0
        sigs.append(f"🟢 [§3.15 Donchian] Harga di lower quartile channel — potential long")
    elif pos >= 0.85:
        score = -3.0
        sigs.append(f"🔴 [§3.15 Donchian] Harga dekat ceiling ({b_up:.5f}) — overbought dalam channel")
    elif pos >= 0.65:
        score = 0.0
    else:
        score = 1.0  # mid-channel, slight positive

    return score, sigs, b_up, b_down, b_mid


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §4.4 — IBS MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _strat_ibs(prev_high: float, prev_low: float, prev_close: float) -> Tuple[float, List[str], float]:
    """
    §4.4 Internal Bar Strength (Eq.370)
    IBS = (PC - PL) / (PH - PL)
    IBS≈0 → close near low → bearish candle → potential bounce up
    IBS≈1 → close near high → bullish candle → potential reversal down

    Returns: (score, signals, ibs_value)
    """
    score, sigs = 0.0, []
    if prev_high <= prev_low or prev_high <= 0:
        return 0.0, sigs, 0.5

    ibs = (prev_close - prev_low) / (prev_high - prev_low)

    if ibs <= 0.10:
        score = 5.0
        sigs.append(f"💚 [§4.4 IBS] Sangat oversold ({ibs:.2f}) — candle tutup di near-low, reversal up probable")
    elif ibs <= 0.25:
        score = 3.0
        sigs.append(f"💚 [§4.4 IBS] IBS rendah ({ibs:.2f}) — bearish candle, potensi mean-reversion")
    elif ibs >= 0.90:
        score = -4.0
        sigs.append(f"🔴 [§4.4 IBS] Sangat overbought ({ibs:.2f}) — candle tutup near-high, beware reversal")
    elif ibs >= 0.75:
        score = -2.0
        sigs.append(f"🟡 [§4.4 IBS] IBS tinggi ({ibs:.2f}) — potensi reversal down")

    return score, sigs, float(ibs)


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §4.1.2 — DUAL MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def _strat_dual_momentum(close: np.ndarray, btc_chg_24h: float) -> Tuple[float, List[str], bool]:
    """
    §4.1.2 Dual Momentum (Antonacci via Kakushadze)
    Relative momentum: coin cumulative return > benchmark (BTC)
    Absolute momentum: broad market (BTC) trending up

    Long only if BOTH conditions met (high-quality filter).
    Returns: (score, signals, passed)
    """
    score, sigs = 0.0, []
    if len(close) < 12: return 0.0, sigs, False

    r_med   = _cum_return(close, min(24, len(close) - 1))
    r_short = _cum_return(close, min(6,  len(close) - 1))
    mkt_pos = btc_chg_24h > 0

    if r_med > 0 and r_short > 0 and mkt_pos:
        score = 8.0
        passed = True
        sigs.append(
            f"🔥 [§4.1.2 DualMom] PASS — coin +{r_med*100:.1f}% (med) & +{r_short*100:.1f}% (short) & BTC positif"
        )
    elif r_med > 0 and mkt_pos:
        score = 4.0
        passed = True
        sigs.append(f"✅ [§4.1.2 DualMom] Partial pass — coin +{r_med*100:.1f}% & BTC positif")
    elif r_med > 0 and not mkt_pos:
        score = 1.0
        passed = False
        sigs.append(f"⚠️ [§4.1.2 DualMom] Coin positif tapi BTC negatif — risiko counter-trend")
    elif r_med < -0.03:
        score = -3.0
        passed = False
    else:
        passed = False

    return max(0.0, score), sigs, passed


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §10.4 — TREND FOLLOWING + TANH SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def _strat_trend_tanh(close: np.ndarray) -> Tuple[float, List[str], float]:
    """
    §10.4 Trend Following with tanh smoothing (Eq.474-475)
    η_i = tanh(R_i / κ)   where κ = volatility * sqrt(T)
    w_i = η_i / σ_i

    tanh smooths out signal instability for small |R| (better than pure sign).
    Returns: (score, signals, eta)
    """
    score, sigs = 0.0, []
    if len(close) < 20: return 0.0, sigs, 0.0

    lb = min(48, len(close) - 1)
    R  = _cum_return(close, lb)
    sigma = _rolling_vol(close, 20)
    if sigma <= 1e-10: return 0.0, sigs, 0.0

    # κ = cross-sectional std proxy = sigma * sqrt(lookback) (§10.4)
    kappa = sigma * (lb ** 0.5)
    eta   = float(np.tanh(R / kappa)) if kappa > 0 else (1.0 if R > 0 else -1.0)

    if eta > 0.65:
        score = 6.0
        sigs.append(f"🚀 [§10.4 Trend] tanh signal = {eta:.2f} — uptrend kuat, momentum confirmed")
    elif eta > 0.30:
        score = 3.5
        sigs.append(f"📈 [§10.4 Trend] tanh signal = {eta:.2f} — uptrend moderat")
    elif eta > 0.05:
        score = 1.0
    elif eta < -0.60:
        score = -4.0
        sigs.append(f"📉 [§10.4 Trend] tanh signal = {eta:.2f} — downtrend kuat")
    elif eta < -0.20:
        score = -2.0

    return score, sigs, eta


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY §10.3.1 — CONTRARIAN + VOLUME ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def _strat_contrarian_vol(close: np.ndarray, volume: np.ndarray) -> Tuple[float, List[str]]:
    """
    §10.3.1 Contrarian + Volume Filter (Eq.470-473)
    v_i = ln(V_now / V_prev)  — volume change ratio
    u_i = ln(OI_now / OI_prev) — open interest proxy

    Buy losers (relative to market) with volume spike — overreaction reversal.
    Returns: (score, signals)
    """
    score, sigs = 0.0, []
    if len(close) < 16 or len(volume) < 16: return 0.0, sigs

    # Weekly window proxy
    r_now  = float(close[-1] / close[-8] - 1.0) if close[-8] > 0 else 0.0
    r_mkt  = float(np.mean(_returns(close[-9:])))

    v_now  = float(np.sum(volume[-7:]))
    v_prev = float(np.sum(volume[-14:-7]))
    vol_chg = float(np.log(v_now / v_prev + 1e-10)) if v_prev > 0 else 0.0

    r_rel = r_now - r_mkt * 7  # relative to market index

    # Contrarian: underperformer + volume spike = oversold bounce (§10.3.1)
    if r_rel < -0.04 and vol_chg > 0.30:
        score = 6.0
        sigs.append(
            f"🔄 [§10.3.1 Contrarian] Oversold ({r_rel*100:.1f}% vs market) + "
            f"volume naik ({vol_chg:.2f}) — reversal setup!"
        )
    elif r_rel < -0.02 and vol_chg > 0.10:
        score = 3.0
        sigs.append(f"🔄 [§10.3.1 Contrarian] Relative losers + moderate volume uptick")
    elif r_rel > 0.06 and vol_chg < -0.25:
        # Overperformer + volume turun = divergence, kurang sustainable
        score = -2.0
        sigs.append(f"⚠️ [§10.3.1 Contrarian] Rally tanpa volume — divergence, risiko reversal")
    elif r_rel > 0.10:
        score = -1.0  # overbought relative

    return score, sigs


# ─────────────────────────────────────────────────────────────────────────────
#  QUANT ENTRY / SL / TP CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _compute_quant_levels(
    price: float,
    pivot_c: float, pivot_r1: float, pivot_r2: float, pivot_s1: float, pivot_s2: float,
    don_upper: float, don_lower: float, don_mid: float,
    ema9: float, ema21: float, ema50: float,
    signal_type: str,
) -> Tuple[float, float, float, float, float, float, bool]:
    """
    Compute entry zone, SL and TP from quant levels.

    Priority hierarchy:
    Entry: Pivot S1-C zone > Donchian lower > EMA21 > default
    SL:    Below Pivot S2 or Donchian lower (tighter of the two, must be >0.3% away)
    TP1:   Pivot R1 (§3.14)
    TP2:   Donchian upper or Pivot R2 (§3.15)
    TP3:   Extended target = R2 + (R2-C)*0.5

    Returns: (entry_low, entry_high, sl, tp1, tp2, tp3, has_levels)
    """
    if price <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False

    entry_low  = 0.0
    entry_high = 0.0
    sl         = 0.0
    tp1        = 0.0
    tp2        = 0.0
    tp3        = 0.0
    has        = False

    if signal_type in ("LONG", "BUY SPOT"):
        # ── Entry Zone ─────────────────────────────────────────────
        # Best: harga antara S1 dan Pivot Center = discount zone (§3.14)
        if pivot_s1 > 0 and pivot_c > 0 and pivot_s1 < price < pivot_c:
            entry_low  = pivot_s1 * 1.001
            entry_high = pivot_c  * 0.999
            has = True
        # Good: near donchian lower = channel floor (§3.15)
        elif don_lower > 0 and (price - don_lower) / price < 0.05:
            entry_low  = don_lower * 1.001
            entry_high = don_mid   * 0.99 if don_mid > 0 else price * 1.005
            has = True
        # Fallback: EMA21 as dynamic support
        elif ema21 > 0 and abs(price - ema21) / price < 0.04:
            entry_low  = ema21 * 0.998
            entry_high = ema21 * 1.012
            has = True

        # ── Stop Loss ──────────────────────────────────────────────
        # SL = tighter of Pivot S2 or Donchian lower (with buffer)
        sl_candidates = []
        if pivot_s2 > 0 and pivot_s2 < price:
            pct = (price - pivot_s2) / price
            if 0.005 < pct < 0.12:
                sl_candidates.append(pivot_s2 * 0.998)
        if don_lower > 0 and don_lower < price:
            pct = (price - don_lower) / price
            if 0.005 < pct < 0.12:
                sl_candidates.append(don_lower * 0.998)
        if pivot_s1 > 0 and pivot_s1 < price * 0.97:
            sl_candidates.append(pivot_s1 * 0.995)

        sl = min(sl_candidates) if sl_candidates else 0.0

        # ── Take Profits ───────────────────────────────────────────
        # TP1 = Pivot R1 (§3.14 — first resistance)
        if pivot_r1 > price:
            r1_pct = (pivot_r1 - price) / price
            if 0.01 < r1_pct < 0.30:
                tp1 = pivot_r1 * 0.998
                has = True

        # TP2 = Donchian upper or Pivot R2 (whichever more realistic)
        don_tp = don_upper * 0.995 if don_upper > price else 0.0
        piv_tp2 = pivot_r2 * 0.998 if pivot_r2 > price else 0.0

        if don_tp > 0 and piv_tp2 > 0:
            # Use closer target to be conservative
            tp2 = min(don_tp, piv_tp2)
        elif don_tp > 0:
            tp2 = don_tp
        elif piv_tp2 > 0:
            tp2 = piv_tp2

        # TP3 = Extended: Pivot R2 + 50% of (R2 - C) range
        if pivot_r2 > price and pivot_c > 0:
            extension = (pivot_r2 - pivot_c) * 0.5
            tp3 = pivot_r2 + extension
        elif tp2 > price:
            tp3 = tp2 * 1.15

    elif signal_type == "SHORT":
        if pivot_r1 > price:
            entry_low  = price * 0.998
            entry_high = pivot_r1 * 0.995
        sl = pivot_r2 * 1.002 if pivot_r2 > price else price * 1.06
        tp1 = pivot_c  if pivot_c < price else price * 0.90
        tp2 = pivot_s1 if pivot_s1 < price else price * 0.80
        tp3 = pivot_s2 if pivot_s2 < price else price * 0.70
        has = sl > 0 and tp1 > 0

    return entry_low, entry_high, sl, tp1, tp2, tp3, has


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_quant(
    symbol        : str,
    price         : float,
    df_1h         : pd.DataFrame,
    df_4h         : pd.DataFrame,
    btc_chg_24h   : float = 0.0,
    signal_type   : str   = "LONG",
) -> QuantResult:
    """
    Run semua 10 strategi dari 151 Trading Strategies dan kembalikan QuantResult.

    Args:
        symbol      : e.g. 'BTCUSDT'
        price       : harga saat ini
        df_1h       : OHLCV DataFrame 1H
        df_4h       : OHLCV DataFrame 4H
        btc_chg_24h : BTC return 24H sebagai market proxy
        signal_type : 'LONG'/'BUY SPOT'/'SHORT' untuk level calculation
    """
    res = QuantResult()
    if price <= 0:
        return res

    # BTC returns array untuk residual momentum (§3.7)
    btc_returns = np.array([btc_chg_24h] * 24, dtype=float)  # simplified proxy

    # Jalankan per timeframe, weighted average (4H lebih berat)
    tf_configs = [("1H", df_1h, 0.35), ("4H", df_4h, 0.65)]

    primary_4h = None  # Simpan hasil 4H untuk level calculation

    for label, df, weight in tf_configs:
        if df is None or df.empty or len(df) < 20:
            continue
        try:
            close  = df["close"].values.astype(float)
            high   = df["high"].values.astype(float)
            low    = df["low"].values.astype(float)
            volume = df["volume"].values.astype(float)

            if len(close) < 2: continue
            prev_h = float(high[-2])
            prev_l = float(low[-2])
            prev_c = float(close[-2])

            # ── §3.1 Price Momentum ──────────────────────────────
            s1, sg1, r_adj = _strat_momentum(close)
            res.momentum_score += s1 * weight
            res.signals.extend(sg1)
            if label == "4H": res.risk_adj_return = r_adj

            # ── §3.4 Low Volatility Anomaly ──────────────────────
            s2, sg2 = _strat_low_vol(close)
            res.lowvol_score += s2 * weight
            res.signals.extend(sg2)

            # ── §3.7 Residual Momentum ───────────────────────────
            s3, sg3, resid = _strat_residual_momentum(close, btc_returns)
            res.residual_score += s3 * weight
            res.signals.extend(sg3)
            if label == "4H": res.residual_mom = resid

            # ── §3.13 Triple MA ───────────────────────────────────
            s4, sg4, bias, e9, e21, e50 = _strat_triple_ma(close)
            res.triple_ma_score += s4 * weight
            res.signals.extend(sg4)
            if label == "4H":
                res.triple_ma_bias = bias
                res.ema9 = e9; res.ema21 = e21; res.ema50 = e50

            # ── §3.14 Pivot Point ─────────────────────────────────
            s5, sg5, C, R1, R2, S1, S2 = _strat_pivot(price, prev_h, prev_l, prev_c)
            res.pivot_score += s5 * weight
            res.signals.extend(sg5)
            if label == "4H":
                res.pivot_c = C; res.pivot_r1 = R1; res.pivot_r2 = R2
                res.pivot_s1 = S1; res.pivot_s2 = S2

            # ── §3.15 Donchian Channel ────────────────────────────
            s6, sg6, d_up, d_dn, d_mid = _strat_donchian(close, high, low, price)
            res.channel_score += s6 * weight
            res.signals.extend(sg6)
            if label == "4H":
                res.donchian_upper = d_up
                res.donchian_lower = d_dn
                res.donchian_mid   = d_mid

            # ── §4.4 IBS Mean Reversion ───────────────────────────
            s7, sg7, ibs = _strat_ibs(prev_h, prev_l, prev_c)
            res.ibs_score += s7 * weight
            res.signals.extend(sg7)
            if label == "1H": res.ibs_value = ibs  # use 1H IBS for intraday

            # ── §4.1.2 Dual Momentum (4H only) ───────────────────
            if label == "4H":
                s8, sg8, d_pass = _strat_dual_momentum(close, btc_chg_24h)
                res.dual_mom_score = s8
                res.dual_mom_pass  = d_pass
                res.signals.extend(sg8)

            # ── §10.4 Trend Following + tanh ─────────────────────
            s9, sg9, eta = _strat_trend_tanh(close)
            res.trend_score += s9 * weight
            res.signals.extend(sg9)
            if label == "4H": res.trend_eta = eta

            # ── §10.3.1 Contrarian + Volume ───────────────────────
            s10, sg10 = _strat_contrarian_vol(close, volume)
            res.contrarian_score += s10 * weight
            res.signals.extend(sg10)

            if label == "4H":
                primary_4h = (close, high, low, volume)

        except Exception as e:
            logger.warning(f"quant [{label}] {symbol}: {e}")
            continue

    # ── Confluence Bonus ──────────────────────────────────────────────────────
    bullish_count = sum([
        1 if res.momentum_score   > 2.0 else 0,
        1 if res.lowvol_score     > 1.0 else 0,
        1 if res.residual_score   > 1.5 else 0,
        1 if res.triple_ma_score  > 2.0 else 0,
        1 if res.pivot_score      > 1.0 else 0,
        1 if res.channel_score    > 2.0 else 0,
        1 if res.ibs_score        > 1.0 else 0,
        1 if res.dual_mom_pass          else 0,
        1 if res.trend_score      > 1.5 else 0,
        1 if res.contrarian_score > 1.5 else 0,
    ])
    res.confluence_count = bullish_count

    raw = (
        res.momentum_score + res.lowvol_score + res.residual_score +
        res.triple_ma_score + res.pivot_score + res.channel_score +
        res.ibs_score + res.dual_mom_score + res.trend_score + res.contrarian_score
    )

    if bullish_count >= 7:
        raw += 10.0
        res.signals.append(f"🌟 [Quant Confluence] {bullish_count}/10 strategi setuju BULLISH — very high conviction!")
    elif bullish_count >= 5:
        raw += 6.0
        res.signals.append(f"✅ [Quant Confluence] {bullish_count}/10 strategi bullish — good alignment")
    elif bullish_count >= 4:
        raw += 3.0

    res.score = float(max(0.0, min(raw, 40.0)))

    # ── Compute Quant Entry/SL/TP Levels ─────────────────────────────────────
    if res.pivot_c > 0 or res.donchian_upper > 0:
        el, eh, sl, tp1, tp2, tp3, has = _compute_quant_levels(
            price,
            res.pivot_c, res.pivot_r1, res.pivot_r2, res.pivot_s1, res.pivot_s2,
            res.donchian_upper, res.donchian_lower, res.donchian_mid,
            res.ema9, res.ema21, res.ema50,
            signal_type,
        )
        res.quant_entry_low  = el
        res.quant_entry_high = eh
        res.quant_sl         = sl
        res.quant_tp1        = tp1
        res.quant_tp2        = tp2
        res.quant_tp3        = tp3
        res.has_quant_levels = has

    # ── Deduplicate signals ───────────────────────────────────────────────────
    seen, deduped = set(), []
    for s in res.signals:
        k = s[:55]
        if k not in seen:
            seen.add(k)
            deduped.append(s)
    res.signals = deduped

    return res
