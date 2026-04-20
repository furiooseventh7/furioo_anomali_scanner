"""
Quant Engine — CMI-ASS (Advanced Strategies from 151 Trading Strategies)
=========================================================================
Implementasi strategi-strategi kuantitatif dari:
  "151 Trading Strategies" — Kakushadze & Serur (2018)
  SSRN: https://ssrn.com/abstract=3247865

Strategi yang diimplementasikan (dipilih berdasarkan relevansi untuk crypto spot/futures):

  1. PRICE MOMENTUM         (§3.1)  — Risk-adjusted cumulative return ranking
  2. RESIDUAL MOMENTUM      (§3.7)  — Momentum setelah strip out market beta
  3. LOW VOLATILITY ANOMALY (§3.4)  — Low-vol coins outperform high-vol
  4. THREE MOVING AVERAGES  (§3.13) — Triple MA confluence filter
  5. PIVOT POINT S/R        (§3.14) — Support/Resistance via pivot point formula
  6. DONCHIAN CHANNEL       (§3.15) — Breakout/mean-reversion via price channel
  7. IBS MEAN REVERSION     (§4.4)  — Internal Bar Strength untuk reversal
  8. DUAL MOMENTUM          (§4.1.2)— Relative + absolute momentum combo
  9. TREND FOLLOWING        (§10.4) — Sign(return)/volatility weight + tanh smoothing
 10. CONTRARIAN + VOLUME    (§10.3) — Mean-reversion diperkuat oleh volume change

Output: QuantResult dengan skor 0–40 (ditambahkan ke confidence score sistem)
"""

import numpy as np
import pandas as pd
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantResult:
    score           : float = 0.0        # 0–40, ditambah ke total confluence score
    signals         : list  = field(default_factory=list)

    # Sub-scores per strategy
    momentum_score  : float = 0.0        # Price + Residual Momentum
    lowvol_score    : float = 0.0        # Low Volatility Anomaly
    ma_score        : float = 0.0        # Triple MA confluence
    pivot_score     : float = 0.0        # Pivot Point S/R
    channel_score   : float = 0.0        # Donchian Channel
    ibs_score       : float = 0.0        # IBS Mean Reversion
    dual_mom_score  : float = 0.0        # Dual Momentum
    trend_score     : float = 0.0        # Trend Following (tanh)
    contrarian_score: float = 0.0        # Contrarian + Volume

    # Key computed values (untuk display di Telegram)
    risk_adj_return : float = 0.0        # Risk-adjusted momentum
    residual_mom    : float = 0.0        # Residual momentum (beta-stripped)
    pivot_center    : float = 0.0        # Pivot point C
    pivot_r1        : float = 0.0        # Resistance R1
    pivot_s1        : float = 0.0        # Support S1
    donchian_upper  : float = 0.0        # Donchian channel upper
    donchian_lower  : float = 0.0        # Donchian channel lower
    ibs_value       : float = 0.5        # IBS 0–1 (0=cheap, 1=rich)
    trend_signal    : float = 0.0        # tanh momentum signal (-1 to +1)
    ma3_bias        : str   = "NEUTRAL"  # THREE_MA bias
    dual_mom_pass   : bool  = False      # Passed dual momentum filter


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_last(arr: np.ndarray) -> float:
    """Return last non-NaN value, or 0.0."""
    if arr is None or len(arr) == 0:
        return 0.0
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if len(valid) > 0 else 0.0


def _sma(close: np.ndarray, n: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full(len(close), np.nan)
    if len(close) < n:
        return result
    for i in range(n - 1, len(close)):
        result[i] = np.mean(close[i - n + 1 : i + 1])
    return result


def _ema(close: np.ndarray, n: int) -> np.ndarray:
    """Exponential Moving Average (λ = 2/(n+1))."""
    result = np.full(len(close), np.nan)
    if len(close) < n:
        return result
    k = 2.0 / (n + 1)
    result[n - 1] = np.mean(close[:n])
    for i in range(n, len(close)):
        result[i] = close[i] * k + result[i - 1] * (1 - k)
    return result


def _rolling_std(arr: np.ndarray, n: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.full(len(arr), np.nan)
    if len(arr) < n:
        return result
    for i in range(n - 1, len(arr)):
        result[i] = float(np.std(arr[i - n + 1 : i + 1], ddof=1))
    return result


def _cumulative_return(close: np.ndarray, lookback: int) -> float:
    """R_cum = P(0)/P(lookback) - 1  (§3.1 Eq.267)"""
    if len(close) <= lookback or close[-(lookback + 1)] == 0:
        return 0.0
    return float(close[-1] / close[-(lookback + 1)] - 1.0)


def _risk_adjusted_return(close: np.ndarray, lookback: int) -> float:
    """R_risk_adj = mean_return / std_return  (§3.1 Eq.268-270)"""
    if len(close) <= lookback:
        return 0.0
    prices = close[-lookback - 1 :]
    rets = np.diff(prices) / (prices[:-1] + 1e-10)
    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1))
    if std_r == 0:
        return 0.0
    return mean_r / std_r


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 1: PRICE + RESIDUAL MOMENTUM  (§3.1, §3.7)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_momentum(close: np.ndarray, volume: np.ndarray) -> tuple[float, list, float, float]:
    """
    Price Momentum (§3.1): risk-adjusted cumulative return.
    Residual Momentum (§3.7): momentum setelah strip out market beta proxy.

    Returns (score 0–10, signals, risk_adj_return, residual_mom)
    """
    score = 0.0
    signals = []

    # --- Price Momentum ---
    # Formation period 12 bars (proxy bulan = candle H4/1H), skip 1
    lookback_long  = min(48, len(close) - 2)  # ~12 periode H4 = 2 hari
    lookback_short = min(6, lookback_long - 1)

    r_cum  = _cumulative_return(close, lookback_long)
    r_radj = _risk_adjusted_return(close, lookback_long)
    r_short = _cumulative_return(close, lookback_short)

    # Momentum signal: positif = bullish
    if r_radj > 0.5:
        score += 4.0
        signals.append(f"📈 [Quant] Momentum kuat: risk-adj return = {r_radj:.2f} (§3.1)")
    elif r_radj > 0.15:
        score += 2.0
        signals.append(f"📈 [Quant] Momentum positif: risk-adj = {r_radj:.2f}")
    elif r_radj < -0.5:
        score -= 3.0
        signals.append(f"📉 [Quant] Momentum lemah: risk-adj = {r_radj:.2f}")

    # --- Residual Momentum: strip out market beta proxy (§3.7) ---
    # Proxy "market" = rata-rata return dari harga sendiri (cross-sectional unavailable)
    # Gunakan BTC-proxy: return coin vs return rata-rata semua bar sebagai benchmark
    if len(close) >= 20:
        rets = np.diff(close[-21:]) / (close[-21:-1] + 1e-10)
        market_ret = float(np.mean(rets))
        # Simple OLS beta proxy: cov(r, mkt) / var(mkt) → simplifikasi = 1 (single coin)
        # Residual = actual return - market_ret (beta=1 assumption)
        actual_short = r_short
        residual = actual_short - market_ret * lookback_short
        if residual > 0.05:
            score += 3.0
            signals.append(f"⚡ [Quant] Residual momentum positif ({residual:.3f}) — alpha murni (§3.7)")
        elif residual < -0.05:
            score -= 2.0
    else:
        residual = 0.0

    return min(score, 10.0), signals, r_radj, float(residual)


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 2: LOW VOLATILITY ANOMALY  (§3.4)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_low_volatility(close: np.ndarray, atr_pct: float) -> tuple[float, list, float]:
    """
    Low Volatility Anomaly (§3.4): low-vol assets outperform high-vol.
    Empirically, coins dengan volatilitas lebih rendah menghasilkan return lebih baik.

    Returns (score -5 to +5, signals, vol_score)
    """
    score = 0.0
    signals = []

    # Hitung realized volatility (rolling 20 bar)
    if len(close) < 20:
        return 0.0, signals, 0.0

    rets = np.diff(close[-21:]) / (close[-21:-1] + 1e-10)
    vol = float(np.std(rets, ddof=1)) * 100  # dalam %

    # Low volatility = positive signal (§3.4)
    # Threshold: crypto normal vol ~3-8% per candle
    if vol < 2.0:
        score += 4.0
        signals.append(f"🧘 [Quant] Volatilitas rendah ({vol:.1f}%) — low-vol anomaly bullish (§3.4)")
    elif vol < 4.0:
        score += 2.0
        signals.append(f"🧘 [Quant] Volatilitas moderat ({vol:.1f}%) — setup stabil")
    elif vol > 10.0:
        score -= 3.0
        signals.append(f"⚠️ [Quant] Volatilitas tinggi ({vol:.1f}%) — low-vol anomaly negatif")
    elif vol > 7.0:
        score -= 1.0

    return max(-5.0, min(score, 5.0)), signals, vol


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 3: THREE MOVING AVERAGES  (§3.13)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_three_ma(close: np.ndarray) -> tuple[float, list, str]:
    """
    Three Moving Averages (§3.13):
    Long if MA(T1) > MA(T2) > MA(T3)   — T1=9, T2=21, T3=50
    Short if MA(T1) < MA(T2) < MA(T3)

    Returns (score -6 to +6, signals, bias)
    """
    score = 0.0
    signals = []
    bias = "NEUTRAL"

    T1, T2, T3 = 9, 21, 50
    if len(close) < T3:
        return 0.0, signals, bias

    ma1 = _safe_last(_ema(close, T1))
    ma2 = _safe_last(_ema(close, T2))
    ma3 = _safe_last(_ema(close, T3))

    if ma1 == 0 or ma2 == 0 or ma3 == 0:
        return 0.0, signals, bias

    # Perfect bullish alignment: MA9 > MA21 > MA50  (§3.13 Eq.324)
    if ma1 > ma2 > ma3:
        score += 6.0
        bias = "BULL"
        signals.append(f"🔺 [Quant] Triple MA Bullish: EMA9>{ma1:.4f} > EMA21>{ma2:.4f} > EMA50 (§3.13)")
    # Partial alignment
    elif ma1 > ma2 and close[-1] > ma3:
        score += 3.0
        bias = "BULL"
        signals.append(f"📈 [Quant] Triple MA partial bullish (EMA9>EMA21, harga>EMA50)")
    # Perfect bearish alignment
    elif ma1 < ma2 < ma3:
        score -= 5.0
        bias = "BEAR"
        signals.append(f"🔻 [Quant] Triple MA Bearish: EMA9 < EMA21 < EMA50 (§3.13)")
    elif ma1 < ma2 and close[-1] < ma3:
        score -= 2.0
        bias = "BEAR"

    return max(-6.0, min(score, 6.0)), signals, bias


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 4: PIVOT POINT SUPPORT & RESISTANCE  (§3.14)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_pivot_point(
    current_price: float,
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> tuple[float, list, float, float, float]:
    """
    Pivot Point S/R (§3.14 Eq.325-328):
    C = (PH + PL + PC) / 3
    R = 2*C - PL
    S = 2*C - PH

    Signal: Long if P > C, Short if P < C
    Liquidate Long at R, Liquidate Short at S

    Returns (score, signals, C, R1, S1)
    """
    score = 0.0
    signals = []

    if prev_high <= 0 or prev_low <= 0 or prev_close <= 0 or current_price <= 0:
        return 0.0, signals, 0.0, 0.0, 0.0

    # Pivot formulas (§3.14)
    C  = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2.0 * C - prev_low
    S1 = 2.0 * C - prev_high

    # Extended levels (R2, S2)
    R2 = C + (prev_high - prev_low)
    S2 = C - (prev_high - prev_low)

    dist_to_s1_pct = (current_price - S1) / current_price * 100
    dist_to_r1_pct = (R1 - current_price) / current_price * 100

    # Signal interpretation (§3.14 Eq.328)
    if current_price > C:
        # Bullish bias (above pivot)
        if dist_to_r1_pct > 0:
            if dist_to_s1_pct < 3.0:
                # Near support from above = strong buy zone
                score += 5.0
                signals.append(
                    f"🎯 [Quant] Harga di atas Pivot ({C:.5f}), dekat S1 ({S1:.5f}) — prime long (§3.14)"
                )
            else:
                score += 2.0
                signals.append(f"✅ [Quant] Harga di atas Pivot ({C:.5f}) — bullish bias")
        else:
            # Price above R1 = approaching resistance, reduce score
            score += 1.0
            signals.append(f"⚠️ [Quant] Harga melewati R1 ({R1:.5f}) — overbought area pivot")
    else:
        # Bearish bias (below pivot)
        score -= 2.0
        if abs(dist_to_s1_pct) < 2.0:
            signals.append(f"🔴 [Quant] Harga di bawah Pivot ({C:.5f}), dekat S1 ({S1:.5f})")
        else:
            signals.append(f"📉 [Quant] Harga di bawah Pivot Center ({C:.5f}) — bearish bias (§3.14)")

    return max(-4.0, min(score, 6.0)), signals, C, R1, S1


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 5: DONCHIAN CHANNEL  (§3.15)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_donchian(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    current_price: float,
    period: int = 20,
) -> tuple[float, list, float, float]:
    """
    Donchian Channel (§3.15 Eq.329-331):
    B_up   = max(P[1..T])
    B_down = min(P[1..T])

    Long di B_down (floor), Short di B_up (ceiling).
    Breakout di atas B_up = momentum bullish.

    Returns (score, signals, upper, lower)
    """
    score = 0.0
    signals = []

    if len(close) < period + 1:
        return 0.0, signals, 0.0, 0.0

    # Gunakan high/low untuk channel (lebih akurat dari close only)
    b_up   = float(np.max(high[-period:]))
    b_down = float(np.min(low[-period:]))

    if b_up <= b_down or b_up == 0:
        return 0.0, signals, b_up, b_down

    channel_width_pct = (b_up - b_down) / b_up * 100
    pos_in_channel    = (current_price - b_down) / (b_up - b_down)  # 0=floor, 1=ceiling

    # Near floor = potential long (mean-reversion) — (§3.15)
    if pos_in_channel <= 0.15:
        score += 6.0
        signals.append(
            f"🟢 [Quant] Harga dekat Donchian Floor ({b_down:.5f}) — mean-reversion buy zone (§3.15)"
        )
    elif pos_in_channel <= 0.30:
        score += 3.0
        signals.append(f"🟢 [Quant] Harga di lower quartile Donchian — potential long")

    # Near ceiling = potential short or breakout
    elif pos_in_channel >= 0.85:
        # Harga di ceiling — bisa short (mean-rev) atau breakout (momentum)
        # Jika volume spike di candle ini, treat as breakout (bullish)
        # Otherwise, treat as resistance
        score -= 2.0
        signals.append(
            f"🔴 [Quant] Harga dekat Donchian Ceiling ({b_up:.5f}) — resistance kuat (§3.15)"
        )

    # Breakout di atas channel = strong bullish trend signal
    elif current_price > b_up * 0.999:
        score += 7.0
        signals.append(
            f"🚀 [Quant] BREAKOUT Donchian Channel! Harga tembus {b_up:.5f} — momentum kuat (§3.15)"
        )

    # Mid-channel: neutral, sedikit positif karena not near resistance
    elif 0.40 <= pos_in_channel <= 0.65:
        score += 1.0

    return max(-4.0, min(score, 8.0)), signals, b_up, b_down


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 6: IBS MEAN REVERSION  (§4.4)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_ibs(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> tuple[float, list, float]:
    """
    Internal Bar Strength Mean Reversion (§4.4 Eq.370):
    IBS = (PC - PL) / (PH - PL)
    IBS near 0 = cheap (buy), IBS near 1 = rich (sell).

    Returns (score -4 to +4, signals, ibs_value)
    """
    score = 0.0
    signals = []

    if prev_high <= prev_low or prev_high == 0:
        return 0.0, signals, 0.5

    ibs = (prev_close - prev_low) / (prev_high - prev_low)

    # IBS thresholds (§4.4)
    if ibs <= 0.15:
        score += 4.0
        signals.append(f"💚 [Quant] IBS sangat rendah ({ibs:.2f}) — candle tutup di bawah, potensi reversal up (§4.4)")
    elif ibs <= 0.30:
        score += 2.0
        signals.append(f"💚 [Quant] IBS rendah ({ibs:.2f}) — bearish candle, potensi reversal")
    elif ibs >= 0.85:
        score -= 3.0
        signals.append(f"🔴 [Quant] IBS sangat tinggi ({ibs:.2f}) — candle bullish kuat, beware reversal down (§4.4)")
    elif ibs >= 0.70:
        score -= 1.0

    return max(-4.0, min(score, 4.0)), signals, float(ibs)


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 7: DUAL MOMENTUM  (§4.1.2)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_dual_momentum(
    close: np.ndarray,
    btc_return_24h: float,
) -> tuple[float, list, bool]:
    """
    Dual Momentum (§4.1.2): Relative momentum + Absolute momentum filter.
    - Relative: cumulative return coin vs BTC-proxy
    - Absolute: hanya long jika BTC (broad market proxy) trending up

    Returns (score 0–8, signals, dual_mom_pass)
    """
    score = 0.0
    signals = []
    passed = False

    if len(close) < 12:
        return 0.0, signals, False

    # Absolute momentum: BTC proxy (broad market) harus positif
    # Jika btc_return_24h > 0 = market sedang uptrend
    market_positive = btc_return_24h > 0

    # Relative momentum: cumulative return coin
    r_cum_medium = _cumulative_return(close, min(24, len(close) - 1))  # ~1 hari H1
    r_cum_short  = _cumulative_return(close, min(6,  len(close) - 1))  # ~6 jam H1

    # Dual momentum condition (§4.1.2): positif di kedua momentum
    if r_cum_medium > 0 and r_cum_short > 0 and market_positive:
        score += 8.0
        passed = True
        signals.append(
            f"🔥 [Quant] DUAL MOMENTUM PASS: coin +{r_cum_medium*100:.1f}% & market positif (§4.1.2)"
        )
    elif r_cum_medium > 0 and market_positive:
        score += 4.0
        passed = True
        signals.append(
            f"✅ [Quant] Dual momentum partial: coin +{r_cum_medium*100:.1f}%, market positif"
        )
    elif r_cum_medium > 0 and not market_positive:
        score += 1.0
        signals.append(
            f"⚠️ [Quant] Coin momentum positif tapi market negatif — risiko lebih tinggi (§4.1.2)"
        )
    elif r_cum_medium < -0.03:
        score -= 2.0

    return max(0.0, min(score, 8.0)), signals, passed


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 8: TREND FOLLOWING dengan TANH SMOOTHING  (§10.4)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_trend_following(close: np.ndarray) -> tuple[float, list, float]:
    """
    Trend Following (§10.4 Eq.474-475) dengan tanh smoothing:
    η_i = tanh(R_i / κ)   where κ = cross-sectional std (single coin: recent vol)
    w_i = η_i / σ_i

    Signal positif = long, negatif = short/avoid.

    Returns (score -4 to +5, signals, trend_signal)
    """
    score = 0.0
    signals = []

    if len(close) < 20:
        return 0.0, signals, 0.0

    # Return over lookback
    lookback = min(48, len(close) - 1)
    R = float(close[-1] / close[-lookback - 1] - 1.0) if close[-lookback - 1] > 0 else 0.0

    # κ = volatility (used as smoothing parameter, §10.4)
    rets = np.diff(close[-21:]) / (close[-21:-1] + 1e-10)
    sigma = float(np.std(rets, ddof=1))
    if sigma == 0:
        return 0.0, signals, 0.0

    # tanh smoothing — mitigates signal instability for small |R| (§10.4)
    kappa = sigma * lookback  # scale kappa ke lookback period
    eta = float(np.tanh(R / kappa)) if kappa > 0 else float(np.sign(R))

    # Trend weight = eta / sigma (§10.4 Eq.474)
    trend_weight = eta / sigma if sigma > 0 else 0.0

    # Score berdasarkan kekuatan trend
    if eta > 0.6:
        score += 5.0
        signals.append(f"🚀 [Quant] Trend Following: tanh signal = {eta:.2f} — uptrend kuat (§10.4)")
    elif eta > 0.25:
        score += 3.0
        signals.append(f"📈 [Quant] Trend Following: tanh signal = {eta:.2f} — uptrend moderat")
    elif eta > 0.05:
        score += 1.0
    elif eta < -0.5:
        score -= 3.0
        signals.append(f"📉 [Quant] Trend Following: tanh signal = {eta:.2f} — downtrend (§10.4)")
    elif eta < -0.15:
        score -= 1.5

    return max(-4.0, min(score, 5.0)), signals, float(eta)


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY 9: CONTRARIAN + VOLUME FILTER  (§10.3, §10.3.1)
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_contrarian_volume(
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[float, list]:
    """
    Contrarian (Mean-Reversion) + Volume Activity Filter (§10.3.1 Eq.470-473):
    - Contrarian: beli loser, jual winner vs market index
    - Volume filter: volume_change = ln(V_now / V_prev) — hanya trade pada volume spike

    Returns (score -3 to +5, signals)
    """
    score = 0.0
    signals = []

    if len(close) < 14 or len(volume) < 14:
        return 0.0, signals

    # Recent return (last 7 bars = "last week" proxy, §10.3)
    r_recent = float(close[-1] / close[-8] - 1.0) if close[-8] > 0 else 0.0
    r_market = float(np.mean(np.diff(close[-8:]) / (close[-8:-1] + 1e-10)))

    # Volume change (§10.3.1 Eq.472): v_i = ln(V_now / V_prev_week)
    v_now  = float(np.sum(volume[-7:]))
    v_prev = float(np.sum(volume[-14:-7]))
    if v_prev > 0:
        volume_change = float(np.log(v_now / v_prev + 1e-10))
    else:
        volume_change = 0.0

    # Contrarian signal: coin yang turun lebih dari market = potential buy (§10.3)
    r_relative = r_recent - r_market
    if r_relative < -0.03 and volume_change > 0.2:
        score += 5.0
        signals.append(
            f"🔄 [Quant] Contrarian setup: coin lemah ({r_relative*100:.1f}%) + "
            f"volume naik ({volume_change:.2f}) — oversold reversal (§10.3.1)"
        )
    elif r_relative < -0.015 and volume_change > 0:
        score += 2.5
        signals.append(f"🔄 [Quant] Contrarian moderate: relative return {r_relative*100:.1f}%")
    elif r_relative > 0.05 and volume_change < -0.2:
        # Overperformer dengan volume turun = divergence, kurang reliable
        score -= 1.0
        signals.append(f"⚠️ [Quant] Divergence: harga naik tapi volume turun")
    elif r_relative > 0.08:
        # Terlalu banyak naik = mean-reversion risk
        score -= 2.0

    return max(-3.0, min(score, 5.0)), signals


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_quant(
    symbol       : str,
    current_price: float,
    df_1h        : pd.DataFrame,
    df_4h        : pd.DataFrame,
    btc_return_24h: float = 0.0,
) -> QuantResult:
    """
    Run semua strategi kuantitatif dan return QuantResult.

    Args:
        symbol        : trading pair, e.g. 'BTCUSDT'
        current_price : harga saat ini
        df_1h         : OHLCV DataFrame 1H (kolom: open, high, low, close, volume)
        df_4h         : OHLCV DataFrame 4H (kolom: open, high, low, close, volume)
        btc_return_24h: return BTC 24H sebagai market proxy (untuk dual momentum)
    """
    res = QuantResult()

    if current_price <= 0:
        return res

    # Pilih timeframe: gunakan 1H untuk sinyal intraday
    # 4H untuk momentum jangka menengah
    for df_label, df in [("1H", df_1h), ("4H", df_4h)]:
        if df is None or df.empty or len(df) < 20:
            continue

        try:
            close  = df["close"].values.astype(float)
            high   = df["high"].values.astype(float)
            low    = df["low"].values.astype(float)
            volume = df["volume"].values.astype(float)

            # Previous bar OHLC (untuk IBS dan Pivot)
            if len(close) < 2:
                continue
            prev_high  = float(high[-2])
            prev_low   = float(low[-2])
            prev_close = float(close[-2])

            # ── Run all strategies ──────────────────────────────
            weight = 0.7 if df_label == "4H" else 0.3  # 4H lebih berat

            # 1. Momentum
            s1, sig1, r_radj, resid = _strategy_momentum(close, volume)
            res.momentum_score += s1 * weight
            res.signals.extend(sig1)
            if df_label == "4H":
                res.risk_adj_return = r_radj
                res.residual_mom    = resid

            # 2. Low Volatility Anomaly
            atr_pct = float(np.mean(np.abs(np.diff(close[-20:])) / (close[-20:-1] + 1e-10))) * 100
            s2, sig2, _ = _strategy_low_volatility(close, atr_pct)
            res.lowvol_score += s2 * weight
            res.signals.extend(sig2)

            # 3. Triple MA
            s3, sig3, ma3_bias = _strategy_three_ma(close)
            res.ma_score += s3 * weight
            res.signals.extend(sig3)
            if df_label == "4H":
                res.ma3_bias = ma3_bias

            # 4. Pivot Point
            s4, sig4, C, R1, S1 = _strategy_pivot_point(current_price, prev_high, prev_low, prev_close)
            res.pivot_score += s4 * weight
            res.signals.extend(sig4)
            if df_label == "4H":
                res.pivot_center = C
                res.pivot_r1     = R1
                res.pivot_s1     = S1

            # 5. Donchian Channel
            s5, sig5, d_up, d_down = _strategy_donchian(close, high, low, current_price)
            res.channel_score += s5 * weight
            res.signals.extend(sig5)
            if df_label == "4H":
                res.donchian_upper = d_up
                res.donchian_lower = d_down

            # 6. IBS
            s6, sig6, ibs_val = _strategy_ibs(prev_high, prev_low, prev_close)
            res.ibs_score += s6 * weight
            res.signals.extend(sig6)
            if df_label == "1H":
                res.ibs_value = ibs_val

            # 7. Dual Momentum (hanya 4H)
            if df_label == "4H":
                s7, sig7, d_pass = _strategy_dual_momentum(close, btc_return_24h)
                res.dual_mom_score = s7
                res.signals.extend(sig7)
                res.dual_mom_pass = d_pass

            # 8. Trend Following
            s8, sig8, eta = _strategy_trend_following(close)
            res.trend_score += s8 * weight
            res.signals.extend(sig8)
            if df_label == "4H":
                res.trend_signal = eta

            # 9. Contrarian + Volume
            s9, sig9 = _strategy_contrarian_volume(close, volume)
            res.contrarian_score += s9 * weight
            res.signals.extend(sig9)

        except Exception as e:
            logger.warning(f"quant [{df_label}] {symbol}: {e}")
            continue

    # ── Final Score (max 40, normalized) ──────────────────────────────────
    raw = (
        res.momentum_score
        + res.lowvol_score
        + res.ma_score
        + res.pivot_score
        + res.channel_score
        + res.ibs_score
        + res.dual_mom_score
        + res.trend_score
        + res.contrarian_score
    )

    # Boost jika banyak strategi setuju (confluence bonus)
    positive_strategies = sum([
        1 if res.momentum_score > 2 else 0,
        1 if res.ma_score > 2 else 0,
        1 if res.channel_score > 2 else 0,
        1 if res.pivot_score > 1 else 0,
        1 if res.trend_score > 1 else 0,
        1 if res.dual_mom_pass else 0,
        1 if res.ibs_score > 1 else 0,
    ])
    if positive_strategies >= 5:
        raw += 8.0
        res.signals.append(f"🌟 [Quant] CONFLUENCE: {positive_strategies}/7 strategi setuju BULLISH!")
    elif positive_strategies >= 4:
        raw += 4.0
        res.signals.append(f"✅ [Quant] Confluence baik: {positive_strategies}/7 strategi bullish")
    elif positive_strategies >= 3:
        raw += 2.0

    res.score = float(max(0.0, min(raw, 40.0)))

    # Deduplicate signals (1H dan 4H kadang menghasilkan sinyal duplikat)
    seen = set()
    deduped = []
    for s in res.signals:
        key = s[:60]  # pakai 60 char pertama sebagai key
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    res.signals = deduped

    return res
