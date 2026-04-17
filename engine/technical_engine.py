"""
Technical Analysis Engine — CMI-ASS
====================================
Modul TAMBAHAN (tidak mengganti engine yang sudah ada).
Menghitung semua indikator teknikal murni dari data OHLCV:

MOMENTUM INDICATORS
  • RSI (14) + RSI Divergence
  • MACD (12,26,9) + Signal + Histogram
  • Stochastic RSI

TREND INDICATORS
  • EMA 9, 21, 50, 200
  • Trend Alignment (Bull/Bear/Sideways)
  • Supertrend (ATR-based)

VOLATILITY
  • Bollinger Bands (20,2)
  • ATR (14)
  • Squeeze Momentum (BB inside KC)

SMART MONEY CONCEPTS (ICT/SMC)
  • Fair Value Gap (FVG) — Bullish & Bearish
  • Order Block (OB) — Bullish & Bearish
  • Break of Structure (BOS) / Change of Character (CHoCH)
  • Liquidity Pool detection

STRUCTURE ANALYSIS
  • Support & Resistance levels (multi-timeframe)
  • Chart Patterns (Double Top/Bottom, Head & Shoulders,
    Ascending/Descending Triangle, Bull/Bear Flag)

CONFLUENCE SCORING
  • Menghitung TA Score (0–30) yang ditambahkan ke total score
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from engine.data_fetcher import get_klines

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class FVGZone:
    """Fair Value Gap zone."""
    kind        : str    # 'BULLISH' atau 'BEARISH'
    top         : float
    bottom      : float
    midpoint    : float
    candle_idx  : int
    filled      : bool = False

@dataclass
class OrderBlock:
    """Order Block zone."""
    kind        : str    # 'BULLISH' atau 'BEARISH'
    top         : float
    bottom      : float
    midpoint    : float
    candle_idx  : int
    valid       : bool = True

@dataclass
class SRLevel:
    """Support atau Resistance level."""
    price       : float
    kind        : str    # 'SUPPORT' atau 'RESISTANCE'
    strength    : int    # berapa kali diuji (touch count)
    timeframe   : str

@dataclass
class ChartPattern:
    """Chart pattern yang terdeteksi."""
    name        : str
    direction   : str   # 'BULLISH' atau 'BEARISH'
    confidence  : float # 0–1
    target_pct  : float # target price move dalam %

@dataclass
class TechnicalResult:
    # ── Momentum ──────────────────────────────────────────
    rsi_14          : float = 50.0
    rsi_signal      : str   = "NEUTRAL"      # OVERSOLD/OVERBOUGHT/NEUTRAL/DIVERGENCE_BULL/DIVERGENCE_BEAR
    macd_line       : float = 0.0
    macd_signal     : float = 0.0
    macd_hist       : float = 0.0
    macd_signal_type: str   = "NEUTRAL"      # BULLISH_CROSS/BEARISH_CROSS/BULLISH/BEARISH/NEUTRAL
    stoch_rsi_k     : float = 50.0
    stoch_rsi_d     : float = 50.0
    stoch_signal    : str   = "NEUTRAL"

    # ── Trend ─────────────────────────────────────────────
    ema9            : float = 0.0
    ema21           : float = 0.0
    ema50           : float = 0.0
    ema200          : float = 0.0
    trend_alignment : str   = "SIDEWAYS"     # STRONG_BULL/BULL/SIDEWAYS/BEAR/STRONG_BEAR
    price_vs_ema200 : str   = "ABOVE"        # ABOVE/BELOW
    supertrend_dir  : str   = "NEUTRAL"      # BULLISH/BEARISH

    # ── Volatility ────────────────────────────────────────
    bb_upper        : float = 0.0
    bb_middle       : float = 0.0
    bb_lower        : float = 0.0
    bb_width        : float = 0.0
    bb_position     : str   = "MIDDLE"       # ABOVE_UPPER/NEAR_UPPER/MIDDLE/NEAR_LOWER/BELOW_LOWER
    atr_14          : float = 0.0
    atr_pct         : float = 0.0           # ATR sebagai % dari harga
    bb_squeeze      : bool  = False         # Bollinger squeeze (volatility rendah, siap meledak)

    # ── Smart Money Concepts ──────────────────────────────
    fvg_zones       : List[FVGZone]   = field(default_factory=list)
    order_blocks    : List[OrderBlock] = field(default_factory=list)
    nearest_bull_fvg: Optional[FVGZone]   = None
    nearest_bear_fvg: Optional[FVGZone]   = None
    nearest_bull_ob : Optional[OrderBlock] = None
    nearest_bear_ob : Optional[OrderBlock] = None
    price_in_fvg    : bool  = False
    price_in_ob     : bool  = False
    bos_detected    : bool  = False    # Break of Structure
    choch_detected  : bool  = False    # Change of Character

    # ── Support & Resistance ──────────────────────────────
    support_levels  : List[SRLevel] = field(default_factory=list)
    resist_levels   : List[SRLevel] = field(default_factory=list)
    nearest_support : float = 0.0
    nearest_resist  : float = 0.0
    support_dist_pct: float = 0.0    # % jarak ke support terdekat
    resist_dist_pct : float = 0.0    # % jarak ke resistance terdekat
    price_at_support: bool  = False   # harga dekat support (potensi bounce)
    price_at_resist : bool  = False   # harga dekat resistance

    # ── Chart Patterns ────────────────────────────────────
    patterns        : List[ChartPattern] = field(default_factory=list)
    dominant_pattern: Optional[ChartPattern] = None

    # ── Confluence Output ─────────────────────────────────
    signals         : List[str] = field(default_factory=list)
    ta_bias         : str   = "NEUTRAL"    # STRONG_BULL/BULL/NEUTRAL/BEAR/STRONG_BEAR
    score           : float = 0.0          # 0–30 (ditambahkan ke total score)
    timeframe_used  : str   = "1h"


# ═══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS — INDICATORS
# ═══════════════════════════════════════════════════════════

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI Wilder Smoothing."""
    delta = np.diff(close)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])

    for i in range(period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period

    rs  = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - (100 / (1 + rs))

    # Pad awal dengan NaN
    result = np.full(len(close), np.nan)
    result[period:] = rsi[period-1:]
    return result

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.full(len(data), np.nan)
    if len(data) < period:
        return result
    k = 2.0 / (period + 1)
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i-1] * (1 - k)
    return result

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full(len(data), np.nan)
    for i in range(period-1, len(data)):
        result[i] = np.mean(data[i-period+1:i+1])
    return result

def _macd(close: np.ndarray, fast=12, slow=26, signal=9):
    """MACD, Signal, Histogram."""
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    # Signal hanya dari bagian yang valid
    valid_mask = ~np.isnan(macd_line)
    signal_line = np.full(len(close), np.nan)
    if valid_mask.sum() >= signal:
        valid_macd = macd_line[valid_mask]
        sig = _ema(valid_macd, signal)
        signal_line[valid_mask] = sig
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(close: np.ndarray, period=20, std_dev=2.0):
    """Bollinger Bands."""
    mid   = _sma(close, period)
    std   = np.array([
        np.std(close[max(0, i-period+1):i+1]) if i >= period-1 else np.nan
        for i in range(len(close))
    ])
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    """Average True Range."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    atr = np.full(len(close), np.nan)
    if len(tr) < period:
        return atr
    atr[period] = np.mean(tr[:period])
    for i in range(period+1, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period
    return atr

def _stoch_rsi(close: np.ndarray, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI."""
    rsi_vals = _rsi(close, rsi_period)
    stoch_k  = np.full(len(close), np.nan)
    for i in range(stoch_period-1, len(rsi_vals)):
        window = rsi_vals[i-stoch_period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        mn, mx = np.min(window), np.max(window)
        stoch_k[i] = (rsi_vals[i] - mn) / (mx - mn) * 100 if mx != mn else 50
    stoch_k_smooth = _sma(stoch_k, k_smooth)
    stoch_d        = _sma(stoch_k_smooth, d_smooth)
    return stoch_k_smooth, stoch_d

def _supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator."""
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    direction = np.full(len(close), 1)  # 1 = bullish, -1 = bearish
    final_up  = np.full(len(close), np.nan)
    final_dn  = np.full(len(close), np.nan)

    for i in range(period, len(close)):
        if np.isnan(atr[i]):
            continue
        basic_up = hl2[i] - multiplier * atr[i]
        basic_dn = hl2[i] + multiplier * atr[i]

        if i == period:
            final_up[i] = basic_up
            final_dn[i] = basic_dn
        else:
            final_up[i] = max(basic_up, final_up[i-1]) if close[i-1] > final_up[i-1] else basic_up
            final_dn[i] = min(basic_dn, final_dn[i-1]) if close[i-1] < final_dn[i-1] else basic_dn

        if close[i] > final_dn[i-1] if not np.isnan(final_dn[i-1]) else True:
            direction[i] = 1
        elif close[i] < final_up[i-1] if not np.isnan(final_up[i-1]) else True:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    return direction, final_up, final_dn


# ═══════════════════════════════════════════════════════════
#  SMART MONEY CONCEPTS
# ═══════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, lookback: int = 30) -> List[FVGZone]:
    """
    Fair Value Gap (Imbalance):
    - BULLISH FVG: candle[i-2].high < candle[i].low  → gap yang belum terisi ke atas
    - BEARISH FVG: candle[i-2].low  > candle[i].high → gap yang belum terisi ke bawah
    """
    fvgs = []
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)
    start = max(0, n - lookback - 2)

    for i in range(start + 2, n):
        # Bullish FVG
        if high[i-2] < low[i]:
            gap_bottom = high[i-2]
            gap_top    = low[i]
            midpoint   = (gap_top + gap_bottom) / 2
            # Cek apakah sudah terisi
            filled = bool(np.any(low[i+1:] <= midpoint)) if i+1 < n else False
            fvgs.append(FVGZone(
                kind="BULLISH", top=gap_top, bottom=gap_bottom,
                midpoint=midpoint, candle_idx=i, filled=filled
            ))
        # Bearish FVG
        if low[i-2] > high[i]:
            gap_top    = low[i-2]
            gap_bottom = high[i]
            midpoint   = (gap_top + gap_bottom) / 2
            filled = bool(np.any(high[i+1:] >= midpoint)) if i+1 < n else False
            fvgs.append(FVGZone(
                kind="BEARISH", top=gap_top, bottom=gap_bottom,
                midpoint=midpoint, candle_idx=i, filled=filled
            ))

    return fvgs

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> List[OrderBlock]:
    """
    Order Block — candle sebelum impulsive move besar:
    - BULLISH OB: bearish candle sebelum bullish impuls besar (>1.5%)
    - BEARISH OB: bullish candle sebelum bearish impuls besar (>1.5%)
    """
    obs   = []
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)
    start = max(0, n - lookback - 1)

    for i in range(start + 1, n - 1):
        candle_range = abs(close[i+1] - open_[i+1])
        price        = close[i]

        # Bullish OB: sebelumnya bearish candle, diikuti bullish impuls kuat
        if (close[i] < open_[i] and close[i+1] > open_[i+1]
                and price > 0
                and (close[i+1] - open_[i+1]) / price > 0.015):
            obs.append(OrderBlock(
                kind="BULLISH", top=high[i], bottom=low[i],
                midpoint=(high[i]+low[i])/2, candle_idx=i
            ))

        # Bearish OB: sebelumnya bullish candle, diikuti bearish impuls kuat
        if (close[i] > open_[i] and close[i+1] < open_[i+1]
                and price > 0
                and (open_[i+1] - close[i+1]) / price > 0.015):
            obs.append(OrderBlock(
                kind="BEARISH", top=high[i], bottom=low[i],
                midpoint=(high[i]+low[i])/2, candle_idx=i
            ))

    return obs

def detect_bos_choch(df: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Break of Structure & Change of Character.
    BOS  = harga break HH (bullish) atau LL (bearish) dengan konfirmasi volume
    CHoCH = market structure berubah arah (LL setelah HH sequence atau sebaliknya)
    """
    if len(df) < 20:
        return False, False

    highs = df["high"].values[-20:]
    lows  = df["low"].values[-20:]
    close = df["close"].values[-20:]

    # Cari pivot highs/lows sederhana
    pivot_highs = []
    pivot_lows  = []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivot_highs.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivot_lows.append((i, lows[i]))

    bos   = False
    choch = False

    # BOS Bullish: current close > last pivot high
    if pivot_highs and close[-1] > pivot_highs[-1][1] * 1.005:
        bos = True

    # CHoCH: setelah sequence HH, kini buat LL (bearish reversal structure)
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        last_hh = pivot_highs[-1][1] > pivot_highs[-2][1]
        last_ll = pivot_lows[-1][1] < pivot_lows[-2][1]
        if last_hh and last_ll and pivot_lows[-1][0] > pivot_highs[-1][0]:
            choch = True

    return bos, choch

def detect_liquidity_pools(df: pd.DataFrame) -> dict:
    """
    Deteksi area likuiditas:
    - Equal highs/lows (tempat stop loss banyak trader)
    - Previous session highs/lows
    """
    if len(df) < 20:
        return {}

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values[-1]
    tolerance = 0.005  # 0.5%

    # Equal Highs
    recent_highs = high[-20:]
    max_high = np.max(recent_highs)
    equal_highs = np.sum(np.abs(recent_highs - max_high) / max_high < tolerance)

    # Equal Lows
    recent_lows = low[-20:]
    min_low = np.min(recent_lows)
    equal_lows = np.sum(np.abs(recent_lows - min_low) / min_low < tolerance)

    return {
        "buy_side_liquidity": float(max_high),   # di atas = BSL (stop loss sellers)
        "sell_side_liquidity": float(min_low),   # di bawah = SSL (stop loss buyers)
        "equal_highs_count": int(equal_highs),
        "equal_lows_count": int(equal_lows),
    }


# ═══════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE
# ═══════════════════════════════════════════════════════════

def detect_support_resistance(df: pd.DataFrame, current_price: float,
                               timeframe: str = "1h") -> Tuple[List[SRLevel], List[SRLevel]]:
    """
    Deteksi S&R berdasarkan pivot points dan price clustering.
    Mengembalikan (support_levels, resistance_levels) diurutkan dari terdekat ke harga.
    """
    if len(df) < 30:
        return [], []

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    tolerance = 0.015  # 2 titik dianggap sama jika dalam range 1.5%

    # Kumpulkan semua pivot points
    pivot_prices = []
    for i in range(3, len(df) - 3):
        # Pivot high
        if (high[i] == max(high[i-3:i+4])):
            pivot_prices.append(("resistance", high[i]))
        # Pivot low
        if (low[i] == min(low[i-3:i+4])):
            pivot_prices.append(("support", low[i]))

    if not pivot_prices:
        return [], []

    # Cluster pivot prices yang berdekatan
    def cluster_levels(prices: list) -> list:
        if not prices:
            return []
        sorted_p = sorted(prices)
        clusters = []
        current_cluster = [sorted_p[0]]
        for p in sorted_p[1:]:
            if abs(p - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(p)
            else:
                clusters.append(current_cluster)
                current_cluster = [p]
        clusters.append(current_cluster)
        return [(np.mean(c), len(c)) for c in clusters]  # (price, strength)

    support_raw = [p for kind, p in pivot_prices if kind == "support" and p < current_price]
    resist_raw  = [p for kind, p in pivot_prices if kind == "resistance" and p > current_price]

    supports = [SRLevel(price=p, kind="SUPPORT", strength=s, timeframe=timeframe)
                for p, s in cluster_levels(support_raw)]
    resists  = [SRLevel(price=p, kind="RESISTANCE", strength=s, timeframe=timeframe)
                for p, s in cluster_levels(resist_raw)]

    # Sort: support descending (terdekat ke atas), resist ascending (terdekat ke bawah)
    supports = sorted(supports, key=lambda x: x.price, reverse=True)[:5]
    resists  = sorted(resists,  key=lambda x: x.price)[:5]

    return supports, resists


# ═══════════════════════════════════════════════════════════
#  CHART PATTERN DETECTION
# ═══════════════════════════════════════════════════════════

def detect_chart_patterns(df: pd.DataFrame, current_price: float) -> List[ChartPattern]:
    """
    Deteksi chart patterns dari OHLCV data.
    Menggunakan pendekatan pivot-based tanpa scipy untuk kompatibilitas.
    """
    patterns = []
    if len(df) < 40:
        return patterns

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(close)

    # Cari pivot highs & lows (window 5)
    def find_pivots(arr, is_high: bool, window: int = 5) -> List[Tuple[int, float]]:
        pivots = []
        for i in range(window, len(arr) - window):
            seg = arr[i-window:i+window+1]
            if is_high and arr[i] == np.max(seg):
                pivots.append((i, arr[i]))
            elif not is_high and arr[i] == np.min(seg):
                pivots.append((i, arr[i]))
        return pivots

    ph = find_pivots(high, True,  window=5)
    pl = find_pivots(low,  False, window=5)

    # ── Double Bottom ─────────────────────────────────────
    if len(pl) >= 2:
        b1_idx, b1 = pl[-2]
        b2_idx, b2 = pl[-1]
        if b2_idx > b1_idx and abs(b1 - b2) / max(b1, 1) < 0.025:
            neckline = max(high[b1_idx:b2_idx+1])
            if current_price > neckline * 0.98:
                target = (neckline - min(b1, b2)) / neckline
                patterns.append(ChartPattern(
                    name="Double Bottom", direction="BULLISH",
                    confidence=0.75, target_pct=target * 100
                ))

    # ── Double Top ────────────────────────────────────────
    if len(ph) >= 2:
        t1_idx, t1 = ph[-2]
        t2_idx, t2 = ph[-1]
        if t2_idx > t1_idx and abs(t1 - t2) / max(t1, 1) < 0.025:
            neckline = min(low[t1_idx:t2_idx+1])
            if current_price < neckline * 1.02:
                target = (max(t1, t2) - neckline) / neckline
                patterns.append(ChartPattern(
                    name="Double Top", direction="BEARISH",
                    confidence=0.72, target_pct=target * 100
                ))

    # ── Head & Shoulders ─────────────────────────────────
    if len(ph) >= 3:
        ls_idx, ls = ph[-3]
        h_idx,  h  = ph[-2]
        rs_idx, rs = ph[-1]
        if (h > ls and h > rs
                and abs(ls - rs) / max(ls, 1) < 0.04
                and rs_idx > h_idx > ls_idx):
            neckline = min(
                min(low[ls_idx:h_idx+1]),
                min(low[h_idx:rs_idx+1])
            )
            if current_price < neckline * 1.02:
                target = (h - neckline) / neckline
                patterns.append(ChartPattern(
                    name="Head & Shoulders", direction="BEARISH",
                    confidence=0.80, target_pct=target * 100
                ))

    # ── Inverse Head & Shoulders ──────────────────────────
    if len(pl) >= 3:
        ls_idx, ls = pl[-3]
        h_idx,  h  = pl[-2]
        rs_idx, rs = pl[-1]
        if (h < ls and h < rs
                and abs(ls - rs) / max(ls, 1) < 0.04
                and rs_idx > h_idx > ls_idx):
            neckline = max(
                max(high[ls_idx:h_idx+1]),
                max(high[h_idx:rs_idx+1])
            )
            if current_price > neckline * 0.98:
                target = (neckline - h) / neckline
                patterns.append(ChartPattern(
                    name="Inv Head & Shoulders", direction="BULLISH",
                    confidence=0.80, target_pct=target * 100
                ))

    # ── Bull Flag ─────────────────────────────────────────
    if n >= 20:
        # Strong bullish move diikuti konsolidasi sempit
        pole_start = close[-20]
        pole_top   = np.max(close[-20:-10])
        flag_high  = np.max(close[-10:])
        flag_low   = np.min(close[-10:])
        pole_move  = (pole_top - pole_start) / pole_start if pole_start > 0 else 0
        flag_range = (flag_high - flag_low) / flag_high if flag_high > 0 else 1

        if pole_move > 0.05 and flag_range < 0.04 and current_price > pole_start:
            patterns.append(ChartPattern(
                name="Bull Flag", direction="BULLISH",
                confidence=0.70, target_pct=pole_move * 100
            ))

    # ── Bear Flag ─────────────────────────────────────────
    if n >= 20:
        pole_start = close[-20]
        pole_bot   = np.min(close[-20:-10])
        flag_high  = np.max(close[-10:])
        flag_low   = np.min(close[-10:])
        pole_move  = (pole_start - pole_bot) / pole_start if pole_start > 0 else 0
        flag_range = (flag_high - flag_low) / flag_high if flag_high > 0 else 1

        if pole_move > 0.05 and flag_range < 0.04 and current_price < pole_start:
            patterns.append(ChartPattern(
                name="Bear Flag", direction="BEARISH",
                confidence=0.70, target_pct=pole_move * 100
            ))

    # ── Ascending Triangle ────────────────────────────────
    if len(ph) >= 2 and len(pl) >= 2:
        high_flat = abs(ph[-1][1] - ph[-2][1]) / max(ph[-2][1], 1) < 0.02
        low_rising = pl[-1][1] > pl[-2][1]
        if high_flat and low_rising:
            patterns.append(ChartPattern(
                name="Ascending Triangle", direction="BULLISH",
                confidence=0.68, target_pct=5.0
            ))

    # ── Descending Triangle ───────────────────────────────
    if len(ph) >= 2 and len(pl) >= 2:
        low_flat    = abs(pl[-1][1] - pl[-2][1]) / max(pl[-2][1], 1) < 0.02
        high_falling = ph[-1][1] < ph[-2][1]
        if low_flat and high_falling:
            patterns.append(ChartPattern(
                name="Descending Triangle", direction="BEARISH",
                confidence=0.65, target_pct=5.0
            ))

    return patterns


# ═══════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════

def analyze_technical(symbol: str, current_price: float) -> TechnicalResult:
    """
    Entry point utama — jalankan semua analisis teknikal.
    Menggunakan 1H candle (200 candle = ~8 hari) untuk analisis utama.
    """
    res = TechnicalResult()

    # ── Ambil data candle ──────────────────────────────────
    df = get_klines(symbol, interval="1h", limit=200)
    if df.empty or len(df) < 50:
        logger.warning(f"TA: data kurang untuk {symbol}")
        return res

    res.timeframe_used = "1h"

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    n     = len(close)

    # ═══════════════════════════════════════════════════════
    #  1. MOMENTUM INDICATORS
    # ═══════════════════════════════════════════════════════

    # RSI
    rsi_vals = _rsi(close, 14)
    valid_rsi = rsi_vals[~np.isnan(rsi_vals)]
    if len(valid_rsi) >= 2:
        res.rsi_14 = float(valid_rsi[-1])
        rsi_prev   = float(valid_rsi[-2])

        if res.rsi_14 <= 30:
            res.rsi_signal = "OVERSOLD"
        elif res.rsi_14 >= 70:
            res.rsi_signal = "OVERBOUGHT"
        elif res.rsi_14 <= 40 and rsi_prev < res.rsi_14:
            res.rsi_signal = "RECOVERING"
        else:
            res.rsi_signal = "NEUTRAL"

        # RSI Divergence (harga buat LL tapi RSI buat HL = bullish divergence)
        if n >= 20:
            price_ll = close[-1] < np.min(close[-20:-1])
            rsi_hl   = res.rsi_14 > float(np.nanmin(rsi_vals[-20:-1]))
            if price_ll and rsi_hl and res.rsi_14 < 50:
                res.rsi_signal = "DIVERGENCE_BULL"

            price_hh = close[-1] > np.max(close[-20:-1])
            rsi_lh   = res.rsi_14 < float(np.nanmax(rsi_vals[-20:-1]))
            if price_hh and rsi_lh and res.rsi_14 > 50:
                res.rsi_signal = "DIVERGENCE_BEAR"

    # MACD
    macd_l, macd_s, macd_h = _macd(close)
    valid_mask = ~(np.isnan(macd_l) | np.isnan(macd_s))
    if valid_mask.sum() >= 2:
        idxs = np.where(valid_mask)[0]
        last, prev = idxs[-1], idxs[-2]
        res.macd_line   = float(macd_l[last])
        res.macd_signal = float(macd_s[last])
        res.macd_hist   = float(macd_h[last]) if not np.isnan(macd_h[last]) else 0.0

        ml_prev = macd_l[prev]
        ms_prev = macd_s[prev]
        mh_prev = macd_h[prev] if not np.isnan(macd_h[prev]) else 0.0

        # Bullish crossover
        if ml_prev <= ms_prev and res.macd_line > res.macd_signal:
            res.macd_signal_type = "BULLISH_CROSS"
        # Bearish crossover
        elif ml_prev >= ms_prev and res.macd_line < res.macd_signal:
            res.macd_signal_type = "BEARISH_CROSS"
        # Histogram growing bullish
        elif res.macd_hist > 0 and res.macd_hist > mh_prev:
            res.macd_signal_type = "BULLISH"
        elif res.macd_hist < 0 and res.macd_hist < mh_prev:
            res.macd_signal_type = "BEARISH"
        else:
            res.macd_signal_type = "NEUTRAL"

    # Stochastic RSI
    stoch_k, stoch_d = _stoch_rsi(close)
    valid_k = stoch_k[~np.isnan(stoch_k)]
    valid_d = stoch_d[~np.isnan(stoch_d)]
    if len(valid_k) >= 1 and len(valid_d) >= 1:
        res.stoch_rsi_k = float(valid_k[-1])
        res.stoch_rsi_d = float(valid_d[-1])
        if res.stoch_rsi_k < 20:
            res.stoch_signal = "OVERSOLD"
        elif res.stoch_rsi_k > 80:
            res.stoch_signal = "OVERBOUGHT"
        elif res.stoch_rsi_k > res.stoch_rsi_d and res.stoch_rsi_k < 50:
            res.stoch_signal = "BULLISH_CROSS"
        else:
            res.stoch_signal = "NEUTRAL"

    # ═══════════════════════════════════════════════════════
    #  2. TREND INDICATORS
    # ═══════════════════════════════════════════════════════

    ema9   = _ema(close, 9)
    ema21  = _ema(close, 21)
    ema50  = _ema(close, 50)
    ema200 = _ema(close, 200)

    def safe_last(arr):
        valid = arr[~np.isnan(arr)]
        return float(valid[-1]) if len(valid) > 0 else 0.0

    res.ema9   = safe_last(ema9)
    res.ema21  = safe_last(ema21)
    res.ema50  = safe_last(ema50)
    res.ema200 = safe_last(ema200)

    # Trend alignment
    price = current_price
    if res.ema200 > 0:
        res.price_vs_ema200 = "ABOVE" if price > res.ema200 else "BELOW"

    bull_count = sum([
        price > res.ema9   > 0,
        price > res.ema21  > 0,
        price > res.ema50  > 0,
        price > res.ema200 > 0,
        res.ema9  > res.ema21  > 0,
        res.ema21 > res.ema50  > 0,
    ])

    if bull_count >= 6:
        res.trend_alignment = "STRONG_BULL"
    elif bull_count >= 4:
        res.trend_alignment = "BULL"
    elif bull_count <= 1:
        res.trend_alignment = "STRONG_BEAR"
    elif bull_count <= 2:
        res.trend_alignment = "BEAR"
    else:
        res.trend_alignment = "SIDEWAYS"

    # Supertrend
    if n >= 15:
        st_dir, _, _ = _supertrend(high, low, close, period=10, multiplier=3.0)
        res.supertrend_dir = "BULLISH" if st_dir[-1] == 1 else "BEARISH"

    # ═══════════════════════════════════════════════════════
    #  3. VOLATILITY
    # ═══════════════════════════════════════════════════════

    bb_upper, bb_mid, bb_lower = _bollinger(close, 20, 2.0)
    atr_vals = _atr(high, low, close, 14)

    if not np.isnan(bb_upper[-1]):
        res.bb_upper  = float(bb_upper[-1])
        res.bb_middle = float(bb_mid[-1])
        res.bb_lower  = float(bb_lower[-1])
        res.bb_width  = (res.bb_upper - res.bb_lower) / res.bb_middle if res.bb_middle > 0 else 0

        if price > res.bb_upper:
            res.bb_position = "ABOVE_UPPER"
        elif price > res.bb_middle + (res.bb_upper - res.bb_middle) * 0.7:
            res.bb_position = "NEAR_UPPER"
        elif price < res.bb_lower:
            res.bb_position = "BELOW_LOWER"
        elif price < res.bb_middle - (res.bb_middle - res.bb_lower) * 0.7:
            res.bb_position = "NEAR_LOWER"
        else:
            res.bb_position = "MIDDLE"

    valid_atr = atr_vals[~np.isnan(atr_vals)]
    if len(valid_atr) >= 2:
        res.atr_14  = float(valid_atr[-1])
        res.atr_pct = res.atr_14 / price * 100 if price > 0 else 0

    # Bollinger Squeeze (BB width < 20% of 20-period average = coiling)
    if n >= 20:
        avg_bb_width = np.nanmean((bb_upper[-20:] - bb_lower[-20:]) / (bb_mid[-20:] + 1e-10))
        current_width = res.bb_width
        if current_width < avg_bb_width * 0.5 and current_width > 0:
            res.bb_squeeze = True

    # ═══════════════════════════════════════════════════════
    #  4. SMART MONEY CONCEPTS
    # ═══════════════════════════════════════════════════════

    fvgs = detect_fvg(df, lookback=40)
    obs  = detect_order_blocks(df, lookback=60)

    res.fvg_zones    = fvgs
    res.order_blocks = obs

    # FVG yang paling dekat ke harga saat ini
    unfilled_bull_fvg = [f for f in fvgs if f.kind == "BULLISH" and not f.filled and f.top < price]
    unfilled_bear_fvg = [f for f in fvgs if f.kind == "BEARISH" and not f.filled and f.bottom > price]

    if unfilled_bull_fvg:
        res.nearest_bull_fvg = max(unfilled_bull_fvg, key=lambda x: x.top)
        dist = (price - res.nearest_bull_fvg.top) / price
        if dist < 0.03:  # dalam 3%
            res.price_in_fvg = True

    if unfilled_bear_fvg:
        res.nearest_bear_fvg = min(unfilled_bear_fvg, key=lambda x: x.bottom)

    # Order Block terdekat
    valid_bull_ob = [o for o in obs if o.kind == "BULLISH" and o.valid and o.top < price]
    valid_bear_ob = [o for o in obs if o.kind == "BEARISH" and o.valid and o.bottom > price]

    if valid_bull_ob:
        res.nearest_bull_ob = max(valid_bull_ob, key=lambda x: x.top)
        dist = (price - res.nearest_bull_ob.top) / price
        if dist < 0.04:  # dalam 4%
            res.price_in_ob = True

    if valid_bear_ob:
        res.nearest_bear_ob = min(valid_bear_ob, key=lambda x: x.bottom)

    # BOS & CHoCH
    res.bos_detected, res.choch_detected = detect_bos_choch(df)

    # ═══════════════════════════════════════════════════════
    #  5. SUPPORT & RESISTANCE
    # ═══════════════════════════════════════════════════════

    supports, resists = detect_support_resistance(df, price, "1h")
    res.support_levels = supports
    res.resist_levels  = resists

    if supports:
        res.nearest_support  = supports[0].price
        res.support_dist_pct = (price - res.nearest_support) / price * 100
        res.price_at_support = res.support_dist_pct < 2.5  # dalam 2.5%

    if resists:
        res.nearest_resist  = resists[0].price
        res.resist_dist_pct = (res.nearest_resist - price) / price * 100
        res.price_at_resist = res.resist_dist_pct < 2.5

    # ═══════════════════════════════════════════════════════
    #  6. CHART PATTERNS
    # ═══════════════════════════════════════════════════════

    res.patterns = detect_chart_patterns(df, price)
    if res.patterns:
        bullish_p = [p for p in res.patterns if p.direction == "BULLISH"]
        bearish_p = [p for p in res.patterns if p.direction == "BEARISH"]
        if bullish_p:
            res.dominant_pattern = max(bullish_p, key=lambda x: x.confidence)
        elif bearish_p:
            res.dominant_pattern = max(bearish_p, key=lambda x: x.confidence)

    # ═══════════════════════════════════════════════════════
    #  7. SCORING & SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════

    score = 0.0

    # ── RSI signals ───────────────────────────────────────
    if res.rsi_signal == "OVERSOLD":
        res.signals.append(f"💚 RSI {res.rsi_14:.1f} — OVERSOLD (zona beli)")
        score += 7
    elif res.rsi_signal == "DIVERGENCE_BULL":
        res.signals.append(f"💚 RSI Bullish DIVERGENCE {res.rsi_14:.1f} — harga LL tapi RSI HL!")
        score += 10
    elif res.rsi_signal == "RECOVERING":
        res.signals.append(f"📈 RSI {res.rsi_14:.1f} — Recovery dari oversold")
        score += 4
    elif res.rsi_signal == "OVERBOUGHT":
        res.signals.append(f"🔴 RSI {res.rsi_14:.1f} — OVERBOUGHT (waspada)")
        score -= 3
    elif res.rsi_signal == "DIVERGENCE_BEAR":
        res.signals.append(f"🔴 RSI Bearish Divergence — harga HH tapi RSI LH")
        score -= 5
    else:
        res.signals.append(f"⚪ RSI {res.rsi_14:.1f} — Neutral")

    # ── MACD signals ──────────────────────────────────────
    if res.macd_signal_type == "BULLISH_CROSS":
        res.signals.append(f"💚 MACD GOLDEN CROSS! Line > Signal (momentum balik BULLISH)")
        score += 10
    elif res.macd_signal_type == "BEARISH_CROSS":
        res.signals.append(f"🔴 MACD DEATH CROSS — momentum balik BEARISH")
        score -= 8
    elif res.macd_signal_type == "BULLISH":
        res.signals.append(f"📈 MACD histogram bullish growing ({res.macd_hist:+.6f})")
        score += 5
    elif res.macd_signal_type == "BEARISH":
        res.signals.append(f"📉 MACD histogram bearish expanding ({res.macd_hist:+.6f})")
        score -= 4

    # ── Stochastic RSI ────────────────────────────────────
    if res.stoch_signal == "OVERSOLD":
        res.signals.append(f"💚 Stoch RSI {res.stoch_rsi_k:.1f} — OVERSOLD")
        score += 5
    elif res.stoch_signal == "BULLISH_CROSS":
        res.signals.append(f"💚 Stoch RSI bullish cross di zona rendah")
        score += 4
    elif res.stoch_signal == "OVERBOUGHT":
        res.signals.append(f"🔴 Stoch RSI {res.stoch_rsi_k:.1f} — OVERBOUGHT")
        score -= 2

    # ── EMA Trend ─────────────────────────────────────────
    if res.trend_alignment == "STRONG_BULL":
        res.signals.append(f"🚀 EMA Alignment STRONG BULL (9>21>50>200, harga di atas semua)")
        score += 8
    elif res.trend_alignment == "BULL":
        res.signals.append(f"📈 EMA Alignment BULLISH (mayoritas EMA bullish)")
        score += 5
    elif res.trend_alignment == "STRONG_BEAR":
        res.signals.append(f"📉 EMA Alignment STRONG BEAR — jangan LONG dulu")
        score -= 6
    elif res.trend_alignment == "BEAR":
        res.signals.append(f"📉 EMA Alignment BEARISH")
        score -= 3

    if res.supertrend_dir == "BULLISH":
        res.signals.append(f"✅ Supertrend: BULLISH (harga di atas supertrend line)")
        score += 4
    elif res.supertrend_dir == "BEARISH":
        res.signals.append(f"⚠️ Supertrend: BEARISH (harga di bawah supertrend line)")
        score -= 3

    # ── Bollinger Bands ───────────────────────────────────
    if res.bb_squeeze:
        res.signals.append(f"🌀 BB SQUEEZE — Volatility sangat rendah, BREAKOUT IMMINENT!")
        score += 8
    if res.bb_position == "BELOW_LOWER":
        res.signals.append(f"💚 Harga di bawah BB Lower — ekstrem oversold, potensi reversal")
        score += 6
    elif res.bb_position == "NEAR_LOWER":
        res.signals.append(f"💚 Harga dekat BB Lower — potential bounce")
        score += 3
    elif res.bb_position == "ABOVE_UPPER":
        res.signals.append(f"⚠️ Harga di atas BB Upper — overbought / momentum kuat")
        score -= 2

    # ── Smart Money Concepts ──────────────────────────────
    if res.price_in_fvg and res.nearest_bull_fvg:
        fvg = res.nearest_bull_fvg
        res.signals.append(
            f"🧲 BULLISH FVG UNFILLED di ${fvg.bottom:.6f}–${fvg.top:.6f} "
            f"(harga masuk zona imbalance)"
        )
        score += 8

    elif res.nearest_bull_fvg:
        fvg = res.nearest_bull_fvg
        dist_fvg = (price - fvg.top) / price * 100
        if dist_fvg < 5:
            res.signals.append(f"🧲 Bullish FVG di bawah {dist_fvg:.1f}% (support kuat)")
            score += 4

    if res.price_in_ob and res.nearest_bull_ob:
        ob = res.nearest_bull_ob
        res.signals.append(
            f"🟩 BULLISH ORDER BLOCK ${ob.bottom:.6f}–${ob.top:.6f} "
            f"(zona institusional support)"
        )
        score += 7

    if res.nearest_bear_fvg:
        dist_bear_fvg = (res.nearest_bear_fvg.bottom - price) / price * 100
        if dist_bear_fvg < 5:
            res.signals.append(f"🟥 Bearish FVG resistance {dist_bear_fvg:.1f}% di atas")
            score -= 2

    if res.bos_detected:
        res.signals.append("🏗️ Break of Structure (BOS) terdeteksi — struktur BULLISH!")
        score += 6

    if res.choch_detected:
        res.signals.append("🔄 Change of Character (CHoCH) — waspada pembalikan struktur")
        score -= 4

    # ── Support & Resistance ──────────────────────────────
    if res.price_at_support and res.nearest_support > 0:
        res.signals.append(
            f"🟢 Harga dekat SUPPORT kuat ${res.nearest_support:.6f} "
            f"(jarak {res.support_dist_pct:.1f}%)"
        )
        score += 6

    if res.price_at_resist and res.nearest_resist > 0:
        res.signals.append(
            f"🔴 Harga dekat RESISTANCE ${res.nearest_resist:.6f} "
            f"(jarak {res.resist_dist_pct:.1f}%) — tunggu breakout"
        )
        score -= 3

    if res.nearest_resist > 0 and res.resist_dist_pct < 8:
        res.signals.append(
            f"📏 Resistance terdekat: ${res.nearest_resist:.6f} "
            f"({res.resist_dist_pct:.1f}% di atas)"
        )
    if res.nearest_support > 0 and res.support_dist_pct < 5:
        res.signals.append(
            f"📏 Support terdekat: ${res.nearest_support:.6f} "
            f"({res.support_dist_pct:.1f}% di bawah)"
        )

    # ── Chart Patterns ────────────────────────────────────
    for pat in res.patterns:
        if pat.direction == "BULLISH":
            res.signals.append(
                f"📐 Chart Pattern: {pat.name} (BULLISH, "
                f"target +{pat.target_pct:.1f}%, conf {pat.confidence*100:.0f}%)"
            )
            score += pat.confidence * 6
        elif pat.direction == "BEARISH":
            res.signals.append(
                f"📐 Chart Pattern: {pat.name} (BEARISH, "
                f"target -{pat.target_pct:.1f}%, conf {pat.confidence*100:.0f}%)"
            )
            score -= pat.confidence * 4

    # ── Final TA Bias ─────────────────────────────────────
    if score >= 25:
        res.ta_bias = "STRONG_BULL"
    elif score >= 12:
        res.ta_bias = "BULL"
    elif score <= -15:
        res.ta_bias = "STRONG_BEAR"
    elif score <= -5:
        res.ta_bias = "BEAR"
    else:
        res.ta_bias = "NEUTRAL"

    res.score = max(0.0, min(score, 30.0))

    return res
