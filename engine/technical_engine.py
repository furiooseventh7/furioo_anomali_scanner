"""
Technical Engine — analisis indikator teknikal klasik.
RSI, MACD, Bollinger Bands, EMA, ATR, Volume Spike.
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from [engine.data](http://engine.data)_fetcher import get_klines

logger = logging.getLogger(__name__)

@dataclass
class TechnicalResult:
    rsi_1h              : float = 50.0
    rsi_4h              : float = 50.0
    macd_signal         : str   = "NEUTRAL"
    macd_histogram      : float = 0.0
    bb_position         : str   = "MID"
    bb_squeeze          : bool  = False
    ema_trend           : str   = "NEUTRAL"
    ema_golden_cross    : bool  = False
    ema_death_cross     : bool  = False
    price_above_ema200  : bool  = False
    atr_pct             : float = 0.0
    volume_spike_20     : float = 1.0
    signals             : list  = field(default_factory=list)
    score               : float = 0.0


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _calc_macd(close: pd.Series):
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_bb(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma     = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper   = sma + std_mult * std_dev
    lower   = sma - std_mult * std_dev
    return upper, sma, lower, std_dev


def analyze_technical(symbol: str) -> TechnicalResult:
    res = TechnicalResult()

    df_1h = get_klines(symbol, interval="1h", limit=200)
    df_4h = get_klines(symbol, interval="4h", limit=100)

    if df_1h.empty or len(df_1h) < 50:
        return res

    close_1h = df_1h["close"].astype(float)
    high_1h  = df_1h["high"].astype(float)
    low_1h   = df_1h["low"].astype(float)
    vol_1h   = df_1h["volume"].astype(float)

    # RSI
    res.rsi_1h = _calc_rsi(close_1h, 14)
    if not df_4h.empty and len(df_4h) >= 20:
        res.rsi_4h = _calc_rsi(df_4h["close"].astype(float), 14)

    # MACD
    macd_line, signal_line, histogram = _calc_macd(close_1h)
    res.macd_histogram = float(histogram.iloc[-1]) if not histogram.empty else 0.0
    if len(histogram) >= 2:
        prev_hist = float(histogram.iloc[-2])
        curr_hist = float(histogram.iloc[-1])
        if prev_hist < 0 and curr_hist > 0:
            res.macd_signal = "BULLISH_CROSS"
        elif prev_hist > 0 and curr_hist < 0:
            res.macd_signal = "BEARISH_CROSS"
        elif curr_hist > 0:
            res.macd_signal = "BULLISH"
        elif curr_hist < 0:
            res.macd_signal = "BEARISH"

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_std = _calc_bb(close_1h)
    price_now = float(close_1h.iloc[-1])
    if not bb_upper.empty:
        upper = float(bb_upper.iloc[-1])
        lower = float(bb_lower.iloc[-1])
        mid   = float(bb_mid.iloc[-1])
        band_width = (upper - lower) / mid if mid > 0 else 0
        [res.bb](http://res.bb)_squeeze = band_width < 0.05
        if price_now <= lower * 1.01:
            [res.bb](http://res.bb)_position = "LOWER"
        elif price_now >= upper * 0.99:
            [res.bb](http://res.bb)_position = "UPPER"
        elif [res.bb](http://res.bb)_squeeze:
            [res.bb](http://res.bb)_position = "SQUEEZE"
        else:
            [res.bb](http://res.bb)_position = "MID"

    # EMA
    ema20  = close_1h.ewm(span=20, adjust=False).mean()
    ema50  = close_1h.ewm(span=50, adjust=False).mean()
    ema200 = close_1h.ewm(span=200, adjust=False).mean()
    e20  = float(ema20.iloc[-1])
    e50  = float(ema50.iloc[-1])
    e200 = float(ema200.iloc[-1])

    res.price_above_ema200 = price_now > e200
    if price_now > e20 > e50 > e200:
        res.ema_trend = "BULLISH"
    elif price_now < e20 < e50 < e200:
        res.ema_trend = "BEARISH"

    if len(ema20) >= 2 and len(ema50) >= 2:
        if float(ema20.iloc[-2]) < float(ema50.iloc[-2]) and e20 > e50:
            res.ema_golden_cross = True
        elif float(ema20.iloc[-2]) > float(ema50.iloc[-2]) and e20 < e50:
            res.ema_death_cross = True

    # ATR
    tr = pd.concat([
        high_1h - low_1h,
        (high_1h - close_1h.shift()).abs(),
        (low_1h  - close_1h.shift()).abs(),
    ], axis=1).
