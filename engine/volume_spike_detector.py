"""
CEX Volume Spike Detector — CMI-ASS v7
=========================================
Mendeteksi lonjakan volume abnormal di CEX (Bitget & MEXC) yang sudah
terintegrasi, dengan analisis mendalam dan konteks yang jelas di alert.

CARA KERJA:
  1. Ambil kline 1H 30 hari terakhir dari Bitget/MEXC (sudah ada di data_fetcher)
  2. Hitung baseline volume: moving average 7D & 30D
  3. Deteksi spike jika volume jam ini > threshold * baseline
  4. Analisis multi-dimensi:
     - Volume spike magnitude (berapa X dari rata-rata)
     - Arah spike: buy-driven vs sell-driven (via taker ratio)
     - Harga saat spike: breakout? retrace? vs support/resistance?
     - Konteks: apakah spike ini dengan atau berlawanan tren?
     - Konsistensi: spike terjadi di beberapa TF? (1H + 4H + 1D)
  5. Klasifikasi spike:
     - BREAKOUT_BUY: volume spike + harga breakout resistance → bullish
     - ACCUMULATION: volume spike moderat + harga stabil / sideways
     - STOP_HUNT: volume spike + harga spike tapi langsung balik
     - DUMP: volume spike + harga turun tajam
     - NOISE: spike kecil, tidak signifikan

THRESHOLD DEFAULT:
  - CRITICAL: > 10x dari rata-rata 7D (extremely rare, very high conviction)
  - HIGH:     > 5x  dari rata-rata 7D
  - MEDIUM:   > 3x  dari rata-rata 7D
  - LOW:      > 2x  dari rata-rata 7D
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from engine.data_fetcher import get_klines, get_agg_trades

logger = logging.getLogger(__name__)

# Threshold multiplier untuk spike detection
SPIKE_THRESHOLDS = {
    "CRITICAL": 10.0,   # > 10x rata-rata
    "HIGH":      5.0,
    "MEDIUM":    3.0,
    "LOW":       2.0,
}

# Minimal volume absolut (USD) supaya tidak false positive pada koin kecil
MIN_SPIKE_VOL_USD = 100_000   # $100K minimum


@dataclass
class VolumeSpikeResult:
    """Hasil analisis volume spike di CEX."""
    symbol              : str   = ""

    # Spike metrics
    has_spike           : bool  = False
    spike_magnitude     : float = 0.0      # berapa X dari rata-rata (misal 7.3x)
    spike_level         : str   = "NONE"   # NONE / LOW / MEDIUM / HIGH / CRITICAL
    spike_type          : str   = "NONE"   # BREAKOUT_BUY / ACCUMULATION / STOP_HUNT / DUMP / NOISE

    # Volume data
    current_vol_usd     : float = 0.0      # volume jam terakhir (USD)
    avg_vol_7d_usd      : float = 0.0      # rata-rata per jam 7 hari
    avg_vol_30d_usd     : float = 0.0      # rata-rata per jam 30 hari
    volume_24h_vs_7d    : float = 0.0      # volume 24H vs rata-rata 7D (ratio)

    # Direction analysis
    buy_spike           : bool  = False    # spike didominasi buyer
    sell_spike          : bool  = False    # spike didominasi seller
    taker_buy_ratio_now : float = 0.5      # taker buy ratio jam terakhir
    taker_buy_ratio_avg : float = 0.5      # taker buy ratio rata-rata 7D

    # Price context
    price_at_spike      : float = 0.0
    price_change_during : float = 0.0     # % perubahan harga selama spike
    above_resistance    : bool  = False   # harga breakout resistance?
    near_support        : bool  = False   # harga di dekat support?
    price_rejected      : bool  = False   # spike tapi harga langsung balik

    # Multi-TF consistency
    spike_1h            : bool  = False
    spike_4h            : bool  = False
    spike_1d            : bool  = False
    tf_consistency      : int   = 0       # berapa TF yang konfirmasi

    # Consecutive spike
    consecutive_spike_hrs: int  = 0       # berapa jam berturut spike
    sustained_volume    : bool  = False   # volume tetap tinggi > 3 jam

    # Output
    signals             : List[str] = field(default_factory=list)
    score               : float = 0.0    # 0–25, masuk ke confidence score
    description         : str   = ""     # narasi untuk Telegram


# ─────────────────────────────────────────────────────────
#  CORE SPIKE ANALYSIS
# ─────────────────────────────────────────────────────────

def _calc_baseline(df: pd.DataFrame, col: str = "volume") -> Tuple[float, float]:
    """
    Hitung baseline volume: rata-rata 7D dan 30D per jam.
    Returns (avg_7d, avg_30d)
    """
    if df.empty or col not in df.columns:
        return 0.0, 0.0
    vol = df[col].replace(0, np.nan).dropna()
    if len(vol) < 10:
        return 0.0, 0.0

    avg_7d  = float(vol.tail(168).mean())   # 7D × 24H = 168 candles
    avg_30d = float(vol.mean())
    return avg_7d, avg_30d


def _classify_spike_type(
    spike_mag: float,
    buy_ratio: float,
    price_chg: float,
    price_rejected: bool,
) -> str:
    """
    Klasifikasi tipe spike berdasarkan direction dan price action.

    Rules:
    - buy_ratio > 0.65 + price_chg > 1% + tidak rejected → BREAKOUT_BUY
    - buy_ratio > 0.60 + price_chg flat (< 1%) → ACCUMULATION
    - buy_ratio < 0.40 + price_chg < -1% → DUMP
    - spike tinggi + price_rejected → STOP_HUNT
    - lainnya → NOISE
    """
    if price_rejected and spike_mag >= 3.0:
        return "STOP_HUNT"
    elif buy_ratio >= 0.65 and price_chg >= 1.0:
        return "BREAKOUT_BUY"
    elif buy_ratio >= 0.60 and abs(price_chg) < 1.5:
        return "ACCUMULATION"
    elif buy_ratio <= 0.40 and price_chg <= -1.5:
        return "DUMP"
    elif buy_ratio >= 0.55 and price_chg >= 0.3:
        return "BULLISH_PRESSURE"
    else:
        return "NOISE"


def _detect_spike_in_df(
    df: pd.DataFrame,
    avg_7d: float,
    min_vol_usd: float = MIN_SPIKE_VOL_USD,
    threshold: float = 2.0,
) -> Tuple[bool, float, int]:
    """
    Deteksi spike dari kline DataFrame.
    Returns: (has_spike, magnitude, consecutive_hours)
    """
    if df.empty or avg_7d <= 0:
        return False, 0.0, 0

    price_col  = "close"
    vol_col    = "volume"
    qvol_col   = "quote_volume" if "quote_volume" in df.columns else None

    # Gunakan quote_volume (USD) jika tersedia, else volume
    if qvol_col and df[qvol_col].iloc[-1] > 0:
        recent_vol = float(df[qvol_col].iloc[-1])
        avg        = float(df[qvol_col].mean())
        if avg <= 0:
            avg = avg_7d
    else:
        recent_vol = float(df[vol_col].iloc[-1])
        avg        = avg_7d

    if recent_vol < min_vol_usd:
        return False, 0.0, 0

    magnitude = recent_vol / avg if avg > 0 else 0

    if magnitude < threshold:
        return False, magnitude, 0

    # Hitung consecutive spike hours
    consec = 0
    for i in range(len(df) - 1, max(len(df) - 24, -1), -1):
        v = float(df[qvol_col].iloc[i]) if qvol_col else float(df[vol_col].iloc[i])
        if v > avg * (threshold * 0.7):   # 70% threshold masih dianggap "spike zone"
            consec += 1
        else:
            break

    return True, magnitude, consec


# ─────────────────────────────────────────────────────────
#  MULTI-TF ANALYSIS
# ─────────────────────────────────────────────────────────

def _analyze_tf(symbol: str, interval: str, limit: int,
                threshold: float) -> Tuple[bool, float, float, float]:
    """
    Analisis satu timeframe.
    Returns: (has_spike, magnitude, taker_ratio, price_chg_pct)
    """
    df = get_klines(symbol, interval=interval, limit=limit)
    if df.empty:
        return False, 0.0, 0.5, 0.0

    avg_7d, _ = _calc_baseline(df, "quote_volume" if "quote_volume" in df.columns else "volume")
    has_spike, mag, _ = _detect_spike_in_df(df, avg_7d, threshold=threshold)

    # Taker buy ratio (jika tersedia)
    tb = 0.5
    if "taker_buy_base" in df.columns and "volume" in df.columns:
        total_tb  = df["taker_buy_base"].tail(5).sum()
        total_vol = df["volume"].tail(5).sum()
        tb = total_tb / total_vol if total_vol > 0 else 0.5

    # Price change dari 5 candle lalu ke sekarang
    if len(df) >= 5:
        price_now  = float(df["close"].iloc[-1])
        price_prev = float(df["close"].iloc[-5])
        price_chg  = (price_now - price_prev) / price_prev * 100 if price_prev > 0 else 0.0
    else:
        price_chg = 0.0

    return has_spike, mag, tb, price_chg


# ─────────────────────────────────────────────────────────
#  PRICE CONTEXT DETECTION
# ─────────────────────────────────────────────────────────

def _check_price_context(df_1h: pd.DataFrame, price: float) -> dict:
    """
    Deteksi konteks harga saat spike terjadi.
    Returns dict: {above_resistance, near_support, price_rejected, support, resistance}
    """
    ctx = {
        "above_resistance": False,
        "near_support":     False,
        "price_rejected":   False,
        "support":          0.0,
        "resistance":       0.0,
    }
    if df_1h.empty or len(df_1h) < 20:
        return ctx

    high  = df_1h["high"].values
    low   = df_1h["low"].values
    close = df_1h["close"].values

    # Simple support: lowest low 20 candle terakhir (kecuali 3 terakhir)
    recent_low  = np.min(low[-20:-3])
    recent_high = np.max(high[-20:-3])

    ctx["support"]    = float(recent_low)
    ctx["resistance"] = float(recent_high)

    # Price rejected: spike tinggi di candle terakhir tapi harga tutup rendah
    last_candle_range = high[-1] - low[-1]
    last_close_pos    = (close[-1] - low[-1]) / last_candle_range if last_candle_range > 0 else 0.5
    if last_close_pos < 0.35:   # tutup di bawah 35% dari range = rejection
        ctx["price_rejected"] = True

    # Above resistance
    if price > recent_high * 1.01:
        ctx["above_resistance"] = True

    # Near support (dalam 2%)
    if abs(price - recent_low) / recent_low < 0.02:
        ctx["near_support"] = True

    return ctx


# ─────────────────────────────────────────────────────────
#  SCORE CALCULATOR
# ─────────────────────────────────────────────────────────

def _build_score_and_signals(result: VolumeSpikeResult) -> None:
    """Hitung skor 0–25 dan bangun daftar signals."""
    score   = 0.0
    signals = []

    if not result.has_spike:
        result.score   = 0.0
        result.signals = []
        return

    # ── Magnitude score ───────────────────────────────────
    m = result.spike_magnitude
    if m >= SPIKE_THRESHOLDS["CRITICAL"]:
        score += 15
        signals.append(
            f"🚨 CRITICAL VOLUME SPIKE: <b>{m:.1f}x</b> rata-rata 7D! "
            f"(${result.current_vol_usd:,.0f} jam ini vs avg ${result.avg_vol_7d_usd:,.0f})"
        )
    elif m >= SPIKE_THRESHOLDS["HIGH"]:
        score += 10
        signals.append(
            f"🔥 Volume spike <b>{m:.1f}x</b> rata-rata 7D "
            f"(${result.current_vol_usd:,.0f} vs avg ${result.avg_vol_7d_usd:,.0f})"
        )
    elif m >= SPIKE_THRESHOLDS["MEDIUM"]:
        score += 6
        signals.append(
            f"📊 Volume spike <b>{m:.1f}x</b> rata-rata 7D"
        )
    elif m >= SPIKE_THRESHOLDS["LOW"]:
        score += 3
        signals.append(f"📈 Volume sedikit meningkat {m:.1f}x rata-rata")

    # ── Direction bonus ───────────────────────────────────
    if result.buy_spike:
        score += 5
        signals.append(
            f"🟢 Spike didominasi BUY: taker ratio {result.taker_buy_ratio_now*100:.0f}% "
            f"(avg {result.taker_buy_ratio_avg*100:.0f}%)"
        )
    elif result.sell_spike:
        score -= 5
        signals.append(
            f"🔴 Spike didominasi SELL: taker ratio {result.taker_buy_ratio_now*100:.0f}%"
        )

    # ── Spike type bonus/penalty ──────────────────────────
    type_bonus = {
        "BREAKOUT_BUY":    8,
        "ACCUMULATION":    5,
        "BULLISH_PRESSURE": 3,
        "STOP_HUNT":       2,   # stop hunt bisa bullish (liquidity sweep)
        "DUMP":           -8,
        "NOISE":          -2,
    }
    bonus = type_bonus.get(result.spike_type, 0)
    score += bonus

    type_signals = {
        "BREAKOUT_BUY":
            f"🚀 BREAKOUT BUY: Volume spike bersamaan dengan breakout harga! "
            f"(+{result.price_change_during:.1f}% selama spike)",
        "ACCUMULATION":
            f"🐋 ACCUMULATION: Volume spike tapi harga stabil → smart money kumpulkan diam-diam",
        "BULLISH_PRESSURE":
            f"📈 Tekanan beli meningkat (+{result.price_change_during:.1f}%)",
        "STOP_HUNT":
            f"💧 STOP HUNT: Volume spike tapi harga langsung balik → liquidity sweep terjadi",
        "DUMP":
            f"📉 DUMP: Volume spike dengan penjualan masif (-{abs(result.price_change_during):.1f}%)",
        "NOISE":
            f"⚠️ Volume spike tapi arah tidak jelas",
    }
    if result.spike_type in type_signals:
        signals.append(type_signals[result.spike_type])

    # ── Multi-TF consistency ──────────────────────────────
    if result.tf_consistency >= 3:
        score += 5
        signals.append(f"🔭 Multi-TF confluence: spike terjadi di {result.tf_consistency} timeframe sekaligus!")
    elif result.tf_consistency == 2:
        score += 2
        signals.append(f"🔭 Spike dikonfirmasi di {result.tf_consistency} TF")

    # ── Sustained volume ──────────────────────────────────
    if result.sustained_volume:
        score += 3
        signals.append(
            f"⏱️ Volume spike SUSTAINED: berlangsung {result.consecutive_spike_hrs} jam berturut-turut "
            f"(bukan fake pump sesaat)"
        )
    elif result.consecutive_spike_hrs >= 2:
        score += 1
        signals.append(f"⏱️ Spike berlangsung {result.consecutive_spike_hrs} jam")

    # ── Price context bonus ───────────────────────────────
    if result.above_resistance and result.buy_spike:
        score += 3
        signals.append(f"🆙 Breakout CONFIRMED: harga di atas resistance dengan volume tinggi!")
    elif result.near_support and result.buy_spike:
        score += 2
        signals.append(f"🟢 Volume spike di area support — kemungkinan bounce")

    # ── 24H volume momentum ───────────────────────────────
    if result.volume_24h_vs_7d >= 3.0:
        score += 2
        signals.append(f"📊 Volume 24H adalah {result.volume_24h_vs_7d:.1f}x lebih besar dari rata-rata 7D")

    result.score   = max(0.0, min(score, 25.0))
    result.signals = signals

    # ── Build description ─────────────────────────────────
    type_desc = {
        "BREAKOUT_BUY":    "Volume Breakout Bullish",
        "ACCUMULATION":    "Akumulasi Diam-diam",
        "BULLISH_PRESSURE": "Tekanan Beli Meningkat",
        "STOP_HUNT":       "Liquidity Sweep / Stop Hunt",
        "DUMP":            "Distribusi / Selling Pressure",
        "NOISE":           "Volume Spike (arah belum jelas)",
    }.get(result.spike_type, "Volume Anomali")

    result.description = (
        f"{m:.1f}x spike | {type_desc} | "
        f"Buy ratio: {result.taker_buy_ratio_now*100:.0f}% | "
        f"${result.current_vol_usd:,.0f} jam ini"
    )


# ─────────────────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────────────────

def analyze_volume_spike(symbol: str, current_price: float = 0.0) -> VolumeSpikeResult:
    """
    Analisis volume spike komprehensif untuk satu simbol.
    Menggunakan Bitget/MEXC data yang sudah ada (gratis, no new API key).
    """
    result = VolumeSpikeResult(symbol=symbol)

    # ── 1. Ambil kline 1H (30 hari = 720 candle) ─────────
    df_1h = get_klines(symbol, interval="1h", limit=720)
    if df_1h.empty or len(df_1h) < 24:
        return result

    # Pilih kolom volume USD
    vol_col = "quote_volume" if "quote_volume" in df_1h.columns and df_1h["quote_volume"].iloc[-1] > 0 else "volume"
    price_col = "close"

    if df_1h[vol_col].iloc[-1] < MIN_SPIKE_VOL_USD and vol_col == "volume":
        # Coba estimasi USD: volume × price
        if current_price > 0:
            df_1h["vol_usd_est"] = df_1h["volume"] * df_1h["close"]
            vol_col = "vol_usd_est"

    # ── 2. Baseline 7D & 30D ──────────────────────────────
    avg_7d, avg_30d = _calc_baseline(df_1h, vol_col)
    result.avg_vol_7d_usd  = avg_7d
    result.avg_vol_30d_usd = avg_30d

    if avg_7d <= 0:
        return result

    result.current_vol_usd = float(df_1h[vol_col].iloc[-1])

    # ── 3. Spike detection 1H ─────────────────────────────
    spike_1h, mag_1h, consec = _detect_spike_in_df(
        df_1h, avg_7d, threshold=SPIKE_THRESHOLDS["LOW"]
    )
    result.spike_1h           = spike_1h
    result.spike_magnitude    = mag_1h
    result.consecutive_spike_hrs = consec
    result.sustained_volume   = consec >= 3

    # ── 4. Taker buy ratio ────────────────────────────────
    if "taker_buy_base" in df_1h.columns and "volume" in df_1h.columns:
        tb_now  = df_1h["taker_buy_base"].iloc[-3:].sum()
        vol_now = df_1h["volume"].iloc[-3:].sum()
        result.taker_buy_ratio_now = tb_now / vol_now if vol_now > 0 else 0.5

        tb_all  = df_1h["taker_buy_base"].tail(168).sum()
        vol_all = df_1h["volume"].tail(168).sum()
        result.taker_buy_ratio_avg = tb_all / vol_all if vol_all > 0 else 0.5
    else:
        # Fallback: estimasi dari agg_trades
        try:
            df_trades = get_agg_trades(symbol, limit=200)
            if not df_trades.empty:
                buy_vol  = df_trades[df_trades["is_aggressive_buy"]]["value"].sum()
                total_vol = df_trades["value"].sum()
                result.taker_buy_ratio_now = buy_vol / total_vol if total_vol > 0 else 0.5
                result.taker_buy_ratio_avg = result.taker_buy_ratio_now
        except Exception:
            pass

    result.buy_spike  = result.taker_buy_ratio_now >= 0.60
    result.sell_spike = result.taker_buy_ratio_now <= 0.40

    # ── 5. Price change during spike ──────────────────────
    if len(df_1h) >= 4:
        p_now  = float(df_1h["close"].iloc[-1])
        p_prev = float(df_1h["close"].iloc[-4])
        result.price_at_spike      = p_now
        result.price_change_during = (p_now - p_prev) / p_prev * 100 if p_prev > 0 else 0.0

    # ── 6. Price context ──────────────────────────────────
    ctx = _check_price_context(df_1h, result.price_at_spike or current_price)
    result.above_resistance = ctx["above_resistance"]
    result.near_support     = ctx["near_support"]
    result.price_rejected   = ctx["price_rejected"]

    # ── 7. Multi-TF confirmation ──────────────────────────
    # 4H
    try:
        spike_4h, mag_4h, tb_4h, chg_4h = _analyze_tf(symbol, "4h", 90, SPIKE_THRESHOLDS["LOW"])
        result.spike_4h = spike_4h
    except Exception:
        spike_4h = False

    # 1D
    try:
        spike_1d, mag_1d, tb_1d, chg_1d = _analyze_tf(symbol, "1d", 30, SPIKE_THRESHOLDS["LOW"])
        result.spike_1d = spike_1d
    except Exception:
        spike_1d = False

    result.tf_consistency = sum([spike_1h, spike_4h, spike_1d])

    # ── 8. 24H vs 7D volume momentum ─────────────────────
    vol_24h = df_1h[vol_col].tail(24).sum()
    vol_7d_daily_avg = avg_7d * 24   # total USD per hari
    result.volume_24h_vs_7d = vol_24h / vol_7d_daily_avg if vol_7d_daily_avg > 0 else 0

    # ── 9. Final spike classification ─────────────────────
    if spike_1h or result.volume_24h_vs_7d >= 2.0:
        result.has_spike = True
        if not spike_1h:
            result.spike_magnitude = result.volume_24h_vs_7d

    if result.has_spike:
        # Spike level
        m = result.spike_magnitude
        if m >= SPIKE_THRESHOLDS["CRITICAL"]:
            result.spike_level = "CRITICAL"
        elif m >= SPIKE_THRESHOLDS["HIGH"]:
            result.spike_level = "HIGH"
        elif m >= SPIKE_THRESHOLDS["MEDIUM"]:
            result.spike_level = "MEDIUM"
        else:
            result.spike_level = "LOW"

        # Spike type
        result.spike_type = _classify_spike_type(
            result.spike_magnitude,
            result.taker_buy_ratio_now,
            result.price_change_during,
            result.price_rejected,
        )

    # ── 10. Build score + signals ─────────────────────────
    _build_score_and_signals(result)

    if result.has_spike:
        logger.info(
            f"VolSpike | {symbol:16s} | {result.spike_level:8s} | "
            f"{result.spike_magnitude:.1f}x | {result.spike_type:20s} | "
            f"BuyRatio:{result.taker_buy_ratio_now*100:.0f}% | "
            f"${result.current_vol_usd:,.0f} | Score:{result.score:.0f}"
        )

    return result


# ─────────────────────────────────────────────────────────
#  BATCH SCANNER — mencari spike di semua koin sekaligus
# ─────────────────────────────────────────────────────────

def scan_volume_spikes_batch(
    ticker_df: pd.DataFrame,
    min_spike_x: float = 3.0,
    max_coins: int = 50,
) -> List[VolumeSpikeResult]:
    """
    Scan cepat volume spike dari ticker 24H tanpa fetch kline.
    Berguna untuk pre-filtering sebelum analisis mendalam.

    Metode: bandingkan quoteVolume 24H dengan estimasi rata-rata historis.
    (Hanya menggunakan data ticker yang sudah di-fetch di main.py)
    """
    results = []
    if ticker_df.empty:
        return results

    # Hitung "anomaly score" dari ticker 24H saja
    # Harga yang naik tinggi dengan volume yang sangat besar = kandidat
    df = ticker_df.copy()
    if "quoteVolume" not in df.columns:
        return results

    # Normalisasi: ranking volume di antara semua koin
    df["vol_rank"] = df["quoteVolume"].rank(pct=True)
    df["chg_abs"]  = df["priceChangePercent"].abs()

    # Kandidat: volume top 15% DAN perubahan harga signifikan
    candidates = df[
        (df["vol_rank"] >= 0.85) &
        (df["chg_abs"] >= 2.0) &
        (df["quoteVolume"] >= MIN_SPIKE_VOL_USD)
    ].nlargest(max_coins, "quoteVolume")

    for _, row in candidates.iterrows():
        try:
            sym    = row["symbol"]
            vol    = float(row["quoteVolume"])
            chg    = float(row["priceChangePercent"])
            price  = float(row["lastPrice"])

            # Quick spike estimate: jika vol_rank > 95% = likely spike
            quick_mag = float(row["vol_rank"]) * 10   # rough estimate
            if quick_mag < min_spike_x:
                continue

            res = VolumeSpikeResult(symbol=sym)
            res.has_spike          = True
            res.spike_magnitude    = quick_mag
            res.current_vol_usd    = vol
            res.price_change_during = chg
            res.price_at_spike     = price
            res.buy_spike          = chg > 0
            res.sell_spike         = chg < 0
            res.taker_buy_ratio_now = 0.65 if chg > 0 else 0.35

            # Quick level
            if quick_mag >= SPIKE_THRESHOLDS["CRITICAL"]:
                res.spike_level = "CRITICAL"
            elif quick_mag >= SPIKE_THRESHOLDS["HIGH"]:
                res.spike_level = "HIGH"
            else:
                res.spike_level = "MEDIUM"

            res.spike_type = _classify_spike_type(
                quick_mag, res.taker_buy_ratio_now, chg, False
            )
            _build_score_and_signals(res)
            results.append(res)
        except Exception:
            continue

    results.sort(key=lambda x: x.spike_magnitude, reverse=True)
    return results


# ─────────────────────────────────────────────────────────
#  TELEGRAM FORMATTER
# ─────────────────────────────────────────────────────────

def format_volume_spike_section(result: VolumeSpikeResult) -> str:
    """Format volume spike untuk bagian di dalam Telegram signal."""
    if not result.has_spike or result.score < 3:
        return ""

    lvl_emoji = {
        "CRITICAL": "🚨", "HIGH": "🔥", "MEDIUM": "📊", "LOW": "📈"
    }.get(result.spike_level, "📈")
    type_emoji = {
        "BREAKOUT_BUY":    "🚀",
        "ACCUMULATION":    "🐋",
        "BULLISH_PRESSURE": "📈",
        "STOP_HUNT":       "💧",
        "DUMP":            "📉",
        "NOISE":           "⚠️",
    }.get(result.spike_type, "📊")

    lines = [f"\n📊 <b>Volume Spike Detector (CEX)</b>"]
    lines.append(
        f"  {lvl_emoji} Level: <b>{result.spike_level}</b>  |  "
        f"{type_emoji} Tipe: <b>{result.spike_type.replace('_',' ')}</b>"
    )
    lines.append(
        f"  📈 Magnitude: <b>{result.spike_magnitude:.1f}x</b> rata-rata 7D"
    )
    lines.append(
        f"  💵 Volume jam ini: <b>${result.current_vol_usd:,.0f}</b>"
        f"  (avg ${result.avg_vol_7d_usd:,.0f})"
    )
    lines.append(
        f"  🟢 Buy ratio: <b>{result.taker_buy_ratio_now*100:.0f}%</b>"
        f"  (avg {result.taker_buy_ratio_avg*100:.0f}%)"
    )

    if result.price_change_during:
        chg_e = "📈" if result.price_change_during > 0 else "📉"
        lines.append(f"  {chg_e} Harga selama spike: <b>{result.price_change_during:+.2f}%</b>")

    if result.sustained_volume:
        lines.append(f"  ⏱️ Volume sustained <b>{result.consecutive_spike_hrs} jam</b> berturut-turut")
    if result.tf_consistency >= 2:
        tfs = []
        if result.spike_1h: tfs.append("1H")
        if result.spike_4h: tfs.append("4H")
        if result.spike_1d: tfs.append("1D")
        lines.append(f"  🔭 Multi-TF spike: <b>{' + '.join(tfs)}</b>")
    if result.above_resistance:
        lines.append(f"  🆙 Breakout di atas resistance!")
    if result.near_support and result.buy_spike:
        lines.append(f"  🟢 Volume spike di area support")
    if result.price_rejected:
        lines.append(f"  ⚡ Harga rejected setelah spike — kemungkinan stop hunt/liquidity sweep")

    if result.volume_24h_vs_7d >= 2:
        lines.append(f"  📊 Volume 24H = <b>{result.volume_24h_vs_7d:.1f}x</b> rata-rata 7D")

    lines.append(f"  📌 Score: <b>{result.score:.0f}/25</b>")
    lines.append("")
    return "\n".join(lines)


def format_standalone_spike_alert(result: VolumeSpikeResult) -> str:
    """Alert standalone untuk spike yang sangat signifikan."""
    if not result.has_spike:
        return ""

    lvl_e = {"CRITICAL": "🚨", "HIGH": "🔥", "MEDIUM": "📊", "LOW": "📈"}.get(result.spike_level, "📊")
    sym   = result.symbol.replace("USDT", "")
    type_desc = {
        "BREAKOUT_BUY":    "🚀 BREAKOUT BUY",
        "ACCUMULATION":    "🐋 AKUMULASI TERSEMBUNYI",
        "BULLISH_PRESSURE": "📈 TEKANAN BELI KUAT",
        "STOP_HUNT":       "💧 STOP HUNT / LIQUIDITY SWEEP",
        "DUMP":            "📉 DUMP / DISTRIBUSI",
    }.get(result.spike_type, "⚠️ VOLUME ANOMALI")

    msg  = f"{lvl_e} <b>VOLUME SPIKE — ${sym}</b> {lvl_e}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Tipe:      <b>{type_desc}</b>\n"
    msg += f"📈 Magnitude: <b>{result.spike_magnitude:.1f}x</b> rata-rata 7D\n"
    msg += f"💵 Volume 1H: <b>${result.current_vol_usd:,.0f}</b>\n"
    msg += f"💵 Avg 7D/H:  ${result.avg_vol_7d_usd:,.0f}\n"
    msg += f"🟢 Buy ratio: {result.taker_buy_ratio_now*100:.0f}% (avg {result.taker_buy_ratio_avg*100:.0f}%)\n"
    if result.price_change_during:
        msg += f"💰 Harga: {result.price_change_during:+.2f}% selama spike\n"
    if result.sustained_volume:
        msg += f"⏱️ Berlangsung {result.consecutive_spike_hrs} jam berturut-turut\n"
    if result.tf_consistency >= 2:
        msg += f"🔭 Dikonfirmasi di {result.tf_consistency} timeframe!\n"
    msg += f"\n📌 Signals:\n"
    for s in result.signals[:5]:
        msg += f"  • {s}\n"
    msg += f"\n⚡ CMI-ASS v7 | CEX Volume Tracker | DYOR!"
    return msg
