"""
Precision Entry Engine — CMI-ASS v6
======================================
Engine khusus untuk menentukan kapan tepatnya MASUK posisi.
Lebih matang dari sekadar "entry zone" — ini mencakup:

1. WHALE ACCUMULATION PHASE DETECTOR
   - Fase 1 (Markup imminent): Akumulasi selesai, harga akan naik SEGERA
   - Fase 2 (Accumulation in progress): Whale akumulasi tapi harga masih akan turun dulu
   - Fase 3 (Distribution): Whale distribusi, jangan masuk

2. WYCKOFF PHASE DETECTION
   - Accumulation Schematic: PS → SC → AR → ST → Spring → SOS → LPS
   - Distribution Schematic: PSY → BC → AR → ST → SOW → LPSY
   - Mendeteksi apakah kita di Spring (best buy zone) atau di LPSY (trap)

3. ENTRY TIMING MATRIX
   Menggabungkan:
   - Liquidity sweep status (stop hunt selesai?)
   - OB/FVG proximity (harga di dalam zona?)
   - Volume profile (accumulation volume vs distribution volume)
   - Funding rate (market overshorted/overlonged?)
   - Momentum divergence (RSI divergence di key level?)

4. OUTPUT:
   - entry_timing: "NOW" | "WAIT_DIP" | "WAIT_BREAKOUT" | "AVOID"
   - wait_level: target harga yang harus dicapai sebelum entry (jika WAIT_DIP)
   - timing_confidence: 0-100
   - timing_reason: penjelasan detail
   - whale_phase: "MARKUP_IMMINENT" | "ACCUMULATING" | "DISTRIBUTING" | "NEUTRAL"
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EntryTimingResult:
    """Hasil analisis timing entry yang presisi."""

    # Core decision
    entry_timing        : str   = "WAIT"          # NOW | WAIT_DIP | WAIT_BREAKOUT | AVOID
    whale_phase         : str   = "NEUTRAL"        # MARKUP_IMMINENT | ACCUMULATING | DISTRIBUTING | NEUTRAL
    timing_confidence   : float = 0.0              # 0-100

    # Target levels
    ideal_entry_price   : float = 0.0              # harga ideal entry (center of zone)
    entry_low           : float = 0.0              # batas bawah zona entry
    entry_high          : float = 0.0              # batas atas zona entry
    wait_target_price   : float = 0.0              # target harga sebelum entry (jika WAIT_DIP)
    wait_reason         : str   = ""               # kenapa harus tunggu

    # Stop Loss yang tepat
    precision_sl        : float = 0.0              # SL berbasis struktur, bukan %
    sl_invalidation     : str   = ""               # kondisi yang invalidate setup

    # Wyckoff context
    wyckoff_phase       : str   = "UNKNOWN"        # PS/SC/AR/ST/SPRING/SOS/LPS/LPSY
    wyckoff_schematic   : str   = "UNKNOWN"        # ACCUMULATION | DISTRIBUTION | NEUTRAL

    # Confluence flags
    liquidity_swept     : bool  = False             # stop hunt sudah selesai?
    in_fvg              : bool  = False             # harga di dalam FVG?
    in_ob               : bool  = False             # harga di dalam Order Block?
    rsi_divergence      : bool  = False             # ada bullish divergence?
    funding_favorable   : bool  = False             # funding rate mendukung long?
    volume_confirms     : bool  = False             # volume profile bullish?
    mtf_aligned         : bool  = False             # multi-TF setuju arah?

    # Signal narrative
    timing_narrative    : str   = ""               # narasi lengkap untuk Telegram
    key_signals         : List[str] = field(default_factory=list)
    warnings            : List[str] = field(default_factory=list)

    # Score contribution
    score               : float = 0.0              # bonus/penalty untuk decision engine


# ─────────────────────────────────────────────────────────
#  WYCKOFF PHASE DETECTOR
# ─────────────────────────────────────────────────────────

def _detect_wyckoff(df: pd.DataFrame, current_price: float) -> Tuple[str, str]:
    """
    Detect Wyckoff accumulation/distribution phase dari price action.

    Returns: (phase, schematic)
    phase: PS | SC | AR | ST | SPRING | SOS | LPS | MARKUP | LPSY | UNKNOWN
    schematic: ACCUMULATION | DISTRIBUTION | MARKUP | NEUTRAL
    """
    if df is None or len(df) < 50:
        return "UNKNOWN", "NEUTRAL"

    try:
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))

        # Kalkulasi rolling stats
        n = len(close)
        win = min(20, n // 3)

        # Price range analysis
        range_high = np.max(high[-win*2:])
        range_low  = np.min(low[-win*2:])
        range_size = range_high - range_low
        if range_size == 0:
            return "UNKNOWN", "NEUTRAL"

        # Current position dalam range
        pos_in_range = (current_price - range_low) / range_size  # 0=bottom, 1=top

        # Volume analysis: akumulasi vs distribusi
        mid = len(close) // 2
        vol_first_half  = np.mean(volume[:mid])
        vol_second_half = np.mean(volume[mid:])
        vol_recent      = np.mean(volume[-5:])
        vol_trend       = vol_second_half / (vol_first_half + 1e-9)

        # Price volatility: makin rendah = makin konsolidasi
        price_std     = np.std(close[-win:])
        price_std_pct = price_std / (np.mean(close[-win:]) + 1e-9)

        # Trend dari 50 candle terakhir
        if len(close) >= 50:
            trend_slope = (close[-1] - close[-50]) / (close[-50] + 1e-9)
        else:
            trend_slope = 0

        # ── ACCUMULATION DETECTION ─────────────────────────
        # Ciri akumulasi:
        # 1. Harga di lower 30% range
        # 2. Volume mengering (tidak banyak yang jual)
        # 3. Sideways / konsolidasi
        # 4. Ada selling climax (volume spike di bawah)

        # Spring detection: harga sempat tembus bawah range tapi langsung balik
        recent_low   = np.min(low[-10:])
        prev_low     = np.min(low[-30:-10]) if len(low) >= 30 else range_low
        spring_cond  = recent_low < prev_low and current_price > prev_low

        # Selling climax: volume tinggi diikuti harga bounce
        max_vol_idx  = np.argmax(volume[-30:]) if len(volume) >= 30 else 0
        sc_price     = close[-(30 - max_vol_idx)] if len(close) >= 30 else close[0]
        sc_cond      = sc_price < np.mean(close[-30:]) and current_price > sc_price

        # SOS: Sign of Strength — breakout dengan volume tinggi
        sos_cond = (current_price > range_high * 0.95 and vol_recent > np.mean(volume) * 1.5)

        # LPS: Last Point of Support — pullback ke support setelah SOS
        lps_cond = (
            current_price < range_high * 0.90 and
            current_price > range_low * 1.05 and
            trend_slope > 0
        )

        if spring_cond and pos_in_range < 0.25:
            return "SPRING", "ACCUMULATION"
        elif sos_cond:
            return "SOS", "ACCUMULATION"
        elif lps_cond and trend_slope > 0.02:
            return "LPS", "ACCUMULATION"
        elif sc_cond and pos_in_range < 0.30:
            return "SC", "ACCUMULATION"
        elif pos_in_range < 0.20 and price_std_pct < 0.03:
            return "ST", "ACCUMULATION"   # Secondary Test / Consolidation bottom

        # ── DISTRIBUTION DETECTION ──────────────────────────
        # Ciri distribusi:
        # 1. Harga di upper 70% range
        # 2. Volume tinggi tapi harga tidak naik (buying climax)
        # 3. SOW: Sign of Weakness — breakdown dengan volume tinggi

        bc_cond  = (pos_in_range > 0.80 and vol_recent > np.mean(volume) * 1.5 and trend_slope < 0.01)
        sow_cond = (current_price < range_low * 1.05 and vol_recent > np.mean(volume) * 1.3)
        lpsy_cond = (pos_in_range > 0.60 and pos_in_range < 0.80 and trend_slope < 0)

        if sow_cond:
            return "SOW", "DISTRIBUTION"
        elif bc_cond:
            return "BC", "DISTRIBUTION"   # Buying Climax
        elif lpsy_cond:
            return "LPSY", "DISTRIBUTION"

        # ── MARKUP / NEUTRAL ────────────────────────────────
        if trend_slope > 0.05 and pos_in_range > 0.50:
            return "MARKUP", "MARKUP"

        return "RANGING", "NEUTRAL"

    except Exception as e:
        logger.debug(f"Wyckoff detection error: {e}")
        return "UNKNOWN", "NEUTRAL"


# ─────────────────────────────────────────────────────────
#  WHALE PHASE ANALYZER
# ─────────────────────────────────────────────────────────

def _analyze_whale_phase(
    whale_res,
    ta_res,
    deriv_res,
    wyckoff_phase: str,
    wyckoff_schematic: str,
) -> Tuple[str, List[str]]:
    """
    Tentukan fase whale: MARKUP_IMMINENT | ACCUMULATING | DISTRIBUTING | NEUTRAL

    MARKUP_IMMINENT: Whale sudah selesai akumulasi, harga akan naik SEGERA
    ACCUMULATING:    Whale masih akumulasi, harga mungkin masih turun dulu
    DISTRIBUTING:    Whale sedang jual, JANGAN masuk
    """
    signals = []
    buy_pressure  = getattr(whale_res, "buy_pressure", 0.5) if whale_res else 0.5
    is_acc        = getattr(whale_res, "is_accumulating", False) if whale_res else False
    is_dist       = getattr(whale_res, "is_distributing", False) if whale_res else False
    ob_imbalance  = getattr(whale_res, "ob_imbalance", 0) if whale_res else 0
    taker_ratio   = getattr(whale_res, "taker_buy_ratio", 0.5) if whale_res else 0.5
    funding       = getattr(deriv_res, "funding_rate", 0) if deriv_res else 0
    liq_swept     = getattr(ta_res, "liquidity_swept", False) if ta_res else False

    acc_score = 0
    dist_score = 0

    # ── Accumulation signals ─────────────────────────────
    if is_acc:
        acc_score += 2
        signals.append("✅ Taker buy meningkat — akumulasi aktif")

    if buy_pressure >= 0.75:
        acc_score += 3
        signals.append(f"🐋 Whale buy pressure {buy_pressure*100:.0f}%")
    elif buy_pressure >= 0.60:
        acc_score += 1
        signals.append(f"🦈 Buy pressure moderat {buy_pressure*100:.0f}%")

    if ob_imbalance >= 0.25:
        acc_score += 2
        signals.append("📚 Bid wall kuat — support terjaga")

    if taker_ratio >= 0.60:
        acc_score += 1

    # Liquidity sweep = stop hunt selesai = ready to move up
    if liq_swept:
        acc_score += 3
        signals.append("💧 Stop hunt SELESAI — liquidity sweep terdeteksi")

    # Wyckoff context
    if wyckoff_phase in ("SPRING", "LPS"):
        acc_score += 3
        signals.append(f"📐 Wyckoff: {wyckoff_phase} — zona beli ideal")
    elif wyckoff_phase in ("SOS",):
        acc_score += 4
        signals.append("📐 Wyckoff: SOS — kekuatan beli terverifikasi")
    elif wyckoff_phase in ("SC",):
        acc_score += 1
        signals.append("📐 Wyckoff: Selling Climax — bisa mulai akumulasi")

    # Funding rate negatif = pasar overshorted = potensi squeeze
    if funding is not None and funding < -0.0005:
        acc_score += 2
        signals.append(f"💰 Funding negatif {funding*100:.4f}% — potensi short squeeze")

    # ── Distribution signals ──────────────────────────────
    if is_dist:
        dist_score += 3
        signals.append("⚠️ Pola DISTRIBUSI terdeteksi")

    if buy_pressure < 0.40:
        dist_score += 2
        signals.append(f"🔴 Whale sell pressure {(1-buy_pressure)*100:.0f}%")

    if wyckoff_phase in ("BC", "SOW", "LPSY"):
        dist_score += 3
        signals.append(f"📐 Wyckoff: {wyckoff_phase} — zona distribusi/lemah")

    if funding is not None and funding > 0.002:
        dist_score += 1
        signals.append(f"⚠️ Funding positif tinggi {funding*100:.4f}% — overlonged")

    # ── Determine phase ───────────────────────────────────
    if dist_score > acc_score and dist_score >= 3:
        return "DISTRIBUTING", signals

    if acc_score >= 8 and dist_score < 2:
        return "MARKUP_IMMINENT", signals
    elif acc_score >= 4:
        return "ACCUMULATING", signals
    else:
        return "NEUTRAL", signals


# ─────────────────────────────────────────────────────────
#  ENTRY TIMING MATRIX
# ─────────────────────────────────────────────────────────

def _calc_entry_timing(
    whale_phase     : str,
    wyckoff_phase   : str,
    ta_res,
    deriv_res,
    current_price   : float,
) -> Tuple[str, float, str]:
    """
    Tentukan: entry NOW, WAIT_DIP, WAIT_BREAKOUT, atau AVOID.
    Returns: (timing, wait_target_price, reason)
    """
    liq_swept   = getattr(ta_res, "liquidity_swept", False) if ta_res else False
    has_entry   = getattr(ta_res, "has_precise_entry", False) if ta_res else False
    entry_low   = getattr(ta_res, "optimal_entry_low", 0) if ta_res else 0
    entry_high  = getattr(ta_res, "optimal_entry_high", 0) if ta_res else 0
    in_disc     = getattr(ta_res, "in_discount_zone", False) if ta_res else False
    in_prem     = getattr(ta_res, "in_premium_zone", False) if ta_res else False
    mtf_agree   = getattr(ta_res, "mtf_agree_count", 0) if ta_res else 0

    # ── AVOID: distribusi jelas ─────────────────────────────
    if whale_phase == "DISTRIBUTING":
        return "AVOID", 0, "Whale sedang DISTRIBUSI — tunggu konfirmasi reversal sebelum entry"

    if wyckoff_phase in ("LPSY", "SOW"):
        return "AVOID", 0, f"Wyckoff {wyckoff_phase} — tanda kelemahan, jangan masuk sekarang"

    # ── NOW: setup sempurna ─────────────────────────────────
    if (whale_phase == "MARKUP_IMMINENT" and
            liq_swept and
            (in_disc or (has_entry and entry_low <= current_price <= entry_high)) and
            mtf_agree >= 2):
        return "NOW", 0, (
            "✅ Setup MATANG: Whale markup imminent + Stop hunt selesai + "
            "Harga di zona ideal + MTF confluence"
        )

    if wyckoff_phase in ("SPRING", "LPS") and whale_phase in ("MARKUP_IMMINENT", "ACCUMULATING"):
        return "NOW", 0, f"Wyckoff {wyckoff_phase} — zona beli terbaik, whale masih akumulasi"

    # ── WAIT_DIP: akumulasi tapi harga masih premium ────────
    if whale_phase == "ACCUMULATING" and in_prem:
        target = entry_low if entry_low > 0 else current_price * 0.95
        return "WAIT_DIP", target, (
            f"Whale akumulasi tapi harga di zona PREMIUM — "
            f"tunggu pullback ke {target:.6f} sebelum entry"
        )

    if whale_phase == "ACCUMULATING" and not liq_swept:
        target = entry_low if entry_low > 0 else current_price * 0.97
        return "WAIT_DIP", target, (
            f"Whale akumulasi tapi stop hunt belum terjadi — "
            f"kemungkinan akan turun dulu ke {target:.6f} untuk sweep liquidity"
        )

    # ── WAIT_BREAKOUT: setup bullish tapi butuh konfirmasi ──
    if wyckoff_phase == "ST" or (whale_phase == "ACCUMULATING" and mtf_agree < 2):
        resist = getattr(ta_res, "nearest_resist", 0) if ta_res else 0
        target = resist if resist > current_price else current_price * 1.03
        return "WAIT_BREAKOUT", target, (
            f"Sedang konsolidasi — tunggu breakout konfirmasi di atas {target:.6f}"
        )

    # ── Default: timing moderat ─────────────────────────────
    if has_entry and entry_low > 0 and current_price > entry_high:
        target = entry_high * 0.995
        return "WAIT_DIP", target, f"Harga di atas zona entry SMC — tunggu retrace ke {target:.6f}"

    return "NOW", 0, "Setup cukup baik untuk entry di zona saat ini"


# ─────────────────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────────────────

def analyze_precision_entry(
    symbol      : str,
    price       : float,
    whale_res,
    ta_res,
    deriv_res,
    df_1h       : Optional[pd.DataFrame] = None,
    df_4h       : Optional[pd.DataFrame] = None,
) -> EntryTimingResult:
    """
    Engine utama untuk precision entry timing.

    Args:
        symbol:    simbol coin
        price:     harga current
        whale_res: WhaleSonarResult
        ta_res:    TechnicalResult
        deriv_res: DerivativesResult
        df_1h:     kline 1H dataframe
        df_4h:     kline 4H dataframe (prioritas untuk Wyckoff)
    """
    result = EntryTimingResult()

    try:
        # 1. Wyckoff detection (gunakan 4H untuk phase, 1H untuk confirmation)
        df_for_wyckoff = df_4h if df_4h is not None and len(df_4h) >= 50 else df_1h
        wyckoff_phase, wyckoff_schematic = _detect_wyckoff(df_for_wyckoff, price)
        result.wyckoff_phase    = wyckoff_phase
        result.wyckoff_schematic = wyckoff_schematic

        # 2. Whale phase
        whale_phase, whale_signals = _analyze_whale_phase(
            whale_res, ta_res, deriv_res, wyckoff_phase, wyckoff_schematic
        )
        result.whale_phase  = whale_phase
        result.key_signals  = whale_signals

        # 3. Confluence flags
        result.liquidity_swept  = getattr(ta_res, "liquidity_swept", False) if ta_res else False
        result.in_fvg           = _check_in_fvg(ta_res, price)
        result.in_ob            = _check_in_ob(ta_res, price)
        result.rsi_divergence   = _check_rsi_divergence(df_1h, price)
        result.funding_favorable = _check_funding(deriv_res)
        result.volume_confirms  = _check_volume(df_1h)
        result.mtf_aligned      = (getattr(ta_res, "mtf_agree_count", 0) >= 2) if ta_res else False

        # 4. Entry timing decision
        timing, wait_target, timing_reason = _calc_entry_timing(
            whale_phase, wyckoff_phase, ta_res, deriv_res, price
        )
        result.entry_timing       = timing
        result.wait_target_price  = wait_target
        result.wait_reason        = timing_reason

        # 5. Entry zone (dari TA atau default)
        el = getattr(ta_res, "optimal_entry_low", 0) if ta_res else 0
        eh = getattr(ta_res, "optimal_entry_high", 0) if ta_res else 0
        if el > 0 and eh > 0:
            result.entry_low  = el
            result.entry_high = eh
            result.ideal_entry_price = (el + eh) / 2
        else:
            # Fallback entry zone ±2% dari harga
            result.entry_low  = price * 0.98
            result.entry_high = price * 1.005
            result.ideal_entry_price = price

        # 6. Precision SL dari structure
        sl_struct = getattr(ta_res, "structure_sl", 0) if ta_res else 0
        if sl_struct > 0:
            result.precision_sl = sl_struct
            result.sl_invalidation = f"Close candle 4H di bawah {sl_struct:.6f}"
        else:
            result.precision_sl = price * 0.95

        # 7. Timing confidence score (0-100)
        conf = 0
        confluence_hits = sum([
            result.liquidity_swept,
            result.in_fvg or result.in_ob,
            result.rsi_divergence,
            result.funding_favorable,
            result.volume_confirms,
            result.mtf_aligned,
            whale_phase in ("MARKUP_IMMINENT", "ACCUMULATING"),
            wyckoff_phase in ("SPRING", "LPS", "SOS"),
        ])
        conf = confluence_hits * 12
        if timing == "NOW":
            conf += 10
        elif timing == "AVOID":
            conf = max(0, conf - 40)
        result.timing_confidence = min(100, conf)

        # 8. Score contribution ke decision engine
        if timing == "AVOID":
            result.score = -10
        elif timing == "NOW" and whale_phase == "MARKUP_IMMINENT":
            result.score = 15
        elif timing == "NOW":
            result.score = 8
        elif timing == "WAIT_DIP":
            result.score = 3   # masih ada potential tapi butuh sabar
        else:
            result.score = 1

        # 9. Build narrative
        result.timing_narrative = _build_narrative(result, price)

        logger.info(
            f"PrecisionEntry | {symbol} | {whale_phase} | Wyckoff:{wyckoff_phase} | "
            f"Timing:{timing} | Conf:{result.timing_confidence:.0f}%"
        )

    except Exception as e:
        logger.error(f"PrecisionEntry error {symbol}: {e}", exc_info=True)

    return result


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────

def _check_in_fvg(ta_res, price: float) -> bool:
    """Cek apakah harga saat ini berada di dalam FVG."""
    try:
        for tf_data in getattr(ta_res, "tf_analyses", {}).values():
            for fvg in getattr(tf_data, "fvg_zones", []):
                if hasattr(fvg, "bottom") and hasattr(fvg, "top"):
                    if fvg.bottom <= price <= fvg.top and not getattr(fvg, "filled", True):
                        return True
    except Exception:
        pass
    return False


def _check_in_ob(ta_res, price: float) -> bool:
    """Cek apakah harga berada di dalam Order Block yang valid."""
    try:
        for tf_data in getattr(ta_res, "tf_analyses", {}).values():
            for ob in getattr(tf_data, "order_blocks", []):
                if hasattr(ob, "bottom") and hasattr(ob, "top"):
                    if ob.bottom <= price <= ob.top and getattr(ob, "valid", True):
                        return True
    except Exception:
        pass
    return False


def _check_rsi_divergence(df: Optional[pd.DataFrame], price: float) -> bool:
    """Deteksi bullish RSI divergence sederhana (harga lower low, RSI higher low)."""
    if df is None or len(df) < 30:
        return False
    try:
        close = df["close"].values
        # RSI 14
        delta = np.diff(close)
        gain  = np.where(delta > 0, delta, 0)
        loss  = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(14)/14, mode="valid")
        avg_loss = np.convolve(loss, np.ones(14)/14, mode="valid")
        rs   = avg_gain / (avg_loss + 1e-9)
        rsi  = 100 - (100 / (1 + rs))

        if len(rsi) < 20:
            return False

        # Cari 2 swing low terakhir
        rsi_recent = rsi[-20:]
        price_recent = close[-20:]
        low1_idx = np.argmin(rsi_recent[:10])
        low2_idx = np.argmin(rsi_recent[10:]) + 10

        price_div = price_recent[low2_idx] < price_recent[low1_idx]   # price: lower low
        rsi_div   = rsi_recent[low2_idx] > rsi_recent[low1_idx]        # RSI: higher low

        return price_div and rsi_div and rsi_recent[low2_idx] < 40
    except Exception:
        return False


def _check_funding(deriv_res) -> bool:
    """Funding rate favorable untuk long (negatif atau mendekati nol)."""
    funding = getattr(deriv_res, "funding_rate", None) if deriv_res else None
    if funding is None:
        return False
    return funding < 0.0001  # negatif atau sangat rendah


def _check_volume(df: Optional[pd.DataFrame]) -> bool:
    """Volume recent lebih rendah dari rata-rata = akumulasi diam (stealth accumulation)."""
    if df is None or len(df) < 20:
        return False
    try:
        vol = df["volume"].values
        recent_vol = np.mean(vol[-5:])
        avg_vol    = np.mean(vol[-20:])
        # Volume rendah di downtrend = selling exhaustion
        return recent_vol < avg_vol * 0.7
    except Exception:
        return False


def _build_narrative(result: EntryTimingResult, price: float) -> str:
    """Build narasi lengkap untuk Telegram."""
    phase_desc = {
        "MARKUP_IMMINENT": "🚀 MARKUP IMMINENT — Whale sudah selesai akumulasi, harga siap naik!",
        "ACCUMULATING":    "🐋 AKUMULASI — Whale masih kumpulkan posisi, harga mungkin turun dulu",
        "DISTRIBUTING":    "🔴 DISTRIBUSI — Whale sedang jual, hindari long sekarang",
        "NEUTRAL":         "⚪ Tidak ada sinyal whale dominan yang jelas",
    }.get(result.whale_phase, "")

    wyckoff_desc = {
        "SPRING":  "💎 Wyckoff SPRING — titik balik terbaik untuk akumulasi",
        "LPS":     "🎯 Wyckoff LPS — Last Point of Support, aman untuk entry",
        "SOS":     "✅ Wyckoff SOS — Sign of Strength dikonfirmasi",
        "SC":      "📉 Wyckoff Selling Climax — bisa mulai akumulasi kecil",
        "ST":      "↔️ Wyckoff Secondary Test — tunggu konfirmasi",
        "MARKUP":  "📈 Wyckoff Markup Phase — trend naik aktif",
        "LPSY":    "⚠️ Wyckoff LPSY — Last Point of Supply, hati-hati distribusi",
        "SOW":     "🔴 Wyckoff Sign of Weakness — jangan beli sekarang",
        "BC":      "🔴 Wyckoff Buying Climax — kemungkinan distribusi",
        "RANGING": "↔️ Sedang ranging/konsolidasi",
    }.get(result.wyckoff_phase, "")

    timing_desc = {
        "NOW":            "⚡ ENTRY SEKARANG — setup sudah matang",
        "WAIT_DIP":       f"⏳ TUNGGU PULLBACK ke {result.wait_target_price:.6g}",
        "WAIT_BREAKOUT":  f"⏳ TUNGGU BREAKOUT di atas {result.wait_target_price:.6g}",
        "AVOID":          "🚫 HINDARI — kondisi tidak mendukung long saat ini",
    }.get(result.entry_timing, "")

    lines = []
    if phase_desc:
        lines.append(phase_desc)
    if wyckoff_desc:
        lines.append(wyckoff_desc)
    if result.wait_reason:
        lines.append(f"💡 {result.wait_reason}")
    if timing_desc:
        lines.append(timing_desc)

    return "\n".join(lines)


def format_entry_timing_section(result: EntryTimingResult, price: float) -> str:
    """Format untuk Telegram message."""
    if result.whale_phase == "NEUTRAL" and result.entry_timing == "WAIT":
        return ""

    lines = ["\n🎯 <b>Precision Entry Analysis</b>"]

    # Whale phase
    phase_emoji = {
        "MARKUP_IMMINENT": "🚀",
        "ACCUMULATING":    "🐋",
        "DISTRIBUTING":    "🔴",
        "NEUTRAL":         "⚪",
    }.get(result.whale_phase, "⚪")
    lines.append(f"  {phase_emoji} <b>Whale Phase: {result.whale_phase.replace('_',' ')}</b>")

    # Wyckoff
    if result.wyckoff_phase not in ("UNKNOWN", "RANGING"):
        wy_emoji = {"SPRING": "💎", "LPS": "🎯", "SOS": "✅", "LPSY": "⚠️", "SOW": "🔴", "BC": "🔴", "MARKUP": "📈"}.get(result.wyckoff_phase, "📐")
        lines.append(f"  {wy_emoji} Wyckoff: <b>{result.wyckoff_phase}</b> ({result.wyckoff_schematic})")

    # Timing decision
    timing_emoji = {"NOW": "⚡", "WAIT_DIP": "⏳", "WAIT_BREAKOUT": "⏳", "AVOID": "🚫"}.get(result.entry_timing, "❓")
    lines.append(f"  {timing_emoji} <b>Timing: {result.entry_timing.replace('_',' ')}</b>  (Confidence: {result.timing_confidence:.0f}%)")

    if result.wait_target_price > 0:
        def _p(v):
            if v >= 1: return f"${v:.4f}"
            elif v >= 0.01: return f"${v:.5f}"
            else: return f"${v:.8f}"
        lines.append(f"  🎯 Target Entry: <b>{_p(result.wait_target_price)}</b>")

    if result.wait_reason and result.entry_timing != "NOW":
        lines.append(f"  💡 {result.wait_reason[:120]}")

    # Confluence checklist (singkat)
    checks = []
    if result.liquidity_swept:  checks.append("✅ Liq.Swept")
    if result.in_fvg:           checks.append("✅ In FVG")
    if result.in_ob:            checks.append("✅ In OB")
    if result.rsi_divergence:   checks.append("✅ RSI Div")
    if result.funding_favorable: checks.append("✅ Funding")
    if result.mtf_aligned:      checks.append("✅ MTF OK")
    if checks:
        lines.append(f"  📋 Confluence: {' | '.join(checks)}")

    if result.sl_invalidation:
        lines.append(f"  ⛔ Invalidasi jika: {result.sl_invalidation}")

    lines.append("")
    return "\n".join(lines)
