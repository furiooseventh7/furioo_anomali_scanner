"""
Signal Validator — CMI-ASS v5
==============================
Layer validasi FINAL sebelum sinyal dikirim ke Telegram.

Fungsi utama:
  1. PROSECUTION CASE — membangun argumen terstruktur KENAPA signal ini diambil
  2. QUALITY GATE — 7 kondisi wajib yang harus terpenuhi sebelum signal dikirim
  3. CONFIDENCE CALIBRATION — menyesuaikan skor akhir berdasarkan konfirmasi faktual
  4. ENTRY PRECISION SCORE — menilai seberapa tepat zona entry yang direkomendasikan
  5. RISK AUDIT — memeriksa apakah SL/TP sudah masuk akal
  6. CONTRADICTION CHECK — membatalkan signal jika ada kondisi yang saling bertentangan
  7. VERDICT — APPROVED / DOWNGRADED / REJECTED dengan alasan lengkap

Cara kerja dalam sistem:
  main.py → scan_single() → make_decision() → validate_signal() → kirim atau skip

Signal yang REJECTED tidak pernah dikirim ke Telegram.
Signal yang DOWNGRADED diturunkan alert level-nya dan ditambahkan warning.
Signal yang APPROVED dikirim dengan Prosecution Case lengkap.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

# Minimum skor per kategori untuk APPROVED
MIN_WHALE_SCORE_FOR_LONG   = 8.0    # whale harus akumulasi
MIN_DERIV_SCORE_FOR_FUTURES = 10.0  # ada konfirmasi dari derivatives
MIN_TA_SCORE_FOR_SIGNAL     = 8.0   # TA harus mendukung
MIN_QUANT_SCORE_FOR_SIGNAL  = 10.0  # quant harus mendukung

# Threshold kontradiksi
MAX_BEARISH_SIGNALS_ALLOWED = 2     # max sinyal bearish sebelum REJECTED
MIN_TF_AGREEMENT            = 2     # minimal TF agree untuk APPROVED

# Risk/Reward minimum
MIN_RR_LONG    = 1.8
MIN_RR_SHORT   = 1.5
MIN_RR_SPOT    = 1.5

# Entry precision: jarak entry ke struktur kunci (max %)
MAX_ENTRY_DIST_FROM_STRUCTURE = 5.0  # % maksimum dari FVG/OB/Support

# Funding rate thresholds
FR_EXTREME_SHORT = -0.001    # FR < ini = extreme short squeeze setup
FR_EXTREME_LONG  = 0.002     # FR > ini = overbought, SHORT setup


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class ValidationEvidence:
    """Satu bukti / argumen untuk/melawan signal."""
    category    : str    # 'WHALE', 'DERIVATIVES', 'TECHNICAL', 'QUANT', 'RISK', 'MARKET'
    polarity    : str    # 'BULLISH', 'BEARISH', 'NEUTRAL', 'WARNING'
    weight      : float  # 0–10 (pentingnya bukti ini)
    description : str

@dataclass
class ProsecutionCase:
    """
    Dokumen lengkap kenapa signal ini diambil.
    Mirip seperti "investment thesis" seorang analis.
    """
    signal_type     : str
    symbol          : str
    price           : float
    verdict         : str    # 'APPROVED', 'DOWNGRADED', 'REJECTED'

    # Argumen pendukung (pro)
    bull_evidence   : List[ValidationEvidence] = field(default_factory=list)
    # Argumen melawan / risiko (kontra)
    bear_evidence   : List[ValidationEvidence] = field(default_factory=list)
    # Warning (bukan penolak tapi perlu diketahui)
    warnings        : List[str] = field(default_factory=list)

    # Skor akhir setelah kalibrasi
    calibrated_score    : float = 0.0
    original_score      : float = 0.0
    score_adjustment    : float = 0.0

    # Kualitas entry
    entry_precision     : str   = "FAIR"   # EXCELLENT/GOOD/FAIR/POOR
    entry_dist_pct      : float = 0.0

    # Risk audit
    rr_ratio            : float = 0.0
    rr_verdict          : str   = "OK"     # OK/TIGHT/NEGATIVE
    sl_quality          : str   = "OK"     # OK/TOO_WIDE/TOO_TIGHT/ARBITRARY

    # Reasons
    rejection_reasons   : List[str] = field(default_factory=list)
    downgrade_reasons   : List[str] = field(default_factory=list)

    # Final prosecution summary (yang muncul di Telegram)
    thesis              : str   = ""   # 1-2 kalimat ringkasan kenapa LONG/SHORT
    key_reasons         : List[str] = field(default_factory=list)    # 3-5 alasan terkuat
    key_risks           : List[str] = field(default_factory=list)    # 1-3 risiko utama

@dataclass
class ValidationResult:
    verdict          : str    # 'APPROVED', 'DOWNGRADED', 'REJECTED'
    case             : ProsecutionCase
    final_alert_level: str
    final_score      : float
    send_signal      : bool   # True = kirim ke Telegram


# ═══════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════

def _g(obj, attr, default=None):
    return getattr(obj, attr, default) if obj is not None else default


def _pct(val: float, price: float) -> float:
    """Hitung % distance dari price."""
    return abs(val - price) / price * 100 if price > 0 else 0.0


# ═══════════════════════════════════════════════════════════
#  STEP 1 — COLLECT EVIDENCE
# ═══════════════════════════════════════════════════════════

def _collect_evidence(sig, ta_res, quant_res, deriv_res, whale_res,
                      supply_res, prepump_res, fear_greed) -> Tuple[List, List, List]:
    """
    Kumpulkan semua bukti dari semua engine.
    Return: (bull_evidence, bear_evidence, warnings)
    """
    bull = []
    bear = []
    warns = []

    price = sig.price

    # ── WHALE EVIDENCE ──────────────────────────────────────
    wp = _g(whale_res, "buy_pressure", 0.5)
    ws = _g(whale_res, "score", 0.0)
    is_accum = _g(whale_res, "is_accumulating", False)

    if wp >= 0.75:
        bull.append(ValidationEvidence("WHALE","BULLISH", 9.0,
            f"Whale buy pressure {wp*100:.0f}% — institusi akumulasi agresif"))
    elif wp >= 0.60:
        bull.append(ValidationEvidence("WHALE","BULLISH", 6.0,
            f"Whale buy pressure {wp*100:.0f}% — tekanan beli dari smart money"))
    elif wp <= 0.40:
        bear.append(ValidationEvidence("WHALE","BEARISH", 7.0,
            f"Whale SELL pressure {(1-wp)*100:.0f}% — distribusi terdeteksi"))
    elif wp <= 0.50:
        warns.append(f"⚠️ Whale buy pressure rendah ({wp*100:.0f}%) — tidak ada konfirmasi akumulasi")

    if is_accum:
        bull.append(ValidationEvidence("WHALE","BULLISH", 5.0,
            "Pola akumulasi terdeteksi (taker buy meningkat 3H terakhir)"))

    ob_imb = _g(whale_res, "ob_imbalance", 0.0)
    if ob_imb >= 0.25:
        bull.append(ValidationEvidence("WHALE","BULLISH", 4.0,
            f"Order book imbalance bid>ask: {ob_imb*100:.0f}% — dinding beli kuat"))

    # ── DERIVATIVES EVIDENCE ─────────────────────────────────
    fr = sig.funding_rate
    oi = sig.oi_change
    ssr = sig.short_squeeze_risk

    if fr is not None:
        if fr <= -0.001:
            bull.append(ValidationEvidence("DERIVATIVES","BULLISH", 10.0,
                f"Funding rate EKSTREM NEGATIF {fr*100:.4f}% — {oi_c_lbl(oi)} short akan kena squeeze"))
        elif fr <= -0.0003:
            bull.append(ValidationEvidence("DERIVATIVES","BULLISH", 7.0,
                f"Funding rate negatif {fr*100:.4f}% — short carry mendukung LONG"))
        elif fr >= 0.002:
            bear.append(ValidationEvidence("DERIVATIVES","BEARISH", 8.0,
                f"Funding rate SANGAT TINGGI {fr*100:.4f}% — market overbought"))
        elif fr >= 0.001:
            warns.append(f"⚠️ Funding rate tinggi {fr*100:.4f}% — LONG berhati-hati")

    if oi is not None:
        if oi >= 40:
            bull.append(ValidationEvidence("DERIVATIVES","BULLISH", 8.0,
                f"Open Interest naik {oi:.1f}% dalam 24H — smart money membuka posisi besar"))
        elif oi >= 20:
            bull.append(ValidationEvidence("DERIVATIVES","BULLISH", 5.0,
                f"OI naik {oi:.1f}% — posisi futures meningkat"))
        elif oi <= -25:
            warns.append(f"⚠️ OI turun {oi:.1f}% — mass liquidation, hati-hati volatility")

    if ssr in ("HIGH","EXTREME"):
        bull.append(ValidationEvidence("DERIVATIVES","BULLISH", 7.0,
            f"Short Squeeze Risk: {ssr} — jika harga naik sedikit, short cover masif"))

    # ── TECHNICAL EVIDENCE ───────────────────────────────────
    ta_bias = _g(ta_res, "ta_bias", "NEUTRAL")
    rsi = sig.rsi_14
    macd = sig.macd_signal_type
    bb_sq = sig.bb_squeeze
    mtf_bias = sig.extra_context.get("mtf_bias","") if hasattr(sig,"extra_context") else ""
    mtf_agree = sig.extra_context.get("mtf_agree_count",0) if hasattr(sig,"extra_context") else 0
    in_discount = sig.extra_context.get("in_discount_zone",False) if hasattr(sig,"extra_context") else False
    liq_swept = sig.extra_context.get("liquidity_swept",False) if hasattr(sig,"extra_context") else False
    bos_dir = sig.extra_context.get("bos_direction","") if hasattr(sig,"extra_context") else ""
    choch_dir = sig.extra_context.get("choch_direction","") if hasattr(sig,"extra_context") else ""

    if ta_bias in ("STRONG_BULL","BULL"):
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 8.0,
            f"TA Bias {ta_bias} — semua indikator teknikal alignment bullish"))
    elif ta_bias in ("STRONG_BEAR","BEAR"):
        bear.append(ValidationEvidence("TECHNICAL","BEARISH", 8.0,
            f"TA Bias {ta_bias} — indikator teknikal bearish"))

    if rsi <= 30:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 6.0,
            f"RSI={rsi:.0f} OVERSOLD — zona beli ekstrem"))
    elif rsi <= 40:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 3.0,
            f"RSI={rsi:.0f} mendekati oversold"))
    elif rsi >= 70:
        bear.append(ValidationEvidence("TECHNICAL","BEARISH", 5.0,
            f"RSI={rsi:.0f} OVERBOUGHT — potensi koreksi"))

    if macd == "BULLISH_CROSS":
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 7.0,
            "MACD Golden Cross — momentum baru bullish dimulai"))
    elif macd == "BEARISH_CROSS":
        bear.append(ValidationEvidence("TECHNICAL","BEARISH", 6.0,
            "MACD Death Cross — momentum berubah bearish"))
    elif macd == "BULLISH":
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 3.0,
            "MACD histogram growing bullish"))

    if bb_sq:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 5.0,
            "BB Squeeze aktif — volatility coiling, breakout imminent"))

    if mtf_bias in ("STRONG_BULLISH","BULLISH") and mtf_agree >= 3:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 9.0,
            f"MTF Consensus {mtf_bias}: {mtf_agree}/4 timeframe setuju arah bullish"))
    elif mtf_agree >= 2:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 5.0,
            f"MTF partial agreement: {mtf_agree}/4 TF bullish"))

    if liq_swept and in_discount:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 8.0,
            "Liquidity Sweep + Discount Zone — setup ICT premium: stop hunt selesai, reversal imminent"))
    elif liq_swept:
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 5.0,
            "Liquidity Swept — stop hunt selesai, potensi reversal"))

    if bos_dir == "BULLISH":
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 6.0,
            "Break of Structure BULLISH dikonfirmasi — struktur pasar berubah ke atas"))
    elif bos_dir == "BEARISH":
        bear.append(ValidationEvidence("TECHNICAL","BEARISH", 6.0,
            "Break of Structure BEARISH — struktur pasar turun"))

    if choch_dir == "BULLISH":
        bull.append(ValidationEvidence("TECHNICAL","BULLISH", 7.0,
            "Change of Character BULLISH — reversal dari downtrend dikonfirmasi"))

    # FVG & OB evidence
    nearest_bull_fvg = sig.extra_context.get("has_precise_entry",False) if hasattr(sig,"extra_context") else False
    if nearest_bull_fvg:
        ol = sig.extra_context.get("optimal_entry_low",0)
        oh = sig.extra_context.get("optimal_entry_high",0)
        if ol > 0:
            fvg_dist = _pct(oh, price)
            bull.append(ValidationEvidence("TECHNICAL","BULLISH", 8.0,
                f"Bullish FVG/OB zone presisi: ${ol:.5f}–${oh:.5f} ({fvg_dist:.1f}% dari harga)"))

    # ── QUANT EVIDENCE ───────────────────────────────────────
    if quant_res is not None:
        q_score = _g(quant_res, "score", 0.0)
        q_conf  = _g(quant_res, "confluence_count", 0)
        q_eta   = _g(quant_res, "trend_eta", 0.0)
        q_ibs   = _g(quant_res, "ibs_value", 0.5)
        q_dual  = _g(quant_res, "dual_mom_pass", False)
        q_r_adj = _g(quant_res, "risk_adj_return", 0.0)
        q_resid = _g(quant_res, "residual_mom", 0.0)
        q_triple = _g(quant_res, "triple_ma_bias", "NEUTRAL")

        if q_score >= 28 and q_conf >= 6:
            bull.append(ValidationEvidence("QUANT","BULLISH", 9.0,
                f"Quant Engine STRONG: {q_conf}/10 strategi bullish (§151 Strategies), score={q_score:.0f}/40"))
        elif q_score >= 18 and q_conf >= 4:
            bull.append(ValidationEvidence("QUANT","BULLISH", 6.0,
                f"Quant Engine bullish: {q_conf}/10 strategi setuju, score={q_score:.0f}/40"))
        elif q_score < 5 and q_conf <= 2:
            bear.append(ValidationEvidence("QUANT","BEARISH", 5.0,
                f"Quant Engine BEARISH: hanya {q_conf}/10 strategi bullish, score={q_score:.0f}/40"))

        if q_ibs < 0.20:
            bull.append(ValidationEvidence("QUANT","BULLISH", 6.0,
                f"IBS §4.4 = {q_ibs:.3f} OVERSOLD — harga close di bawah range (mean-reversion setup)"))
        elif q_ibs > 0.80:
            bear.append(ValidationEvidence("QUANT","BEARISH", 5.0,
                f"IBS §4.4 = {q_ibs:.3f} OVERBOUGHT — hindari beli di puncak candle"))

        if q_dual:
            bull.append(ValidationEvidence("QUANT","BULLISH", 5.0,
                "Dual Momentum §4.1.2 PASS — momentum relatif DAN absolut positif"))

        if q_r_adj > 0.3:
            bull.append(ValidationEvidence("QUANT","BULLISH", 5.0,
                f"Risk-adjusted momentum §3.1 = {q_r_adj:.2f} — strong upward trend"))
        elif q_r_adj < -0.3:
            bear.append(ValidationEvidence("QUANT","BEARISH", 4.0,
                f"Risk-adjusted momentum §3.1 = {q_r_adj:.2f} — trending turun"))

        if q_eta > 0.3:
            bull.append(ValidationEvidence("QUANT","BULLISH", 4.0,
                f"Trend signal §10.4 η={q_eta:.2f} — tanh-smoothed momentum bullish"))
        elif q_eta < -0.3:
            bear.append(ValidationEvidence("QUANT","BEARISH", 3.0,
                f"Trend signal §10.4 η={q_eta:.2f} — bearish"))

        if q_triple == "BULL":
            bull.append(ValidationEvidence("QUANT","BULLISH", 5.0,
                "Triple EMA §3.13: EMA9>EMA21>EMA50 — perfect bullish alignment"))
        elif q_triple == "BEAR":
            bear.append(ValidationEvidence("QUANT","BEARISH", 4.0,
                "Triple EMA §3.13: EMA9<EMA21<EMA50 — perfect bearish alignment"))

    # ── SUPPLY / MARKET CAP EVIDENCE ─────────────────────────
    cat = _g(supply_res, "category", "")
    sp  = _g(supply_res, "selling_pressure", "")
    fdv = _g(supply_res, "fdv_mc_ratio", 1.0)
    ath_pct = _g(supply_res, "ath_change_pct", 0.0)

    if cat == "MICRO":
        bull.append(ValidationEvidence("MARKET","BULLISH", 5.0,
            f"Micro cap — pump potential tinggi, sedikit modal bisa gerakkan harga"))
    elif cat == "SMALL":
        bull.append(ValidationEvidence("MARKET","BULLISH", 3.0,
            "Small cap — upside potential lebih besar dari large cap"))

    if fdv >= 5.0:
        bear.append(ValidationEvidence("MARKET","BEARISH", 6.0,
            f"FDV/MC ratio {fdv:.1f}x TINGGI — risiko dilusi dan selling pressure dari unlock"))
    elif fdv <= 1.2:
        bull.append(ValidationEvidence("MARKET","BULLISH", 4.0,
            f"FDV/MC ratio {fdv:.1f}x rendah — supply hampir fully diluted, risiko dilusi minimal"))

    if ath_pct <= -80:
        bull.append(ValidationEvidence("MARKET","BULLISH", 5.0,
            f"{ath_pct:.0f}% dari ATH — extreme discount, upside potential besar"))
    elif ath_pct <= -50:
        bull.append(ValidationEvidence("MARKET","BULLISH", 2.0,
            f"{ath_pct:.0f}% dari ATH — significant discount"))

    # ── SENTIMENT EVIDENCE ───────────────────────────────────
    fg = fear_greed.get("value", 50)
    fg_label = fear_greed.get("label","Neutral")
    if fg <= 20:
        bull.append(ValidationEvidence("MARKET","BULLISH", 6.0,
            f"Fear & Greed {fg}/100 (EXTREME FEAR) — pasar capitulation, historically best buy point"))
    elif fg <= 30:
        bull.append(ValidationEvidence("MARKET","BULLISH", 4.0,
            f"Fear & Greed {fg}/100 (FEAR) — sentiment rendah = peluang masuk"))
    elif fg >= 80:
        bear.append(ValidationEvidence("MARKET","BEARISH", 5.0,
            f"Fear & Greed {fg}/100 (EXTREME GREED) — pasar euphoria, risiko koreksi"))
    elif fg >= 70:
        warns.append(f"⚠️ F&G {fg}/100 (Greed) — pertimbangkan risiko reversal")

    return bull, bear, warns


def oi_c_lbl(oi):
    if oi is None: return ""
    if oi >= 30: return f"OI+{oi:.0f}%"
    return ""


# ═══════════════════════════════════════════════════════════
#  STEP 2 — QUALITY GATE (7 kondisi)
# ═══════════════════════════════════════════════════════════

def _quality_gate(sig, bull_ev, bear_ev, quant_res, ta_res) -> Tuple[bool, List[str], List[str]]:
    """
    7 Quality Gate checks.
    Return: (passed, rejection_reasons, downgrade_reasons)
    """
    rejections = []
    downgrades = []

    price = sig.price
    st = sig.signal_type

    # ── GATE 1: Contradiction Check ──────────────────────────
    # Jika bear evidence lebih kuat dari bull evidence → reject
    bull_weight = sum(e.weight for e in bull_ev)
    bear_weight = sum(e.weight for e in bear_ev)

    if bear_weight > bull_weight * 1.5 and st in ("LONG", "BUY SPOT"):
        rejections.append(
            f"GATE-1 FAILED: Bear evidence ({bear_weight:.0f}) > Bull ({bull_weight:.0f}) × 1.5 — "
            f"terlalu banyak sinyal melawan arah LONG"
        )

    # ── GATE 2: Minimum Bull Evidence ────────────────────────
    if len(bull_ev) < 3 and st in ("LONG", "BUY SPOT"):
        rejections.append(
            f"GATE-2 FAILED: Hanya {len(bull_ev)} bukti bullish (minimum 3) — "
            "sinyal tidak cukup terkonfirmasi"
        )

    # ── GATE 3: Whale Confirmation ────────────────────────────
    wp = _g(sig, "_whale_buy_pressure_raw", None)
    whale_score = sig.whale_score
    if whale_score < MIN_WHALE_SCORE_FOR_LONG and st in ("LONG", "BUY SPOT"):
        downgrades.append(
            f"GATE-3: Whale score rendah ({whale_score:.0f}/{MIN_WHALE_SCORE_FOR_LONG}) — "
            "tidak ada konfirmasi akumulasi institusional"
        )

    # ── GATE 4: Risk/Reward Check ─────────────────────────────
    rr = sig.risk_reward
    min_rr = MIN_RR_LONG if st == "LONG" else (MIN_RR_SHORT if st == "SHORT" else MIN_RR_SPOT)
    if rr < min_rr and rr > 0:
        downgrades.append(
            f"GATE-4: R/R ratio {rr:.1f} di bawah minimum {min_rr} untuk {st} — "
            "reward tidak sebanding dengan risiko"
        )
    elif rr <= 0:
        rejections.append(
            f"GATE-4 FAILED: R/R ratio {rr:.1f} tidak valid — "
            "SL/TP setup tidak masuk akal"
        )

    # ── GATE 5: SL Validity ───────────────────────────────────
    sl = sig.stop_loss
    sl_dist = _pct(sl, price)
    if sl_dist > 20:
        rejections.append(
            f"GATE-5 FAILED: Stop Loss terlalu jauh ({sl_dist:.1f}%) — "
            "risiko per trade tidak terkontrol"
        )
    elif sl_dist > 15:
        downgrades.append(
            f"GATE-5: SL jauh ({sl_dist:.1f}%) — pertimbangkan position size lebih kecil"
        )
    elif sl_dist < 0.5:
        rejections.append(
            f"GATE-5 FAILED: SL terlalu dekat ({sl_dist:.1f}%) — "
            "akan kena stop loss oleh normal market noise"
        )

    # ── GATE 6: Anti-FOMO Filter ─────────────────────────────
    chg = sig.price_change_24h
    if chg > 40 and st in ("LONG", "BUY SPOT"):
        rejections.append(
            f"GATE-6 FAILED: Harga sudah naik {chg:.1f}% dalam 24H — "
            "FOMO setup, risiko beli di puncak sangat tinggi"
        )
    elif chg > 20 and st in ("LONG", "BUY SPOT"):
        downgrades.append(
            f"GATE-6: Harga naik {chg:.1f}% 24H — entry lebih berisiko, waspada late entry"
        )

    # ── GATE 7: Contradicting Bearish Structures ─────────────
    ec = getattr(sig, "extra_context", {}) or {}
    near_bear_fvg = ec.get("near_bear_fvg", False)
    bos_bearish = ec.get("bos_direction", "") == "BEARISH"
    daily_bearish = ec.get("daily_bias", "") == "BEARISH"

    contradictions = 0
    if near_bear_fvg:
        contradictions += 1
        downgrades.append("GATE-7: Bearish FVG sangat dekat di atas — resistance institusional kuat")
    if bos_bearish and st in ("LONG", "BUY SPOT"):
        contradictions += 1
        downgrades.append("GATE-7: BOS Bearish aktif — struktur pasar masih turun")
    if daily_bearish and st in ("LONG", "BUY SPOT"):
        contradictions += 1
        downgrades.append("GATE-7: Bias daily BEARISH — counter-trend trade berisiko tinggi")

    if contradictions >= 3 and st in ("LONG", "BUY SPOT"):
        rejections.append(
            "GATE-7 FAILED: 3+ kontradiksi bearish struktural — "
            "signal LONG berlawanan dengan arah pasar yang lebih besar"
        )

    passed = len(rejections) == 0
    return passed, rejections, downgrades


# ═══════════════════════════════════════════════════════════
#  STEP 3 — CONFIDENCE CALIBRATION
# ═══════════════════════════════════════════════════════════

def _calibrate_score(original_score: float, bull_ev: List, bear_ev: List,
                     gate_downgrades: List) -> Tuple[float, float]:
    """
    Kalibrasi skor berdasarkan kekuatan bukti.
    Return: (calibrated_score, adjustment)
    """
    bull_weight = sum(e.weight for e in bull_ev)
    bear_weight = sum(e.weight for e in bear_ev)

    # Bonus jika banyak bukti kuat
    high_weight_bull = [e for e in bull_ev if e.weight >= 7.0]
    bonus = min(len(high_weight_bull) * 2.0, 10.0)

    # Penalty untuk bear evidence
    penalty = min(bear_weight * 0.5, 20.0)

    # Penalty untuk setiap downgrade
    gate_penalty = len(gate_downgrades) * 3.0

    adjustment = bonus - penalty - gate_penalty
    calibrated = original_score + adjustment
    calibrated = max(0.0, min(calibrated, 100.0))

    return calibrated, adjustment


# ═══════════════════════════════════════════════════════════
#  STEP 4 — ENTRY PRECISION SCORING
# ═══════════════════════════════════════════════════════════

def _score_entry_precision(sig) -> Tuple[str, float]:
    """
    Nilai seberapa presisi zona entry.
    EXCELLENT: entry di dalam FVG/OB institusional
    GOOD: entry dekat support penting / pivot
    FAIR: entry di zona yang reasonable
    POOR: entry terlalu jauh dari struktur kunci
    """
    ec = getattr(sig, "extra_context", {}) or {}
    price = sig.price
    entry_low  = sig.entry_zone_low
    entry_high = sig.entry_zone_high

    # Apakah entry di SMC zone
    has_precise = ec.get("has_precise_entry", False)
    opt_low  = ec.get("optimal_entry_low", 0.0)
    opt_high = ec.get("optimal_entry_high", 0.0)

    if has_precise and opt_low > 0:
        entry_dist = _pct(opt_low, price)
        if entry_dist < 1.0:
            return "EXCELLENT", entry_dist
        elif entry_dist < 3.0:
            return "GOOD", entry_dist
        else:
            return "FAIR", entry_dist

    # Pivot support
    qec = ec.get("quant")
    if qec is not None:
        ps1 = _g(qec, "pivot_s1", 0.0)
        if ps1 > 0:
            ps_dist = _pct(ps1, price)
            if ps_dist < 1.5:
                return "GOOD", ps_dist
            elif ps_dist < 4.0:
                return "FAIR", ps_dist

    # Support dari TA
    ns = sig.nearest_support
    if ns > 0:
        ns_dist = _pct(ns, price)
        if ns_dist < 1.5:
            return "GOOD", ns_dist
        elif ns_dist < 3.5:
            return "FAIR", ns_dist

    # Fallback: lihat seberapa dekat entry_low dengan price
    if entry_low > 0:
        spread_pct = _pct(entry_low, entry_high) if entry_high > 0 else 5.0
        if spread_pct < 0.5:
            return "POOR", spread_pct  # terlalu sempit/arbitrary
        elif spread_pct < 2.0:
            return "FAIR", spread_pct

    return "POOR", 10.0


# ═══════════════════════════════════════════════════════════
#  STEP 5 — BUILD PROSECUTION CASE
# ═══════════════════════════════════════════════════════════

def _build_case(sig, bull_ev, bear_ev, warns, gate_ok, rejections,
                downgrades, calib_score, orig_score, adj,
                entry_precision, entry_dist) -> ProsecutionCase:
    """
    Build the Prosecution Case — dokumen final kenapa signal ini diambil.
    """
    price = sig.price
    st = sig.signal_type
    sym = sig.symbol.replace("USDT", "")

    # Verdict
    if not gate_ok:
        verdict = "REJECTED"
    elif len(downgrades) >= 3 or (len(downgrades) >= 2 and calib_score < 55):
        verdict = "DOWNGRADED"
    else:
        verdict = "APPROVED"

    # Build key reasons (3-5 alasan terkuat dari bull evidence)
    sorted_bull = sorted(bull_ev, key=lambda x: x.weight, reverse=True)
    key_reasons = [e.description for e in sorted_bull[:5]]

    # Build key risks (dari bear evidence + warnings)
    sorted_bear = sorted(bear_ev, key=lambda x: x.weight, reverse=True)
    key_risks = [e.description for e in sorted_bear[:3]]
    if warns:
        key_risks.extend(warns[:2])

    # Thesis — 1 kalimat ringkasan
    if st in ("LONG","BUY SPOT"):
        primary_bull = sorted_bull[0].description if sorted_bull else "Confluence indicators bullish"
        thesis = f"${sym} menunjukkan setup {st} karena: {primary_bull[:120]}"
        if len(sorted_bull) > 1:
            thesis += f", dikonfirmasi {len(sorted_bull)-1} bukti tambahan"
    elif st == "SHORT":
        primary_bear = sorted_bear[0].description if sorted_bear else "Bearish structure terkonfirmasi"
        thesis = f"${sym} SHORT setup: {primary_bear[:120]}"
    else:
        thesis = f"${sym} WATCH — menunggu konfirmasi lebih lanjut"

    rr = sig.risk_reward
    rr_verdict = "OK" if rr >= MIN_RR_LONG else ("TIGHT" if rr > 0 else "NEGATIVE")

    sl_dist = _pct(sig.stop_loss, price)
    if sl_dist < 0.5:
        sl_quality = "TOO_TIGHT"
    elif sl_dist > 15:
        sl_quality = "TOO_WIDE"
    else:
        sl_quality = "OK"

    return ProsecutionCase(
        signal_type=st, symbol=sig.symbol, price=price, verdict=verdict,
        bull_evidence=bull_ev, bear_evidence=bear_ev, warnings=warns,
        calibrated_score=calib_score, original_score=orig_score, score_adjustment=adj,
        entry_precision=entry_precision, entry_dist_pct=entry_dist,
        rr_ratio=rr, rr_verdict=rr_verdict, sl_quality=sl_quality,
        rejection_reasons=rejections, downgrade_reasons=downgrades,
        thesis=thesis, key_reasons=key_reasons, key_risks=key_risks,
    )


# ═══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def validate_signal(
    sig,
    ta_res    = None,
    quant_res = None,
    deriv_res = None,
    whale_res = None,
    supply_res = None,
    prepump_res = None,
    fear_greed : dict = None,
) -> ValidationResult:
    """
    Validasi signal sebelum dikirim ke Telegram.
    Return: ValidationResult dengan verdict final.
    """
    if fear_greed is None:
        fear_greed = {"value": 50, "label": "Neutral"}

    price = sig.price
    st    = sig.signal_type

    # ── Kumpulkan bukti ──────────────────────────────────────
    bull_ev, bear_ev, warns = _collect_evidence(
        sig, ta_res, quant_res, deriv_res, whale_res, supply_res, prepump_res, fear_greed
    )

    # ── Quality Gate ─────────────────────────────────────────
    gate_ok, rejections, downgrades = _quality_gate(sig, bull_ev, bear_ev, quant_res, ta_res)

    # ── Calibrate Score ──────────────────────────────────────
    orig_score = sig.confidence_score
    calib_score, adj = _calibrate_score(orig_score, bull_ev, bear_ev, downgrades)

    # ── Entry Precision ──────────────────────────────────────
    entry_prec, entry_dist = _score_entry_precision(sig)

    # ── Build Case ───────────────────────────────────────────
    case = _build_case(
        sig, bull_ev, bear_ev, warns, gate_ok, rejections, downgrades,
        calib_score, orig_score, adj, entry_prec, entry_dist
    )

    # ── Final alert level (bisa turun karena downgrade) ──────
    orig_level  = sig.alert_level
    final_level = orig_level

    if case.verdict == "REJECTED":
        final_level = "LOW"
        send = False
    elif case.verdict == "DOWNGRADED":
        levels = ["LOW","MEDIUM","HIGH","CRITICAL"]
        idx = levels.index(orig_level)
        final_level = levels[max(0, idx-1)]
        send = True
    else:
        send = True

    return ValidationResult(
        verdict=case.verdict,
        case=case,
        final_alert_level=final_level,
        final_score=calib_score,
        send_signal=send,
    )
