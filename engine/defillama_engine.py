"""
DefiLlama Engine — CMI-ASS v6
================================
Mendeteksi koin UNDERVALUED menggunakan data fundamental dari DefiLlama.

RUMUS UNDERVALUED (Precision Valuation Framework):
──────────────────────────────────────────────────
1. P/S Ratio (Price-to-Sales) — METRIK UTAMA
   P/S = Market Cap / Annualized Revenue
   
   Interpretasi:
   P/S < 1   → SANGAT undervalued (bayar $1, dapat revenue $1/tahun)
   P/S 1-3   → Undervalued
   P/S 3-10  → Fair value
   P/S 10-25 → Mahal
   P/S > 25  → Sangat mahal / spekulatif

2. P/F Ratio (Price-to-Fees) — protokol DeFi
   Sama dengan P/S tapi pakai fee yang ke protokol (bukan total volume)

3. Revenue Growth Rate (MoM / QoQ)
   Revenue naik sementara harga turun = sangat undervalued

4. TVL-to-MCap Ratio (khusus DeFi)
   TVL/MCap > 1 = aset yang TVL-nya melebihi market cap → sangat undervalued
   TVL/MCap 0.5-1 = undervalued
   TVL/MCap < 0.1 = MCap jauh lebih besar dari TVL → mahal

5. Composite Undervalue Score (0-100)
   Menggabungkan P/S + TVL ratio + revenue growth

API ENDPOINTS DEFILLAMA (gratis, no key):
  - Protocols:     https://api.llama.fi/protocols
  - Protocol:      https://api.llama.fi/protocol/{slug}
  - Revenue:       https://api.llama.fi/overview/fees/{protocol}
  - TVL:           https://api.llama.fi/tvl/{protocol}
  - Chains:        https://api.llama.fi/v2/chains
  - Token prices:  https://coins.llama.fi/prices/current/{chains}:{address}
"""

import requests
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

LLAMA_BASE    = "https://api.llama.fi"
LLAMA_COINS   = "https://coins.llama.fi"
TIMEOUT       = 8
HEADERS       = {"User-Agent": "CMI-ASS/6.0", "Accept": "application/json"}

# Cache protocols list (berat ~2MB, refresh 1x per run)
_PROTOCOLS_CACHE: Optional[List[dict]] = None
_PROTOCOLS_TS: float = 0


# ─────────────────────────────────────────────────────────
#  VALUATION THRESHOLDS
# ─────────────────────────────────────────────────────────

PS_THRESHOLDS = {
    "EXTREME_UNDERVALUE": 1.0,    # P/S < 1 → sangat murah
    "UNDERVALUE":         3.0,    # P/S 1-3
    "FAIR":              10.0,    # P/S 3-10
    "OVERVALUE":         25.0,    # P/S 10-25
    # P/S > 25 → sangat mahal
}

TVL_MC_THRESHOLDS = {
    "EXTREME_UNDERVALUE": 1.0,    # TVL > MCap
    "UNDERVALUE":         0.5,    # TVL = 50%-100% MCap
    "FAIR":               0.1,    # TVL = 10%-50% MCap
    # < 0.1 → MCap jauh lebih besar dari TVL (mahal)
}


@dataclass
class FundamentalData:
    """Data fundamental yang diambil dari DefiLlama."""
    protocol_name     : str   = ""
    symbol            : str   = ""
    category          : str   = ""          # DEX, Lending, CDP, Bridge, etc.
    chain             : str   = ""
    tvl               : float = 0.0         # Total Value Locked (USD)
    tvl_7d_change     : float = 0.0         # % perubahan TVL 7 hari
    tvl_30d_change    : float = 0.0
    market_cap        : float = 0.0
    revenue_30d       : float = 0.0         # Revenue 30 hari terakhir (to protocol)
    revenue_90d       : float = 0.0
    fees_30d          : float = 0.0         # Total fees 30 hari
    fees_90d          : float = 0.0
    revenue_annualized: float = 0.0         # revenue_30d * 12
    fees_annualized   : float = 0.0
    ps_ratio          : float = 999.0       # Market Cap / Annualized Revenue
    pf_ratio          : float = 999.0       # Market Cap / Annualized Fees
    tvl_mc_ratio      : float = 0.0         # TVL / Market Cap
    revenue_growth_mom: float = 0.0         # MoM revenue growth %
    found_on_llama    : bool  = False


@dataclass
class ValuationResult:
    """Hasil penilaian undervalued dari DefiLlama."""
    symbol              : str   = ""
    found               : bool  = False
    fundamental         : Optional[FundamentalData] = None

    # Valuation metrics
    ps_ratio            : float = 999.0
    pf_ratio            : float = 999.0
    tvl_mc_ratio        : float = 0.0
    revenue_growth_mom  : float = 0.0

    # Verdict
    ps_verdict          : str   = "UNKNOWN"   # EXTREME_UNDERVALUE/UNDERVALUE/FAIR/OVERVALUE/EXTREME_OVERVALUE
    tvl_verdict         : str   = "UNKNOWN"
    overall_verdict     : str   = "UNKNOWN"
    undervalue_score    : float = 0.0         # 0-35 (kontribusi ke confidence score)

    # Human readable signals
    signals             : List[str] = field(default_factory=list)
    score               : float = 0.0        # alias undervalue_score untuk sistem


# ─────────────────────────────────────────────────────────
#  HTTP HELPERS
# ─────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> Optional[dict | list]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"DefiLlama GET {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────
#  PROTOCOL DISCOVERY
# ─────────────────────────────────────────────────────────

def _get_protocols() -> List[dict]:
    """Ambil semua protokol dari DefiLlama. Di-cache per session."""
    global _PROTOCOLS_CACHE, _PROTOCOLS_TS
    now = time.time()
    if _PROTOCOLS_CACHE and (now - _PROTOCOLS_TS) < 3600:
        return _PROTOCOLS_CACHE

    data = _get(f"{LLAMA_BASE}/protocols")
    if data and isinstance(data, list):
        _PROTOCOLS_CACHE = data
        _PROTOCOLS_TS = now
        logger.info(f"DefiLlama: loaded {len(data)} protocols")
    return _PROTOCOLS_CACHE or []


def _find_protocol(symbol: str) -> Optional[dict]:
    """
    Cari protokol berdasarkan ticker symbol.
    Matching: exact symbol → symbol substring → name substring
    """
    protocols = _get_protocols()
    sym_upper = symbol.upper().replace("USDT", "").replace("USDC", "")

    # Pass 1: exact symbol match
    for p in protocols:
        if p.get("symbol", "").upper() == sym_upper:
            return p

    # Pass 2: symbol substring (e.g. "UNI" matches "UNI-V2")
    for p in protocols:
        p_sym = p.get("symbol", "").upper()
        if p_sym and sym_upper in p_sym and len(p_sym) <= len(sym_upper) + 3:
            return p

    # Pass 3: name matches (e.g. "AAVE" → name="Aave")
    sym_lower = sym_upper.lower()
    for p in protocols:
        if sym_lower in p.get("name", "").lower():
            return p

    return None


# ─────────────────────────────────────────────────────────
#  REVENUE / FEES DATA
# ─────────────────────────────────────────────────────────

def _get_revenue_fees(slug: str) -> Dict[str, float]:
    """
    Ambil data revenue dan fees 30d + 90d dari endpoint /overview/fees/{protocol}.
    Returns dict: {revenue_30d, revenue_90d, fees_30d, fees_90d, revenue_growth_mom}
    """
    result = {
        "revenue_30d": 0.0,
        "revenue_90d": 0.0,
        "fees_30d":    0.0,
        "fees_90d":    0.0,
        "revenue_growth_mom": 0.0,
    }

    # Coba endpoint fees overview
    data = _get(f"{LLAMA_BASE}/summary/fees/{slug}")
    if not data:
        # fallback: protocol detail
        data = _get(f"{LLAMA_BASE}/protocol/{slug}")
        if not data:
            return result

    # Revenue dari field terstandarisasi DefiLlama
    # Field: revenue30d, revenue90d, totalRevenue30d (tergantung endpoint)
    for key_rev, key_30, key_90 in [
        ("revenue", "revenue30d", "revenue90d"),
        ("protocolRevenue", "protocolRevenue30d", "protocolRevenue90d"),
    ]:
        r30 = data.get(key_30, 0) or data.get("totalRevenue30d", 0) or 0
        r90 = data.get(key_90, 0) or data.get("totalRevenue90d", 0) or 0
        if r30 > 0:
            result["revenue_30d"] = float(r30)
            result["revenue_90d"] = float(r90)
            break

    # Fees
    f30 = data.get("total30d", 0) or data.get("fees30d", 0) or 0
    f90 = data.get("total90d", 0) or data.get("fees90d", 0) or 0
    result["fees_30d"] = float(f30)
    result["fees_90d"] = float(f90)

    # Revenue growth MoM: bandingkan 30d pertama vs 30d kedua dalam 90d
    # MoM = (revenue_30d - revenue_30to60d) / revenue_30to60d
    rev_30to60 = result["revenue_90d"] - result["revenue_30d"] - (result["revenue_90d"] - 2*result["revenue_30d"])
    # Pendekatan lebih sederhana: bandingkan 30d vs rata-rata 90d/3
    if result["revenue_90d"] > 0:
        avg_monthly_90d = result["revenue_90d"] / 3
        if avg_monthly_90d > 0:
            result["revenue_growth_mom"] = (result["revenue_30d"] - avg_monthly_90d) / avg_monthly_90d * 100

    return result


# ─────────────────────────────────────────────────────────
#  TVL DATA
# ─────────────────────────────────────────────────────────

def _get_tvl_data(protocol_data: dict) -> Dict[str, float]:
    """Extract TVL data dari protocol dict."""
    result = {"tvl": 0.0, "tvl_7d_change": 0.0, "tvl_30d_change": 0.0}

    result["tvl"] = float(protocol_data.get("tvl", 0) or 0)

    # % change dari DefiLlama
    change_7d  = protocol_data.get("change_7d")
    change_1m  = protocol_data.get("change_1m")
    if change_7d is not None:
        result["tvl_7d_change"] = float(change_7d)
    if change_1m is not None:
        result["tvl_30d_change"] = float(change_1m)

    return result


# ─────────────────────────────────────────────────────────
#  VALUATION CALCULATION
# ─────────────────────────────────────────────────────────

def _calc_ps_verdict(ps: float) -> str:
    if ps <= 0 or ps == 999.0:
        return "UNKNOWN"
    if ps < PS_THRESHOLDS["EXTREME_UNDERVALUE"]:
        return "EXTREME_UNDERVALUE"
    elif ps < PS_THRESHOLDS["UNDERVALUE"]:
        return "UNDERVALUE"
    elif ps < PS_THRESHOLDS["FAIR"]:
        return "FAIR"
    elif ps < PS_THRESHOLDS["OVERVALUE"]:
        return "OVERVALUE"
    else:
        return "EXTREME_OVERVALUE"


def _calc_tvl_verdict(ratio: float) -> str:
    if ratio <= 0:
        return "UNKNOWN"
    if ratio >= TVL_MC_THRESHOLDS["EXTREME_UNDERVALUE"]:
        return "EXTREME_UNDERVALUE"
    elif ratio >= TVL_MC_THRESHOLDS["UNDERVALUE"]:
        return "UNDERVALUE"
    elif ratio >= TVL_MC_THRESHOLDS["FAIR"]:
        return "FAIR"
    else:
        return "OVERVALUE"


def _calc_undervalue_score(vr: ValuationResult) -> float:
    """
    Hitung skor undervalued 0-35 untuk kontribusi ke confidence score.

    Breakdown:
    - P/S Ratio:        0-15 poin (metrik utama)
    - TVL/MCap:         0-10 poin
    - Revenue Growth:   0-7  poin
    - Category bonus:   0-3  poin
    """
    score = 0.0
    signals = []

    mc  = vr.fundamental.market_cap if vr.fundamental else 0
    cat = vr.fundamental.category if vr.fundamental else ""

    # ── P/S Scoring ──────────────────────────────────────
    ps = vr.ps_ratio
    if ps < PS_THRESHOLDS["EXTREME_UNDERVALUE"]:      # P/S < 1
        score += 15
        signals.append(f"🔥 P/S Ratio {ps:.2f}x — SANGAT UNDERVALUED! (tiap $1 MCap = ${1/ps:.2f} revenue/tahun)")
    elif ps < PS_THRESHOLDS["UNDERVALUE"]:              # P/S 1-3
        score += 10
        signals.append(f"💚 P/S Ratio {ps:.2f}x — Undervalued (revenue yield {100/ps:.1f}%/tahun)")
    elif ps < PS_THRESHOLDS["FAIR"]:                    # P/S 3-10
        score += 4
        signals.append(f"⚪ P/S Ratio {ps:.2f}x — Fair value")
    elif ps < PS_THRESHOLDS["OVERVALUE"]:               # P/S 10-25
        score -= 2
        signals.append(f"🟡 P/S Ratio {ps:.2f}x — Sedikit mahal")
    elif ps != 999.0:                                    # P/S > 25
        score -= 5
        signals.append(f"🔴 P/S Ratio {ps:.2f}x — MAHAL (revenue yield hanya {100/ps:.2f}%/tahun)")

    # ── TVL/MCap Scoring ─────────────────────────────────
    tvl_ratio = vr.tvl_mc_ratio
    if tvl_ratio >= TVL_MC_THRESHOLDS["EXTREME_UNDERVALUE"]:    # TVL > MCap
        score += 10
        signals.append(f"🏦 TVL/MCap {tvl_ratio:.2f}x — TVL MELEBIHI Market Cap! (sangat undervalued)")
    elif tvl_ratio >= TVL_MC_THRESHOLDS["UNDERVALUE"]:           # TVL 50-100% MCap
        score += 6
        signals.append(f"🏦 TVL/MCap {tvl_ratio:.2f}x — TVL kuat relatif MCap")
    elif tvl_ratio >= TVL_MC_THRESHOLDS["FAIR"]:                 # TVL 10-50% MCap
        score += 2
        signals.append(f"🏦 TVL/MCap {tvl_ratio:.2f}x — TVL moderat")
    elif tvl_ratio > 0:
        score -= 2
        signals.append(f"⚠️ TVL/MCap {tvl_ratio:.2f}x — MCap jauh lebih besar dari TVL")

    # ── Revenue Growth Scoring ───────────────────────────
    growth = vr.revenue_growth_mom
    if growth > 50:
        score += 7
        signals.append(f"🚀 Revenue growth +{growth:.0f}% MoM — tumbuh pesat!")
    elif growth > 20:
        score += 4
        signals.append(f"📈 Revenue growth +{growth:.0f}% MoM — bullish fundamental")
    elif growth > 0:
        score += 1
        signals.append(f"📊 Revenue tumbuh +{growth:.0f}% MoM")
    elif growth < -30:
        score -= 4
        signals.append(f"📉 Revenue turun {growth:.0f}% MoM — fundamental melemah")

    # ── P/F vs P/S bonus ─────────────────────────────────
    pf = vr.pf_ratio
    if 0 < pf < 5:
        score += 2
        signals.append(f"💹 P/F (Price-to-Fees) {pf:.2f}x — fee-based sangat murah")

    # ── Category bonus untuk high-revenue categories ─────
    high_rev_cats = ["DEX", "Lending", "CDP", "Derivatives", "Staking", "Yield"]
    if any(c.lower() in cat.lower() for c in high_rev_cats):
        score += 1

    vr.signals = signals
    return max(0.0, min(score, 35.0))


# ─────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────

def analyze_defillama(symbol: str, market_cap: float = 0.0) -> ValuationResult:
    """
    Analisis fundamental menggunakan DefiLlama untuk token/protokol DeFi.

    Args:
        symbol:     simbol coin (e.g. "AAVEUSDT" atau "UNI")
        market_cap: market cap dari CoinGecko (fallback jika llama tidak punya)

    Returns:
        ValuationResult dengan skor undervalued 0-35
    """
    vr = ValuationResult(symbol=symbol)
    base = symbol.upper().replace("USDT", "").replace("USDC", "").replace("BTC", "")

    try:
        # 1. Cari protokol di DefiLlama
        protocol = _find_protocol(base)
        if not protocol:
            logger.debug(f"DefiLlama: {base} tidak ditemukan di protocol list")
            vr.overall_verdict = "NOT_FOUND"
            return vr

        vr.found = True
        slug = protocol.get("slug", "").lower()
        name = protocol.get("name", base)

        # 2. Ambil TVL data
        tvl_data = _get_tvl_data(protocol)

        # 3. Ambil Revenue/Fees data
        rev_data = _get_revenue_fees(slug)
        time.sleep(0.3)  # rate limit DefiLlama gratis

        # 4. Market cap — prioritaskan dari DefiLlama, fallback ke CoinGecko
        mc_llama = float(protocol.get("mcap", 0) or 0)
        mc = mc_llama if mc_llama > 0 else market_cap

        # 5. Buat FundamentalData
        fd = FundamentalData(
            protocol_name      = name,
            symbol             = base,
            category           = protocol.get("category", ""),
            chain              = protocol.get("chain", ""),
            tvl                = tvl_data["tvl"],
            tvl_7d_change      = tvl_data["tvl_7d_change"],
            tvl_30d_change     = tvl_data["tvl_30d_change"],
            market_cap         = mc,
            revenue_30d        = rev_data["revenue_30d"],
            revenue_90d        = rev_data["revenue_90d"],
            fees_30d           = rev_data["fees_30d"],
            fees_90d           = rev_data["fees_90d"],
            revenue_growth_mom = rev_data["revenue_growth_mom"],
            found_on_llama     = True,
        )

        # 6. Kalkulasi annualized
        fd.revenue_annualized = fd.revenue_30d * 12
        fd.fees_annualized    = fd.fees_30d * 12

        # 7. P/S Ratio
        if mc > 0 and fd.revenue_annualized > 0:
            fd.ps_ratio = mc / fd.revenue_annualized
        else:
            fd.ps_ratio = 999.0

        # 8. P/F Ratio
        if mc > 0 and fd.fees_annualized > 0:
            fd.pf_ratio = mc / fd.fees_annualized
        else:
            fd.pf_ratio = 999.0

        # 9. TVL/MCap Ratio
        if mc > 0 and fd.tvl > 0:
            fd.tvl_mc_ratio = fd.tvl / mc
        else:
            fd.tvl_mc_ratio = 0.0

        # 10. Isi ValuationResult
        vr.fundamental       = fd
        vr.ps_ratio          = fd.ps_ratio
        vr.pf_ratio          = fd.pf_ratio
        vr.tvl_mc_ratio      = fd.tvl_mc_ratio
        vr.revenue_growth_mom = fd.revenue_growth_mom

        vr.ps_verdict  = _calc_ps_verdict(fd.ps_ratio)
        vr.tvl_verdict = _calc_tvl_verdict(fd.tvl_mc_ratio)

        # 11. Overall verdict
        verdicts = [vr.ps_verdict, vr.tvl_verdict]
        uv_count = sum(1 for v in verdicts if "UNDERVALUE" in v)
        ov_count = sum(1 for v in verdicts if "OVERVALUE" in v)
        if uv_count >= 2:
            vr.overall_verdict = "EXTREME_UNDERVALUE" if "EXTREME_UNDERVALUE" in verdicts else "UNDERVALUE"
        elif uv_count == 1 and ov_count == 0:
            vr.overall_verdict = "UNDERVALUE"
        elif ov_count >= 1:
            vr.overall_verdict = "OVERVALUE"
        else:
            vr.overall_verdict = "FAIR"

        # 12. Score calculation
        vr.undervalue_score = _calc_undervalue_score(vr)
        vr.score = vr.undervalue_score

        logger.info(
            f"DefiLlama | {base:10s} | P/S:{fd.ps_ratio:.1f}x | "
            f"TVL/MC:{fd.tvl_mc_ratio:.2f}x | Rev30d:${fd.revenue_30d:,.0f} | "
            f"Score:{vr.score:.1f} | {vr.overall_verdict}"
        )

    except Exception as e:
        logger.error(f"DefiLlama analyze error {symbol}: {e}", exc_info=True)

    return vr


# ─────────────────────────────────────────────────────────
#  BATCH SCAN (untuk top DeFi protocols)
# ─────────────────────────────────────────────────────────

def scan_undervalued_defi(min_tvl: float = 1_000_000) -> List[ValuationResult]:
    """
    Scan semua protokol DefiLlama dan return yang undervalued.
    Berguna untuk menemukan gems yang belum terlacak signal lain.

    min_tvl: minimum TVL dalam USD (default $1M)
    """
    results = []
    protocols = _get_protocols()

    for p in protocols:
        tvl = float(p.get("tvl", 0) or 0)
        if tvl < min_tvl:
            continue
        mc = float(p.get("mcap", 0) or 0)
        if mc <= 0:
            continue

        sym = p.get("symbol", "")
        if not sym:
            continue

        vr = analyze_defillama(sym, mc)
        if vr.found and vr.overall_verdict in ("EXTREME_UNDERVALUE", "UNDERVALUE"):
            results.append(vr)
        time.sleep(0.2)

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def format_defillama_section(vr: ValuationResult) -> str:
    """Format bagian DefiLlama untuk Telegram message."""
    if not vr.found or not vr.fundamental:
        return ""

    fd = vr.fundamental
    verdict_emoji = {
        "EXTREME_UNDERVALUE": "🔥",
        "UNDERVALUE":         "💚",
        "FAIR":               "⚪",
        "OVERVALUE":          "🟡",
        "EXTREME_OVERVALUE":  "🔴",
        "UNKNOWN":            "❓",
    }.get(vr.overall_verdict, "❓")

    lines = [f"\n📊 <b>Fundamental (DefiLlama)</b> — {fd.category}"]

    # TVL
    if fd.tvl > 0:
        tvl_chg = f" ({fd.tvl_7d_change:+.1f}% 7D)" if fd.tvl_7d_change else ""
        lines.append(f"  🏦 TVL: <b>${fd.tvl:,.0f}</b>{tvl_chg}")

    # Revenue
    if fd.revenue_30d > 0:
        lines.append(f"  💰 Revenue 30D: <b>${fd.revenue_30d:,.0f}</b>  →  Annualized: ${fd.revenue_annualized:,.0f}")
    if fd.fees_30d > 0:
        lines.append(f"  💹 Fees 30D: <b>${fd.fees_30d:,.0f}</b>")

    # Ratios
    if vr.ps_ratio < 999:
        ps_e = "🔥" if vr.ps_ratio < 1 else ("💚" if vr.ps_ratio < 3 else ("🟡" if vr.ps_ratio < 10 else "🔴"))
        lines.append(f"  {ps_e} P/S Ratio: <b>{vr.ps_ratio:.2f}x</b>  ({vr.ps_verdict.replace('_',' ')})")

    if vr.pf_ratio < 999 and vr.pf_ratio != vr.ps_ratio:
        lines.append(f"  📐 P/F Ratio: <b>{vr.pf_ratio:.2f}x</b>")

    if vr.tvl_mc_ratio > 0:
        tvl_e = "🔥" if vr.tvl_mc_ratio >= 1 else ("💚" if vr.tvl_mc_ratio >= 0.5 else "⚪")
        lines.append(f"  {tvl_e} TVL/MCap: <b>{vr.tvl_mc_ratio:.2f}x</b>  ({vr.tvl_verdict.replace('_',' ')})")

    if vr.revenue_growth_mom != 0:
        g_e = "🚀" if vr.revenue_growth_mom > 30 else ("📈" if vr.revenue_growth_mom > 0 else "📉")
        lines.append(f"  {g_e} Revenue Growth MoM: <b>{vr.revenue_growth_mom:+.1f}%</b>")

    lines.append(f"  {verdict_emoji} <b>Valuation: {vr.overall_verdict.replace('_',' ')}</b>  |  Score: {vr.score:.0f}/35")
    lines.append("")

    return "\n".join(lines)
