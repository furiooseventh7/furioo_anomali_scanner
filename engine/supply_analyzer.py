"""
Supply Analyzer — analisis on-chain supply metrics:
Circulating/Max Supply ratio, FDV vs Market Cap,
ATH distance, dan implikasi selling pressure.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class SupplyResult:
    market_cap          : float = 0.0
    fully_diluted_val   : float = 0.0
    circulating_supply  : float = 0.0
    max_supply          : float = 0.0
    supply_ratio        : float = 1.0      # circulating / max (1 = fully diluted)
    fdv_mc_ratio        : float = 1.0      # FDV / Market Cap (>2 = high dilution risk)
    ath_change_pct      : float = 0.0      # % dari ATH
    selling_pressure    : str   = "UNKNOWN"  # LOW / MEDIUM / HIGH
    category            : str   = "UNKNOWN"  # MICRO / SMALL / MID / LARGE
    signals             : list  = field(default_factory=list)
    score               : float = 0.0       # kontribusi 0–20

def analyze_supply(symbol_base: str, cg_data: dict) -> SupplyResult:
    """
    symbol_base: simbol tanpa USDT (e.g. 'BTC', 'ENA', 'RAVE')
    cg_data: dict hasil get_coingecko_data()
    """
    res = SupplyResult()
    coin = cg_data.get(symbol_base.upper())
    if not coin:
        return res

    res.market_cap        = coin["market_cap"]
    res.fully_diluted_val = coin["fully_diluted_val"]
    res.circulating_supply = coin["circulating_supply"]
    res.max_supply         = coin["max_supply"]
    res.ath_change_pct     = coin["ath_change_pct"]

    # Supply ratio
    if res.max_supply and res.max_supply > 0:
        res.supply_ratio = res.circulating_supply / res.max_supply
    else:
        res.supply_ratio = 1.0   # fully diluted assumption

    # FDV / Market Cap ratio
    if res.market_cap and res.market_cap > 0 and res.fully_diluted_val and res.fully_diluted_val > 0:
        res.fdv_mc_ratio = res.fully_diluted_val / res.market_cap
    else:
        res.fdv_mc_ratio = 1.0

    # Kategorisasi market cap
    mc = res.market_cap
    if mc < 10_000_000:
        res.category = "MICRO"      # < $10M — paling explosive
    elif mc < 100_000_000:
        res.category = "SMALL"      # $10M–$100M
    elif mc < 1_000_000_000:
        res.category = "MID"        # $100M–$1B
    else:
        res.category = "LARGE"      # > $1B

    # Selling pressure assessment
    if res.supply_ratio >= 0.90 and res.fdv_mc_ratio <= 1.15:
        res.selling_pressure = "LOW"      # hampir fully diluted, risiko dilusi rendah
    elif res.fdv_mc_ratio >= 5:
        res.selling_pressure = "HIGH"     # banyak supply belum beredar → dump risk
    elif res.fdv_mc_ratio >= 2.5:
        res.selling_pressure = "MEDIUM"
    else:
        res.selling_pressure = "LOW"

    # ── Scoring ───────────────────────────────────────────
    score = 0.0

    # Small/Micro cap → lebih mudah digerakkan whale
    if res.category == "MICRO":
        res.signals.append(f"🚀 MICRO CAP ${mc/1e6:.2f}M (Explosive pump potential!)")
        score += 20
    elif res.category == "SMALL":
        res.signals.append(f"📊 Small cap ${mc/1e6:.1f}M (Good pump potential)")
        score += 12

    # FDV ratio
    if res.fdv_mc_ratio <= 1.15:
        res.signals.append(f"✅ Supply hampir fully diluted (FDV/MC {res.fdv_mc_ratio:.1f}x) → LOW dilution risk")
        score += 8
    elif res.fdv_mc_ratio >= 5:
        res.signals.append(f"⚠️ Dilution risk TINGGI (FDV/MC {res.fdv_mc_ratio:.1f}x) → holding risk")
        score -= 5
    elif res.fdv_mc_ratio >= 2.5:
        res.signals.append(f"🟡 Dilution risk MEDIUM (FDV/MC {res.fdv_mc_ratio:.1f}x)")
        score -= 2

    # ATH distance (jauh dari ATH = upside besar)
    if res.ath_change_pct <= -80:
        res.signals.append(f"📉 {res.ath_change_pct:.0f}% dari ATH → Extreme discount")
        score += 8
    elif res.ath_change_pct <= -50:
        res.signals.append(f"📉 {res.ath_change_pct:.0f}% dari ATH → Significant discount")
        score += 4

    res.score = max(0.0, min(score, 20.0))
    return res
