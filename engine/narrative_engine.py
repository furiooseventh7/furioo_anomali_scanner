"""
Narrative Engine — CMI-ASS
Memetakan setiap ticker ke narasi/sektor crypto (AI, RWA, DeFi, dll.)
dan menghitung Narrative Hype Score berdasarkan performa harga rata-rata
semua coin dalam narasi yang sama dalam 24 jam terakhir.

Cara kerja:
  1. Setiap ticker dipetakan ke satu atau lebih narasi (NARRATIVE_MAP).
  2. get_narrative_scores(df_ticker) menghitung hype score per narasi:
       - Ambil semua coin dalam narasi tersebut yang ada di df_ticker
       - Rata-rata price_change_24h dan volume_spike mereka
       - Normalisasi ke 0–100
  3. Hasilnya: NarrativeResult(name, hype_score, description, coins)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  PETA TICKER → NARASI
#  Satu ticker bisa punya beberapa narasi (list), ambil yang pertama sebagai
#  narasi utama. Update daftar ini sesuai perkembangan pasar.
# ─────────────────────────────────────────────────────────────────────────────
NARRATIVE_MAP: Dict[str, List[str]] = {
    # ── AI & Agents ──────────────────────────────────────────────────────────
    "TAO"       : ["AI", "DePIN"],
    "NEAR"      : ["AI", "Layer1"],
    "FET"       : ["AI", "DePIN"],
    "AGIX"      : ["AI"],
    "OCEAN"     : ["AI", "DePIN"],
    "RNDR"      : ["AI", "DePIN"],
    "RENDER"    : ["AI", "DePIN"],
    "AKT"       : ["AI", "DePIN"],
    "WLD"       : ["AI"],
    "GRT"       : ["AI", "DeFi"],
    "IO"        : ["AI", "DePIN"],
    "VIRTUAL"   : ["AI", "Gaming"],
    "AI16Z"     : ["AI", "Meme"],
    "AIXBT"     : ["AI", "Meme"],
    "GRIFFAIN"  : ["AI", "Meme"],
    "ZEREBRO"   : ["AI"],
    "ALCH"      : ["AI"],
    "GOAT"      : ["AI", "Meme"],
    "TURBO"     : ["AI", "Meme"],
    "PROMPT"    : ["AI"],

    # ── RWA (Real World Assets) ───────────────────────────────────────────────
    "ONDO"      : ["RWA", "DeFi"],
    "POLYX"     : ["RWA"],
    "CFG"       : ["RWA", "DeFi"],
    "MPL"       : ["RWA", "DeFi"],
    "RIO"       : ["RWA"],
    "LEND"      : ["RWA", "DeFi"],
    "OPEN"      : ["RWA"],
    "TRU"       : ["RWA", "DeFi"],
    "CPOOL"     : ["RWA", "DeFi"],
    "TOKEN"     : ["RWA"],
    "GHO"       : ["RWA", "Stablecoin"],
    "SNX"       : ["RWA", "DeFi"],
    "MKR"       : ["RWA", "DeFi"],
    "AAVE"      : ["RWA", "DeFi"],
    "ETHENA"    : ["RWA", "Stablecoin"],
    "ENA"       : ["RWA", "Stablecoin"],
    "USUAL"     : ["RWA", "Stablecoin"],
    "SKY"       : ["RWA", "DeFi"],

    # ── DeFi ─────────────────────────────────────────────────────────────────
    "UNI"       : ["DeFi"],
    "CAKE"      : ["DeFi"],
    "CRV"       : ["DeFi"],
    "CVX"       : ["DeFi"],
    "BAL"       : ["DeFi"],
    "SUSHI"     : ["DeFi"],
    "1INCH"     : ["DeFi"],
    "LDO"       : ["DeFi", "LSD"],
    "RPL"       : ["DeFi", "LSD"],
    "FXS"       : ["DeFi", "Stablecoin"],
    "FRAX"      : ["DeFi", "Stablecoin"],
    "PENDLE"    : ["DeFi", "Yield"],
    "EIGEN"     : ["DeFi", "LSD"],
    "ETHFI"     : ["DeFi", "LSD"],
    "RSETH"     : ["DeFi", "LSD"],
    "RENZO"     : ["DeFi", "LSD"],
    "VELO"      : ["DeFi"],
    "AERO"      : ["DeFi"],
    "FLUID"     : ["DeFi"],
    "MORPHO"    : ["DeFi"],
    "EULER"     : ["DeFi"],

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    "ETH"       : ["Layer1", "LSD"],
    "SOL"       : ["Layer1"],
    "ADA"       : ["Layer1"],
    "AVAX"      : ["Layer1"],
    "DOT"       : ["Layer1"],
    "ATOM"      : ["Layer1", "Interop"],
    "ALGO"      : ["Layer1"],
    "SUI"       : ["Layer1"],
    "APT"       : ["Layer1"],
    "INJ"       : ["Layer1", "DeFi"],
    "SEI"       : ["Layer1"],
    "TIA"       : ["Layer1", "Modular"],
    "HYPE"      : ["Layer1", "DeFi"],
    "MONAD"     : ["Layer1"],
    "BERACHAIN" : ["Layer1", "DeFi"],
    "BERA"      : ["Layer1", "DeFi"],
    "S"         : ["Layer1"],
    "FTM"       : ["Layer1", "DeFi"],

    # ── Layer 2 & Scaling ─────────────────────────────────────────────────────
    "MATIC"     : ["Layer2"],
    "POL"       : ["Layer2"],
    "ARB"       : ["Layer2"],
    "OP"        : ["Layer2"],
    "IMX"       : ["Layer2", "Gaming"],
    "STRK"      : ["Layer2"],
    "MANTA"     : ["Layer2", "DeFi"],
    "ZKSYNC"    : ["Layer2"],
    "ZK"        : ["Layer2"],
    "SCROLL"    : ["Layer2"],
    "LINEA"     : ["Layer2"],
    "BLAST"     : ["Layer2", "DeFi"],
    "METIS"     : ["Layer2"],
    "BOBA"      : ["Layer2"],
    "BASE"      : ["Layer2"],

    # ── Meme ─────────────────────────────────────────────────────────────────
    "DOGE"      : ["Meme"],
    "SHIB"      : ["Meme"],
    "PEPE"      : ["Meme"],
    "FLOKI"     : ["Meme"],
    "WIF"       : ["Meme"],
    "BONK"      : ["Meme"],
    "MEME"      : ["Meme"],
    "POPCAT"    : ["Meme"],
    "FARTCOIN"  : ["Meme"],
    "PNUT"      : ["Meme"],
    "MOG"       : ["Meme"],
    "NEIRO"     : ["Meme"],
    "CHILLGUY"  : ["Meme"],
    "PONKE"     : ["Meme"],
    "BRETT"     : ["Meme"],
    "TRUMP"     : ["Meme", "Political"],
    "MELANIA"   : ["Meme", "Political"],
    "MAGA"      : ["Meme", "Political"],

    # ── Gaming & Metaverse ────────────────────────────────────────────────────
    "AXS"       : ["Gaming"],
    "SAND"      : ["Gaming", "Metaverse"],
    "MANA"      : ["Gaming", "Metaverse"],
    "GALA"      : ["Gaming"],
    "ILV"       : ["Gaming"],
    "BEAM"      : ["Gaming"],
    "RON"       : ["Gaming"],
    "PYR"       : ["Gaming"],
    "MAGIC"     : ["Gaming"],
    "YGG"       : ["Gaming"],
    "PRIME"     : ["Gaming"],
    "PORTAL"    : ["Gaming"],
    "PIXEL"     : ["Gaming"],
    "NYAN"      : ["Gaming"],

    # ── DePIN (Decentralized Physical Infrastructure) ─────────────────────────
    "HNT"       : ["DePIN"],
    "MOBILE"    : ["DePIN"],
    "IOTX"      : ["DePIN"],
    "POKT"      : ["DePIN"],
    "GLM"       : ["DePIN"],
    "AR"        : ["DePIN", "Storage"],
    "FIL"       : ["DePIN", "Storage"],
    "STORJ"     : ["DePIN", "Storage"],
    "ROSE"      : ["DePIN", "Privacy"],
    "DIMO"      : ["DePIN"],
    "GEODNET"   : ["DePIN"],
    "WNT"       : ["DePIN"],
    "NATIX"     : ["DePIN"],

    # ── Bitcoin Ecosystem ─────────────────────────────────────────────────────
    "BTC"       : ["Bitcoin"],
    "WBTC"      : ["Bitcoin", "DeFi"],
    "STX"       : ["Bitcoin", "Layer2"],
    "ORDI"      : ["Bitcoin", "Ordinals"],
    "SATS"      : ["Bitcoin", "Ordinals"],
    "RUNE"      : ["Bitcoin", "Ordinals"],
    "RATS"      : ["Bitcoin", "Ordinals"],
    "MERLIN"    : ["Bitcoin", "Layer2"],
    "B2"        : ["Bitcoin", "Layer2"],
    "MERL"      : ["Bitcoin", "Layer2"],

    # ── Payments & Privacy ────────────────────────────────────────────────────
    "XRP"       : ["Payments"],
    "XLM"       : ["Payments"],
    "LTC"       : ["Payments"],
    "BCH"       : ["Payments"],
    "DASH"      : ["Payments", "Privacy"],
    "ZEC"       : ["Privacy"],
    "XMR"       : ["Privacy"],
    "SCRT"      : ["Privacy"],

    # ── Interoperability & Bridge ─────────────────────────────────────────────
    "LINK"      : ["Oracle", "Interop"],
    "BAND"      : ["Oracle"],
    "API3"      : ["Oracle"],
    "AXL"       : ["Interop"],
    "W"         : ["Interop"],
    "PYTH"      : ["Oracle"],
    "ARKM"      : ["Intelligence"],

    # ── Stablecoins & Yield ───────────────────────────────────────────────────
    "USDE"      : ["Stablecoin"],
    "SUSD"      : ["Stablecoin"],
    "LUSD"      : ["Stablecoin"],
    "RAI"       : ["Stablecoin"],
    "CRVUSD"    : ["Stablecoin"],

    # ── Social & Creator ──────────────────────────────────────────────────────
    "DESO"      : ["SocialFi"],
    "CYB"       : ["SocialFi"],
    "FRIEND"    : ["SocialFi"],
    "BONSAI"    : ["SocialFi"],

    # ── Liquid Staking (LSD) ──────────────────────────────────────────────────
    "STETH"     : ["LSD"],
    "RETH"      : ["LSD"],
    "CBETH"     : ["LSD"],
    "SFRXETH"   : ["LSD"],
    "SWETH"     : ["LSD"],
    "ANKR"      : ["LSD"],
    "SSV"       : ["LSD"],
    "OETH"      : ["LSD"],

    # ── Exchange Tokens ───────────────────────────────────────────────────────
    "BNB"       : ["Exchange", "Layer1"],
    "OKB"       : ["Exchange"],
    "CRO"       : ["Exchange"],
    "KCS"       : ["Exchange"],
    "GT"        : ["Exchange"],
    "BGB"       : ["Exchange"],
    "MX"        : ["Exchange"],
    "HT"        : ["Exchange"],
    "LEO"       : ["Exchange"],
    "WBT"       : ["Exchange"],

    # ── Modular & DA ─────────────────────────────────────────────────────────
    "AVAIL"     : ["Modular"],
    "EIGEN"     : ["Modular", "LSD"],
    "ALT"       : ["Modular"],
    "BLOBSCRIPTIONS": ["Modular"],

    # ── SocialFi / NFT ────────────────────────────────────────────────────────
    "APE"       : ["NFT", "Gaming"],
    "BLUR"      : ["NFT"],
    "LOOKS"     : ["NFT"],
    "X2Y2"      : ["NFT"],
    "FLOW"      : ["NFT", "Gaming"],
}

# Deskripsi narasi untuk display
NARRATIVE_DESC: Dict[str, str] = {
    "AI"        : "Artificial Intelligence & Autonomous Agents",
    "RWA"       : "Real World Assets Tokenization",
    "DeFi"      : "Decentralized Finance",
    "Layer1"    : "Layer 1 Blockchain",
    "Layer2"    : "Layer 2 & Scaling Solutions",
    "Meme"      : "Meme Coins",
    "Gaming"    : "GameFi & Metaverse",
    "DePIN"     : "Decentralized Physical Infrastructure",
    "Bitcoin"   : "Bitcoin Ecosystem & Ordinals",
    "Payments"  : "Payments & Remittance",
    "Privacy"   : "Privacy Coins",
    "Oracle"    : "Oracle Networks",
    "Interop"   : "Cross-chain Interoperability",
    "LSD"       : "Liquid Staking Derivatives",
    "Stablecoin": "Stablecoins & Yield-Bearing Stable",
    "Exchange"  : "Exchange Tokens",
    "Modular"   : "Modular Blockchain & Data Availability",
    "NFT"       : "NFT & Digital Collectibles",
    "SocialFi"  : "Social Finance & Creator Economy",
    "Yield"     : "Yield Optimization",
    "Ordinals"  : "Bitcoin Ordinals & BRC-20",
    "Storage"   : "Decentralized Storage",
    "Political" : "Political Meme Tokens",
    "Intelligence": "On-chain Intelligence & Analytics",
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeResult:
    name            : str   = "Unknown"
    hype_score      : int   = 0          # 0–100
    hype_label      : str   = "UNKNOWN"  # COLD / COOL / NEUTRAL / WARM / HOT / 🔥 HYPE
    description     : str   = ""
    avg_change_24h  : float = 0.0        # rata2 perubahan harga 24H semua coin di narasi
    top_gainers     : list  = field(default_factory=list)  # [(symbol, chg), ...]
    coin_count      : int   = 0          # jumlah coin aktif dalam narasi ini
    secondary       : str   = ""         # narasi sekunder (jika ada)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_narrative(symbol_base: str) -> Optional[str]:
    """Kembalikan narasi utama untuk sebuah ticker (tanpa USDT)."""
    narr = NARRATIVE_MAP.get(symbol_base.upper())
    return narr[0] if narr else None


def get_secondary_narrative(symbol_base: str) -> Optional[str]:
    """Kembalikan narasi sekunder jika ada."""
    narr = NARRATIVE_MAP.get(symbol_base.upper())
    return narr[1] if narr and len(narr) > 1 else None


def _score_to_label(score: int) -> str:
    if score >= 85: return "🔥 HYPE PEAK"
    if score >= 70: return "🚀 VERY HOT"
    if score >= 55: return "🌶️ HOT"
    if score >= 40: return "♨️ WARM"
    if score >= 25: return "😐 NEUTRAL"
    if score >= 10: return "🧊 COOL"
    return "❄️ COLD"


def compute_narrative_scores(df_ticker: pd.DataFrame) -> Dict[str, NarrativeResult]:
    """
    Hitung hype score untuk semua narasi berdasarkan df_ticker (hasil get_all_tickers_24h).
    df_ticker harus punya kolom: 'symbol', 'priceChangePercent', 'quoteVolume'

    Returns: dict {narrative_name: NarrativeResult}
    """
    if df_ticker.empty:
        return {}

    # Build lookup: symbol_base → row
    try:
        df = df_ticker[["symbol", "priceChangePercent", "quoteVolume"]].copy()
        df["base"] = df["symbol"].str.replace("USDT", "", regex=False)
        df["chg"]  = pd.to_numeric(df["priceChangePercent"], errors="coerce").fillna(0.0)
        df["vol"]  = pd.to_numeric(df["quoteVolume"], errors="coerce").fillna(0.0)
    except Exception as e:
        logger.warning(f"narrative: gagal proses df_ticker: {e}")
        return {}

    # Kumpulkan data per narasi
    narr_data: Dict[str, list] = {}  # narr_name → [(base, chg, vol)]
    for _, row in df.iterrows():
        base = row["base"]
        narratives = NARRATIVE_MAP.get(base.upper())
        if not narratives:
            continue
        for narr in narratives:
            if narr not in narr_data:
                narr_data[narr] = []
            narr_data[narr].append((base, float(row["chg"]), float(row["vol"])))

    if not narr_data:
        return {}

    # Hitung rata-rata chg dan vol per narasi, normalisasi ke hype score
    results: Dict[str, NarrativeResult] = {}

    # Kita butuh global max_avg_chg untuk normalisasi — ambil dari semua narasi
    avg_changes = {}
    for narr, coins in narr_data.items():
        if not coins:
            continue
        avg_chg = sum(c[1] for c in coins) / len(coins)
        avg_changes[narr] = avg_chg

    if not avg_changes:
        return {}

    # Normalisasi: max avg_chg jadi ~80 score, flat distribution sisanya
    # Gunakan range [-20%, +20%] sebagai baseline; di atas itu makin tinggi
    max_chg = max(avg_changes.values())
    min_chg = min(avg_changes.values())
    chg_range = max(max_chg - min_chg, 1.0)  # hindari divide by zero

    for narr, coins in narr_data.items():
        avg_chg = avg_changes.get(narr, 0.0)
        count   = len(coins)

        # Hype score formula:
        # Base: normalisasi avg_chg ke 0-70 (relative to range)
        # Bonus: +10 jika avg_chg > 5%, +15 jika > 10%, +20 jika > 20%
        # Bonus: +5 jika banyak coin (>= 5) semua naik
        normalized = ((avg_chg - min_chg) / chg_range) * 70.0
        bonus = 0
        if avg_chg > 20:  bonus = 25
        elif avg_chg > 10: bonus = 18
        elif avg_chg > 5:  bonus = 10
        elif avg_chg > 2:  bonus = 5
        elif avg_chg < -10: bonus = -15
        elif avg_chg < -5:  bonus = -8

        # Coin count bonus — narasi dengan banyak coin aktif lebih valid
        if count >= 8: normalized += 5
        elif count >= 5: normalized += 3

        raw_score = normalized + bonus
        hype_score = int(min(max(raw_score, 0), 100))

        # Top gainers dalam narasi ini (max 3)
        sorted_coins = sorted(coins, key=lambda x: x[1], reverse=True)
        top_gainers  = [(c[0], c[1]) for c in sorted_coins[:3]]

        results[narr] = NarrativeResult(
            name           = narr,
            hype_score     = hype_score,
            hype_label     = _score_to_label(hype_score),
            description    = NARRATIVE_DESC.get(narr, narr),
            avg_change_24h = round(avg_chg, 2),
            top_gainers    = top_gainers,
            coin_count     = count,
        )

    return results


def get_ticker_narrative_result(
    symbol_base: str,
    narrative_scores: Dict[str, NarrativeResult]
) -> Optional[NarrativeResult]:
    """
    Ambil NarrativeResult untuk ticker tertentu.
    symbol_base: tanpa USDT (e.g. 'ONDO', 'TAO')
    """
    narr_list = NARRATIVE_MAP.get(symbol_base.upper())
    if not narr_list:
        return None
    # Cari narasi dengan hype score tertinggi dari list narasi ticker ini
    best = None
    for narr in narr_list:
        nr = narrative_scores.get(narr)
        if nr and (best is None or nr.hype_score > best.hype_score):
            best = nr
            best.secondary = narr_list[1] if len(narr_list) > 1 and narr_list[0] == narr else ""
    return best
