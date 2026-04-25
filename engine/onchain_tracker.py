"""
On-Chain DEX Whale Tracker — CMI-ASS v7
==========================================
Membaca transaksi besar di DEX on-chain secara otomatis menggunakan
Flipside Crypto API (gratis, no credit card, 1000 query/hari).

ARSITEKTUR:
  PRIMARY   → Flipside Crypto (SQL query on-chain, gratis, no key for basic)
  SECONDARY → Dune Analytics  (via API key jika ada)
  TERTIARY  → DexScreener     (fallback gratis tanpa API key, covers all chains)

CHAIN YANG DICAKUP (13 blockchain):
  ┌─────────────────────────────────────────────────────────┐
  │  EVM-Compatible (12 chain):                             │
  │   • Ethereum     — Uniswap V2/V3, Curve, Balancer       │
  │   • Arbitrum     — Camelot, GMX, Ramses, Uniswap V3     │
  │   • BSC          — PancakeSwap V2/V3, Biswap, DODO      │
  │   • Polygon      — QuickSwap, Uniswap V3, SushiSwap     │
  │   • Optimism     — Velodrome, Uniswap V3, Synthetix     │
  │   • Avalanche    — Trader Joe, Pangolin, GMX             │
  │   • Base         — Aerodrome, Uniswap V3, BaseSwap      │
  │   • Blast        — Thruster, Ring Protocol               │
  │   • Mantle       — FusionX, Agni Finance                 │
  │   • Scroll       — Izumi, SyncSwap                       │
  │   • Gnosis       — Honeyswap, Swapr, Curve               │
  │   • Fantom       — SpookySwap, SpiritSwap, Beethoven X   │
  │                                                          │
  │  Non-EVM (1 chain):                                      │
  │   • Solana       — Raydium, Orca, Jupiter, Meteora       │
  └─────────────────────────────────────────────────────────┘

QUERY SQL YANG DIJALANKAN OTOMATIS:
  "Cari semua transaksi swap di SEMUA DEX di 13 blockchain
   dalam 4 jam terakhir yang nilainya di atas $100.000.
   Urutkan berdasarkan koin apa yang paling banyak dibeli,
   dan flagging wallet baru (< 7 hari) sebagai STEALTH ACCUMULATION."

CARA KERJA:
  1. Query SQL dikirim via HTTP POST ke Flipside API (UNION ALL 13 chain)
  2. Hasil: top 25 token dengan total whale buys terbesar dalam 4 jam
  3. Jika wallet baru + volume masif → flag STEALTH ACCUMULATION
  4. Alert dikirim ke Telegram beserta chart TradingView
  5. Score on-chain (0–30) menambah confidence score sinyal CEX

FLIPSIDE CRYPTO:
  - URL: https://api-v2.flipsidecrypto.xyz/json-rpc
  - Gratis tanpa API key (rate limit 5 req/menit)
  - Dengan API key (free tier): 1000 query/hari, lebih cepat
  - Daftar: https://flipsidecrypto.xyz

DUNE ANALYTICS:
  - URL: https://api.dune.com/api/v1/query/{query_id}/results
  - Butuh API key (free tier tersedia)
  - Daftar: https://dune.com/settings/api

DEXSCREENER (no key needed):
  - Covers 50+ chains termasuk semua EVM + Solana + TON + SUI + Aptos
  - https://api.dexscreener.com/token-boosts/latest/v1
"""

import os
import re
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

TIMEOUT      = 15
HEADERS_JSON = {"Content-Type": "application/json", "User-Agent": "CMI-ASS/7.0"}

# ─── API Keys (dari environment / GitHub Secrets) ─────────
FLIPSIDE_KEY  = os.getenv("FLIPSIDE_API_KEY", "")     # opsional
DUNE_KEY      = os.getenv("DUNE_API_KEY", "")         # opsional

FLIPSIDE_URL  = "https://api-v2.flipsidecrypto.xyz/json-rpc"
DUNE_URL      = "https://api.dune.com/api/v1"
DEXSCREENER_URL = "https://api.dexscreener.com"

# Threshold transaksi yang dianggap "whale"
WHALE_TX_USD     = 100_000   # min $100K per transaksi
NEW_WALLET_DAYS  = 7         # wallet baru = kurang dari 7 hari aktivitas
MIN_WHALE_TXS    = 3         # minimal 3 tx whale dalam 4 jam = STEALTH flag


# ─────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class OnChainWhaleEvent:
    """Satu event akumulasi whale on-chain pada sebuah token."""
    symbol          : str   = ""
    token_address   : str   = ""
    chain           : str   = ""           # ethereum, bsc, arbitrum, dll
    total_bought_usd: float = 0.0          # total USD dibeli whale dalam 4H
    whale_tx_count  : int   = 0            # jumlah transaksi whale
    new_wallet_count: int   = 0            # jumlah wallet baru yang beli
    unique_whales   : int   = 0            # jumlah unique whale wallets
    avg_tx_usd      : float = 0.0          # rata-rata per transaksi
    largest_tx_usd  : float = 0.0          # transaksi terbesar
    time_window_hrs : float = 4.0          # window analisis (jam)
    is_stealth_acc  : bool  = False        # wallet baru + volume besar = stealth
    dex_name        : str   = ""           # Uniswap / PancakeSwap / dll
    source          : str   = ""           # flipside / dune / dexscreener


@dataclass
class OnChainResult:
    """Hasil keseluruhan on-chain whale tracking untuk satu simbol."""
    symbol          : str   = ""
    found_onchain   : bool  = False
    events          : List[OnChainWhaleEvent] = field(default_factory=list)

    # Aggregated metrics
    total_bought_usd: float = 0.0
    whale_tx_count  : int   = 0
    new_wallet_count: int   = 0
    unique_whales   : int   = 0
    is_stealth_acc  : bool  = False
    chains_active   : List[str] = field(default_factory=list)

    # Signal outputs
    signals         : List[str] = field(default_factory=list)
    score           : float = 0.0           # 0–30, masuk ke confidence score
    alert_level     : str   = "NONE"        # NONE / WATCH / HIGH / CRITICAL


# ─────────────────────────────────────────────────────────
#  FLIPSIDE CRYPTO — SQL QUERY ENGINE
# ─────────────────────────────────────────────────────────

def _build_flipside_query(symbol_or_address: str, hours: int = 4) -> str:
    """
    Build SQL query untuk Flipside Crypto — ALL CHAINS.

    Chain yang dicakup (semua yang didukung Flipside dengan ez_dex_swaps):
      EVM   : ethereum, arbitrum, bsc (Binance Smart Chain), polygon,
               optimism, avalanche, base, blast, mantle, scroll, gnosis, fantom
      Non-EVM: solana (via solana.defi.ez_dex_swaps, schema berbeda)

    Wallet age detection dilakukan terhadap masing-masing chain asal transaksi.
    Solana menggunakan signers bukan origin_from_address.
    """
    base = symbol_or_address.upper().replace("USDT", "").replace("USDC", "")

    # ── EVM chains: schema identik, bisa UNION ALL langsung ──────
    # Setiap chain di Flipside punya namespace sendiri:
    # ethereum.defi.ez_dex_swaps, arbitrum.defi.ez_dex_swaps, dst.
    evm_chains = [
        ("ethereum",  "ethereum"),
        ("arbitrum",  "arbitrum"),
        ("bsc",       "bsc"),            # Binance Smart Chain
        ("polygon",   "polygon"),
        ("optimism",  "optimism"),
        ("avalanche", "avalanche"),
        ("base",      "base"),           # Base (Coinbase L2)
        ("blast",     "blast"),          # Blast L2
        ("mantle",    "mantle"),         # Mantle Network
        ("scroll",    "scroll"),         # Scroll ZK-rollup
        ("gnosis",    "gnosis"),         # Gnosis Chain (xDai)
        ("fantom",    "fantom"),         # Fantom
    ]

    # Build UNION ALL untuk semua EVM chain
    evm_blocks = []
    for schema, chain_label in evm_chains:
        block = f"""  SELECT
    block_timestamp,
    tx_hash,
    origin_from_address  AS wallet,
    token_out_symbol     AS token_symbol,
    token_out            AS token_address,
    amount_out_usd       AS amount_usd,
    platform             AS dex_name,
    '{chain_label}'      AS chain,
    'evm'                AS chain_type
  FROM {schema}.defi.ez_dex_swaps
  WHERE
    block_timestamp >= DATEADD(hour, -{hours}, CURRENT_TIMESTAMP)
    AND token_out_symbol ILIKE '%{base}%'
    AND amount_out_usd >= {WHALE_TX_USD}"""
        evm_blocks.append(block)

    evm_union = "\n  UNION ALL\n".join(evm_blocks)

    # ── Solana: schema berbeda (signers, swap_from/to) ────────────
    # Solana menggunakan swapper (bukan origin_from_address)
    # token_out_mint → token address, token_out_symbol untuk nama
    solana_block = f"""  SELECT
    block_timestamp,
    tx_id                AS tx_hash,
    swapper              AS wallet,
    token_out_symbol     AS token_symbol,
    token_out_mint       AS token_address,
    amount_out_usd       AS amount_usd,
    program_id           AS dex_name,
    'solana'             AS chain,
    'solana'             AS chain_type
  FROM solana.defi.ez_dex_swaps
  WHERE
    block_timestamp >= DATEADD(hour, -{hours}, CURRENT_TIMESTAMP)
    AND token_out_symbol ILIKE '%{base}%'
    AND amount_out_usd >= {WHALE_TX_USD}"""

    query = f"""
WITH recent_swaps AS (
{evm_union}
  UNION ALL
{solana_block}
),
-- Wallet age: EVM chains (cross-chain via ethereum sebagai proxy)
-- Untuk akurasi wallet age per chain, kita gunakan ethereum sebagai baseline
-- karena most whale wallets aktif di ethereum
evm_wallet_age AS (
  SELECT
    origin_from_address AS wallet,
    MIN(block_timestamp) AS first_tx,
    'evm'               AS chain_type
  FROM ethereum.core.fact_transactions
  WHERE origin_from_address IN (
    SELECT DISTINCT wallet FROM recent_swaps WHERE chain_type = 'evm'
  )
  GROUP BY 1, 3
),
-- Solana wallet age
sol_wallet_age AS (
  SELECT
    signers[0]           AS wallet,
    MIN(block_timestamp) AS first_tx,
    'solana'             AS chain_type
  FROM solana.core.fact_transactions
  WHERE signers[0] IN (
    SELECT DISTINCT wallet FROM recent_swaps WHERE chain_type = 'solana'
  )
  GROUP BY 1, 3
),
all_wallet_age AS (
  SELECT wallet, first_tx, chain_type FROM evm_wallet_age
  UNION ALL
  SELECT wallet, first_tx, chain_type FROM sol_wallet_age
)
SELECT
  rs.token_symbol,
  rs.token_address,
  rs.chain,
  rs.dex_name,
  COUNT(DISTINCT rs.tx_hash)    AS whale_tx_count,
  COUNT(DISTINCT rs.wallet)     AS unique_whales,
  SUM(rs.amount_usd)            AS total_bought_usd,
  AVG(rs.amount_usd)            AS avg_tx_usd,
  MAX(rs.amount_usd)            AS largest_tx_usd,
  COUNT(DISTINCT CASE
    WHEN wa.first_tx >= DATEADD(day, -{NEW_WALLET_DAYS}, CURRENT_TIMESTAMP)
    THEN rs.wallet
  END) AS new_wallet_count
FROM recent_swaps rs
LEFT JOIN all_wallet_age wa
  ON rs.wallet = wa.wallet AND rs.chain_type = wa.chain_type
GROUP BY 1, 2, 3, 4
ORDER BY total_bought_usd DESC
LIMIT 25
"""
    return query.strip()


def _run_flipside_query(sql: str) -> Optional[List[dict]]:
    """
    Kirim SQL query ke Flipside Crypto API dan tunggu hasilnya.
    Flipside menggunakan async pattern: create → poll → fetch results.
    """
    headers = HEADERS_JSON.copy()
    if FLIPSIDE_KEY:
        headers["x-api-key"] = FLIPSIDE_KEY

    # Step 1: Create query
    payload = {
        "jsonrpc": "2.0",
        "method":  "createQuery",
        "params":  {"sql": sql, "ttlMinutes": 10},
        "id":      1,
    }
    try:
        r = requests.post(FLIPSIDE_URL, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            logger.warning(f"Flipside createQuery error: {body['error']}")
            return None
        token = body.get("result", {}).get("token")
        if not token:
            return None
    except Exception as e:
        logger.debug(f"Flipside createQuery: {e}")
        return None

    # Step 2: Poll untuk hasil (max 30 detik)
    poll_url = FLIPSIDE_URL
    poll_payload = {
        "jsonrpc": "2.0",
        "method":  "getQueryResults",
        "params":  {"token": token, "pageNumber": 1, "pageSize": 100},
        "id":      2,
    }
    for _ in range(6):   # max 6x poll × 5 detik = 30 detik
        time.sleep(5)
        try:
            r = requests.post(poll_url, json=poll_payload, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            body = r.json()
            result = body.get("result", {})
            status = result.get("status", "")

            if status == "finished":
                rows = result.get("rows", [])
                cols = result.get("columnNames", [])
                if rows and cols:
                    return [dict(zip(cols, row)) for row in rows]
                return []
            elif status in ("error", "failed"):
                logger.warning(f"Flipside query failed: {result.get('errorMessage','')}")
                return None
            # status = "running" → lanjut poll
        except Exception as e:
            logger.debug(f"Flipside poll: {e}")

    logger.warning("Flipside query timeout (>30s)")
    return None


def _parse_flipside_results(rows: List[dict], symbol: str) -> List[OnChainWhaleEvent]:
    """Parse hasil Flipside query menjadi list OnChainWhaleEvent."""
    events = []
    for row in rows:
        try:
            ev = OnChainWhaleEvent(
                symbol           = row.get("TOKEN_SYMBOL", symbol),
                token_address    = row.get("TOKEN_ADDRESS", ""),
                chain            = row.get("CHAIN", "ethereum"),
                dex_name         = row.get("DEX_NAME", "DEX"),
                whale_tx_count   = int(row.get("WHALE_TX_COUNT", 0) or 0),
                unique_whales    = int(row.get("UNIQUE_WHALES", 0) or 0),
                total_bought_usd = float(row.get("TOTAL_BOUGHT_USD", 0) or 0),
                avg_tx_usd       = float(row.get("AVG_TX_USD", 0) or 0),
                largest_tx_usd   = float(row.get("LARGEST_TX_USD", 0) or 0),
                new_wallet_count = int(row.get("NEW_WALLET_COUNT", 0) or 0),
                time_window_hrs  = 4.0,
                source           = "flipside",
            )
            # Flag stealth accumulation
            if (ev.new_wallet_count >= 2 and
                    ev.whale_tx_count >= MIN_WHALE_TXS and
                    ev.total_bought_usd >= 500_000):
                ev.is_stealth_acc = True
            events.append(ev)
        except Exception as e:
            logger.debug(f"Flipside parse row error: {e}")
    return events


# ─────────────────────────────────────────────────────────
#  DUNE ANALYTICS — Pre-built Query Execution
# ─────────────────────────────────────────────────────────
# Query Dune yang sudah dibuat dan di-share publik:
# - 3358740: DEX large trades > $100K, 4H rolling
# - 1258228: New wallet accumulation pattern
# Anda bisa fork query ini di https://dune.com dan ganti dengan query_id kamu sendiri

DUNE_QUERY_IDS = {
    "large_dex_buys" : 3358740,    # query publik: large DEX buys
    "new_wallet_acc" : 1258228,    # query publik: new wallet accumulation
}


def _run_dune_query(query_id: int, params: dict = None) -> Optional[List[dict]]:
    """Execute Dune query dan ambil hasilnya."""
    if not DUNE_KEY:
        return None

    headers = {"X-DUNE-API-KEY": DUNE_KEY, **HEADERS_JSON}

    # Execute query
    exec_url = f"{DUNE_URL}/query/{query_id}/execute"
    try:
        body = {"performance": "medium"}
        if params:
            body["query_parameters"] = params
        r = requests.post(exec_url, json=body, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        exec_id = r.json().get("execution_id")
        if not exec_id:
            return None
    except Exception as e:
        logger.debug(f"Dune execute {query_id}: {e}")
        return None

    # Poll hasil (max 30 detik)
    status_url = f"{DUNE_URL}/execution/{exec_id}/status"
    result_url = f"{DUNE_URL}/execution/{exec_id}/results"
    for _ in range(6):
        time.sleep(5)
        try:
            r = requests.get(status_url, headers=headers, timeout=TIMEOUT)
            status = r.json().get("state", "")
            if status == "QUERY_STATE_COMPLETED":
                r2 = requests.get(result_url, headers=headers, timeout=TIMEOUT)
                rows = r2.json().get("result", {}).get("rows", [])
                return rows
            elif "FAILED" in status or "CANCELLED" in status:
                return None
        except Exception as e:
            logger.debug(f"Dune poll: {e}")

    return None


def _run_dune_with_token_filter(symbol: str) -> Optional[List[dict]]:
    """Run Dune large buys query dengan filter token."""
    base = symbol.upper().replace("USDT", "").replace("USDC", "")
    params = {
        "token_symbol"  : base,
        "min_usd"       : str(WHALE_TX_USD),
        "hours_back"    : "4",
    }
    return _run_dune_query(DUNE_QUERY_IDS["large_dex_buys"], params)


# ─────────────────────────────────────────────────────────
#  DEXSCREENER — No-key Fallback
# ─────────────────────────────────────────────────────────

def _get_dexscreener_pairs(symbol: str) -> Optional[List[dict]]:
    """
    Ambil data pair dari DexScreener — ALL CHAINS.
    Endpoint /latest/dex/search mencari di 50+ blockchain sekaligus:
    ETH, BSC, ARB, SOL, BASE, AVAX, MATIC, OP, FTM, BLAST, MANTLE,
    SCROLL, TON, SUI, APTOS, NEAR, CRO, KAVA, CELO, AURORA, dll.

    Gratis, no API key, rate limit generous (~300 req/menit).
    """
    base = symbol.upper().replace("USDT", "").replace("USDC", "")
    try:
        url = f"{DEXSCREENER_URL}/latest/dex/search"
        r = requests.get(url, params={"q": base}, headers=HEADERS_JSON, timeout=TIMEOUT)
        r.raise_for_status()
        pairs = r.json().get("pairs", [])
        # Filter: hanya pair aktif dengan volume 24H > $50K (semua chain)
        filtered = [
            p for p in pairs
            if float(p.get("volume", {}).get("h24", 0) or 0) >= 50_000
        ]
        # Log chain coverage
        if filtered:
            chains_found = list({p.get("chainId", "?") for p in filtered})
            logger.debug(f"DexScreener {base}: {len(filtered)} pairs across "
                         f"{len(chains_found)} chains: {chains_found[:6]}")
        return filtered
    except Exception as e:
        logger.debug(f"DexScreener search {symbol}: {e}")
        return None


def _get_dexscreener_boosted() -> Optional[List[dict]]:
    """
    Ambil token yang sedang boosted/trending di DexScreener.
    Endpoint gratis: /token-boosts/latest/v1
    Ini adalah indikator token yang mendapatkan perhatian besar tiba-tiba.
    """
    try:
        url = f"{DEXSCREENER_URL}/token-boosts/latest/v1"
        r = requests.get(url, headers=HEADERS_JSON, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        logger.debug(f"DexScreener boosted: {e}")
        return None


def _dexscreener_to_event(pairs: List[dict], symbol: str) -> List[OnChainWhaleEvent]:
    """Convert DexScreener pairs ke OnChainWhaleEvent (estimasi dari volume data)."""
    events = []
    aggregated: Dict[str, dict] = {}

    for pair in pairs:
        chain    = pair.get("chainId", "unknown")
        dex      = pair.get("dexId", "unknown")
        vol_1h   = float(pair.get("volume", {}).get("h1", 0) or 0)
        vol_6h   = float(pair.get("volume", {}).get("h6", 0) or 0)
        vol_24h  = float(pair.get("volume", {}).get("h24", 0) or 0)
        price    = float(pair.get("priceUsd", 0) or 0)
        liq      = float(pair.get("liquidity", {}).get("usd", 0) or 0)
        txns_1h  = pair.get("txns", {}).get("h1", {})
        buys_1h  = int(txns_1h.get("buys", 0) or 0)

        # Estimasi: jika volume 1H sangat tinggi relatif ke 6H avg → spike
        avg_vol_per_hour = vol_6h / 6 if vol_6h > 0 else 1
        vol_ratio = vol_1h / avg_vol_per_hour if avg_vol_per_hour > 0 else 1

        # Estimasi whale trades: asumsi top 10% trades = whale
        # Jika vol_1h > $500K dan txn count rendah → beberapa transaksi besar
        estimated_whale_vol = 0.0
        estimated_whale_txs = 0
        if buys_1h > 0 and vol_1h > 0:
            avg_per_tx = vol_1h / max(buys_1h, 1)
            if avg_per_tx > WHALE_TX_USD:
                # Semua transaksi tergolong whale (rata-rata sudah > $100K)
                estimated_whale_vol = vol_1h
                estimated_whale_txs = buys_1h
            elif vol_1h > WHALE_TX_USD * 3:
                # Estimasi: 10% transaksi = whale
                estimated_whale_vol = vol_1h * 0.10
                estimated_whale_txs = max(1, buys_1h // 10)

        key = f"{chain}_{dex}"
        if key not in aggregated:
            aggregated[key] = {
                "chain": chain, "dex": dex,
                "whale_vol": 0, "whale_txs": 0,
                "vol_ratio": vol_ratio, "liquidity": liq,
            }
        aggregated[key]["whale_vol"] += estimated_whale_vol
        aggregated[key]["whale_txs"] += estimated_whale_txs
        aggregated[key]["vol_ratio"] = max(aggregated[key]["vol_ratio"], vol_ratio)

    for key, ag in aggregated.items():
        if ag["whale_vol"] < 50_000:
            continue
        ev = OnChainWhaleEvent(
            symbol           = symbol.replace("USDT", ""),
            chain            = ag["chain"],
            dex_name         = ag["dex"],
            total_bought_usd = ag["whale_vol"],
            whale_tx_count   = ag["whale_txs"],
            unique_whales    = max(1, ag["whale_txs"] // 2),
            avg_tx_usd       = ag["whale_vol"] / max(ag["whale_txs"], 1),
            largest_tx_usd   = ag["whale_vol"] * 0.4,
            time_window_hrs  = 1.0,
            source           = "dexscreener",
        )
        if ag["vol_ratio"] >= 3 and ag["whale_vol"] >= 200_000:
            ev.is_stealth_acc = True
        events.append(ev)

    return events


# ─────────────────────────────────────────────────────────
#  SCORE CALCULATOR
# ─────────────────────────────────────────────────────────

def _calculate_onchain_score(result: OnChainResult) -> float:
    """
    Hitung skor 0–30 dari aktivitas whale on-chain.

    Breakdown:
    - Total USD dibeli:     0–12 poin
    - Stealth accumulation: 0–10 poin
    - Wallet diversity:     0–5  poin
    - Multi-chain:          0–3  poin
    """
    if not result.found_onchain:
        return 0.0

    score = 0.0
    signals = []

    # ── Total USD bought ─────────────────────────────────
    usd = result.total_bought_usd
    if usd >= 10_000_000:         # > $10M
        score += 12
        signals.append(f"🔥 ON-CHAIN: ${usd/1e6:.1f}M dibeli whale di DEX dalam 4H!")
    elif usd >= 5_000_000:        # > $5M
        score += 10
        signals.append(f"🐋 ON-CHAIN: ${usd/1e6:.1f}M whale accumulation di DEX")
    elif usd >= 1_000_000:        # > $1M
        score += 7
        signals.append(f"🐋 ON-CHAIN: ${usd/1e6:.1f}M transaksi whale di DEX")
    elif usd >= 500_000:          # > $500K
        score += 4
        signals.append(f"🦈 ON-CHAIN: ${usd/1000:.0f}K whale buys di DEX")
    elif usd >= 100_000:          # > $100K
        score += 2
        signals.append(f"🦈 ON-CHAIN: ${usd/1000:.0f}K transaksi besar di DEX")

    # ── Stealth accumulation (wallet baru) ───────────────
    if result.is_stealth_acc:
        score += 10
        signals.append(
            f"🕵️ STEALTH ACCUMULATION: {result.new_wallet_count} wallet baru "
            f"melakukan akumulasi masif! Indikasi smart money masuk diam-diam"
        )
    elif result.new_wallet_count >= 2:
        score += 4
        signals.append(f"👀 {result.new_wallet_count} wallet baru aktif beli")

    # ── Unique whale wallets ─────────────────────────────
    if result.unique_whales >= 10:
        score += 5
        signals.append(f"🐋 {result.unique_whales} unique whale wallets aktif!")
    elif result.unique_whales >= 5:
        score += 3
        signals.append(f"🦈 {result.unique_whales} whale wallets beli secara terkoordinasi")
    elif result.unique_whales >= 2:
        score += 1

    # ── Multi-chain activity ─────────────────────────────
    if len(result.chains_active) >= 3:
        score += 3
        signals.append(f"⛓️ Aktivitas di {len(result.chains_active)} chain sekaligus: {', '.join(result.chains_active)}")
    elif len(result.chains_active) == 2:
        score += 1

    result.signals = signals

    # Alert level
    if score >= 20:
        result.alert_level = "CRITICAL"
    elif score >= 12:
        result.alert_level = "HIGH"
    elif score >= 5:
        result.alert_level = "WATCH"
    else:
        result.alert_level = "NONE"

    return min(score, 30.0)


# ─────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────

def analyze_onchain(symbol: str) -> OnChainResult:
    """
    Analisis on-chain whale activity untuk satu simbol.
    Prioritas: Flipside → Dune → DexScreener
    """
    result = OnChainResult(symbol=symbol)
    all_events: List[OnChainWhaleEvent] = []
    base = symbol.upper().replace("USDT", "").replace("USDC", "")

    # ── 1. Flipside Crypto (SQL on-chain) ─────────────────
    try:
        sql = _build_flipside_query(base, hours=4)
        rows = _run_flipside_query(sql)
        if rows is not None and len(rows) > 0:
            events = _parse_flipside_results(rows, symbol)
            all_events.extend(events)
            logger.info(f"OnChain Flipside | {symbol}: {len(events)} events, "
                        f"${sum(e.total_bought_usd for e in events):,.0f} total bought")
    except Exception as e:
        logger.debug(f"Flipside analyze {symbol}: {e}")

    # ── 2. Dune Analytics (jika Flipside kosong & DUNE_KEY ada) ──
    if not all_events and DUNE_KEY:
        try:
            rows = _run_dune_with_token_filter(symbol)
            if rows:
                # Dune rows sudah dalam format dict — parse sederhana
                for row in rows:
                    ev = OnChainWhaleEvent(
                        symbol           = base,
                        chain            = str(row.get("blockchain", "unknown")),
                        dex_name         = str(row.get("project", "DEX")),
                        total_bought_usd = float(row.get("total_usd", 0) or 0),
                        whale_tx_count   = int(row.get("tx_count", 0) or 0),
                        unique_whales    = int(row.get("unique_wallets", 0) or 0),
                        avg_tx_usd       = float(row.get("avg_usd", 0) or 0),
                        new_wallet_count = int(row.get("new_wallets", 0) or 0),
                        time_window_hrs  = 4.0,
                        source           = "dune",
                    )
                    if ev.total_bought_usd > 0:
                        all_events.append(ev)
                logger.info(f"OnChain Dune | {symbol}: {len(all_events)} events")
        except Exception as e:
            logger.debug(f"Dune analyze {symbol}: {e}")

    # ── 3. DexScreener (fallback gratis) ──────────────────
    if not all_events:
        try:
            pairs = _get_dexscreener_pairs(symbol)
            if pairs:
                events = _dexscreener_to_event(pairs, symbol)
                all_events.extend(events)
                if events:
                    logger.info(f"OnChain DexScreener | {symbol}: "
                                f"${sum(e.total_bought_usd for e in events):,.0f} est. whale vol")
        except Exception as e:
            logger.debug(f"DexScreener analyze {symbol}: {e}")

    # ── 4. Aggregate results ──────────────────────────────
    if not all_events:
        return result

    result.found_onchain   = True
    result.events          = all_events
    result.total_bought_usd = sum(e.total_bought_usd for e in all_events)
    result.whale_tx_count   = sum(e.whale_tx_count for e in all_events)
    result.new_wallet_count = sum(e.new_wallet_count for e in all_events)
    result.unique_whales    = sum(e.unique_whales for e in all_events)
    result.is_stealth_acc   = any(e.is_stealth_acc for e in all_events)
    result.chains_active    = list({e.chain for e in all_events if e.chain})

    # ── 5. Score ──────────────────────────────────────────
    result.score = _calculate_onchain_score(result)

    return result


# ─────────────────────────────────────────────────────────
#  PROACTIVE SCANNER — cari gem on-chain tanpa CEX signal
# ─────────────────────────────────────────────────────────

def _build_allchain_top_query(hours: int = 4) -> str:
    """
    Build SQL proactive scan — cari token terpanas di SEMUA chain sekaligus.
    Query ini tidak filter by symbol — ambil top 20 token berdasarkan
    total USD whale buys, agregasi dari semua chain yang didukung Flipside.

    Chain yang dicakup (13 total):
      EVM  : ethereum, arbitrum, bsc, polygon, optimism, avalanche,
              base, blast, mantle, scroll, gnosis, fantom
      Non-EVM: solana
    """

    # EVM chains dengan schema identik
    evm_chains = [
        "ethereum", "arbitrum", "bsc", "polygon", "optimism",
        "avalanche", "base", "blast", "mantle", "scroll", "gnosis", "fantom",
    ]

    # Stablecoin + wrapped tokens yang dikecualikan
    excludes = "('USDT','USDC','DAI','WETH','WBTC','BUSD','WBNB','WMATIC','WAVAX','WFTM','WXDAI','USDbC','USDB','MNT','WSCROLL')"

    evm_blocks = []
    for chain in evm_chains:
        block = f"""  SELECT
    token_out_symbol     AS token_symbol,
    token_out            AS token_address,
    '{chain}'            AS chain,
    platform             AS dex_name,
    tx_hash,
    origin_from_address  AS wallet,
    amount_out_usd       AS amount_usd,
    block_timestamp
  FROM {chain}.defi.ez_dex_swaps
  WHERE
    block_timestamp >= DATEADD(hour, -{hours}, CURRENT_TIMESTAMP)
    AND amount_out_usd >= {WHALE_TX_USD}
    AND token_out_symbol IS NOT NULL
    AND token_out_symbol NOT IN {excludes}"""
        evm_blocks.append(block)

    evm_union = "\n  UNION ALL\n".join(evm_blocks)

    # Solana dengan schema berbeda
    solana_block = f"""  SELECT
    token_out_symbol     AS token_symbol,
    token_out_mint       AS token_address,
    'solana'             AS chain,
    program_id           AS dex_name,
    tx_id                AS tx_hash,
    swapper              AS wallet,
    amount_out_usd       AS amount_usd,
    block_timestamp
  FROM solana.defi.ez_dex_swaps
  WHERE
    block_timestamp >= DATEADD(hour, -{hours}, CURRENT_TIMESTAMP)
    AND amount_out_usd >= {WHALE_TX_USD}
    AND token_out_symbol IS NOT NULL
    AND token_out_symbol NOT IN {excludes}"""

    query = f"""
WITH all_chains AS (
{evm_union}
  UNION ALL
{solana_block}
)
SELECT
  token_symbol,
  token_address,
  chain,
  dex_name,
  COUNT(DISTINCT tx_hash)   AS whale_tx_count,
  COUNT(DISTINCT wallet)    AS unique_whales,
  SUM(amount_usd)           AS total_bought_usd,
  AVG(amount_usd)           AS avg_tx_usd,
  MAX(amount_usd)           AS largest_tx_usd,
  0                         AS new_wallet_count
FROM all_chains
GROUP BY 1, 2, 3, 4
HAVING COUNT(DISTINCT tx_hash) >= {MIN_WHALE_TXS}
ORDER BY total_bought_usd DESC
LIMIT 25
"""
    return query.strip()


def scan_top_dex_whales(min_usd: float = 500_000, hours: int = 4) -> List[OnChainResult]:
    """
    Proactive scan: cari SEMUA token yang sedang diakumulasi whale
    di DEX di SELURUH chain dalam N jam terakhir.

    Chain yang dicakup (13 chain):
      EVM  : Ethereum · Arbitrum · BSC · Polygon · Optimism · Avalanche
              Base · Blast · Mantle · Scroll · Gnosis · Fantom
      Non-EVM: Solana

    Returns top 15 token berdasarkan total USD dibeli.
    Berguna untuk menemukan gem yang belum ada di radar CEX.
    """
    results = []

    # ── 1. Flipside all-chain SQL query ──────────────────
    try:
        sql  = _build_allchain_top_query(hours)
        rows = _run_flipside_query(sql)
        if rows:
            seen_symbols = set()
            for row in rows:
                sym = str(row.get("TOKEN_SYMBOL", "")).strip().upper()
                if not sym or len(sym) > 12 or sym in seen_symbols:
                    continue
                seen_symbols.add(sym)

                total_usd = float(row.get("TOTAL_BOUGHT_USD", 0) or 0)
                if total_usd < min_usd:
                    continue

                ev = OnChainWhaleEvent(
                    symbol           = sym,
                    token_address    = str(row.get("TOKEN_ADDRESS", "")),
                    chain            = str(row.get("CHAIN", "unknown")),
                    dex_name         = str(row.get("DEX_NAME", "DEX")),
                    whale_tx_count   = int(row.get("WHALE_TX_COUNT", 0) or 0),
                    unique_whales    = int(row.get("UNIQUE_WHALES", 0) or 0),
                    total_bought_usd = total_usd,
                    avg_tx_usd       = float(row.get("AVG_TX_USD", 0) or 0),
                    largest_tx_usd   = float(row.get("LARGEST_TX_USD", 0) or 0),
                    time_window_hrs  = float(hours),
                    source           = "flipside_allchain",
                )
                # Flag stealth: banyak tx besar dalam waktu singkat
                if ev.whale_tx_count >= MIN_WHALE_TXS * 2 and total_usd >= 1_000_000:
                    ev.is_stealth_acc = True

                res = OnChainResult(symbol=sym + "USDT")
                res.found_onchain    = True
                res.events           = [ev]
                res.total_bought_usd = total_usd
                res.whale_tx_count   = ev.whale_tx_count
                res.unique_whales    = ev.unique_whales
                res.is_stealth_acc   = ev.is_stealth_acc
                res.chains_active    = [ev.chain]
                res.score = _calculate_onchain_score(res)
                results.append(res)

            logger.info(f"Flipside all-chain scan: {len(results)} tokens found "
                        f"(13 chains × {hours}H window)")
    except Exception as e:
        logger.warning(f"Flipside all-chain top scan error: {e}")

    # ── 2. DexScreener boosted sebagai fallback ───────────
    # DexScreener /token-boosts mencakup semua chain yang terdaftar
    # termasuk: ETH, BSC, ARB, SOL, BASE, AVAX, MATIC, FTM, dll.
    if not results:
        try:
            boosted = _get_dexscreener_boosted()
            if boosted:
                for item in boosted[:15]:
                    sym   = str(item.get("tokenSymbol", "")).upper().strip()
                    chain = str(item.get("chainId", "unknown"))
                    total = float(item.get("totalAmount", 0) or 0)
                    if not sym or len(sym) > 12 or total < min_usd:
                        continue

                    ev = OnChainWhaleEvent(
                        symbol           = sym,
                        chain            = chain,
                        dex_name         = "dexscreener_trending",
                        total_bought_usd = total,
                        source           = "dexscreener_boost",
                    )
                    res = OnChainResult(symbol=sym + "USDT")
                    res.found_onchain    = True
                    res.events           = [ev]
                    res.total_bought_usd = total
                    res.chains_active    = [chain]
                    res.score = _calculate_onchain_score(res)
                    results.append(res)

            logger.info(f"DexScreener boost fallback: {len(results)} tokens")
        except Exception as e:
            logger.debug(f"DexScreener boost scan: {e}")

    results.sort(key=lambda x: x.total_bought_usd, reverse=True)
    return results[:15]


# ─────────────────────────────────────────────────────────
#  TELEGRAM FORMATTER
# ─────────────────────────────────────────────────────────

def format_onchain_section(result: OnChainResult) -> str:
    """Format on-chain data untuk Telegram message."""
    if not result.found_onchain or result.score < 2:
        return ""

    al_emoji = {"CRITICAL": "🚨", "HIGH": "🔥", "WATCH": "👀", "NONE": ""}.get(result.alert_level, "")
    lines = [f"\n⛓️ <b>On-Chain Whale Activity</b> {al_emoji}"]

    lines.append(f"  💰 Total DEX Buys (4H): <b>${result.total_bought_usd:,.0f}</b>")
    lines.append(f"  🐋 Whale Transactions:  <b>{result.whale_tx_count}</b>")
    lines.append(f"  👛 Unique Whale Wallets: <b>{result.unique_whales}</b>")

    if result.new_wallet_count > 0:
        lines.append(f"  🆕 New Wallets (<7d):   <b>{result.new_wallet_count}</b>")

    if result.chains_active:
        lines.append(f"  ⛓️ Active Chains: <b>{', '.join(result.chains_active)}</b>")

    if result.is_stealth_acc:
        lines.append(f"  🕵️ <b>STEALTH ACCUMULATION DETECTED!</b>")
        lines.append(f"     ↳ Wallet baru masuk masif → smart money diam-diam mengumpulkan")

    # Top events
    top_events = sorted(result.events, key=lambda e: e.total_bought_usd, reverse=True)[:2]
    for ev in top_events:
        src = ev.source.upper()
        lines.append(
            f"  📍 [{ev.chain.upper()}] {ev.dex_name}: "
            f"${ev.total_bought_usd:,.0f} via {ev.whale_tx_count} tx "
            f"(avg ${ev.avg_tx_usd:,.0f}) [{src}]"
        )

    lines.append(f"  📊 On-Chain Score: <b>{result.score:.0f}/30</b> | {result.alert_level}")
    lines.append("")
    return "\n".join(lines)


def format_standalone_onchain_alert(result: OnChainResult) -> str:
    """
    Format alert standalone untuk on-chain whale event yang terdeteksi
    tanpa ada CEX signal — proactive alert.
    """
    if not result.found_onchain:
        return ""

    al_emoji = {"CRITICAL": "🚨", "HIGH": "🔥", "WATCH": "👀"}.get(result.alert_level, "⚪")
    sym = result.symbol.replace("USDT", "")

    msg  = f"{al_emoji} <b>ON-CHAIN WHALE ALERT — ${sym}</b> {al_emoji}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 Total DEX Buys (4H): <b>${result.total_bought_usd:,.0f}</b>\n"
    msg += f"🐋 Whale Tx Count:      <b>{result.whale_tx_count}</b>\n"
    msg += f"👛 Unique Whales:       <b>{result.unique_whales}</b>\n"
    if result.new_wallet_count > 0:
        msg += f"🆕 New Wallets (<7d):   <b>{result.new_wallet_count}</b>\n"
    if result.chains_active:
        msg += f"⛓️ Chains: <b>{', '.join(result.chains_active)}</b>\n"
    if result.is_stealth_acc:
        msg += f"\n🕵️ <b>⚠️ STEALTH ACCUMULATION!</b>\n"
        msg += f"   Wallet-wallet baru masuk secara masif.\n"
        msg += f"   Ini ciri khas smart money sebelum pump besar.\n"
    msg += f"\n📊 On-Chain Score: <b>{result.score:.0f}/30</b>\n"
    for s in result.signals:
        msg += f"  • {s}\n"
    msg += f"\n⚡ CMI-ASS v7 | On-Chain Tracker | DYOR!"
    return msg
