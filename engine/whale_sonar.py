"""
Whale Sonar v7 — CEX + On-Chain + Volume Spike
=================================================
Menggabungkan:
  1. CEX whale detection (agg trades + order book) — sudah ada
  2. On-Chain DEX tracking (Flipside / Dune / DexScreener) — NEW
  3. CEX Volume Spike Detection (Bitget/MEXC multi-TF) — NEW
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from engine.data_fetcher import get_agg_trades, get_order_book, get_klines

logger = logging.getLogger(__name__)

@dataclass
class WhaleSonarResult:
    buy_pressure        : float = 0.5
    sell_pressure       : float = 0.5
    avg_whale_trade_usd : float = 0.0
    ob_imbalance        : float = 0.0
    bid_ask_ratio       : float = 1.0
    bid_wall_usd        : float = 0.0
    is_accumulating     : bool  = False
    is_distributing     : bool  = False
    aggressive_buy_pct  : float = 0.0
    taker_buy_ratio     : float = 0.5
    signals             : list  = field(default_factory=list)
    score               : float = 0.0   # 0–25 CEX whale score

    # v7 additions — diisi oleh main.py setelah analyze_whale()
    onchain_score       : float = 0.0   # 0–30
    volume_spike_score  : float = 0.0   # 0–25
    is_stealth_acc      : bool  = False  # on-chain stealth
    total_whale_score   : float = 0.0   # combined 0–80


def analyze_whale(symbol: str) -> WhaleSonarResult:
    """CEX whale detection (sama seperti v6, on-chain ditambahkan di main.py)."""
    res = WhaleSonarResult()

    # ── 1. Aggregate Trades ───────────────────────────────
    df_trades = get_agg_trades(symbol, limit=500)
    if not df_trades.empty:
        q90 = df_trades["value"].quantile(0.90)
        whale_df = df_trades[df_trades["value"] >= q90]
        if not whale_df.empty:
            whale_buy  = whale_df[whale_df["is_aggressive_buy"]]["value"].sum()
            whale_sell = whale_df[~whale_df["is_aggressive_buy"]]["value"].sum()
            total_w    = whale_buy + whale_sell
            res.buy_pressure  = whale_buy  / total_w if total_w > 0 else 0.5
            res.sell_pressure = whale_sell / total_w if total_w > 0 else 0.5
            res.avg_whale_trade_usd = whale_df["value"].mean()
            total_vol   = df_trades["value"].sum()
            agg_buy_vol = df_trades[df_trades["is_aggressive_buy"]]["value"].sum()
            res.aggressive_buy_pct = agg_buy_vol / total_vol if total_vol > 0 else 0.5

    # ── 2. Kline Taker Buy Ratio ──────────────────────────
    df_kline = get_klines(symbol, interval="1h", limit=24)
    if not df_kline.empty:
        total_buy = df_kline["taker_buy_base"].sum()
        total_vol = df_kline["volume"].sum()
        res.taker_buy_ratio = total_buy / total_vol if total_vol > 0 else 0.5
        last_3h_buy = df_kline["taker_buy_base"].iloc[-3:].sum()
        last_3h_vol = df_kline["volume"].iloc[-3:].sum()
        recent_ratio = last_3h_buy / last_3h_vol if last_3h_vol > 0 else 0.5
        prev_ratio   = (total_buy - last_3h_buy) / (total_vol - last_3h_vol + 1e-9)
        if recent_ratio > prev_ratio * 1.25:
            res.is_accumulating = True

    # ── 3. Order Book Imbalance ───────────────────────────
    ob = get_order_book(symbol, limit=100)
    if ob:
        res.ob_imbalance  = ob["imbalance"]
        res.bid_ask_ratio = ob["bid_ask_ratio"]
        res.bid_wall_usd  = ob["max_bid_wall"]
        if ob["imbalance"] > 0.25 and ob["bid_ask_ratio"] > 1.5:
            res.is_accumulating = True

    # ── 4. Distribusi detection ───────────────────────────
    if res.sell_pressure > 0.70 or res.ob_imbalance < -0.25:
        res.is_distributing = True

    # ── 5. Score + Signals ────────────────────────────────
    score = 0.0
    if res.buy_pressure >= 0.80:
        res.signals.append(f"🐋 WHALE BUY PRESSURE {res.buy_pressure*100:.0f}% (avg ${res.avg_whale_trade_usd:,.0f}/trade)")
        score += 25
    elif res.buy_pressure >= 0.70:
        res.signals.append(f"🐋 Whale buy dominan {res.buy_pressure*100:.0f}%")
        score += 17
    elif res.buy_pressure >= 0.60:
        res.signals.append(f"🦈 Tekanan beli moderate {res.buy_pressure*100:.0f}%")
        score += 10

    if res.taker_buy_ratio >= 0.65:
        res.signals.append(f"📈 Taker buy ratio {res.taker_buy_ratio*100:.0f}% (24H)")
        score += 5

    if res.ob_imbalance >= 0.30:
        res.signals.append(f"📚 BID WALL kuat! Rasio {res.bid_ask_ratio:.2f}x (${res.bid_wall_usd:,.0f} bid)")
        score += 7
    elif res.ob_imbalance >= 0.15:
        res.signals.append(f"📚 Order book condong buy {res.ob_imbalance*100:.0f}%")
        score += 3

    if res.is_accumulating:
        res.signals.append("✅ Pola AKUMULASI terdeteksi (taker buy meningkat)")
        score += 5

    if res.is_distributing:
        res.signals.append("⚠️ Pola DISTRIBUSI terdeteksi — waspada!")
        score -= 8

    if res.sell_pressure > 0.70:
        res.signals.append(f"🔴 WHALE SELL PRESSURE {res.sell_pressure*100:.0f}%!")
        score -= 10

    res.score = max(0.0, min(score, 25.0))
    return res
