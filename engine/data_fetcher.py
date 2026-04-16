"""
Data Fetcher — mengambil semua data market dari API gratis.
Binance Spot, Binance Futures, CoinGecko, Fear & Greed Index.
"""
import requests
import pandas as pd
import logging
import time
from typing import Optional
from config import BINANCE_SPOT, BINANCE_FUTURES, COINGECKO, FEAR_GREED_API

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "CMI-ASS/1.0"}
TIMEOUT = 15

def _get(url: str, params: dict = None) -> Optional[dict | list]:
    """HTTP GET dengan error handling."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"GET {url}: {e}")
        return None

# ─────────────────────────────────────────────────────────
#  SPOT MARKET DATA
# ─────────────────────────────────────────────────────────

def get_all_tickers_24h() -> pd.DataFrame:
    """Ambil ticker 24H semua pasangan USDT dari Binance Spot."""
    data = _get(f"{BINANCE_SPOT}/api/v3/ticker/24hr")
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")].copy()

    # Buang stablecoin
    stables = ["BUSD", "USDC", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "EUR", "GBP"]
    mask = ~df["symbol"].str[:-4].isin(stables)
    df = df[mask]

    # Konversi numerik
    cols = ["lastPrice", "priceChangePercent", "volume", "quoteVolume",
            "highPrice", "lowPrice", "openPrice", "count"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)

def get_klines(symbol: str, interval: str = "1h", limit: int = 168) -> pd.DataFrame:
    """Ambil candlestick data (default 7 hari hourly)."""
    data = _get(f"{BINANCE_SPOT}/api/v3/klines",
                {"symbol": symbol, "interval": interval, "limit": limit})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def get_agg_trades(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Ambil aggregate trades untuk whale detection."""
    data = _get(f"{BINANCE_SPOT}/api/v3/aggTrades",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df["qty"]   = pd.to_numeric(df["q"], errors="coerce")
    df["value"] = df["price"] * df["qty"]
    df["time"]  = pd.to_datetime(df["T"], unit="ms")
    df["is_buyer_maker"] = df["m"]            # True = sell pressure (buyer is passive)
    df["is_aggressive_buy"] = ~df["m"]        # True = aggressive buy (buyer is taker)
    return df

def get_order_book(symbol: str, limit: int = 100) -> dict:
    """Ambil order book depth."""
    data = _get(f"{BINANCE_SPOT}/api/v3/depth", {"symbol": symbol, "limit": limit})
    if not data:
        return {}

    bids = pd.DataFrame(data["bids"], columns=["price","qty"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price","qty"], dtype=float)

    bid_val = (bids["price"] * bids["qty"]).sum()
    ask_val = (asks["price"] * asks["qty"]).sum()
    total   = bid_val + ask_val

    # Deteksi bid wall (whale limit order beli)
    bids["val"] = bids["price"] * bids["qty"]
    asks["val"] = asks["price"] * asks["qty"]
    max_bid_wall = bids["val"].max()
    max_ask_wall = asks["val"].max()

    return {
        "bid_value"    : bid_val,
        "ask_value"    : ask_val,
        "imbalance"    : (bid_val - ask_val) / total if total > 0 else 0,
        "bid_ask_ratio": bid_val / ask_val if ask_val > 0 else 1,
        "max_bid_wall" : max_bid_wall,
        "max_ask_wall" : max_ask_wall,
        "best_bid"     : float(data["bids"][0][0]) if data["bids"] else 0,
        "best_ask"     : float(data["asks"][0][0]) if data["asks"] else 0,
    }

# ─────────────────────────────────────────────────────────
#  FUTURES / DERIVATIVES DATA
# ─────────────────────────────────────────────────────────

def get_futures_symbols() -> set:
    """Ambil semua simbol aktif di Binance Futures."""
    data = _get(f"{BINANCE_FUTURES}/fapi/v1/exchangeInfo")
    if not data:
        return set()
    return {s["symbol"] for s in data.get("symbols", []) if s["status"] == "TRADING"}

def get_funding_rate_history(symbol: str, limit: int = 10) -> list:
    """Ambil histori funding rate."""
    data = _get(f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return []
    return [float(d["fundingRate"]) for d in data]

def get_open_interest_hist(symbol: str, period: str = "1h", limit: int = 24) -> pd.DataFrame:
    """Ambil historical Open Interest (nilai USD)."""
    data = _get(f"{BINANCE_FUTURES}/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["sumOpenInterest"]      = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def get_long_short_ratio(symbol: str, period: str = "1h", limit: int = 10) -> pd.DataFrame:
    """Ambil rasio long/short account dari Binance Futures."""
    data = _get(f"{BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": period, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    df["longAccount"]    = pd.to_numeric(df["longAccount"], errors="coerce")
    df["shortAccount"]   = pd.to_numeric(df["shortAccount"], errors="coerce")
    return df

def get_liquidations(symbol: str) -> dict:
    """Ambil data likuidasi 24H dari Binance Futures."""
    data = _get(f"{BINANCE_FUTURES}/fapi/v1/ticker/24hr", {"symbol": symbol})
    if not data:
        return {}
    return {
        "price"         : float(data.get("lastPrice", 0)),
        "volume"        : float(data.get("volume", 0)),
        "quote_volume"  : float(data.get("quoteVolume", 0)),
    }

# ─────────────────────────────────────────────────────────
#  ON-CHAIN & MARKET CAP DATA
# ─────────────────────────────────────────────────────────

def get_coingecko_data(symbols: list) -> dict:
    """
    Ambil data market cap, supply, FDV dari CoinGecko.
    Mengembalikan dict: {SYMBOL_UPPER: coin_data}
    """
    # CoinGecko butuh coin_id bukan symbol; kita query top coins dulu
    data = _get(f"{COINGECKO}/coins/markets", {
        "vs_currency" : "usd",
        "order"       : "volume_desc",
        "per_page"    : 250,
        "page"        : 1,
        "sparkline"   : "false",
    })
    if not isinstance(data, list):
        return {}

    result = {}
    symbols_upper = {s.upper().replace("USDT","") for s in symbols}
    for coin in data:
        sym = coin.get("symbol","").upper()
        if sym in symbols_upper:
            result[sym] = {
                "market_cap"         : coin.get("market_cap") or 0,
                "fully_diluted_val"  : coin.get("fully_diluted_valuation") or 0,
                "circulating_supply" : coin.get("circulating_supply") or 0,
                "max_supply"         : coin.get("max_supply") or 0,
                "total_supply"       : coin.get("total_supply") or 0,
                "volume_24h"         : coin.get("total_volume") or 0,
                "price"              : coin.get("current_price") or 0,
                "price_change_24h"   : coin.get("price_change_percentage_24h") or 0,
                "ath"                : coin.get("ath") or 0,
                "ath_change_pct"     : coin.get("ath_change_percentage") or 0,
            }
    return result

def get_fear_greed_index() -> dict:
    """Ambil Fear & Greed Index dari alternative.me."""
    data = _get(FEAR_GREED_API, {"limit": 1})
    if not data or not data.get("data"):
        return {"value": 50, "label": "Neutral"}
    d = data["data"][0]
    return {
        "value": int(d.get("value", 50)),
        "label": d.get("value_classification", "Neutral"),
    }
