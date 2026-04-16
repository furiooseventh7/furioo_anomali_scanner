"""
Data Fetcher — mengambil semua data market dari API gratis.
Binance Spot, Binance Futures, CoinGecko, Fear & Greed Index.

Catatan penting:
  Binance memblokir IP dari cloud datacenter (GitHub Actions = Azure) dengan HTTP 451.
  Solusi: _get_with_fallback() mencoba beberapa base URL secara berurutan.
  Jika SEMUA Binance URL gagal, get_all_tickers_24h() fallback ke CoinGecko
  sehingga scanner tetap berjalan.
"""
import requests
import pandas as pd
import logging
import time
from typing import Optional

from config import (
    BINANCE_SPOT_URLS, BINANCE_FUTURES_URLS,
    BINANCE_SPOT, BINANCE_FUTURES,
    COINGECKO, FEAR_GREED_API
)

# logger HARUS didefinisikan di sini, sebelum fungsi apapun
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "CMI-ASS/1.0"}
TIMEOUT = 15


# ─────────────────────────────────────────────────────────
#  HELPER: HTTP GET
# ─────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> Optional[dict | list]:
    """HTTP GET biasa — untuk endpoint non-Binance (CoinGecko, dll)."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"GET {url}: {e}")
        return None


def _get_with_fallback(path: str, base_urls: list, params: dict = None) -> Optional[dict | list]:
    """
    HTTP GET dengan fallback ke beberapa base URL.
    Dipakai untuk Binance endpoint yang bisa diblokir di cloud runner (HTTP 451).

    Args:
        path      : Bagian setelah domain, misal "/api/v3/ticker/24hr"
        base_urls : List base URL yang akan dicoba satu per satu
        params    : Query params (opsional)

    Returns:
        JSON response (dict/list) jika berhasil, None jika semua gagal.
    """
    last_error = None
    for base in base_urls:
        url = f"{base}{path}"
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            logger.debug(f"Berhasil dari {base}")
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            logger.warning(f"HTTP {status} dari {base}{path} — coba URL berikutnya")
            last_error = e
        except requests.exceptions.RequestException as e:
            logger.warning(f"Connection error dari {base}{path}: {e} — coba URL berikutnya")
            last_error = e

    logger.error(
        f"Semua {len(base_urls)} Binance endpoint gagal untuk path '{path}'. "
        f"Error terakhir: {last_error}"
    )
    return None


# ─────────────────────────────────────────────────────────
#  SPOT MARKET DATA
# ─────────────────────────────────────────────────────────

def _build_ticker_df_from_binance(data: list) -> pd.DataFrame:
    """Konversi raw Binance ticker JSON ke DataFrame standar."""
    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")].copy()

    stables = ["BUSD", "USDC", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "EUR", "GBP"]
    mask = ~df["symbol"].str[:-4].isin(stables)
    df = df[mask]

    cols = ["lastPrice", "priceChangePercent", "volume", "quoteVolume",
            "highPrice", "lowPrice", "openPrice", "count"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def _build_ticker_df_from_coingecko() -> pd.DataFrame:
    """
    FALLBACK: Bangun DataFrame ticker dari CoinGecko jika semua Binance URL gagal.
    Kolom yang dihasilkan disesuaikan dengan yang dipakai engine lain.
    """
    logger.warning("⚠️ Menggunakan CoinGecko sebagai fallback ticker (Binance tidak bisa diakses)")
    data = _get(f"{COINGECKO}/coins/markets", {
        "vs_currency" : "usd",
        "order"       : "volume_desc",
        "per_page"    : 250,
        "page"        : 1,
        "sparkline"   : "false",
    })
    if not isinstance(data, list) or not data:
        logger.error("CoinGecko fallback juga gagal — tidak ada data ticker sama sekali")
        return pd.DataFrame()

    rows = []
    for coin in data:
        sym = coin.get("symbol", "").upper() + "USDT"
        price = coin.get("current_price") or 0
        chg   = coin.get("price_change_percentage_24h") or 0
        vol   = coin.get("total_volume") or 0
        high  = coin.get("high_24h") or price
        low   = coin.get("low_24h") or price
        rows.append({
            "symbol"             : sym,
            "lastPrice"          : float(price),
            "priceChangePercent" : float(chg),
            "volume"             : float(vol / price) if price > 0 else 0,
            "quoteVolume"        : float(vol),
            "highPrice"          : float(high),
            "lowPrice"           : float(low),
            "openPrice"          : float(price / (1 + chg / 100)) if chg != -100 else 0,
            "count"              : 0,
        })

    df = pd.DataFrame(rows)
    stables = ["BUSD", "USDC", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "EUR", "GBP"]
    mask = ~df["symbol"].str[:-4].isin(stables)
    return df[mask].reset_index(drop=True)


def get_all_tickers_24h() -> pd.DataFrame:
    """
    Ambil ticker 24H semua pasangan USDT dari Binance Spot.
    Otomatis fallback ke CoinGecko jika semua Binance URL diblokir (HTTP 451).
    """
    data = _get_with_fallback("/api/v3/ticker/24hr", BINANCE_SPOT_URLS)
    if data:
        return _build_ticker_df_from_binance(data)

    # Binance tidak bisa diakses → pakai CoinGecko
    return _build_ticker_df_from_coingecko()


def get_klines(symbol: str, interval: str = "1h", limit: int = 168) -> pd.DataFrame:
    """Ambil candlestick data (default 7 hari hourly)."""
    data = _get_with_fallback(
        f"/api/v3/klines",
        BINANCE_SPOT_URLS,
        {"symbol": symbol, "interval": interval, "limit": limit}
    )
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
    data = _get_with_fallback(
        "/api/v3/aggTrades",
        BINANCE_SPOT_URLS,
        {"symbol": symbol, "limit": limit}
    )
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df["qty"]   = pd.to_numeric(df["q"], errors="coerce")
    df["value"] = df["price"] * df["qty"]
    df["time"]  = pd.to_datetime(df["T"], unit="ms")
    df["is_buyer_maker"]    = df["m"]
    df["is_aggressive_buy"] = ~df["m"]
    return df


def get_order_book(symbol: str, limit: int = 100) -> dict:
    """Ambil order book depth."""
    data = _get_with_fallback(
        "/api/v3/depth",
        BINANCE_SPOT_URLS,
        {"symbol": symbol, "limit": limit}
    )
    if not data:
        return {}

    bids = pd.DataFrame(data["bids"], columns=["price","qty"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price","qty"], dtype=float)

    bid_val = (bids["price"] * bids["qty"]).sum()
    ask_val = (asks["price"] * asks["qty"]).sum()
    total   = bid_val + ask_val

    bids["val"] = bids["price"] * bids["qty"]
    asks["val"] = asks["price"] * asks["qty"]

    return {
        "bid_value"    : bid_val,
        "ask_value"    : ask_val,
        "imbalance"    : (bid_val - ask_val) / total if total > 0 else 0,
        "bid_ask_ratio": bid_val / ask_val if ask_val > 0 else 1,
        "max_bid_wall" : bids["val"].max(),
        "max_ask_wall" : asks["val"].max(),
        "best_bid"     : float(data["bids"][0][0]) if data["bids"] else 0,
        "best_ask"     : float(data["asks"][0][0]) if data["asks"] else 0,
    }


# ─────────────────────────────────────────────────────────
#  FUTURES / DERIVATIVES DATA
# ─────────────────────────────────────────────────────────

def get_futures_symbols() -> set:
    """Ambil semua simbol aktif di Binance Futures."""
    data = _get_with_fallback("/fapi/v1/exchangeInfo", BINANCE_FUTURES_URLS)
    if not data:
        logger.warning("⚠️ Futures symbols tidak bisa diambil — scan tetap jalan tanpa data futures.")
        return set()
    return {s["symbol"] for s in data.get("symbols", []) if s["status"] == "TRADING"}


def get_funding_rate_history(symbol: str, limit: int = 10) -> list:
    """Ambil histori funding rate."""
    data = _get_with_fallback(
        "/fapi/v1/fundingRate",
        BINANCE_FUTURES_URLS,
        {"symbol": symbol, "limit": limit}
    )
    if not isinstance(data, list):
        return []
    return [float(d["fundingRate"]) for d in data]


def get_open_interest_hist(symbol: str, period: str = "1h", limit: int = 24) -> pd.DataFrame:
    """Ambil historical Open Interest (nilai USD)."""
    data = _get_with_fallback(
        "/futures/data/openInterestHist",
        BINANCE_FUTURES_URLS,
        {"symbol": symbol, "period": period, "limit": limit}
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["sumOpenInterest"]      = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_long_short_ratio(symbol: str, period: str = "1h", limit: int = 10) -> pd.DataFrame:
    """Ambil rasio long/short account dari Binance Futures."""
    data = _get_with_fallback(
        "/futures/data/globalLongShortAccountRatio",
        BINANCE_FUTURES_URLS,
        {"symbol": symbol, "period": period, "limit": limit}
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    df["longAccount"]    = pd.to_numeric(df["longAccount"], errors="coerce")
    df["shortAccount"]   = pd.to_numeric(df["shortAccount"], errors="coerce")
    return df


def get_liquidations(symbol: str) -> dict:
    """Ambil data likuidasi 24H dari Binance Futures."""
    data = _get_with_fallback(
        "/fapi/v1/ticker/24hr",
        BINANCE_FUTURES_URLS,
        {"symbol": symbol}
    )
    if not data:
        return {}
    return {
        "price"       : float(data.get("lastPrice", 0)),
        "volume"      : float(data.get("volume", 0)),
        "quote_volume": float(data.get("quoteVolume", 0)),
    }


# ─────────────────────────────────────────────────────────
#  ON-CHAIN & MARKET CAP DATA
# ─────────────────────────────────────────────────────────

def get_coingecko_data(symbols: list) -> dict:
    """
    Ambil data market cap, supply, FDV dari CoinGecko.
    Mengembalikan dict: {SYMBOL_UPPER: coin_data}
    """
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
