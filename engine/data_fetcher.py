"""
Data Fetcher — mengambil semua data market dari API publik.

Arsitektur sumber data:
  PRIMARY   → Bitget  (spot & futures, tidak blokir cloud/GitHub Actions)
  SECONDARY → MEXC    (spot & futures, tidak blokir cloud/GitHub Actions)
  TERTIARY  → CoinGecko (fallback ticker saja, rate-limit 30 req/mnt)

Kenapa TIDAK pakai Binance:
  Binance memblokir IP dari cloud datacenter (GitHub Actions = Azure) dengan HTTP 451.
  Bitget dan MEXC menyediakan public REST API yang identik strukturnya dan
  dapat diakses bebas dari mana pun termasuk GitHub Actions runner.
"""
import requests
import pandas as pd
import logging
import time
from typing import Optional

from config import (
    BITGET_SPOT_BASE, BITGET_FUTURES_BASE,
    MEXC_SPOT_BASE, MEXC_FUTURES_BASE,
    COINGECKO, FEAR_GREED_API,
)

logger  = logging.getLogger(__name__)
HEADERS = {"User-Agent": "CMI-ASS/1.0", "Content-Type": "application/json"}
TIMEOUT = 15


# ─────────────────────────────────────────────────────────
#  HELPER: HTTP GET
# ─────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> Optional[dict | list]:
    """HTTP GET biasa — untuk endpoint non-exchange (CoinGecko, Fear&Greed)."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"GET {url}: {e}")
        return None


def _get_bitget(path: str, params: dict = None) -> Optional[dict | list]:
    """
    GET ke Bitget REST API.
    Bitget membungkus semua respons dalam:  {"code":"00000","msg":"success","data":...}
    Fungsi ini mengekstrak `.data` secara otomatis.
    """
    url = f"{BITGET_SPOT_BASE}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if body.get("code") == "00000":
            return body.get("data")
        logger.warning(f"Bitget API error [{body.get('code')}] {path}: {body.get('msg')}")
        return None
    except Exception as e:
        logger.warning(f"Bitget GET {path}: {e}")
        return None


def _get_bitget_futures(path: str, params: dict = None) -> Optional[dict | list]:
    """GET ke Bitget Futures (mix endpoint)."""
    url = f"{BITGET_FUTURES_BASE}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if body.get("code") == "00000":
            return body.get("data")
        logger.warning(f"Bitget Futures error [{body.get('code')}] {path}: {body.get('msg')}")
        return None
    except Exception as e:
        logger.warning(f"Bitget Futures GET {path}: {e}")
        return None


def _get_mexc(path: str, params: dict = None) -> Optional[dict | list]:
    """
    GET ke MEXC REST API (Binance-compatible interface).
    MEXC spot API kompatibel dengan Binance spot API — path dan response format sama.
    """
    url = f"{MEXC_SPOT_BASE}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"MEXC GET {path}: {e}")
        return None


def _get_mexc_futures(path: str, params: dict = None) -> Optional[dict | list]:
    """GET ke MEXC Futures REST API."""
    url = f"{MEXC_FUTURES_BASE}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        # MEXC futures kadang wrap dalam {"success":true,"data":...}
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body
    except Exception as e:
        logger.warning(f"MEXC Futures GET {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────
#  SPOT MARKET DATA
# ─────────────────────────────────────────────────────────

def _build_ticker_df_from_bitget(data: list) -> pd.DataFrame:
    """Konversi raw Bitget ticker list ke DataFrame standar."""
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        sym = item.get("symbol", "")          # format Bitget: "BTCUSDT"
        if not sym.endswith("USDT"):
            continue
        rows.append({
            "symbol"             : sym,
            "lastPrice"          : float(item.get("lastPr", 0) or 0),
            "priceChangePercent" : float(item.get("change24h", 0) or 0) * 100,
            "volume"             : float(item.get("baseVolume", 0) or 0),
            "quoteVolume"        : float(item.get("quoteVolume", 0) or 0),
            "highPrice"          : float(item.get("high24h", 0) or 0),
            "lowPrice"           : float(item.get("low24h", 0) or 0),
            "openPrice"          : float(item.get("openUtc0", 0) or 0),
            "count"              : int(item.get("tradeCount", 0) or 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    stables = ["BUSD", "USDC", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "EUR", "GBP"]
    mask = ~df["symbol"].str[:-4].isin(stables)
    return df[mask].reset_index(drop=True)


def _build_ticker_df_from_mexc(data: list) -> pd.DataFrame:
    """
    Konversi raw MEXC ticker list ke DataFrame standar.
    MEXC spot /api/v3/ticker/24hr kompatibel dengan Binance — field sama.
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")].copy()

    stables = ["BUSD", "USDC", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "EUR", "GBP"]
    mask = ~df["symbol"].str[:-4].isin(stables)
    df = df[mask].copy()

    cols = ["lastPrice", "priceChangePercent", "volume", "quoteVolume",
            "highPrice", "lowPrice", "openPrice"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "count" not in df.columns:
        df["count"] = 0

    return df.reset_index(drop=True)


def _build_ticker_df_from_coingecko() -> pd.DataFrame:
    """
    TERTIARY FALLBACK: Bangun DataFrame ticker dari CoinGecko jika Bitget & MEXC gagal.
    """
    logger.warning("⚠️ Menggunakan CoinGecko sebagai fallback ticker (Bitget & MEXC tidak bisa diakses)")
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
        sym   = coin.get("symbol", "").upper() + "USDT"
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
    Ambil ticker 24H semua pasangan USDT.
    Urutan: Bitget → MEXC → CoinGecko
    """
    # ── 1. Coba Bitget ─────────────────────────────────────
    data = _get_bitget("/api/v2/spot/market/tickers")
    if data:
        df = _build_ticker_df_from_bitget(data)
        if not df.empty:
            logger.info(f"✅ Ticker dari Bitget: {len(df)} pasangan USDT")
            return df

    # ── 2. Coba MEXC ───────────────────────────────────────
    logger.warning("Bitget ticker gagal → mencoba MEXC")
    data = _get_mexc("/api/v3/ticker/24hr")
    if data:
        df = _build_ticker_df_from_mexc(data)
        if not df.empty:
            logger.info(f"✅ Ticker dari MEXC: {len(df)} pasangan USDT")
            return df

    # ── 3. CoinGecko fallback ──────────────────────────────
    logger.warning("MEXC ticker gagal → mencoba CoinGecko")
    return _build_ticker_df_from_coingecko()


def get_klines(symbol: str, interval: str = "1h", limit: int = 168) -> pd.DataFrame:
    """
    Ambil candlestick data (default 7 hari hourly).
    Urutan: Bitget → MEXC
    """
    # ── Bitget ─────────────────────────────────────────────
    # Bitget interval: "1H" (uppercase) | granularity mapping
    interval_map = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1H", "4h": "4H", "1d": "1Dutc",
    }
    bg_interval = interval_map.get(interval, "1H")
    data = _get_bitget("/api/v2/spot/market/candles", {
        "symbol"      : symbol,
        "granularity" : bg_interval,
        "limit"       : limit,
    })
    if data and isinstance(data, list):
        # Bitget candle: [timestamp, open, high, low, close, volume, quoteVolume]
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", "quote_volume",
        ])
        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"]    = pd.to_datetime(df["open_time"].astype(float), unit="ms")
        df["taker_buy_base"]  = 0.0
        df["taker_buy_quote"] = 0.0
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    # ── MEXC fallback (Binance-compatible) ─────────────────
    logger.warning(f"Bitget klines gagal untuk {symbol} → mencoba MEXC")
    data = _get_mexc("/api/v3/klines", {
        "symbol"   : symbol,
        "interval" : interval,
        "limit"    : limit,
    })
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def get_agg_trades(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Ambil aggregate trades untuk whale detection.
    Urutan: Bitget → MEXC
    """
    # ── Bitget fills ───────────────────────────────────────
    data = _get_bitget("/api/v2/spot/market/fills", {
        "symbol": symbol,
        "limit" : limit,
    })
    if data and isinstance(data, list):
        rows = []
        for t in data:
            price = float(t.get("price", 0) or 0)
            qty   = float(t.get("size", 0) or 0)
            side  = t.get("side", "buy")
            rows.append({
                "price"           : price,
                "qty"             : qty,
                "value"           : price * qty,
                "time"            : pd.to_datetime(t.get("ts", 0), unit="ms"),
                "is_buyer_maker"  : (side == "sell"),
                "is_aggressive_buy": (side == "buy"),
            })
        return pd.DataFrame(rows)

    # ── MEXC aggTrades (Binance-compatible) ────────────────
    logger.warning(f"Bitget fills gagal untuk {symbol} → mencoba MEXC aggTrades")
    data = _get_mexc("/api/v3/aggTrades", {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["price"] = pd.to_numeric(df["p"], errors="coerce")
    df["qty"]   = pd.to_numeric(df["q"], errors="coerce")
    df["value"] = df["price"] * df["qty"]
    df["time"]  = pd.to_datetime(df["T"], unit="ms")
    df["is_buyer_maker"]     = df["m"]
    df["is_aggressive_buy"]  = ~df["m"]
    return df


def get_order_book(symbol: str, limit: int = 100) -> dict:
    """
    Ambil order book depth.
    Urutan: Bitget → MEXC
    """
    # ── Bitget ─────────────────────────────────────────────
    data = _get_bitget("/api/v2/spot/market/orderbook", {
        "symbol": symbol,
        "limit" : limit,
    })
    if data and "bids" in data and "asks" in data:
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
    else:
        # ── MEXC fallback ──────────────────────────────────
        logger.warning(f"Bitget orderbook gagal untuk {symbol} → mencoba MEXC")
        data = _get_mexc("/api/v3/depth", {"symbol": symbol, "limit": limit})
        if not data:
            return {}
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)

    bid_val = (bids["price"] * bids["qty"]).sum()
    ask_val = (asks["price"] * asks["qty"]).sum()
    total   = bid_val + ask_val

    bids["val"] = bids["price"] * bids["qty"]
    asks["val"] = asks["price"] * asks["qty"]

    best_bid = float(bids["price"].max()) if not bids.empty else 0
    best_ask = float(asks["price"].min()) if not asks.empty else 0

    return {
        "bid_value"     : bid_val,
        "ask_value"     : ask_val,
        "imbalance"     : (bid_val - ask_val) / total if total > 0 else 0,
        "bid_ask_ratio" : bid_val / ask_val if ask_val > 0 else 1,
        "max_bid_wall"  : bids["val"].max(),
        "max_ask_wall"  : asks["val"].max(),
        "best_bid"      : best_bid,
        "best_ask"      : best_ask,
    }


# ─────────────────────────────────────────────────────────
#  FUTURES / DERIVATIVES DATA
# ─────────────────────────────────────────────────────────

def _to_bitget_futures_symbol(symbol: str) -> str:
    """Konversi 'BTCUSDT' → 'BTCUSDT' (Bitget USDT-M perpetual format sama)."""
    return symbol


def get_futures_symbols() -> set:
    """
    Ambil semua simbol aktif di pasar futures.
    Urutan: Bitget → MEXC
    """
    # ── Bitget mix perpetual ────────────────────────────────
    data = _get_bitget_futures("/api/v2/mix/market/tickers", {"productType": "USDT-FUTURES"})
    if data and isinstance(data, list):
        symbols = {
            item["symbol"].replace("_UMCBL", "").replace("-USDT", "USDT")
            for item in data
            if item.get("symbol")
        }
        logger.info(f"✅ Futures symbols dari Bitget: {len(symbols)}")
        return symbols

    # ── MEXC futures ───────────────────────────────────────
    logger.warning("Bitget futures symbols gagal → mencoba MEXC")
    data = _get_mexc_futures("/api/v1/contract/detail")
    if data and isinstance(data, list):
        symbols = {
            item["symbol"].replace("_USDT", "USDT")
            for item in data
            if item.get("symbol", "").endswith("_USDT") and item.get("state") == 1
        }
        logger.info(f"✅ Futures symbols dari MEXC: {len(symbols)}")
        return symbols

    logger.warning("⚠️ Futures symbols tidak bisa diambil — scan tetap jalan tanpa data futures.")
    return set()


def get_funding_rate_history(symbol: str, limit: int = 10) -> list:
    """
    Ambil histori funding rate.
    Urutan: Bitget → MEXC
    """
    # ── Bitget ─────────────────────────────────────────────
    data = _get_bitget_futures("/api/v2/mix/market/history-fund-rate", {
        "symbol"      : symbol,
        "productType" : "USDT-FUTURES",
        "pageSize"    : limit,
    })
    if data and isinstance(data, list):
        rates = []
        for d in data:
            try:
                rates.append(float(d.get("fundingRate", 0)))
            except (ValueError, TypeError):
                pass
        if rates:
            return rates[-limit:]  # urutan terlama → terbaru

    # ── MEXC fallback ──────────────────────────────────────
    logger.warning(f"Bitget funding rate gagal untuk {symbol} → mencoba MEXC")
    mx_sym = symbol.replace("USDT", "_USDT")
    data = _get_mexc_futures(f"/api/v1/contract/funding_rate/{mx_sym}")
    if isinstance(data, dict):
        rate = data.get("fundingRate")
        if rate is not None:
            return [float(rate)] * min(limit, 3)

    return []


def get_open_interest_hist(symbol: str, period: str = "1h", limit: int = 24) -> pd.DataFrame:
    """
    Ambil historical Open Interest (nilai USD).
    Urutan: Bitget → MEXC
    """
    # ── Bitget ─────────────────────────────────────────────
    # Bitget: period map "1h" → "1H"
    period_map = {"1h": "1H", "4h": "4H", "1d": "1D"}
    bg_period = period_map.get(period, "1H")
    data = _get_bitget_futures("/api/v2/mix/market/open-interest", {
        "symbol"      : symbol,
        "productType" : "USDT-FUTURES",
        "period"      : bg_period,
        "limit"       : limit,
    })
    if data and isinstance(data, list):
        rows = []
        for d in data:
            try:
                rows.append({
                    "sumOpenInterestValue": float(d.get("openInterestValue", 0) or 0),
                    "sumOpenInterest"     : float(d.get("openInterest", 0) or 0),
                    "timestamp"           : pd.to_datetime(int(d.get("ts", 0)), unit="ms"),
                })
            except (ValueError, TypeError):
                pass
        if rows:
            return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # ── MEXC fallback ──────────────────────────────────────
    logger.warning(f"Bitget OI hist gagal untuk {symbol} → mencoba MEXC")
    mx_sym = symbol.replace("USDT", "_USDT")
    data = _get_mexc_futures(f"/api/v1/contract/kline/openInterest/{mx_sym}", {
        "interval" : period,
        "limit"    : limit,
    })
    if isinstance(data, dict) and "dataList" in data:
        rows = []
        for d in data["dataList"]:
            try:
                oi_val = float(d.get("openInterest", 0) or 0)
                rows.append({
                    "sumOpenInterestValue": oi_val,
                    "sumOpenInterest"     : oi_val,
                    "timestamp"           : pd.to_datetime(int(d.get("time", 0)), unit="s"),
                })
            except (ValueError, TypeError):
                pass
        if rows:
            return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    return pd.DataFrame()


def get_long_short_ratio(symbol: str, period: str = "1h", limit: int = 10) -> pd.DataFrame:
    """
    Ambil rasio long/short account.
    Urutan: Bitget → MEXC
    """
    # ── Bitget ─────────────────────────────────────────────
    period_map = {"1h": "1H", "4h": "4H", "1d": "1D"}
    bg_period = period_map.get(period, "1H")
    data = _get_bitget_futures("/api/v2/mix/market/long-short-ratio", {
        "symbol"      : symbol,
        "productType" : "USDT-FUTURES",
        "period"      : bg_period,
        "limit"       : limit,
    })
    if data and isinstance(data, list):
        rows = []
        for d in data:
            try:
                long_pct  = float(d.get("longAccountRatio", 0.5) or 0.5)
                short_pct = 1.0 - long_pct
                rows.append({
                    "longShortRatio": long_pct / short_pct if short_pct > 0 else 1.0,
                    "longAccount"   : long_pct,
                    "shortAccount"  : short_pct,
                })
            except (ValueError, TypeError):
                pass
        if rows:
            return pd.DataFrame(rows)

    # ── MEXC fallback ──────────────────────────────────────
    logger.warning(f"Bitget L/S ratio gagal untuk {symbol} → mencoba MEXC")
    mx_sym = symbol.replace("USDT", "_USDT")
    data = _get_mexc_futures(f"/api/v1/contract/long_short/{mx_sym}")
    if isinstance(data, dict):
        try:
            long_pct  = float(data.get("longRatio", 0.5) or 0.5)
            short_pct = float(data.get("shortRatio", 0.5) or 0.5)
            ratio     = long_pct / short_pct if short_pct > 0 else 1.0
            return pd.DataFrame([{
                "longShortRatio": ratio,
                "longAccount"   : long_pct,
                "shortAccount"  : short_pct,
            }])
        except (ValueError, TypeError):
            pass

    return pd.DataFrame()


def get_liquidations(symbol: str) -> dict:
    """
    Ambil data ticker futures 24H (proxy untuk volume & harga).
    Urutan: Bitget → MEXC
    """
    # ── Bitget ─────────────────────────────────────────────
    data = _get_bitget_futures("/api/v2/mix/market/ticker", {
        "symbol"      : symbol,
        "productType" : "USDT-FUTURES",
    })
    if data:
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            return {
                "price"        : float(data.get("lastPr", 0) or 0),
                "volume"       : float(data.get("baseVolume", 0) or 0),
                "quote_volume" : float(data.get("quoteVolume", 0) or 0),
            }

    # ── MEXC fallback ──────────────────────────────────────
    logger.warning(f"Bitget futures ticker gagal untuk {symbol} → mencoba MEXC")
    mx_sym = symbol.replace("USDT", "_USDT")
    data = _get_mexc_futures(f"/api/v1/contract/ticker?symbol={mx_sym}")
    if isinstance(data, dict):
        return {
            "price"        : float(data.get("lastPrice", 0) or 0),
            "volume"       : float(data.get("volume24", 0) or 0),
            "quote_volume" : float(data.get("amount24", 0) or 0),
        }

    return {}


# ─────────────────────────────────────────────────────────
#  ON-CHAIN & MARKET CAP DATA  (tetap CoinGecko)
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
    symbols_upper = {s.upper().replace("USDT", "") for s in symbols}
    for coin in data:
        sym = coin.get("symbol", "").upper()
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
