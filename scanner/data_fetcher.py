import requests
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL_BINANCE = "https://api.binance.com"
BASE_URL_FUTURES = "https://fapi.binance.com"
BASE_URL_COINGECKO = "https://api.coingecko.com/api/v3"

def get_all_usdt_symbols() -> list:
    """Ambil semua pasangan USDT dari Binance Spot"""
    try:
        url = f"{BASE_URL_BINANCE}/api/v3/exchangeInfo"
        r = requests.get(url, timeout=15)
        data = r.json()
        symbols = [
            s['symbol'] for s in data['symbols']
            if s['quoteAsset'] == 'USDT'
            and s['status'] == 'TRADING'
            and s['isSpotTradingAllowed']
        ]
        logger.info(f"Total USDT pairs: {len(symbols)}")
        return symbols
    except Exception as e:
        logger.error(f"Error get symbols: {e}")
        return []

def get_ticker_24h(symbols: list = None) -> pd.DataFrame:
    """Ambil data ticker 24 jam untuk semua atau symbol tertentu"""
    try:
        url = f"{BASE_URL_BINANCE}/api/v3/ticker/24hr"
        r = requests.get(url, timeout=15)
        data = r.json()
        df = pd.DataFrame(data)
        
        # Filter hanya USDT pairs
        df = df[df['symbol'].str.endswith('USDT')]
        
        # Convert ke numeric
        numeric_cols = ['priceChange', 'priceChangePercent', 'lastPrice',
                       'volume', 'quoteVolume', 'highPrice', 'lowPrice',
                       'openPrice', 'count']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out stablecoins dan pair tidak relevan
        stables = ['BUSD', 'USDC', 'TUSD', 'USDP', 'DAI', 'FDUSD', 'USDS']
        for stable in stables:
            df = df[~df['symbol'].str.startswith(stable)]
        
        return df
    except Exception as e:
        logger.error(f"Error get ticker: {e}")
        return pd.DataFrame()

def get_klines(symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    """Ambil candlestick data"""
    try:
        url = f"{BASE_URL_BINANCE}/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error get klines {symbol}: {e}")
        return pd.DataFrame()

def get_futures_symbols() -> list:
    """Ambil semua symbol di Binance Futures"""
    try:
        url = f"{BASE_URL_FUTURES}/fapi/v1/exchangeInfo"
        r = requests.get(url, timeout=15)
        data = r.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
        return symbols
    except Exception as e:
        logger.error(f"Error get futures symbols: {e}")
        return []

def get_funding_rate(symbol: str) -> Optional[float]:
    """Ambil funding rate dari Binance Futures"""
    try:
        url = f"{BASE_URL_FUTURES}/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': 10}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if isinstance(data, list) and len(data) > 0:
            latest = float(data[-1]['fundingRate'])
            prev = float(data[-2]['fundingRate']) if len(data) > 1 else latest
            return latest, prev
        return None, None
    except:
        return None, None

def get_open_interest(symbol: str) -> Optional[dict]:
    """Ambil Open Interest dari Futures"""
    try:
        url = f"{BASE_URL_FUTURES}/fapi/v1/openInterest"
        params = {'symbol': symbol}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return float(data.get('openInterest', 0))
    except:
        return None

def get_open_interest_hist(symbol: str, period: str = '1h', limit: int = 10) -> pd.DataFrame:
    """Ambil historical Open Interest"""
    try:
        url = f"{BASE_URL_FUTURES}/futures/data/openInterestHist"
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'])
            df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'])
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def get_large_trades(symbol: str, limit: int = 50) -> pd.DataFrame:
    """Ambil large/aggregate trades untuk deteksi whale"""
    try:
        url = f"{BASE_URL_BINANCE}/api/v3/aggTrades"
        params = {'symbol': symbol, 'limit': limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if not isinstance(data, list):
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df['p'] = pd.to_numeric(df['p'])  # price
        df['q'] = pd.to_numeric(df['q'])  # quantity
        df['value'] = df['p'] * df['q']
        df['time'] = pd.to_datetime(df['T'], unit='ms')
        df['is_buyer'] = ~df['m']  # buyer is maker = sell pressure, buyer is taker = buy pressure
        return df
    except:
        return pd.DataFrame()

def get_coingecko_market_data(coin_ids: list) -> dict:
    """Ambil data dari CoinGecko (market cap, dll)"""
    try:
        ids_str = ','.join(coin_ids[:50])  # max 50 per request
        url = f"{BASE_URL_COINGECKO}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ids_str,
            'order': 'market_cap_asc',
            'per_page': 50,
            'page': 1,
            'sparkline': False
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        return {coin['symbol'].upper(): coin for coin in data}
    except Exception as e:
        logger.error(f"CoinGecko error: {e}")
        return {}

def get_order_book_imbalance(symbol: str, limit: int = 20) -> dict:
    """Cek ketidakseimbangan order book (indikasi whale)"""
    try:
        url = f"{BASE_URL_BINANCE}/api/v3/depth"
        params = {'symbol': symbol, 'limit': limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        bids = pd.DataFrame(data['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(data['asks'], columns=['price', 'qty'], dtype=float)
        
        bid_volume = (bids['price'] * bids['qty']).sum()
        ask_volume = (asks['price'] * asks['qty']).sum()
        
        total = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,  # positif = tekanan beli, negatif = tekanan jual
            'ratio': bid_volume / ask_volume if ask_volume > 0 else 0
        }
    except:
        return {}
