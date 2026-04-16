import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional
from scanner.data_fetcher import (
    get_klines, get_funding_rate, get_open_interest_hist,
    get_large_trades, get_order_book_imbalance, get_futures_symbols
)
import time

logger = logging.getLogger(__name__)

@dataclass
class AnomalySignal:
    symbol: str
    score: float           # Total score 0-100
    signals: List[str]     # Daftar sinyal yang terdeteksi
    price: float
    price_change_24h: float
    volume_24h: float
    volume_spike: float    # Rasio volume vs rata-rata
    funding_rate: Optional[float]
    oi_change: Optional[float]
    whale_buy_pressure: Optional[float]
    ob_imbalance: Optional[float]
    market_cap: Optional[float]
    alert_level: str       # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

def calculate_volume_spike(symbol: str, current_volume: float) -> tuple:
    """Hitung volume spike vs rata-rata 7 hari"""
    try:
        df = get_klines(symbol, interval='1h', limit=168)  # 7 hari
        if df.empty or len(df) < 24:
            return 1.0, []
        
        # Rata-rata volume per jam (7 hari terakhir, exclude jam terakhir)
        avg_volume = df['volume'].iloc[:-1].mean()
        std_volume = df['volume'].iloc[:-1].std()
        
        # Volume jam terakhir
        last_volume = df['volume'].iloc[-1]
        
        # Volume spike ratio
        spike_ratio = last_volume / avg_volume if avg_volume > 0 else 1
        
        # Z-score
        z_score = (last_volume - avg_volume) / std_volume if std_volume > 0 else 0
        
        # Cek konsistensi - apakah 3 jam terakhir volume naik terus
        last_3h = df['volume'].iloc[-4:-1].mean()
        trend_ratio = last_volume / last_3h if last_3h > 0 else 1
        
        signals = []
        if spike_ratio >= 5:
            signals.append(f"🔥 VOLUME SPIKE {spike_ratio:.1f}x vs avg 7D")
        elif spike_ratio >= 3:
            signals.append(f"⚡ Volume naik {spike_ratio:.1f}x vs avg 7D")
        
        if z_score > 3:
            signals.append(f"📊 Z-Score volume: {z_score:.1f} (sangat abnormal)")
        
        return spike_ratio, signals
    except Exception as e:
        logger.error(f"Volume spike error {symbol}: {e}")
        return 1.0, []

def analyze_price_consolidation(symbol: str) -> tuple:
    """Deteksi konsolidasi harga sebelum breakout (seperti $RAVE)"""
    try:
        df = get_klines(symbol, interval='4h', limit=30)
        if df.empty or len(df) < 10:
            return False, []
        
        # Cek 7 candle terakhir untuk konsolidasi
        recent = df.iloc[-8:-1]
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        
        # Candle terakhir (potensi breakout)
        last_candle = df.iloc[-1]
        prev_high = recent['high'].max()
        
        signals = []
        is_consolidating = price_range < 0.08  # Range < 8% = konsolidasi ketat
        
        if is_consolidating:
            signals.append(f"📦 Konsolidasi ketat {price_range*100:.1f}% selama 7 candle 4H")
        
        # Breakout dari konsolidasi
        if last_candle['close'] > prev_high * 1.02:
            signals.append(f"🚀 BREAKOUT dari konsolidasi! +{((last_candle['close']/prev_high)-1)*100:.1f}%")
        
        # Cek apakah volume konfirmasi breakout
        avg_vol = df['volume'].iloc[-10:-1].mean()
        if last_candle['volume'] > avg_vol * 2 and last_candle['close'] > prev_high:
            signals.append("✅ Volume konfirmasi breakout!")
        
        return is_consolidating, signals
    except Exception as e:
        logger.error(f"Consolidation error {symbol}: {e}")
        return False, []

def analyze_funding_rate(symbol: str, futures_symbols: list) -> tuple:
    """Analisis funding rate untuk deteksi reversal atau squeeze"""
    try:
        # Cek apakah ada di futures
        futures_symbol = symbol.replace('USDT', '') + 'USDT'
        if futures_symbol not in futures_symbols:
            return None, None, []
        
        latest_fr, prev_fr = get_funding_rate(futures_symbol)
        if latest_fr is None:
            return None, None, []
        
        signals = []
        
        # Funding rate sangat negatif = short squeeze potential
        if latest_fr < -0.001:
            signals.append(f"💥 Funding Rate NEGATIF {latest_fr*100:.3f}% → Potensi SHORT SQUEEZE")
        elif latest_fr < -0.0005:
            signals.append(f"⚠️ Funding Rate negatif {latest_fr*100:.3f}%")
        
        # Funding rate balik dari negatif ke positif (seperti pre-pump RAVE)
        if prev_fr is not None and prev_fr < 0 and latest_fr > 0:
            signals.append(f"🔄 Funding Rate REVERSAL: {prev_fr*100:.3f}% → {latest_fr*100:.3f}% (BULLISH)")
        
        # Funding rate sangat tinggi = overbought warning
        if latest_fr > 0.003:
            signals.append(f"⚠️ Funding Rate sangat tinggi {latest_fr*100:.3f}% (Overbought)")
        
        return latest_fr, prev_fr, signals
    except Exception as e:
        logger.error(f"Funding rate error {symbol}: {e}")
        return None, None, []

def analyze_open_interest(symbol: str, futures_symbols: list) -> tuple:
    """Analisis Open Interest - whale accumulation"""
    try:
        if symbol not in futures_symbols:
            return None, []
        
        df_oi = get_open_interest_hist(symbol, period='1h', limit=24)
        if df_oi.empty:
            return None, []
        
        oi_latest = df_oi['sumOpenInterestValue'].iloc[-1]
        oi_24h_ago = df_oi['sumOpenInterestValue'].iloc[0]
        
        oi_change = (oi_latest - oi_24h_ago) / oi_24h_ago * 100 if oi_24h_ago > 0 else 0
        
        signals = []
        if oi_change > 50:
            signals.append(f"🐋 OI naik {oi_change:.1f}% dalam 24H (Whale MASUK!)")
        elif oi_change > 25:
            signals.append(f"📈 OI naik {oi_change:.1f}% dalam 24H")
        elif oi_change < -30:
            signals.append(f"📉 OI turun {oi_change:.1f}% (Mass liquidation)")
        
        return oi_change, signals
    except Exception as e:
        logger.error(f"OI error {symbol}: {e}")
        return None, []

def analyze_whale_trades(symbol: str) -> tuple:
    """Deteksi whale buy pressure dari large trades"""
    try:
        df = get_large_trades(symbol, limit=500)
        if df.empty:
            return None, []
        
        # Filter hanya trade besar (top 10% by value)
        threshold = df['value'].quantile(0.90)
        large_trades = df[df['value'] >= threshold]
        
        if large_trades.empty:
            return None, []
        
        # Hitung buy vs sell pressure dari whale
        whale_buy = large_trades[large_trades['is_buyer']]['value'].sum()
        whale_sell = large_trades[~large_trades['is_buyer']]['value'].sum()
        total_whale = whale_buy + whale_sell
        
        buy_pressure = whale_buy / total_whale if total_whale > 0 else 0.5
        
        # Nilai rata-rata transaksi whale
        avg_whale_size = large_trades['value'].mean()
        
        signals = []
        if buy_pressure > 0.75:
            signals.append(f"🐋 WHALE BUY PRESSURE {buy_pressure*100:.0f}%! (Avg ${avg_whale_size:,.0f}/trade)")
        elif buy_pressure > 0.65:
            signals.append(f"🐋 Whale buy dominan {buy_pressure*100:.0f}%")
        elif buy_pressure < 0.35:
            signals.append(f"⚠️ Whale SELL PRESSURE {(1-buy_pressure)*100:.0f}%!")
        
        return buy_pressure, signals
    except Exception as e:
        logger.error(f"Whale trades error {symbol}: {e}")
        return None, []

def analyze_order_book(symbol: str) -> tuple:
    """Analisis order book untuk deteksi dinding beli besar"""
    try:
        ob = get_order_book_imbalance(symbol, limit=50)
        if not ob:
            return None, []
        
        imbalance = ob.get('imbalance', 0)
        ratio = ob.get('ratio', 1)
        
        signals = []
        if imbalance > 0.3:
            signals.append(f"📚 Order Book: BID WALL kuat! Ratio {ratio:.2f}x")
        elif imbalance > 0.2:
            signals.append(f"📚 Order Book: Tekanan beli {imbalance*100:.0f}%")
        elif imbalance < -0.3:
            signals.append(f"📚 Order Book: ASK WALL besar (tekanan jual)")
        
        return imbalance, signals
    except Exception as e:
        logger.error(f"Order book error {symbol}: {e}")
        return None, []

def calculate_rave_similarity_score(
    price_change_24h: float,
    volume_spike: float,
    funding_rate: Optional[float],
    oi_change: Optional[float],
    whale_buy_pressure: Optional[float],
    ob_imbalance: Optional[float],
    is_consolidating: bool,
    market_cap: Optional[float]
) -> float:
    """
    Hitung score mirip kondisi $RAVE sebelum pump
    Score 0-100, makin tinggi makin mirip kondisi pre-pump
    """
    score = 0
    
    # 1. Volume Spike (30 poin max) - faktor terpenting
    if volume_spike >= 10:
        score += 30
    elif volume_spike >= 5:
        score += 22
    elif volume_spike >= 3:
        score += 15
    elif volume_spike >= 2:
        score += 8
    
    # 2. Whale Activity (25 poin max)
    if whale_buy_pressure is not None:
        if whale_buy_pressure >= 0.80:
            score += 25
        elif whale_buy_pressure >= 0.70:
            score += 18
        elif whale_buy_pressure >= 0.60:
            score += 10
    
    # 3. Funding Rate Signal (20 poin max)
    if funding_rate is not None:
        if funding_rate < -0.001:  # Short squeeze potential
            score += 20
        elif funding_rate < -0.0005:
            score += 15
        elif -0.0005 <= funding_rate <= 0.0005:  # Netral = fresh position
            score += 8
    
    # 4. Open Interest (15 poin max)
    if oi_change is not None:
        if oi_change >= 50:
            score += 15
        elif oi_change >= 25:
            score += 10
        elif oi_change >= 10:
            score += 5
    
    # 5. Price Action (5 poin max)
    if -5 <= price_change_24h <= 15:  # Belum pump besar tapi ada pergerakan
        score += 5
    elif price_change_24h < -10:  # Oversold
        score += 3
    
    # 6. Konsolidasi sebelum breakout (5 poin)
    if is_consolidating:
        score += 5
    
    # 7. Small Market Cap bonus (naik multiplier)
    if market_cap is not None:
        if market_cap < 10_000_000:  # < $10M
            score *= 1.3
        elif market_cap < 50_000_000:  # < $50M
            score *= 1.15
    
    return min(score, 100)

def scan_single_coin(row: pd.Series, futures_symbols: list) -> Optional[AnomalySignal]:
    """Scan satu coin untuk anomali"""
    symbol = row['symbol']
    price = row['lastPrice']
    price_change_24h = row['priceChangePercent']
    volume_24h = row['quoteVolume']
    
    try:
        # Filter awal - hanya coin dengan volume minimal $100K/24H
        if volume_24h < 100_000:
            return None
        
        # Filter harga terlalu rendah (dust coins)
        if price < 0.0000001:
            return None
        
        all_signals = []
        
        # 1. Volume Spike Analysis
        volume_spike, vol_signals = calculate_volume_spike(symbol, volume_24h)
        all_signals.extend(vol_signals)
        time.sleep(0.1)
        
        # Jika volume spike tidak signifikan, skip (optimasi)
        if volume_spike < 2.0:
            return None
        
        # 2. Price Consolidation
        is_consolidating, cons_signals = analyze_price_consolidation(symbol)
        all_signals.extend(cons_signals)
        time.sleep(0.1)
        
        # 3. Funding Rate (jika ada di futures)
        funding_rate, prev_fr, fr_signals = analyze_funding_rate(symbol, futures_symbols)
        all_signals.extend(fr_signals)
        
        # 4. Open Interest
        oi_change, oi_signals = analyze_open_interest(symbol, futures_symbols)
        all_signals.extend(oi_signals)
        
        # 5. Whale Trades
        whale_buy_pressure, whale_signals = analyze_whale_trades(symbol)
        all_signals.extend(whale_signals)
        time.sleep(0.1)
        
        # 6. Order Book
        ob_imbalance, ob_signals = analyze_order_book(symbol)
        all_signals.extend(ob_signals)
        
        # Hitung RAVE Similarity Score
        score = calculate_rave_similarity_score(
            price_change_24h=price_change_24h,
            volume_spike=volume_spike,
            funding_rate=funding_rate,
            oi_change=oi_change,
            whale_buy_pressure=whale_buy_pressure,
            ob_imbalance=ob_imbalance,
            is_consolidating=is_consolidating,
            market_cap=None
        )
        
        # Tentukan alert level
        if score >= 75:
            alert_level = "CRITICAL"
        elif score >= 55:
            alert_level = "HIGH"
        elif score >= 35:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"
        
        # Hanya return jika score minimal MEDIUM
        if score < 35 and volume_spike < 3:
            return None
        
        return AnomalySignal(
            symbol=symbol,
            score=score,
            signals=all_signals,
            price=price,
            price_change_24h=price_change_24h,
            volume_24h=volume_24h,
            volume_spike=volume_spike,
            funding_rate=funding_rate,
            oi_change=oi_change,
            whale_buy_pressure=whale_buy_pressure,
            ob_imbalance=ob_imbalance,
            market_cap=None,
            alert_level=alert_level
        )
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None
