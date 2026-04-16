import os
import time
import logging
import pandas as pd
from scanner.data_fetcher import get_ticker_24h, get_futures_symbols
from scanner.anomaly_detector import scan_single_coin, AnomalySignal
from scanner.telegram_alert import send_telegram, format_signal_message, send_summary_alert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Konfigurasi dari environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Threshold konfigurasi
MIN_VOLUME_24H = float(os.getenv('MIN_VOLUME_24H', '100000'))      # Min $100K
MIN_SCORE = float(os.getenv('MIN_SCORE', '35'))                     # Min score
MAX_COINS_TO_SCAN = int(os.getenv('MAX_COINS_TO_SCAN', '200'))     # Max coin
ALERT_LEVEL_FILTER = os.getenv('ALERT_LEVEL_FILTER', 'MEDIUM')    # Min level alert

ALERT_LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

def should_send_alert(alert_level: str, min_level: str) -> bool:
    """Cek apakah level alert memenuhi threshold"""
    return ALERT_LEVELS.index(alert_level) >= ALERT_LEVELS.index(min_level)

def run_scanner():
    """Main scanner function"""
    logger.info("🚀 Starting Crypto Anomaly Scanner...")
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("❌ TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID tidak ditemukan!")
        return
    
    # 1. Ambil semua ticker 24H
    logger.info("📡 Fetching ticker data...")
    df_ticker = get_ticker_24h()
    if df_ticker.empty:
        logger.error("Failed to get ticker data")
        return
    
    # 2. Filter awal berdasarkan volume
    df_ticker = df_ticker[df_ticker['quoteVolume'] >= MIN_VOLUME_24H]
    
    # 3. Sort by volume spike potential (volume tinggi + belum pump besar)
    df_ticker = df_ticker[df_ticker['priceChangePercent'] < 50]  # Exclude yang sudah pump
    df_ticker = df_ticker.sort_values('quoteVolume', ascending=False)
    df_ticker = df_ticker.head(MAX_COINS_TO_SCAN)
    
    logger.info(f"📊 Scanning {len(df_ticker)} coins...")
    
    # 4. Ambil futures symbols sekali saja
    futures_symbols = get_futures_symbols()
    logger.info(f"📈 Futures pairs available: {len(futures_symbols)}")
    
    # 5. Scan setiap coin
    anomalies = []
    scanned = 0
    
    for _, row in df_ticker.iterrows():
        try:
            signal = scan_single_coin(row, futures_symbols)
            if signal and signal.score >= MIN_SCORE:
                anomalies.append(signal)
                logger.info(f"🎯 ANOMALY: {signal.symbol} | Score: {signal.score:.0f} | Level: {signal.alert_level}")
            
            scanned += 1
            if scanned % 20 == 0:
                logger.info(f"Progress: {scanned}/{len(df_ticker)}")
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error scanning {row.get('symbol', 'UNKNOWN')}: {e}")
            continue
    
    # 6. Sort by score
    anomalies.sort(key=lambda x: x.score, reverse=True)
    
    logger.info(f"✅ Scan complete! Found {len(anomalies)} anomalies")
    
    # 7. Kirim alerts ke Telegram
    alerts_sent = 0
    for signal in anomalies:
        if should_send_alert(signal.alert_level, ALERT_LEVEL_FILTER):
            message = format_signal_message(signal)
            success = send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message)
            if success:
                alerts_sent += 1
                time.sleep(1)  # Jeda antar pesan
    
    # 8. Kirim summary
    send_summary_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, anomalies, scanned)
    
    logger.info(f"📨 Sent {alerts_sent} alerts to Telegram")

if __name__ == "__main__":
    run_scanner()
