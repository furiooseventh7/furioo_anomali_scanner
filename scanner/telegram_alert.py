import requests
import logging
from scanner.anomaly_detector import AnomalySignal
import os

logger = logging.getLogger(__name__)

def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    """Kirim pesan ke Telegram"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            logger.info(f"Telegram sent: {r.json().get('ok')}")
            return True
        else:
            logger.error(f"Telegram error: {r.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram exception: {e}")
        return False

def format_signal_message(signal: AnomalySignal) -> str:
    """Format pesan alert yang informatif"""
    
    level_emoji = {
        'CRITICAL': '🚨🚨🚨',
        'HIGH': '🔴🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    emoji = level_emoji.get(signal.alert_level, '⚪')
    
    # Header
    msg = f"{emoji} <b>CRYPTO ANOMALY ALERT!</b> {emoji}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🪙 <b>{signal.symbol}</b> | Score: <b>{signal.score:.0f}/100</b>\n"
    msg += f"⚠️ Level: <b>{signal.alert_level}</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Price Info
    change_emoji = "📈" if signal.price_change_24h > 0 else "📉"
    msg += f"💰 <b>Harga:</b> ${signal.price:.6f}\n"
    msg += f"{change_emoji} <b>Change 24H:</b> {signal.price_change_24h:+.2f}%\n"
    msg += f"💹 <b>Volume 24H:</b> ${signal.volume_24h:,.0f}\n"
    msg += f"🔥 <b>Volume Spike:</b> {signal.volume_spike:.1f}x\n\n"
    
    # Futures Data
    if signal.funding_rate is not None:
        fr_emoji = "💚" if signal.funding_rate < 0 else "🔴"
        msg += f"{fr_emoji} <b>Funding Rate:</b> {signal.funding_rate*100:.4f}%\n"
    
    if signal.oi_change is not None:
        oi_emoji = "📈" if signal.oi_change > 0 else "📉"
        msg += f"{oi_emoji} <b>OI Change 24H:</b> {signal.oi_change:+.1f}%\n"
    
    if signal.whale_buy_pressure is not None:
        wp_emoji = "🐋" if signal.whale_buy_pressure > 0.6 else "🦈"
        msg += f"{wp_emoji} <b>Whale Buy:</b> {signal.whale_buy_pressure*100:.0f}%\n"
    
    if signal.ob_imbalance is not None:
        ob_emoji = "📗" if signal.ob_imbalance > 0 else "📕"
        msg += f"{ob_emoji} <b>OB Imbalance:</b> {signal.ob_imbalance*100:+.1f}%\n"
    
    # Signals Detected
    if signal.signals:
        msg += f"\n📋 <b>Sinyal Terdeteksi:</b>\n"
        for s in signal.signals[:6]:  # Max 6 sinyal
            msg += f"  • {s}\n"
    
    # Links
    base_symbol = signal.symbol.replace('USDT', '')
    msg += f"\n🔗 <a href='https://www.binance.com/en/trade/{signal.symbol}'>Binance</a>"
    msg += f" | <a href='https://coinmarketcap.com/currencies/{base_symbol.lower()}/'>CMC</a>"
    msg += f" | <a href='https://www.tradingview.com/chart/?symbol=BINANCE:{signal.symbol}'>TradingView</a>\n"
    
    # Disclaimer
    msg += f"\n⚡ <i>Deteksi: GitHub Actions Bot</i>\n"
    msg += f"⚠️ <i>BUKAN financial advice! DYOR!</i>"
    
    return msg

def send_summary_alert(bot_token: str, chat_id: str, signals: list, scan_count: int):
    """Kirim summary scan"""
    msg = f"✅ <b>Scan Selesai</b>\n"
    msg += f"📊 Coin dianalisis: {scan_count}\n"
    msg += f"🎯 Anomali ditemukan: {len(signals)}\n\n"
    
    if signals:
        msg += "<b>Top Anomalies:</b>\n"
        for s in sorted(signals, key=lambda x: x.score, reverse=True)[:5]:
            msg += f"• {s.symbol}: {s.score:.0f}pt ({s.alert_level})\n"
    else:
        msg += "🔍 Tidak ada anomali signifikan saat ini."
    
    send_telegram(bot_token, chat_id, msg)
