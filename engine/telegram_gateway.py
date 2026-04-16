"""
Telegram Gateway — format pesan clean, informatif, dengan
Reasoning Log lengkap dan parameter manajemen risiko.
"""
import requests
import logging
from typing import List
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

def _send(message: str, parse_mode: str = "HTML") -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id"                 : TELEGRAM_CHAT_ID,
            "text"                    : message,
            "parse_mode"              : parse_mode,
            "disable_web_page_preview": True,
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
        return False

def _fmt_price(p: float) -> str:
    """Format harga adaptif sesuai magnitude."""
    if p >= 1000:
        return f"${p:,.2f}"
    elif p >= 1:
        return f"${p:.4f}"
    elif p >= 0.01:
        return f"${p:.5f}"
    else:
        return f"${p:.8f}"

def format_and_send_signal(sig) -> bool:
    """Format FinalSignal menjadi pesan Telegram yang clean dan kirim."""

    # ── Header ────────────────────────────────────────────
    signal_emoji = {
        "LONG"    : "🟢",
        "SHORT"   : "🔴",
        "BUY SPOT": "🔵",
        "WATCH"   : "🟡",
        "NEUTRAL" : "⚪",
    }
    level_emoji = {
        "CRITICAL": "🚨",
        "HIGH"    : "🔴",
        "MEDIUM"  : "🟡",
        "LOW"     : "🟢",
    }

    se = signal_emoji.get(sig.signal_type, "⚪")
    le = level_emoji.get(sig.alert_level, "⚪")

    msg  = f"{le} <b>CMI-ASS SIGNAL</b> {le}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"{se} <b>{sig.signal_type}</b>  |  <b>${sig.symbol.replace('USDT','')}</b>\n"
    msg += f"🎯 Confidence Score: <b>{sig.confidence_score:.0f}/100</b>  |  {sig.alert_level}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # ── Market Snapshot ───────────────────────────────────
    chg_emoji = "📈" if sig.price_change_24h >= 0 else "📉"
    msg += f"💰 <b>Harga:</b> {_fmt_price(sig.price)}\n"
    msg += f"{chg_emoji} <b>24H:</b> {sig.price_change_24h:+.2f}%\n"
    msg += f"💹 <b>Volume 24H:</b> ${sig.volume_24h:,.0f}\n"
    msg += f"🔥 <b>Volume Spike:</b> {sig.volume_spike:.1f}x avg 7D\n"

    if sig.market_cap:
        msg += f"🏦 <b>Market Cap:</b> ${sig.market_cap:,.0f} ({sig.supply_category})\n"
        msg += f"⚖️ <b>Selling Pressure:</b> {sig.selling_pressure}\n"

    msg += "\n"

    # ── Derivatives ───────────────────────────────────────
    if sig.funding_rate is not None:
        fr_arrow = "🟢" if sig.funding_rate < 0 else "🔴"
        msg += f"{fr_arrow} <b>Funding Rate:</b> {sig.funding_rate*100:.4f}%\n"
        msg += f"📊 <b>Short Squeeze Risk:</b> {sig.short_squeeze_risk}\n"

    if sig.oi_change is not None:
        oi_arrow = "📈" if sig.oi_change > 0 else "📉"
        msg += f"{oi_arrow} <b>OI Change 24H:</b> {sig.oi_change:+.1f}%\n"

    msg += "\n"

    # ── Risk Management ───────────────────────────────────
    msg += f"🎯 <b>ENTRY ZONE:</b> {_fmt_price(sig.entry_zone_low)} – {_fmt_price(sig.entry_zone_high)}\n"
    msg += f"🛡️ <b>STOP LOSS:</b> {_fmt_price(sig.stop_loss)}\n"
    msg += f"✅ <b>TP1:</b> {_fmt_price(sig.tp1)}\n"
    msg += f"✅ <b>TP2:</b> {_fmt_price(sig.tp2)}\n"
    msg += f"🚀 <b>TP3:</b> {_fmt_price(sig.tp3)}\n"
    msg += f"⚖️ <b>Risk/Reward:</b> 1:{sig.risk_reward:.1f}\n\n"

    # ── Scoring Breakdown ─────────────────────────────────
    msg += f"📋 <b>Score Breakdown:</b>\n"
    msg += f"  🐋 Whale Sonar:   {sig.whale_score:.0f}/25\n"
    msg += f"  📈 Derivatives:   {sig.derivatives_score:.0f}/30\n"
    msg += f"  🏦 Supply:        {sig.supply_score:.0f}/20\n"
    msg += f"  🔥 Pre-Pump:      {sig.pre_pump_score:.0f}/25\n\n"

    # ── Reasoning Log ─────────────────────────────────────
    if sig.reasoning_log:
        msg += f"🧠 <b>Reasoning Log:</b>\n"
        for reason in sig.reasoning_log[:8]:   # max 8 alasan
            msg += f"  • {reason}\n"
        msg += "\n"

    # ── Links ─────────────────────────────────────────────
    base = sig.symbol.replace("USDT","")
    msg += f"🔗 "
    msg += f"<a href='https://www.binance.com/en/trade/{sig.symbol}'>Binance</a> | "
    msg += f"<a href='https://www.tradingview.com/chart/?symbol=BINANCE:{sig.symbol}'>TradingView</a> | "
    msg += f"<a href='https://coinmarketcap.com/currencies/{base.lower()}/'>CMC</a>\n\n"

    # ── Disclaimer ────────────────────────────────────────
    msg += f"⚡ <i>CMI-ASS Bot | GitHub Actions 24/7</i>\n"
    msg += f"⚠️ <i>BUKAN financial advice. Always DYOR & manage risk!</i>"

    return _send(msg)

def send_scan_summary(total_scanned: int, signals: list, fear_greed: dict) -> bool:
    """Kirim summary setiap selesai scan."""
    fg_val   = fear_greed.get("value", 50)
    fg_label = fear_greed.get("label", "Neutral")
    fg_emoji = "😱" if fg_val < 30 else ("🤑" if fg_val > 70 else "😐")

    msg  = f"✅ <b>CMI-ASS Scan Complete</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Coin dianalisis: <b>{total_scanned}</b>\n"
    msg += f"🎯 Sinyal ditemukan: <b>{len(signals)}</b>\n"
    msg += f"{fg_emoji} Fear & Greed: <b>{fg_val}/100</b> ({fg_label})\n"

    if signals:
        msg += f"\n<b>Top Signals:</b>\n"
        sorted_sigs = sorted(signals, key=lambda x: x.confidence_score, reverse=True)
        for s in sorted_sigs[:5]:
            se = {"LONG":"🟢","SHORT":"🔴","BUY SPOT":"🔵","WATCH":"🟡"}.get(s.signal_type,"⚪")
            msg += (f"  {se} {s.symbol.replace('USDT','')} | "
                    f"{s.signal_type} | Score: {s.confidence_score:.0f} | {s.alert_level}\n")
    else:
        msg += "\n🔍 Belum ada anomali signifikan saat ini."

    return _send(msg)

def send_startup_message() -> bool:
    """Kirim notifikasi saat scanner baru dijalankan."""
    msg  = "🚀 <b>CMI-ASS Scanner AKTIF</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "✅ GitHub Actions berhasil dijalankan\n"
    msg += "🔍 Memulai scanning pasar crypto...\n"
    msg += "📡 Mengambil data Binance + CoinGecko\n"
    msg += "🐋 Whale Sonar: ON\n"
    msg += "📈 Derivatives Engine: ON\n"
    msg += "🏦 Supply Analyzer: ON\n"
    msg += "🔥 Pre-Pump Detector: ON\n"
    msg += "\n⏳ Tunggu hasil scan dalam beberapa menit..."
    return _send(msg)
