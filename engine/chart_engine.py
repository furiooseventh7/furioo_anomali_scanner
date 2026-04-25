"""
Chart Engine — CMI-ASS v6
==========================
Mengambil screenshot chart TradingView dan mengirim ke Telegram sebagai foto
dengan anotasi TA yang tertulis langsung di atas chart.

STRATEGI:
  TradingView Advanced Chart tersedia via screenshot widget gratis:
  https://s.tradingview.com/widgetembed/?symbol=BITGET:{SYMBOL}&interval={TF}&...

  Kita gunakan endpoint screenshot dari TradingView (tidak perlu login)
  atau fallback ke service screenshot publik: screenshotlayer, htmlcsstoimage

FALLBACK CHAIN:
  1. TradingView Widget Screenshot (via puppeteer-style URL snapshot)
  2. screenshotone.com (free tier 100/bulan)
  3. Kirim link chart saja jika semua gagal (non-blocking)

CHART ANNOTATION:
  Anotasi ditulis di caption Telegram (bukan di image).
  TradingView widget URL bisa di-customize dengan indicator overlay.
"""

import requests
import logging
import os
import time
from typing import Optional, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)

TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CMI-ASS/6.0)"}

# ─────────────────────────────────────────────────────────
#  TRADINGVIEW URL BUILDER
# ─────────────────────────────────────────────────────────

# Mapping timeframe signal ke TradingView interval
TF_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "1h": "60", "2h": "120", "4h": "240",
    "6h": "360", "12h": "720", "1d": "D", "1w": "W",
}


def _build_tv_widget_url(symbol: str, interval: str = "240",
                          studies: list = None) -> str:
    """
    Build TradingView Advanced Chart Widget URL.
    Interval: 60=1H, 240=4H, D=1D
    Studies: list of indicator names untuk overlay (EMA, RSI, MACD, dll)
    """
    base_sym = symbol.replace("USDT", "").replace("USDC", "")
    # Bitget format: BITGET:BTCUSDT
    tv_symbol = f"BITGET:{symbol}"

    # Default studies yang selalu ada
    default_studies = [
        "EMA@tv-basicstudies",       # EMA
        "RSI@tv-basicstudies",       # RSI
        "MACD@tv-basicstudies",      # MACD
        "BB@tv-basicstudies",        # Bollinger Bands
        "Volume@tv-basicstudies",    # Volume
    ]
    all_studies = studies or default_studies
    studies_str = "%1F".join(all_studies)

    url = (
        f"https://s.tradingview.com/widgetembed/"
        f"?frameElementId=tradingview_chart"
        f"&symbol={tv_symbol}"
        f"&interval={interval}"
        f"&hidesidetoolbar=1"
        f"&hidetoptoolbar=0"
        f"&saveimage=1"
        f"&toolbarbg=f1f3f6"
        f"&studies={studies_str}"
        f"&theme=dark"
        f"&style=1"
        f"&locale=en"
        f"&timezone=Asia%2FJakarta"
        f"&withdateranges=1"
        f"&showpopupbutton=1"
    )
    return url


def _build_tv_chart_url(symbol: str, interval: str = "240") -> str:
    """URL chart TradingView langsung (untuk link di Telegram)."""

    return f"https://www.tradingview.com/chart/?symbol=BITGET:{symbol}&interval={interval}"


def _build_tv_screenshot_url(symbol: str, interval: str = "240") -> str:
    """
    TradingView snapshot URL (PNG langsung dari widget).
    Gunakan screenshotapi jika ada, fallback ke chart link.
    """
    widget_url = _build_tv_widget_url(symbol, interval)

    # Cek apakah SCREENSHOTONE_KEY tersedia di environment
    ss_key = os.getenv("SCREENSHOTONE_KEY", "")
    if ss_key:
        # screenshotone.com free tier: 100 shots/bulan
        import urllib.parse
        encoded = urllib.parse.quote(widget_url, safe="")
        return (
            f"https://api.screenshotone.com/take"
            f"?url={encoded}"
            f"&access_key={ss_key}"
            f"&viewport_width=1200"
            f"&viewport_height=700"
            f"&full_page=false"
            f"&delay=4"
            f"&format=jpg"
            f"&image_quality=85"
        )

    return ""


# ─────────────────────────────────────────────────────────
#  SCREENSHOT FETCHER
# ─────────────────────────────────────────────────────────

def get_chart_image(symbol: str, timeframe: str = "4h") -> Optional[bytes]:
    """
    Ambil screenshot chart dalam bentuk bytes (JPEG).
    Returns None jika gagal.
    """
    interval = TF_MAP.get(timeframe, "240")
    ss_url   = _build_tv_screenshot_url(symbol, interval)

    if not ss_url:
        logger.debug(f"Chart screenshot: SCREENSHOTONE_KEY tidak ada, skip image")
        return None

    try:
        r = requests.get(ss_url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and len(r.content) > 5000:
            logger.info(f"Chart screenshot OK: {symbol} {timeframe} ({len(r.content)//1024}KB)")
            return r.content
        else:
            logger.warning(f"Chart screenshot gagal: {symbol} status={r.status_code} size={len(r.content)}")
    except Exception as e:
        logger.warning(f"Chart screenshot error: {symbol}: {e}")

    return None


# ─────────────────────────────────────────────────────────
#  TELEGRAM PHOTO SENDER
# ─────────────────────────────────────────────────────────

def send_chart_to_telegram(
    bot_token  : str,
    chat_id    : str,
    symbol     : str,
    caption    : str,
    timeframe  : str = "4h",
    reply_to   : Optional[int] = None,
) -> Tuple[bool, Optional[int]]:
    """
    Kirim chart TradingView ke Telegram sebagai foto dengan caption TA.

    Returns:
        (success: bool, message_id: Optional[int])
    """
    image_bytes = get_chart_image(symbol, timeframe)

    if image_bytes:
        # Kirim sebagai foto
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        try:
            files = {"photo": (f"{symbol}_{timeframe}.jpg", BytesIO(image_bytes), "image/jpeg")}
            data  = {
                "chat_id"    : chat_id,
                "caption"    : caption[:1024],  # Telegram caption limit
                "parse_mode" : "HTML",
            }
            if reply_to:
                data["reply_to_message_id"] = reply_to

            r = requests.post(url, files=files, data=data, timeout=20)
            if r.status_code == 200:
                msg_id = r.json().get("result", {}).get("message_id")
                logger.info(f"Chart photo sent: {symbol} → chat {chat_id}")
                return True, msg_id
            else:
                logger.warning(f"sendPhoto gagal: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.error(f"sendPhoto error {symbol}: {e}")

    # Fallback: kirim link chart saja
    return False, None


def build_chart_caption(sig, vr_defillama=None) -> str:
    """
    Buat caption untuk chart yang dikirim ke Telegram.
    Caption berisi ringkasan TA, entry zone, dan key levels.
    Ini yang akan muncul di bawah gambar chart.
    """
    base = sig.symbol.replace("USDT", "")
    se   = {"LONG": "🟢", "SHORT": "🔴", "BUY SPOT": "🔵", "WATCH": "🟡"}.get(sig.signal_type, "⚪")

    def _p(v):
        if v >= 1000:   return f"${v:,.2f}"
        elif v >= 1:    return f"${v:.4f}"
        elif v >= 0.01: return f"${v:.5f}"
        else:            return f"${v:.8f}"

    caption  = f"{se} <b>${base} — {sig.signal_type}</b> | Score: {sig.confidence_score:.0f}/100\n"
    caption += f"━━━━━━━━━━━━━━━━━━━━━━\n"

    # TA Summary
    ec = sig.extra_context if hasattr(sig, "extra_context") else {}
    mb = ec.get("mtf_bias", sig.ta_bias)
    tbe = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "⚪", "BEAR": "📉", "STRONG_BEAR": "🔻"}.get(mb, "⚪")
    caption += f"📊 Bias: {tbe} <b>{mb}</b>  |  RSI: {sig.rsi_14:.0f}  |  MACD: {sig.macd_signal_type}\n"

    # SMC Key Levels
    if ec.get("has_precise_entry"):
        el = ec.get("optimal_entry_low", 0)
        eh = ec.get("optimal_entry_high", 0)
        if el > 0:
            caption += f"🎯 SMC Entry: <b>{_p(el)} – {_p(eh)}</b>\n"

    caption += f"🎯 Entry: {_p(sig.entry_zone_low)} – {_p(sig.entry_zone_high)}\n"
    caption += f"🛡️ SL: {_p(sig.stop_loss)}  |  TP1: {_p(sig.tp1)}  |  TP3: {_p(sig.tp3)}\n"

    # Whale context
    if sig.is_accumulating:
        caption += f"🐋 Whale AKUMULASI terdeteksi\n"

    # DefiLlama undervalue
    if vr_defillama and vr_defillama.found and vr_defillama.overall_verdict in ("EXTREME_UNDERVALUE", "UNDERVALUE"):
        verd = vr_defillama.overall_verdict.replace("_", " ")
        ps   = vr_defillama.ps_ratio
        if ps < 999:
            caption += f"💚 Fundamental: P/S {ps:.1f}x ({verd})\n"
        else:
            caption += f"💚 Fundamental: {verd}\n"

    caption += f"\n⚡ CMI-ASS v6 | {sig.alert_level}"
    return caption


# ─────────────────────────────────────────────────────────
#  CHART LINK BUILDER (untuk text message, selalu ada)
# ─────────────────────────────────────────────────────────

def get_chart_links(symbol: str) -> str:
    """Return HTML link chart untuk berbagai timeframe."""
    links = [
        f"<a href='{_build_tv_chart_url(symbol, '60')}'>1H</a>",
        f"<a href='{_build_tv_chart_url(symbol, '240')}'>4H</a>",
        f"<a href='{_build_tv_chart_url(symbol, 'D')}'>1D</a>",
    ]
    return " | ".join(links)
