"""
Telegram Gateway — CMI-ASS
Format pesan clean dengan Reasoning Log + TA section baru.
(SMC GOD-TIER OPTIMIZED & CRASH-PROOF)
"""
import requests
import logging
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
    if p >= 1000:   return f"${p:,.2f}"
    elif p >= 1:    return f"${p:.4f}"
    elif p >= 0.01: return f"${p:.5f}"
    else:           return f"${p:.8f}"


def format_and_send_signal(sig) -> bool:
    """Format FinalSignal → pesan Telegram clean dengan TA section."""

    signal_emoji = {"LONG":"🟢","SHORT":"🔴","BUY SPOT":"🔵","WATCH":"🟡","NEUTRAL":"⚪"}
    level_emoji  = {"CRITICAL":"🚨","HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}
    se = signal_emoji.get(sig.signal_type, "⚪")
    le = level_emoji.get(sig.alert_level, "⚪")

    # ── Header ────────────────────────────────────────────
    msg  = f"{le} <b>CMI-ASS SIGNAL</b> {le}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"{se} <b>{sig.signal_type}</b>  |  <b>${sig.symbol.replace('USDT','')}</b>\n"
    msg += f"🎯 Confidence: <b>{sig.confidence_score:.0f}/100</b>  |  {sig.alert_level}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # ── Market Snapshot ───────────────────────────────────
    chg_emoji = "📈" if sig.price_change_24h >= 0 else "📉"
    msg += f"💰 <b>Harga Saat Ini:</b> {_fmt_price(sig.price)}\n"
    msg += f"{chg_emoji} <b>24H Change:</b> {sig.price_change_24h:+.2f}%\n"
    msg += f"💹 <b>Volume 24H:</b> ${sig.volume_24h:,.0f}\n"
    msg += f"🔥 <b>Volume Spike:</b> {getattr(sig, 'volume_spike', 0):.1f}x avg 7D\n"
    
    mc = getattr(sig, 'market_cap', None)
    if mc:
        msg += f"🏦 <b>Market Cap:</b> ${mc:,.0f} ({getattr(sig, 'supply_category', 'UNKNOWN')})\n"
        msg += f"⚖️ <b>Selling Pressure:</b> {getattr(sig, 'selling_pressure', 'UNKNOWN')}\n"
    msg += "\n"

    # ── Derivatives ───────────────────────────────────────
    fr = getattr(sig, 'funding_rate', None)
    if fr is not None:
        fr_arrow = "🟢" if fr < 0 else "🔴"
        msg += f"{fr_arrow} <b>Funding Rate:</b> {fr*100:.4f}%\n"
        msg += f"📊 <b>Short Squeeze Risk:</b> {getattr(sig, 'short_squeeze_risk', 'UNKNOWN')}\n"
        
    oi = getattr(sig, 'oi_change', None)
    if oi is not None:
        oi_arrow = "📈" if oi > 0 else "📉"
        msg += f"{oi_arrow} <b>OI Change 24H:</b> {oi:+.1f}%\n"
    msg += "\n"

    # ── Technical Analysis Summary ────────────────────────
    ta_bias_emoji = {
        "STRONG_BULL": "🚀","BULL": "📈","NEUTRAL": "⚪",
        "BEAR": "📉","STRONG_BEAR": "🔻"
    }
    bias = getattr(sig, 'ta_bias', 'UNKNOWN')
    tbe = ta_bias_emoji.get(bias, "⚪")

    msg += f"📊 <b>Technical & SMC Analysis</b>\n"
    msg += f"  {tbe} <b>TA Bias:</b> {bias}\n"
    
    rsi = getattr(sig, 'rsi_14', 50.0)
    msg += f"  📈 <b>RSI (14):</b> {rsi:.1f}"
    if rsi <= 30:
        msg += " ⬇️ OVERSOLD\n"
    elif rsi >= 70:
        msg += " ⬆️ OVERBOUGHT\n"
    else:
        msg += "\n"

    macd = getattr(sig, 'macd_signal_type', 'UNKNOWN')
    macd_emoji = {"BULLISH_CROSS":"💚","BEARISH_CROSS":"🔴","BULLISH":"📈","BEARISH":"📉","NEUTRAL":"⚪"}
    msg += f"  {macd_emoji.get(macd,'⚪')} <b>MACD:</b> {macd}\n"

    # Pengambilan data dengan getattr agar tidak crash jika None
    trend_align = getattr(sig, 'trend_alignment', 'UNKNOWN')
    if trend_align != 'UNKNOWN':
        trend_emoji = {"STRONG_BULL":"🚀","BULL":"📈","SIDEWAYS":"↔️","BEAR":"📉","STRONG_BEAR":"🔻"}
        msg += f"  {trend_emoji.get(trend_align,'⚪')} <b>Trend (EMA):</b> {trend_align}\n"

    if getattr(sig, 'bb_squeeze', False):
        msg += f"  🌀 <b>BB Squeeze:</b> AKTIF — breakout imminent!\n"

    pattern = getattr(sig, 'dominant_pattern', None)
    if pattern:
        msg += f"  📐 <b>Pattern:</b> {pattern}\n"

    sup = getattr(sig, 'nearest_support', 0)
    if sup > 0:
        msg += f"  🟢 <b>Support:</b> {_fmt_price(sup)}\n"
        
    res = getattr(sig, 'nearest_resist', 0)
    if res > 0:
        msg += f"  🔴 <b>Resistance:</b> {_fmt_price(res)}\n"
    msg += "\n"

    # ── Risk Management (LIMIT ORDER SMC) ─────────────────
    msg += f"🎯 <b>LIMIT ENTRY ZONE:</b>\n"
    msg += f"   {_fmt_price(sig.entry_zone_low)} – {_fmt_price(sig.entry_zone_high)}\n"
    msg += f"🛡️ <b>STOP LOSS (Strict):</b> {_fmt_price(sig.stop_loss)}\n"
    msg += f"✅ <b>TP1 (Liquidity):</b> {_fmt_price(sig.tp1)}\n"
    msg += f"✅ <b>TP2 (Structure):</b> {_fmt_price(sig.tp2)}\n"
    msg += f"🚀 <b>TP3 (Moon Bag):</b> {_fmt_price(sig.tp3)}\n"
    msg += f"⚖️ <b>Risk/Reward Ratio:</b> 1 : {sig.risk_reward:.1f}\n\n"

    # ── Reasoning Log ─────────────────────────────────────
    if sig.reasoning_log:
        msg += f"🧠 <b>Analisis Uang Pintar (Smart Money):</b>\n"
        for reason in sig.reasoning_log[:7]:
            msg += f"  • {reason}\n"
        msg += "\n"

    # ── Score Breakdown ───────────────────────────────────
    msg += f"📋 <b>Score Breakdown:</b>\n"
    msg += f"  🐋 Whale Sonar:   {getattr(sig, 'whale_score', 0):.0f}/25\n"
    msg += f"  📈 Derivatives:   {getattr(sig, 'derivatives_score', 0):.0f}/30\n"
    msg += f"  🏦 Supply:        {getattr(sig, 'supply_score', 0):.0f}/20\n"
    msg += f"  🔥 Pre-Pump:      {getattr(sig, 'pre_pump_score', 0):.0f}/25\n"
    msg += f"  📊 Technical:     {getattr(sig, 'ta_score', 0):.0f}/30\n"
    msg += f"  📌 Total Confidence:{sig.confidence_score:.0f}/100\n\n"

    # ── Links ─────────────────────────────────────────────
    base = sig.symbol.replace("USDT","")
    msg += f"🔗 "
    msg += f"<a href='https://www.binance.com/en/trade/{sig.symbol}'>Binance</a> | "
    msg += f"<a href='https://www.tradingview.com/chart/?symbol=BINANCE:{sig.symbol}'>TradingView</a> | "
    msg += f"<a href='https://coinmarketcap.com/currencies/{base.lower()}/'>CMC</a>\n\n"

    msg += f"⚡ <i>CMI-ASS Bot v2 | by Furiooseventh</i>\n"
    msg += f"⚠️ <i>BUKAN financial advice. Jaga MM (Money Management)!</i>\n"
    msg += f"⚠️ <i>NEK LAGI TRADE OJO NGELAMUN</i>"

    return _send(msg)


def send_scan_summary(total_scanned: int, signals: list, fear_greed: dict) -> bool:
    fg_val   = fear_greed.get("value", 50)
    fg_label = fear_greed.get("label", "Neutral")
    fg_emoji = "😱" if fg_val < 30 else ("🤑" if fg_val > 70 else "😐")

    msg  = f"✅ <b>CMI-ASS v2 Scan Complete</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Coin dianalisis: <b>{total_scanned}</b>\n"
    msg += f"🎯 Sinyal potensial: <b>{len(signals)}</b>\n"
    msg += f"{fg_emoji} Fear & Greed: <b>{fg_val}/100</b> ({fg_label})\n"

    if signals:
        msg += f"\n<b>🔥 Top SMC Signals:</b>\n"
        sorted_sigs = sorted(signals, key=lambda x: x.confidence_score, reverse=True)
        for s in sorted_sigs[:5]:
            se = {"LONG":"🟢","SHORT":"🔴","BUY SPOT":"🔵","WATCH":"🟡"}.get(s.signal_type,"⚪")
            bias = getattr(s, 'ta_bias', '⚪')
            tbe = {"STRONG_BULL":"🚀","BULL":"📈","NEUTRAL":"⚪","BEAR":"📉","STRONG_BEAR":"🔻"}.get(bias,"⚪")
            msg += (f"  {se} {s.symbol.replace('USDT','')} | "
                    f"{s.signal_type} | Score: {s.confidence_score:.0f} | "
                    f"TA: {tbe}{bias}\n")
    else:
        msg += "\n🔍 Belum ada setup SMC yang matang saat ini."

    return _send(msg)


def send_startup_message() -> bool:
    msg  = "🚀 <b>CMI-ASS SMC Scanner AKTIF</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "✅ Mesin Dewa Trader berhasil dipanaskan\n"
    msg += "🔍 Memulai pelacakan jejak Whale...\n"
    msg += "📡 Target Data: Global Crypto Market\n"
    msg += "🐋 Whale Sonar: ON\n"
    msg += "📈 Derivatives Engine: ON\n"
    msg += "🏦 Supply Analyzer: ON\n"
    msg += "🔥 Pre-Pump Detector: ON\n"
    msg += "📊 SMC Technical Analysis: ON\n"
    msg += "  ↳ Limit Order via Order Block (OB)\n"
    msg += "  ↳ Fair Value Gap (FVG) Detection\n"
    msg += "  ↳ Liquidity Sweep Logic\n"
    msg += "\n⏳ Tunggu hasil scan masuk ke Telegram..."
    return _send(msg)
