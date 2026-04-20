"""
Telegram Gateway — CMI-ASS
Format pesan clean dengan Reasoning Log + TA section baru.
"""
import requests
import logging
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

# Support multiple chat IDs dipisah koma
# Contoh: TELEGRAM_CHAT_ID = "123456789,987654321"
_CHAT_IDS = [cid.strip() for cid in str(TELEGRAM_CHAT_ID).split(",") if cid.strip()]


def _send(message: str, parse_mode: str = "HTML") -> bool:
    """Kirim pesan ke semua chat ID yang terdaftar."""
    if not _CHAT_IDS:
        logger.error("Tidak ada TELEGRAM_CHAT_ID yang valid")
        return False

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    success = False
    for chat_id in _CHAT_IDS:
        try:
            r = requests.post(url, json={
                "chat_id"                 : chat_id,
                "text"                    : message,
                "parse_mode"              : parse_mode,
                "disable_web_page_preview": True,
            }, timeout=10)
            if r.status_code == 200:
                success = True
            else:
                logger.warning(f"Telegram send ke {chat_id} gagal: {r.status_code} {r.text[:100]}")
        except Exception as e:
            logger.error(f"Telegram send error ke {chat_id}: {e}")
    return success


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

    # ── [BARU] Narrative Hype Section ─────────────────────
    ec = sig.extra_context if hasattr(sig, "extra_context") and sig.extra_context else {}
    narr = ec.get("narrative")
    if narr is not None:
        hype_bar = "█" * (narr.hype_score // 10) + "░" * (10 - narr.hype_score // 10)
        msg += f"🏷️ <b>Narasi: {narr.name}</b> — {narr.description}\n"
        msg += f"   📊 Hype Score: <b>{narr.hype_score}/100</b>  {narr.hype_label}\n"
        msg += f"   [{hype_bar}]\n"
        if narr.avg_change_24h != 0:
            chg_arrow = "📈" if narr.avg_change_24h >= 0 else "📉"
            msg += f"   {chg_arrow} Rata-rata sektor 24H: <b>{narr.avg_change_24h:+.1f}%</b>\n"
        if narr.top_gainers:
            gainers_str = "  ".join(
                f"${g[0]} {g[1]:+.1f}%" for g in narr.top_gainers[:3]
            )
            msg += f"   🏆 Top: {gainers_str}\n"
        if narr.secondary:
            msg += f"   🔖 Tag tambahan: #{narr.secondary}\n"
        msg += "\n"

    # ── [BARU] Technical Analysis Summary ─────────────────
    ta_bias_emoji = {
        "STRONG_BULL": "🚀","BULL": "📈","NEUTRAL": "⚪",
        "BEAR": "📉","STRONG_BEAR": "🔻"
    }
    tbe = ta_bias_emoji.get(sig.ta_bias, "⚪")

    msg += f"📊 <b>Technical Analysis — Multi-Timeframe</b>\n"
    msg += f"  {tbe} <b>TA Bias:</b> {sig.ta_bias}\n"

    # MTF Consensus (v3 field — aman jika tidak ada)
    ta_obj = getattr(sig, "_ta_ref", None)  # fallback
    mtf_bias_disp  = getattr(sig, "ta_bias", "NEUTRAL")
    mtf_agree      = getattr(sig, "rsi_14", 50)  # fallback placeholder

    msg += f"  📈 <b>RSI (14):</b> {sig.rsi_14:.1f}"
    if sig.rsi_14 <= 30: msg += " ⬇️ OVERSOLD\n"
    elif sig.rsi_14 >= 70: msg += " ⬆️ OVERBOUGHT\n"
    else: msg += "\n"

    macd_emoji = {"BULLISH_CROSS":"💚","BEARISH_CROSS":"🔴","BULLISH":"📈","BEARISH":"📉","NEUTRAL":"⚪"}
    msg += f"  {macd_emoji.get(sig.macd_signal_type,'⚪')} <b>MACD:</b> {sig.macd_signal_type}\n"
    trend_emoji = {"STRONG_BULL":"🚀","BULL":"📈","SIDEWAYS":"↔️","BEAR":"📉","STRONG_BEAR":"🔻"}
    msg += f"  {trend_emoji.get(sig.trend_alignment,'⚪')} <b>EMA Trend:</b> {sig.trend_alignment}\n"

    if sig.bb_squeeze:
        msg += f"  🌀 <b>BB Squeeze AKTIF</b> — volatility sangat rendah!\n"

    if sig.dominant_pattern:
        msg += f"  📐 <b>Pattern:</b> {sig.dominant_pattern}\n"

    if sig.nearest_support > 0:
        msg += f"  🟢 <b>Support:</b> {_fmt_price(sig.nearest_support)}\n"
    if sig.nearest_resist > 0:
        msg += f"  🔴 <b>Resistance:</b> {_fmt_price(sig.nearest_resist)}\n"
    msg += "\n"

    # ── [v3] SMC Precision Zone ───────────────────────────────
    ec = sig.extra_context if hasattr(sig, "extra_context") and sig.extra_context else {}
    has_pe = ec.get("has_precise_entry", False)
    opt_low  = ec.get("optimal_entry_low", 0.0)
    opt_high = ec.get("optimal_entry_high", 0.0)
    struct_sl_price = ec.get("structure_sl", 0.0)
    in_discount  = ec.get("in_discount_zone", False)
    in_premium   = ec.get("in_premium_zone", False)
    liq_swept    = ec.get("liquidity_swept", False)
    mtf_bias_val = ec.get("mtf_bias", "")
    mtf_agree_n  = ec.get("mtf_agree_count", 0)

    smc_lines = []
    if mtf_bias_val: smc_lines.append(f"🔭 MTF: <b>{mtf_bias_val}</b> ({mtf_agree_n}/4 TF setuju)")
    if liq_swept: smc_lines.append("💧 <b>Liquidity Swept</b> — stop hunt selesai, reversal setup!")
    if in_discount: smc_lines.append("💚 ICT: <b>Discount Zone</b> — zona ideal beli")
    elif in_premium: smc_lines.append("⚠️ ICT: <b>Premium Zone</b> — harga mahal, risiko tinggi")
    if has_pe and opt_low > 0:
        smc_lines.append(f"🎯 SMC Entry: {_fmt_price(opt_low)} – {_fmt_price(opt_high)}")
    if struct_sl_price > 0:
        smc_lines.append(f"🛡️ Structure SL: {_fmt_price(struct_sl_price)}")

    if smc_lines:
        msg += "🧩 <b>Smart Money Context</b>\n"
        for line in smc_lines:
            msg += f"  {line}\n"
        msg += "\n"

    # ── Invalidation Warnings ─────────────────────────────────
    inv_list = ec.get("invalidation", [])
    if inv_list:
        msg += "⚠️ <b>Invalidation Risks:</b>\n"
        for inv in inv_list[:2]:
            msg += f"  {inv}\n"
        msg += "\n"

    # ── Risk Management ───────────────────────────────────
    msg += f"🎯 <b>ENTRY ZONE:</b> {_fmt_price(sig.entry_zone_low)} – {_fmt_price(sig.entry_zone_high)}\n"
    msg += f"🛡️ <b>STOP LOSS:</b> {_fmt_price(sig.stop_loss)}\n"
    msg += f"✅ <b>TP1:</b> {_fmt_price(sig.tp1)}\n"
    msg += f"✅ <b>TP2:</b> {_fmt_price(sig.tp2)}\n"
    msg += f"🚀 <b>TP3:</b> {_fmt_price(sig.tp3)}\n"
    msg += f"⚖️ <b>Risk/Reward:</b> 1:{sig.risk_reward:.1f}\n\n"

    # ── Score Breakdown (termasuk TA + Quant score) ──────
    quant_obj = ec.get("quant")
    msg += f"📋 <b>Score Breakdown:</b>\n"
    msg += f"  🐋 Whale Sonar:   {sig.whale_score:.0f}/25\n"
    msg += f"  📈 Derivatives:   {sig.derivatives_score:.0f}/30\n"
    msg += f"  🏦 Supply:        {sig.supply_score:.0f}/20\n"
    msg += f"  🔥 Pre-Pump:      {sig.pre_pump_score:.0f}/25\n"
    msg += f"  📊 Technical:     {sig.ta_score:.0f}/30\n"
    quant_score_disp = getattr(sig, "quant_score", 0.0)
    msg += f"  🔬 Quant (151):   {quant_score_disp:.0f}/40\n"
    msg += f"  📌 Total:         {sig.confidence_score:.0f}/100\n\n"

    # ── [BARU] Quant Engine Section (151 Trading Strategies) ─
    if quant_obj is not None:
        q = quant_obj
        msg += f"🔬 <b>Quant Analysis</b> <i>(151 Trading Strategies)</i>\n"

        # MA3 bias
        ma3_emoji = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "↔️"}.get(q.ma3_bias, "↔️")
        msg += f"  {ma3_emoji} Triple MA (EMA9/21/50): <b>{q.ma3_bias}</b>\n"

        # Trend signal (tanh)
        if abs(q.trend_signal) > 0.05:
            trend_dir = "🚀 BULLISH" if q.trend_signal > 0 else "🔻 BEARISH"
            msg += f"  📡 Trend Signal (tanh): <b>{q.trend_signal:+.2f}</b> {trend_dir}\n"

        # Dual momentum
        if q.dual_mom_pass:
            msg += f"  ✅ Dual Momentum: <b>PASS</b> — relative & absolute positif\n"
        else:
            msg += f"  ❌ Dual Momentum: <b>FAIL</b> — salah satu filter tidak lolos\n"

        # IBS
        ibs_label = "OVERSOLD 💚" if q.ibs_value < 0.25 else ("OVERBOUGHT 🔴" if q.ibs_value > 0.75 else "NEUTRAL ↔️")
        msg += f"  📊 IBS (Internal Bar Strength): <b>{q.ibs_value:.2f}</b> — {ibs_label}\n"

        # Donchian Channel
        if q.donchian_upper > 0 and q.donchian_lower > 0:
            msg += f"  📦 Donchian: Floor {_fmt_price(q.donchian_lower)} | Ceiling {_fmt_price(q.donchian_upper)}\n"

        # Pivot Point
        if q.pivot_center > 0:
            msg += f"  🎯 Pivot: C={_fmt_price(q.pivot_center)} | R1={_fmt_price(q.pivot_r1)} | S1={_fmt_price(q.pivot_s1)}\n"

        # Risk-adjusted momentum
        if abs(q.risk_adj_return) > 0.05:
            radj_emoji = "📈" if q.risk_adj_return > 0 else "📉"
            msg += f"  {radj_emoji} Risk-Adj Momentum: <b>{q.risk_adj_return:.2f}</b>\n"

        # Residual momentum
        if abs(q.residual_mom) > 0.02:
            res_emoji = "⚡" if q.residual_mom > 0 else "⬇️"
            msg += f"  {res_emoji} Residual Momentum (alpha): <b>{q.residual_mom:.3f}</b>\n"

        msg += "\n"

    # ── Reasoning Log ─────────────────────────────────────
    if sig.reasoning_log:
        msg += f"🧠 <b>Reasoning Log:</b>\n"
        for reason in sig.reasoning_log[:12]:
            msg += f"  • {reason}\n"
        msg += "\n"

    # ── Links ─────────────────────────────────────────────
    base = sig.symbol.replace("USDT","")
    msg += f"🔗 "
    msg += f"<a href='https://www.binance.com/en/trade/{sig.symbol}'>Binance</a> | "
    msg += f"<a href='https://www.tradingview.com/chart/?symbol=BINANCE:{sig.symbol}'>TradingView</a> | "
    msg += f"<a href='https://coinmarketcap.com/currencies/{base.lower()}/'>CMC</a>\n\n"

    msg += f"⚡ <i>CMI-ASS Bot v2 | by Furiooseventh</i>\n"
    msg += f"⚠️ <i>BUKAN financial advice. Always DYOR & manage risk!</i>"
    msg += f"⚠️ <i>NEK LAGI TRADE OJO NGELAMUN</i>"

    return _send(msg)


def send_scan_summary(total_scanned: int, signals: list, fear_greed: dict) -> bool:
    fg_val   = fear_greed.get("value", 50)
    fg_label = fear_greed.get("label", "Neutral")
    fg_emoji = "😱" if fg_val < 30 else ("🤑" if fg_val > 70 else "😐")

    msg  = f"✅ <b>CMI-ASS v2 Scan Complete</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Coin dianalisis: <b>{total_scanned}</b>\n"
    msg += f"🎯 Sinyal ditemukan: <b>{len(signals)}</b>\n"
    msg += f"{fg_emoji} Fear & Greed: <b>{fg_val}/100</b> ({fg_label})\n"

    if signals:
        msg += f"\n<b>Top Signals:</b>\n"
        sorted_sigs = sorted(signals, key=lambda x: x.confidence_score, reverse=True)
        for s in sorted_sigs[:5]:
            se = {"LONG":"🟢","SHORT":"🔴","BUY SPOT":"🔵","WATCH":"🟡"}.get(s.signal_type,"⚪")
            tbe = {"STRONG_BULL":"🚀","BULL":"📈","NEUTRAL":"⚪","BEAR":"📉","STRONG_BEAR":"🔻"}.get(s.ta_bias,"⚪")
            msg += (f"  {se} {s.symbol.replace('USDT','')} | "
                    f"{s.signal_type} | Score: {s.confidence_score:.0f} | "
                    f"TA: {tbe}{s.ta_bias}\n")
    else:
        msg += "\n🔍 Belum ada anomali signifikan saat ini."

    return _send(msg)


def send_startup_message() -> bool:
    msg  = "🚀 <b>CMI-ASS v3 Scanner AKTIF</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "✅ Screener berhasil dijalankan\n"
    msg += "🔍 Memulai scanning pasar crypto...\n"
    msg += "📡 Data: Bitget + MEXC (cloud-safe)\n"
    msg += "🐋 Whale Sonar: ON\n"
    msg += "📈 Derivatives Engine: ON\n"
    msg += "🏦 Supply Analyzer: ON\n"
    msg += "🔥 Pre-Pump Detector: ON\n"
    msg += "📊 Technical Analysis (SMC v3): ON\n"
    msg += "  ↳ RSI · MACD · EMA · BB · Stoch\n"
    msg += "  ↳ FVG · Order Block · BOS/CHoCH\n"
    msg += "  ↳ Multi-Timeframe Confluence Gate\n"
    msg += "🏷️ Narrative Engine: ON\n"
    msg += "  ↳ 150+ tickers → 20+ narasi (AI, RWA, DeFi...)\n"
    msg += "  ↳ Hype Score real-time per sektor\n"
    msg += "🔬 <b>Quant Engine (151 Trading Strategies): ON ← NEW</b>\n"
    msg += "  ↳ Price Momentum (§3.1) — risk-adjusted\n"
    msg += "  ↳ Residual Momentum / Alpha (§3.7)\n"
    msg += "  ↳ Low Volatility Anomaly (§3.4)\n"
    msg += "  ↳ Triple Moving Average (§3.13)\n"
    msg += "  ↳ Pivot Point S/R (§3.14)\n"
    msg += "  ↳ Donchian Channel Breakout (§3.15)\n"
    msg += "  ↳ IBS Mean Reversion (§4.4)\n"
    msg += "  ↳ Dual Momentum Filter (§4.1.2)\n"
    msg += "  ↳ Trend Following + tanh (§10.4)\n"
    msg += "  ↳ Contrarian + Volume (§10.3.1)\n"
    msg += "\n⏳ Tunggu hasil scan dalam beberapa menit..."
    return _send(msg)
