"""
Telegram Gateway — CMI-ASS v6
Multi-chat support + Chart Photo + DefiLlama + Precision Entry
"""
import requests
import logging
from io import BytesIO
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger    = logging.getLogger(__name__)
_CHAT_IDS = [c.strip() for c in str(TELEGRAM_CHAT_ID).split(",") if c.strip()]

# ─────────────────────────────────────────────────────────
#  LOW-LEVEL SENDERS
# ─────────────────────────────────────────────────────────

def _send(msg: str, parse_mode: str = "HTML") -> bool:
    if not _CHAT_IDS: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    ok = False
    for cid in _CHAT_IDS:
        try:
            r = requests.post(url, json={
                "chat_id": cid, "text": msg,
                "parse_mode": parse_mode, "disable_web_page_preview": True
            }, timeout=10)
            if r.status_code == 200: ok = True
            else: logger.warning(f"Telegram {cid}: {r.status_code}")
        except Exception as e:
            logger.error(f"Telegram {cid}: {e}")
    return ok


def _send_photo(image_bytes: bytes, caption: str) -> bool:
    """Kirim foto ke semua chat ID."""
    if not _CHAT_IDS: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    ok = False
    for cid in _CHAT_IDS:
        try:
            files = {"photo": ("chart.jpg", BytesIO(image_bytes), "image/jpeg")}
            data  = {"chat_id": cid, "caption": caption[:1024], "parse_mode": "HTML"}
            r = requests.post(url, files=files, data=data, timeout=20)
            if r.status_code == 200:
                ok = True
                logger.info(f"Chart photo sent → {cid}")
            else:
                logger.warning(f"sendPhoto {cid}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            logger.error(f"sendPhoto {cid}: {e}")
    return ok


def _p(v: float) -> str:
    if v >= 1000:   return f"${v:,.2f}"
    elif v >= 1:    return f"${v:.4f}"
    elif v >= 0.01: return f"${v:.5f}"
    else:           return f"${v:.8f}"


# ─────────────────────────────────────────────────────────
#  CHART CAPTION (singkat, untuk foto)
# ─────────────────────────────────────────────────────────

def _build_chart_caption(sig, vr_llama=None, entry_timing=None) -> str:
    base = sig.symbol.replace("USDT", "")
    se   = {"LONG": "🟢", "SHORT": "🔴", "BUY SPOT": "🔵", "WATCH": "🟡"}.get(sig.signal_type, "⚪")
    le   = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sig.alert_level, "⚪")
    tbe  = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "⚪", "BEAR": "📉", "STRONG_BEAR": "🔻"}.get(sig.ta_bias, "⚪")

    c  = f"{le} <b>${base} — {sig.signal_type}</b> | Score: {sig.confidence_score:.0f}/100\n"
    c += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    c += f"💰 {_p(sig.price)}  |  {tbe} {sig.ta_bias}  |  RSI: {sig.rsi_14:.0f}\n"

    # Whale + Entry Timing
    ec = sig.extra_context if hasattr(sig, "extra_context") else {}
    if entry_timing:
        phase = entry_timing.whale_phase
        timing = entry_timing.entry_timing
        phase_e = {"MARKUP_IMMINENT": "🚀", "ACCUMULATING": "🐋", "DISTRIBUTING": "🔴", "NEUTRAL": "⚪"}.get(phase, "⚪")
        timing_e = {"NOW": "⚡", "WAIT_DIP": "⏳", "WAIT_BREAKOUT": "⏳", "AVOID": "🚫"}.get(timing, "❓")
        c += f"{phase_e} Whale: <b>{phase.replace('_',' ')}</b>  |  {timing_e} Entry: <b>{timing.replace('_',' ')}</b>\n"
        if timing == "WAIT_DIP" and entry_timing.wait_target_price > 0:
            c += f"⏳ Tunggu pullback ke <b>{_p(entry_timing.wait_target_price)}</b>\n"
        if entry_timing.wyckoff_phase not in ("UNKNOWN", "RANGING", "NEUTRAL"):
            c += f"📐 Wyckoff: <b>{entry_timing.wyckoff_phase}</b>\n"

    # Key levels
    c += f"🎯 Entry: {_p(sig.entry_zone_low)} – {_p(sig.entry_zone_high)}\n"
    c += f"🛡️ SL: {_p(sig.stop_loss)}  |  TP1: {_p(sig.tp1)}  |  TP3: {_p(sig.tp3)}\n"

    # SMC
    if ec.get("has_precise_entry"):
        el = ec.get("optimal_entry_low", 0)
        eh = ec.get("optimal_entry_high", 0)
        if el > 0:
            c += f"🧩 SMC Entry: {_p(el)} – {_p(eh)}\n"
    if ec.get("liquidity_swept"):
        c += "💧 Liquidity Swept ✅\n"
    if ec.get("in_discount_zone"):
        c += "💚 ICT Discount Zone ✅\n"
    mb = ec.get("mtf_bias", "")
    if mb:
        c += f"🔭 MTF: {mb} ({ec.get('mtf_agree_count',0)}/4)\n"

    # DefiLlama
    if vr_llama and vr_llama.found and vr_llama.overall_verdict in ("EXTREME_UNDERVALUE", "UNDERVALUE"):
        verd = vr_llama.overall_verdict.replace("_", " ")
        if vr_llama.ps_ratio < 999:
            c += f"💚 P/S: {vr_llama.ps_ratio:.1f}x ({verd})\n"
        if vr_llama.tvl_mc_ratio > 0:
            c += f"🏦 TVL/MCap: {vr_llama.tvl_mc_ratio:.2f}x\n"

    # MACD
    macd_e = {"BULLISH_CROSS": "💚", "BEARISH_CROSS": "🔴", "BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "⚪"}.get(sig.macd_signal_type, "⚪")
    c += f"MACD: {macd_e}{sig.macd_signal_type}  |  R/R: 1:{sig.risk_reward:.1f}\n"
    c += f"\n⚡ CMI-ASS v6 | {sig.alert_level} | DYOR!"
    return c


# ─────────────────────────────────────────────────────────
#  CHART LINKS (selalu tersedia bahkan tanpa screenshot)
# ─────────────────────────────────────────────────────────

def _chart_links(symbol: str) -> str:
    base = f"https://www.tradingview.com/chart/?symbol=BITGET:{symbol}&interval="
    links = [
        f"<a href='{base}60'>📊 1H</a>",
        f"<a href='{base}240'>📊 4H</a>",
        f"<a href='{base}D'>📊 1D</a>",
        f"<a href='https://coinmarketcap.com/currencies/{symbol.replace('USDT','').lower()}/'>CMC</a>",
    ]
    return " | ".join(links)


# ─────────────────────────────────────────────────────────
#  MAIN SIGNAL SENDER
# ─────────────────────────────────────────────────────────

def format_and_send_signal(sig, vr_llama=None, entry_timing=None) -> bool:
    """
    Kirim sinyal ke Telegram:
    1. Foto chart TradingView (jika screenshot tersedia)
    2. Pesan text lengkap dengan semua analisis
    """
    se = {"LONG": "🟢", "SHORT": "🔴", "BUY SPOT": "🔵", "WATCH": "🟡", "NEUTRAL": "⚪"}.get(sig.signal_type, "⚪")
    le = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sig.alert_level, "⚪")
    base = sig.symbol.replace("USDT", "")

    # ── 1. KIRIM CHART PHOTO ──────────────────────────────
    chart_sent = False
    try:
        from engine.chart_engine import get_chart_image
        img_bytes = get_chart_image(sig.symbol, timeframe="4h")
        if img_bytes:
            caption   = _build_chart_caption(sig, vr_llama, entry_timing)
            chart_sent = _send_photo(img_bytes, caption)
    except Exception as e:
        logger.warning(f"Chart fetch/send error: {e}")

    # ── 2. PESAN TEXT LENGKAP ─────────────────────────────
    msg  = f"{le} <b>CMI-ASS v6 SIGNAL</b> {le}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"{se} <b>{sig.signal_type}</b>  |  <b>${base}</b>\n"
    msg += f"🎯 Confidence: <b>{sig.confidence_score:.0f}/100</b>  |  {sig.alert_level}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    chg_e = "📈" if sig.price_change_24h >= 0 else "📉"
    msg += f"💰 <b>Harga:</b> {_p(sig.price)}\n"
    msg += f"{chg_e} <b>24H:</b> {sig.price_change_24h:+.2f}%\n"
    msg += f"💹 <b>Volume 24H:</b> ${sig.volume_24h:,.0f}\n"
    msg += f"🔥 <b>Volume Spike:</b> {sig.volume_spike:.1f}x avg 7D\n"
    if sig.market_cap:
        msg += f"🏦 <b>Market Cap:</b> ${sig.market_cap:,.0f} ({sig.supply_category})\n"
        msg += f"⚖️ <b>Selling Pressure:</b> {sig.selling_pressure}\n"
    msg += "\n"

    if sig.funding_rate is not None:
        fr_e = "🟢" if sig.funding_rate < 0 else "🔴"
        msg += f"{fr_e} <b>Funding Rate:</b> {sig.funding_rate*100:.4f}%\n"
        msg += f"📊 <b>Short Squeeze Risk:</b> {sig.short_squeeze_risk}\n"
    if sig.oi_change is not None:
        oi_e = "📈" if sig.oi_change > 0 else "📉"
        msg += f"{oi_e} <b>OI Change 24H:</b> {sig.oi_change:+.1f}%\n"
    if sig.funding_rate is not None or sig.oi_change is not None:
        msg += "\n"

    # ── DefiLlama Fundamental ─────────────────────────────
    if vr_llama and vr_llama.found:
        try:
            from engine.defillama_engine import format_defillama_section
            msg += format_defillama_section(vr_llama)
        except Exception as e:
            logger.debug(f"DefiLlama format error: {e}")

    # ── Precision Entry Timing ────────────────────────────
    if entry_timing:
        try:
            from engine.precision_entry_engine import format_entry_timing_section
            msg += format_entry_timing_section(entry_timing, sig.price)
        except Exception as e:
            logger.debug(f"PrecisionEntry format error: {e}")

    # ── Technical Analysis ────────────────────────────────
    tbe = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "⚪", "BEAR": "📉", "STRONG_BEAR": "🔻"}.get(sig.ta_bias, "⚪")
    msg += f"📊 <b>Technical Analysis — SMC v3</b>\n"
    msg += f"  {tbe} <b>TA Bias:</b> {sig.ta_bias}\n"
    msg += f"  📈 <b>RSI(14):</b> {sig.rsi_14:.1f}"
    if sig.rsi_14 <= 30:   msg += " ⬇️ OVERSOLD\n"
    elif sig.rsi_14 >= 70: msg += " ⬆️ OVERBOUGHT\n"
    else:                   msg += "\n"
    macd_e = {"BULLISH_CROSS": "💚", "BEARISH_CROSS": "🔴", "BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "⚪"}.get(sig.macd_signal_type, "⚪")
    msg += f"  {macd_e} <b>MACD:</b> {sig.macd_signal_type}\n"
    tr_e = {"STRONG_BULL": "🚀", "BULL": "📈", "SIDEWAYS": "↔️", "BEAR": "📉", "STRONG_BEAR": "🔻"}.get(sig.trend_alignment, "⚪")
    msg += f"  {tr_e} <b>EMA Trend:</b> {sig.trend_alignment}\n"
    if sig.bb_squeeze:      msg += f"  🌀 <b>BB Squeeze AKTIF</b> — volatility sangat rendah!\n"
    if sig.dominant_pattern: msg += f"  📐 <b>Pattern:</b> {sig.dominant_pattern}\n"
    if sig.nearest_support > 0: msg += f"  🟢 <b>Support:</b> {_p(sig.nearest_support)}\n"
    if sig.nearest_resist > 0:  msg += f"  🔴 <b>Resistance:</b> {_p(sig.nearest_resist)}\n"
    msg += "\n"

    ec = sig.extra_context if hasattr(sig, "extra_context") and sig.extra_context else {}
    smc = []
    mb = ec.get("mtf_bias", ""); mn = ec.get("mtf_agree_count", 0)
    if mb: smc.append(f"🔭 MTF: <b>{mb}</b> ({mn}/4 TF setuju)")
    if ec.get("liquidity_swept"):    smc.append("💧 <b>Liquidity Swept</b> — stop hunt selesai!")
    if ec.get("in_discount_zone"):   smc.append("💚 ICT: <b>Discount Zone</b> — zona ideal beli")
    elif ec.get("in_premium_zone"):  smc.append("⚠️ ICT: <b>Premium Zone</b> — harga mahal")
    ol = ec.get("optimal_entry_low", 0); oh = ec.get("optimal_entry_high", 0)
    if ec.get("has_precise_entry") and ol > 0:
        smc.append(f"🎯 SMC Entry: {_p(ol)} – {_p(oh)}")
    ss = ec.get("structure_sl", 0)
    if ss > 0: smc.append(f"🛡️ Structure SL: {_p(ss)}")
    if smc:
        msg += "🧩 <b>Smart Money Context</b>\n"
        for line in smc: msg += f"  {line}\n"
        msg += "\n"

    inv = ec.get("invalidation", [])
    if inv:
        msg += "⚠️ <b>Invalidation Risks:</b>\n"
        for i in inv[:2]: msg += f"  {i}\n"
        msg += "\n"

    # ── Quant Section ─────────────────────────────────────
    q = ec.get("quant")
    if q is not None:
        q_score = getattr(q, "score", 0)
        q_cnt   = getattr(q, "confluence_count", 0)
        msg += f"🔬 <b>Quant Engine</b> <i>(151 Strategies)</i> — Score: <b>{q_score:.0f}/40</b>\n"
        ma_bias = getattr(q, "triple_ma_bias", "NEUTRAL")
        ma_e = {"BULL": "🔺", "BEAR": "🔻", "NEUTRAL": "↔️"}.get(ma_bias, "↔️")
        e9 = getattr(q, "ema9", 0); e21 = getattr(q, "ema21", 0)
        if e9 > 0 and e21 > 0:
            msg += f"  {ma_e} Triple MA: <b>{ma_bias}</b> | EMA9={_p(e9)} EMA21={_p(e21)}\n"
        eta = getattr(q, "trend_eta", 0)
        if abs(eta) > 0.05:
            tr_dir = "🚀 BULLISH" if eta > 0 else "🔻 BEARISH"
            msg += f"  📡 Trend Signal: <b>{eta:+.2f}</b> {tr_dir}\n"
        dp = getattr(q, "dual_mom_pass", False)
        msg += f"  {'✅' if dp else '❌'} Dual Momentum: <b>{'PASS' if dp else 'FAIL'}</b>\n"
        ibs = getattr(q, "ibs_value", 0.5)
        ibs_lbl = "OVERSOLD 💚" if ibs < 0.25 else ("OVERBOUGHT 🔴" if ibs > 0.75 else "NEUTRAL ↔️")
        msg += f"  📊 IBS: <b>{ibs:.2f}</b> — {ibs_lbl}\n"
        pc = getattr(q, "pivot_c", 0); pr1 = getattr(q, "pivot_r1", 0); ps1 = getattr(q, "pivot_s1", 0)
        if pc > 0:
            msg += f"  🎯 Pivot: C={_p(pc)} | R1={_p(pr1)} | S1={_p(ps1)}\n"
        msg += f"  🏆 Confluence: <b>{q_cnt}/10</b> strategi bullish\n\n"

    # ── Risk Management ───────────────────────────────────
    msg += f"🎯 <b>ENTRY ZONE:</b> {_p(sig.entry_zone_low)} – {_p(sig.entry_zone_high)}\n"
    msg += f"🛡️ <b>STOP LOSS:</b> {_p(sig.stop_loss)}\n"
    sl_pct = abs(sig.price - sig.stop_loss) / sig.price * 100 if sig.price > 0 else 0
    msg += f"   ↳ SL distance: {sl_pct:.1f}% from entry\n"
    msg += f"✅ <b>TP1:</b> {_p(sig.tp1)}  (+{abs(sig.tp1-sig.price)/sig.price*100:.1f}%)\n"
    msg += f"✅ <b>TP2:</b> {_p(sig.tp2)}  (+{abs(sig.tp2-sig.price)/sig.price*100:.1f}%)\n"
    msg += f"🚀 <b>TP3:</b> {_p(sig.tp3)}  (+{abs(sig.tp3-sig.price)/sig.price*100:.1f}%)\n"
    msg += f"⚖️ <b>Risk/Reward:</b> 1:{sig.risk_reward:.1f}\n\n"

    # ── Score Breakdown ───────────────────────────────────
    q_score  = getattr(sig, "quant_score", 0.0)
    ll_score = getattr(sig, "llama_score", 0.0)
    msg += f"📋 <b>Score Breakdown:</b>\n"
    msg += f"  🐋 Whale:         {sig.whale_score:.0f}/25\n"
    msg += f"  📈 Derivatives:   {sig.derivatives_score:.0f}/30\n"
    msg += f"  🏦 Supply:        {sig.supply_score:.0f}/20\n"
    msg += f"  🔥 Pre-Pump:      {sig.pre_pump_score:.0f}/25\n"
    msg += f"  📊 Technical:     {sig.ta_score:.0f}/30\n"
    msg += f"  🔬 Quant(151):    {q_score:.0f}/40\n"
    if ll_score > 0:
        msg += f"  💚 Fundamental:   {ll_score:.0f}/35\n"
    msg += f"  📌 Total:         {sig.confidence_score:.0f}/100\n\n"

    # ── Prosecution Case ──────────────────────────────────
    ec2 = sig.extra_context if hasattr(sig, "extra_context") and sig.extra_context else {}
    case = ec2.get("prosecution")
    if case is not None:
        v_verdict = ec2.get("validator_verdict", "APPROVED")
        v_emoji   = {"APPROVED": "✅", "DOWNGRADED": "⚡", "REJECTED": "🚫"}.get(v_verdict, "✅")
        adj = getattr(case, "score_adjustment", 0.0)
        adj_str = f"+{adj:.1f}" if adj >= 0 else f"{adj:.1f}"
        ep  = getattr(case, "entry_precision", "FAIR")
        ep_emoji = {"EXCELLENT": "🎯", "GOOD": "✅", "FAIR": "⚠️", "POOR": "❌"}.get(ep, "⚠️")
        ep_dist = getattr(case, "entry_dist_pct", 0.0)
        rr_v = getattr(case, "rr_verdict", "OK")
        sl_q = getattr(case, "sl_quality", "OK")
        msg += f"{v_emoji} <b>Signal Verdict: {v_verdict}</b>"
        if v_verdict == "DOWNGRADED":
            msg += f" (score dikoreksi {adj_str})"
        msg += f"\n"
        thesis = getattr(case, "thesis", "")
        if thesis:
            msg += f"💡 <b>Thesis:</b> {thesis[:200]}\n\n"
        key_r = getattr(case, "key_reasons", [])[:4]
        if key_r:
            msg += f"📌 <b>Alasan UTAMA:</b>\n"
            for i, r in enumerate(key_r, 1):
                msg += f"  {i}. {r[:130]}\n"
            msg += "\n"
        msg += f"{ep_emoji} <b>Entry Precision:</b> {ep}"
        if ep_dist > 0:
            msg += f" ({ep_dist:.1f}% dari struktur kunci)"
        msg += "\n"
        rr_e = "✅" if rr_v == "OK" else ("⚠️" if rr_v == "TIGHT" else "❌")
        sl_e = "✅" if sl_q == "OK" else "⚠️"
        msg += f"{rr_e} R/R: {sig.risk_reward:.1f}  |  {sl_e} SL quality: {sl_q}\n"
        key_k = getattr(case, "key_risks", [])[:3]
        if key_k:
            msg += f"\n⚠️ <b>Risiko Utama:</b>\n"
            for r in key_k:
                msg += f"  • {r[:120]}\n"
        msg += "\n"

    if sig.reasoning_log:
        msg += f"🧠 <b>Signal Evidence:</b>\n"
        for r in sig.reasoning_log[:8]: msg += f"  • {r}\n"
        msg += "\n"

    # ── Chart Links ───────────────────────────────────────
    msg += f"📈 Chart: {_chart_links(sig.symbol)}\n\n"
    if chart_sent:
        msg += f"🖼️ <i>Chart 4H dikirim di atas ↑</i>\n"
    msg += f"⚡ <i>CMI-ASS v6 | DefiLlama+SMC+Whale+Quant | by Furiooseventh</i>\n"
    msg += f"⚠️ <i>BUKAN financial advice. DYOR & manage risk!</i>\n"
    msg += f"⚠️ <i>NEK LAGI TRADE OJO NGELAMUN</i>"

    return _send(msg)


# ─────────────────────────────────────────────────────────
#  SUMMARY & STARTUP
# ─────────────────────────────────────────────────────────

def send_scan_summary(total_scanned: int, signals: list, fear_greed: dict) -> bool:
    fg_val   = fear_greed.get("value", 50)
    fg_label = fear_greed.get("label", "Neutral")
    fg_emoji = "😱" if fg_val < 30 else ("🤑" if fg_val > 70 else "😐")

    msg  = f"✅ <b>CMI-ASS v6 Scan Complete</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Dianalisis: <b>{total_scanned}</b> coins\n"
    msg += f"🎯 Sinyal: <b>{len(signals)}</b>\n"
    msg += f"{fg_emoji} Fear & Greed: <b>{fg_val}/100</b> ({fg_label})\n"
    if signals:
        msg += f"\n<b>Top Signals:</b>\n"
        for s in sorted(signals, key=lambda x: x.confidence_score, reverse=True)[:5]:
            se  = {"LONG": "🟢", "SHORT": "🔴", "BUY SPOT": "🔵", "WATCH": "🟡"}.get(s.signal_type, "⚪")
            tbe = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "⚪", "BEAR": "📉", "STRONG_BEAR": "🔻"}.get(s.ta_bias, "⚪")
            qs  = getattr(s, "quant_score", 0)
            ls  = getattr(s, "llama_score", 0)
            ll_str = f" | 💚F:{ls:.0f}" if ls > 5 else ""
            msg += f"  {se} {s.symbol.replace('USDT','')} | {s.signal_type} | {s.confidence_score:.0f}/100 | TA:{tbe}{s.ta_bias} | Q:{qs:.0f}{ll_str}\n"
    else:
        msg += "\n🔍 Belum ada anomali signifikan saat ini."
    return _send(msg)


def send_startup_message() -> bool:
    msg  = "🚀 <b>CMI-ASS v6 Scanner AKTIF</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🐋 Whale Sonar: ON\n"
    msg += "📈 Derivatives Engine: ON\n"
    msg += "🏦 Supply Analyzer: ON\n"
    msg += "🔥 Pre-Pump Detector: ON\n"
    msg += "📊 Technical Analysis SMC v3: ON\n"
    msg += "  ↳ FVG · OB · BOS/CHoCH · MTF Confluence\n"
    msg += "🔬 Quant Engine (151 Strategies): ON\n"
    msg += "💚 <b>DefiLlama Fundamental Engine: ON ← NEW</b>\n"
    msg += "  ↳ P/S Ratio · P/F Ratio · TVL/MCap · Revenue Growth\n"
    msg += "🎯 <b>Precision Entry Engine: ON ← NEW</b>\n"
    msg += "  ↳ Wyckoff Phase · Whale Phase · Entry Timing Matrix\n"
    msg += "  ↳ MARKUP_IMMINENT · ACCUMULATING · DISTRIBUTING\n"
    msg += "🖼️ <b>TradingView Chart Screenshot: ON ← NEW</b>\n"
    msg += "  ↳ Chart 4H dikirim bersama setiap alert\n"
    msg += "\n⏳ Scanning dalam beberapa menit..."
    return _send(msg)
