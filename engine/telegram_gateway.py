"""
Telegram Gateway — CMI-ASS v4
Multi-chat support + Quant section
"""
import requests, logging
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)
_CHAT_IDS = [c.strip() for c in str(TELEGRAM_CHAT_ID).split(",") if c.strip()]

def _send(msg: str, parse_mode: str = "HTML") -> bool:
    if not _CHAT_IDS: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    ok = False
    for cid in _CHAT_IDS:
        try:
            r = requests.post(url, json={"chat_id":cid,"text":msg,"parse_mode":parse_mode,"disable_web_page_preview":True}, timeout=10)
            if r.status_code == 200: ok = True
            else: logger.warning(f"Telegram {cid}: {r.status_code}")
        except Exception as e: logger.error(f"Telegram {cid}: {e}")
    return ok

def _p(v: float) -> str:
    if v >= 1000:   return f"${v:,.2f}"
    elif v >= 1:    return f"${v:.4f}"
    elif v >= 0.01: return f"${v:.5f}"
    else:           return f"${v:.8f}"

def format_and_send_signal(sig) -> bool:
    se = {"LONG":"🟢","SHORT":"🔴","BUY SPOT":"🔵","WATCH":"🟡","NEUTRAL":"⚪"}.get(sig.signal_type,"⚪")
    le = {"CRITICAL":"🚨","HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(sig.alert_level,"⚪")
    base = sig.symbol.replace("USDT","")

    msg  = f"{le} <b>CMI-ASS v4 SIGNAL</b> {le}\n"
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

    tbe = {"STRONG_BULL":"🚀","BULL":"📈","NEUTRAL":"⚪","BEAR":"📉","STRONG_BEAR":"🔻"}.get(sig.ta_bias,"⚪")
    msg += f"📊 <b>Technical Analysis — SMC v3</b>\n"
    msg += f"  {tbe} <b>TA Bias:</b> {sig.ta_bias}\n"
    msg += f"  📈 <b>RSI(14):</b> {sig.rsi_14:.1f}"
    if sig.rsi_14<=30: msg += " ⬇️ OVERSOLD\n"
    elif sig.rsi_14>=70: msg += " ⬆️ OVERBOUGHT\n"
    else: msg += "\n"
    macd_e = {"BULLISH_CROSS":"💚","BEARISH_CROSS":"🔴","BULLISH":"📈","BEARISH":"📉","NEUTRAL":"⚪"}.get(sig.macd_signal_type,"⚪")
    msg += f"  {macd_e} <b>MACD:</b> {sig.macd_signal_type}\n"
    tr_e = {"STRONG_BULL":"🚀","BULL":"📈","SIDEWAYS":"↔️","BEAR":"📉","STRONG_BEAR":"🔻"}.get(sig.trend_alignment,"⚪")
    msg += f"  {tr_e} <b>EMA Trend:</b> {sig.trend_alignment}\n"
    if sig.bb_squeeze: msg += f"  🌀 <b>BB Squeeze AKTIF</b> — volatility sangat rendah!\n"
    if sig.dominant_pattern: msg += f"  📐 <b>Pattern:</b> {sig.dominant_pattern}\n"
    if sig.nearest_support>0: msg += f"  🟢 <b>Support:</b> {_p(sig.nearest_support)}\n"
    if sig.nearest_resist>0:  msg += f"  🔴 <b>Resistance:</b> {_p(sig.nearest_resist)}\n"
    msg += "\n"

    ec = sig.extra_context if hasattr(sig,"extra_context") and sig.extra_context else {}
    # SMC Context
    smc = []
    mb = ec.get("mtf_bias",""); mn = ec.get("mtf_agree_count",0)
    if mb: smc.append(f"🔭 MTF: <b>{mb}</b> ({mn}/4 TF setuju)")
    if ec.get("liquidity_swept"): smc.append("💧 <b>Liquidity Swept</b> — stop hunt selesai!")
    if ec.get("in_discount_zone"): smc.append("💚 ICT: <b>Discount Zone</b> — zona ideal beli")
    elif ec.get("in_premium_zone"): smc.append("⚠️ ICT: <b>Premium Zone</b> — harga mahal")
    ol=ec.get("optimal_entry_low",0); oh=ec.get("optimal_entry_high",0)
    if ec.get("has_precise_entry") and ol>0:
        smc.append(f"🎯 SMC Entry: {_p(ol)} – {_p(oh)}")
    ss=ec.get("structure_sl",0)
    if ss>0: smc.append(f"🛡️ Structure SL: {_p(ss)}")
    if smc:
        msg += "🧩 <b>Smart Money Context</b>\n"
        for line in smc: msg += f"  {line}\n"
        msg += "\n"

    inv = ec.get("invalidation",[])
    if inv:
        msg += "⚠️ <b>Invalidation Risks:</b>\n"
        for i in inv[:2]: msg += f"  {i}\n"
        msg += "\n"

    # ── Quant Engine Section ───────────────────────────────────────────────
    q = ec.get("quant")
    if q is not None:
        q_score = getattr(q,"score",0)
        q_cnt   = getattr(q,"confluence_count",0)
        msg += f"🔬 <b>Quant Engine</b> <i>(151 Trading Strategies)</i> — Score: <b>{q_score:.0f}/40</b>\n"

        # Triple MA (§3.13)
        ma_bias = getattr(q,"triple_ma_bias","NEUTRAL")
        ma_e = {"BULL":"🔺","BEAR":"🔻","NEUTRAL":"↔️"}.get(ma_bias,"↔️")
        e9=getattr(q,"ema9",0); e21=getattr(q,"ema21",0); e50=getattr(q,"ema50",0)
        if e9>0 and e21>0 and e50>0:
            msg += f"  {ma_e} Triple MA (§3.13): <b>{ma_bias}</b> | EMA9={_p(e9)} EMA21={_p(e21)}\n"
        else:
            msg += f"  {ma_e} Triple MA (§3.13): <b>{ma_bias}</b>\n"

        # Trend following tanh (§10.4)
        eta = getattr(q,"trend_eta",0)
        if abs(eta) > 0.05:
            tr_dir = "🚀 BULLISH" if eta > 0 else "🔻 BEARISH"
            msg += f"  📡 Trend Signal (§10.4): <b>{eta:+.2f}</b> {tr_dir}\n"

        # Dual Momentum (§4.1.2)
        dp = getattr(q,"dual_mom_pass",False)
        msg += f"  {'✅' if dp else '❌'} Dual Momentum (§4.1.2): <b>{'PASS' if dp else 'FAIL'}</b>\n"

        # IBS (§4.4)
        ibs = getattr(q,"ibs_value",0.5)
        ibs_lbl = "OVERSOLD 💚" if ibs<0.25 else ("OVERBOUGHT 🔴" if ibs>0.75 else "NEUTRAL ↔️")
        msg += f"  📊 IBS (§4.4): <b>{ibs:.2f}</b> — {ibs_lbl}\n"

        # Pivot Point (§3.14)
        pc=getattr(q,"pivot_c",0); pr1=getattr(q,"pivot_r1",0); ps1=getattr(q,"pivot_s1",0)
        if pc>0:
            msg += f"  🎯 Pivot (§3.14): C={_p(pc)} | R1={_p(pr1)} | S1={_p(ps1)}\n"

        # Donchian Channel (§3.15)
        du=getattr(q,"donchian_upper",0); dl=getattr(q,"donchian_lower",0)
        if du>0 and dl>0:
            msg += f"  📦 Donchian (§3.15): Floor={_p(dl)} | Ceiling={_p(du)}\n"

        # Momentum metrics
        radj = getattr(q,"risk_adj_return",0)
        resid= getattr(q,"residual_mom",0)
        if abs(radj)>0.05:
            msg += f"  {'📈' if radj>0 else '📉'} Risk-Adj Momentum (§3.1): <b>{radj:.2f}</b>\n"
        if abs(resid)>0.002:
            msg += f"  {'⚡' if resid>0 else '⬇️'} Alpha/Residual (§3.7): <b>{resid:.4f}</b>\n"

        msg += f"  🏆 Confluence: <b>{q_cnt}/10</b> strategi bullish\n\n"

    # Risk Management
    msg += f"🎯 <b>ENTRY ZONE:</b> {_p(sig.entry_zone_low)} – {_p(sig.entry_zone_high)}\n"
    msg += f"🛡️ <b>STOP LOSS:</b> {_p(sig.stop_loss)}\n"
    sl_pct = abs(sig.price-sig.stop_loss)/sig.price*100 if sig.price>0 else 0
    msg += f"   ↳ SL distance: {sl_pct:.1f}% from entry\n"
    msg += f"✅ <b>TP1:</b> {_p(sig.tp1)}  (+{abs(sig.tp1-sig.price)/sig.price*100:.1f}%)\n"
    msg += f"✅ <b>TP2:</b> {_p(sig.tp2)}  (+{abs(sig.tp2-sig.price)/sig.price*100:.1f}%)\n"
    msg += f"🚀 <b>TP3:</b> {_p(sig.tp3)}  (+{abs(sig.tp3-sig.price)/sig.price*100:.1f}%)\n"
    msg += f"⚖️ <b>Risk/Reward:</b> 1:{sig.risk_reward:.1f}\n\n"

    # Score Breakdown
    q_score = getattr(sig,"quant_score",0.0)
    msg += f"📋 <b>Score Breakdown:</b>\n"
    msg += f"  🐋 Whale:      {sig.whale_score:.0f}/25\n"
    msg += f"  📈 Derivatives:{sig.derivatives_score:.0f}/30\n"
    msg += f"  🏦 Supply:     {sig.supply_score:.0f}/20\n"
    msg += f"  🔥 Pre-Pump:   {sig.pre_pump_score:.0f}/25\n"
    msg += f"  📊 Technical:  {sig.ta_score:.0f}/30\n"
    msg += f"  🔬 Quant(151): {q_score:.0f}/40\n"
    msg += f"  📌 Total:      {sig.confidence_score:.0f}/100\n\n"

    # ── Prosecution Case (Validator v5) ──────────────────────────────────────
    ec2 = sig.extra_context if hasattr(sig,"extra_context") and sig.extra_context else {}
    case = ec2.get("prosecution")
    if case is not None:
        v_verdict  = ec2.get("validator_verdict","APPROVED")
        v_emoji    = {"APPROVED":"✅","DOWNGRADED":"⚡","REJECTED":"🚫"}.get(v_verdict,"✅")
        adj        = getattr(case,"score_adjustment",0.0)
        adj_str    = f"+{adj:.1f}" if adj>=0 else f"{adj:.1f}"
        ep         = getattr(case,"entry_precision","FAIR")
        ep_emoji   = {"EXCELLENT":"🎯","GOOD":"✅","FAIR":"⚠️","POOR":"❌"}.get(ep,"⚠️")
        ep_dist    = getattr(case,"entry_dist_pct",0.0)
        rr_v       = getattr(case,"rr_verdict","OK")
        sl_q       = getattr(case,"sl_quality","OK")

        msg += f"{v_emoji} <b>Signal Verdict: {v_verdict}</b>"
        if v_verdict == "DOWNGRADED":
            msg += f" (score dikoreksi {adj_str})"
        msg += f"\n"

        # Thesis
        thesis = getattr(case,"thesis","")
        if thesis:
            msg += f"💡 <b>Thesis:</b> {thesis[:200]}\n\n"

        # Key Reasons (top 4 strongest)
        key_r = getattr(case,"key_reasons",[])[:4]
        if key_r:
            msg += f"📌 <b>Alasan UTAMA:</b>\n"
            for i,r in enumerate(key_r,1):
                msg += f"  {i}. {r[:130]}\n"
            msg += "\n"

        # Entry precision
        msg += f"{ep_emoji} <b>Entry Precision:</b> {ep}"
        if ep_dist > 0:
            msg += f" ({ep_dist:.1f}% dari struktur kunci)"
        msg += "\n"

        # RR and SL quality
        rr_e = "✅" if rr_v=="OK" else ("⚠️" if rr_v=="TIGHT" else "❌")
        sl_e = "✅" if sl_q=="OK" else "⚠️"
        msg += f"{rr_e} R/R: {sig.risk_reward:.1f}  |  {sl_e} SL quality: {sl_q}\n"

        # Key risks (max 3)
        key_k = getattr(case,"key_risks",[])[:3]
        if key_k:
            msg += f"\n⚠️ <b>Risiko Utama:</b>\n"
            for r in key_k:
                msg += f"  • {r[:120]}\n"
        msg += "\n"

        # Downgrade reasons if any
        dg = getattr(case,"downgrade_reasons",[])[:2]
        if dg:
            msg += "🔻 <b>Catatan Validator:</b>\n"
            for d in dg: msg += f"  • {d[:100]}\n"
            msg += "\n"

    if sig.reasoning_log:
        msg += f"🧠 <b>Signal Evidence:</b>\n"
        for r in sig.reasoning_log[:10]: msg += f"  • {r}\n"
        msg += "\n"

    msg += f"🔗 <a href='https://www.tradingview.com/chart/?symbol=BITGET:{sig.symbol}'>TradingView</a> | "
    msg += f"<a href='https://coinmarketcap.com/currencies/{base.lower()}/'>CMC</a>\n\n"
    msg += f"⚡ <i>CMI-ASS v4 | Quant+SMC+Whale | by Furiooseventh</i>\n"
    msg += f"⚠️ <i>BUKAN financial advice. DYOR & manage risk!</i>\n"
    msg += f"⚠️ <i>NEK LAGI TRADE OJO NGELAMUN</i>"

    return _send(msg)


def send_scan_summary(total_scanned: int, signals: list, fear_greed: dict) -> bool:
    fg_val   = fear_greed.get("value", 50)
    fg_label = fear_greed.get("label", "Neutral")
    fg_emoji = "😱" if fg_val<30 else ("🤑" if fg_val>70 else "😐")

    msg  = f"✅ <b>CMI-ASS v4 Scan Complete</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Dianalisis: <b>{total_scanned}</b> coins\n"
    msg += f"🎯 Sinyal: <b>{len(signals)}</b>\n"
    msg += f"{fg_emoji} Fear & Greed: <b>{fg_val}/100</b> ({fg_label})\n"

    if signals:
        msg += f"\n<b>Top Signals:</b>\n"
        for s in sorted(signals, key=lambda x: x.confidence_score, reverse=True)[:5]:
            se = {"LONG":"🟢","SHORT":"🔴","BUY SPOT":"🔵","WATCH":"🟡"}.get(s.signal_type,"⚪")
            tbe= {"STRONG_BULL":"🚀","BULL":"📈","NEUTRAL":"⚪","BEAR":"📉","STRONG_BEAR":"🔻"}.get(s.ta_bias,"⚪")
            qs = getattr(s,"quant_score",0)
            msg += f"  {se} {s.symbol.replace('USDT','')} | {s.signal_type} | {s.confidence_score:.0f}/100 | TA:{tbe}{s.ta_bias} | Q:{qs:.0f}\n"
    else:
        msg += "\n🔍 Belum ada anomali signifikan saat ini."
    return _send(msg)


def send_startup_message() -> bool:
    msg  = "🚀 <b>CMI-ASS v4 Scanner AKTIF</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🐋 Whale Sonar: ON\n"
    msg += "📈 Derivatives Engine: ON\n"
    msg += "🏦 Supply Analyzer: ON\n"
    msg += "🔥 Pre-Pump Detector: ON\n"
    msg += "📊 Technical Analysis SMC v3: ON\n"
    msg += "  ↳ FVG · OB · BOS/CHoCH · MTF Confluence Gate\n"
    msg += "🔬 <b>Quant Engine (151 Trading Strategies): ON ← NEW</b>\n"
    msg += "  ↳ §3.1  Price Momentum (risk-adjusted)\n"
    msg += "  ↳ §3.4  Low Volatility Anomaly\n"
    msg += "  ↳ §3.7  Residual Momentum / Alpha\n"
    msg += "  ↳ §3.13 Triple EMA Alignment\n"
    msg += "  ↳ §3.14 Pivot Point S/R → Entry+SL+TP\n"
    msg += "  ↳ §3.15 Donchian Channel → Breakout+Reversal\n"
    msg += "  ↳ §4.4  IBS Mean Reversion\n"
    msg += "  ↳ §4.1.2 Dual Momentum Filter\n"
    msg += "  ↳ §10.4 Trend Following + tanh smoothing\n"
    msg += "  ↳ §10.3.1 Contrarian + Volume Activity\n"
    msg += "\n⏳ Scanning dalam beberapa menit..."
    return _send(msg)
