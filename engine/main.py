"""
Main Orchestrator — CMI-ASS
Mengkoordinasi semua modul: fetch → analyze → decide → alert Telegram.
v2: Ditambahkan Technical Analysis Engine (RSI, MACD, FVG, OB, S/R, Patterns)
"""
import os
import time
import logging
import pandas as pd
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    MIN_VOLUME_24H_USD, MAX_COINS_TO_SCAN,
    MIN_CONFLUENCE_SCORE, ALERT_MIN_LEVEL, ALERT_LEVELS,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)
from engine.data_fetcher import (
    get_all_tickers_24h, get_futures_symbols,
    get_coingecko_data, get_fear_greed_index
)
from engine.whale_sonar        import analyze_whale
from engine.derivatives_engine import analyze_derivatives
from engine.supply_analyzer    import analyze_supply
from engine.pre_pump_detector  import analyze_pre_pump
from engine.technical_engine   import analyze_technical
from engine.decision_engine    import make_decision, FinalSignal
from engine.narrative_engine   import compute_narrative_scores, get_ticker_narrative_result
from engine.telegram_gateway   import (
    format_and_send_signal, send_scan_summary, send_startup_message
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("CMI-ASS")


def should_alert(level: str, min_level: str) -> bool:
    return ALERT_LEVELS.index(level) >= ALERT_LEVELS.index(min_level)


def scan_single(row: pd.Series, futures_symbols: set,
                cg_data: dict, fear_greed: dict,
                narrative_scores: dict = None) -> Optional[FinalSignal]:
    symbol    = row["symbol"]
    price     = float(row["lastPrice"])
    chg_24h   = float(row["priceChangePercent"])
    vol_24h   = float(row["quoteVolume"])
    base      = symbol.replace("USDT", "")

    # ── Filter kasar ──────────────────────────────────────
    if price <= 0 or vol_24h < MIN_VOLUME_24H_USD:
        return None
    if chg_24h > 80:
        return None

    try:
        # ── Engine lama (tidak diubah sama sekali) ────────
        whale_res   = analyze_whale(symbol)
        time.sleep(0.15)

        deriv_res   = analyze_derivatives(symbol, futures_symbols)
        time.sleep(0.15)

        supply_res  = analyze_supply(base, cg_data)

        prepump_res = analyze_pre_pump(symbol)
        time.sleep(0.15)

        # ── Engine baru: Technical Analysis ──────────────
        ta_res = analyze_technical(symbol, price)
        time.sleep(0.15)

        # ── Narrative ─────────────────────────────────────
        narr_res = None
        if narrative_scores:
            narr_res = get_ticker_narrative_result(base, narrative_scores)

        # ── Buat keputusan ────────────────────────────────
        signal = make_decision(
            symbol        = symbol,
            price         = price,
            price_chg_24h = chg_24h,
            volume_24h    = vol_24h,
            whale_res     = whale_res,
            deriv_res     = deriv_res,
            supply_res    = supply_res,
            prepump_res   = prepump_res,
            fear_greed    = fear_greed,
            ta_res        = ta_res,
            narr_res      = narr_res,
        )

        # Simpan v3 SMC context ke extra_context agar gateway bisa tampilkan
        if ta_res is not None:
            signal.extra_context.update({
                "has_precise_entry" : getattr(ta_res, "has_precise_entry", False),
                "optimal_entry_low" : getattr(ta_res, "optimal_entry_low", 0.0),
                "optimal_entry_high": getattr(ta_res, "optimal_entry_high", 0.0),
                "structure_sl"      : getattr(ta_res, "structure_sl", 0.0),
                "in_discount_zone"  : getattr(ta_res, "in_discount_zone", False),
                "in_premium_zone"   : getattr(ta_res, "in_premium_zone", False),
                "liquidity_swept"   : getattr(ta_res, "liquidity_swept", False),
                "mtf_bias"          : getattr(ta_res, "mtf_bias", "NEUTRAL"),
                "mtf_agree_count"   : getattr(ta_res, "mtf_agree_count", 0),
                "is_gated"          : getattr(ta_res, "is_gated", False),
                "invalidation"      : getattr(ta_res, "invalidation", []),
            })

        # Simpan narrative result ke extra_context
        if narr_res is not None:
            signal.extra_context["narrative"] = narr_res

        if signal.confidence_score >= MIN_CONFLUENCE_SCORE and signal.signal_type != "NEUTRAL":
            narr_info = f"{narr_res.name} {narr_res.hype_score}/100" if narr_res else "N/A"
            logger.info(
                f"SIGNAL | {symbol:16s} | {signal.signal_type:8s} | "
                f"Score {signal.confidence_score:.0f} | {signal.alert_level} | "
                f"TA: {signal.ta_bias} | Narr: {narr_info}"
            )
            return signal
        else:
            logger.debug(
                f"SKIP   | {symbol:16s} | {signal.signal_type:8s} | "
                f"Score {signal.confidence_score:.1f} (min {MIN_CONFLUENCE_SCORE}) | "
                f"W:{signal.whale_score:.1f} D:{signal.derivatives_score:.1f} "
                f"S:{signal.supply_score:.1f} P:{signal.pre_pump_score:.1f} "
                f"TA:{signal.ta_score:.1f} | TA_bias:{signal.ta_bias}"
            )

    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}", exc_info=True)

    return None


def run():
    logger.info("=" * 60)
    logger.info("CMI-ASS v2 — Technical Analysis Engine ACTIVE")
    logger.info("=" * 60)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.critical("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tidak ada di secrets!")
        return

    send_startup_message()

    # ── Fetch global data ─────────────────────────────────
    logger.info("📡 Fetching global market data...")
    df_ticker   = get_all_tickers_24h()
    futures_set = get_futures_symbols()
    fear_greed  = get_fear_greed_index()

    if df_ticker.empty:
        logger.error("Gagal ambil ticker data")
        return

    logger.info(f"✅ Tickers: {len(df_ticker)} | Futures: {len(futures_set)} | F&G: {fear_greed}")

    logger.info("🦎 Fetching CoinGecko supply data...")
    top_symbols = df_ticker.nlargest(250, "quoteVolume")["symbol"].tolist()
    cg_data = get_coingecko_data(top_symbols)
    logger.info(f"✅ CoinGecko data: {len(cg_data)} coins")

    # ── Narrative Hype Scores ─────────────────────────────
    logger.info("📖 Computing narrative hype scores...")
    narrative_scores = compute_narrative_scores(df_ticker)
    if narrative_scores:
        top_narr = sorted(narrative_scores.values(), key=lambda x: x.hype_score, reverse=True)[:3]
        top_str  = " | ".join(f"{n.name} {n.hype_score}/100" for n in top_narr)
        logger.info(f"✅ Narratives: {len(narrative_scores)} tracked | Top: {top_str}")

    # ── Filter & prioritisasi ─────────────────────────────
    candidates = df_ticker[
        (df_ticker["quoteVolume"] >= MIN_VOLUME_24H_USD) &
        (df_ticker["priceChangePercent"].abs() < 80)
    ].copy()

    candidates["vol_score"] = candidates["quoteVolume"] / (candidates["priceChangePercent"].abs() + 1)
    candidates = candidates.nlargest(MAX_COINS_TO_SCAN, "vol_score")

    logger.info(f"🔍 Akan scan {len(candidates)} coins (dengan TA engine)...")

    # ── Main scan loop (parallel) ─────────────────────────
    signals: List[FinalSignal] = []
    scanned = 0
    rows = [row for _, row in candidates.iterrows()]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_map = {
            executor.submit(scan_single, row, futures_set, cg_data, fear_greed, narrative_scores): row
            for row in rows
        }
        for future in as_completed(futures_map):
            try:
                sig = future.result()
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"Error in parallel scan: {e}")
            scanned += 1
            if scanned % 20 == 0:
                logger.info(f"Progress: {scanned}/{len(candidates)} | Signals: {len(signals)}")

    # ── Sort & kirim alert ────────────────────────────────
    signals.sort(key=lambda x: x.confidence_score, reverse=True)

    alerts_sent = 0
    for sig in signals:
        if should_alert(sig.alert_level, ALERT_MIN_LEVEL):
            success = format_and_send_signal(sig)
            if success:
                alerts_sent += 1
            time.sleep(1.5)

    send_scan_summary(scanned, signals, fear_greed)
    logger.info(f"✅ DONE | Scanned: {scanned} | Signals: {len(signals)} | Alerts sent: {alerts_sent}")


if __name__ == "__main__":
    run()
