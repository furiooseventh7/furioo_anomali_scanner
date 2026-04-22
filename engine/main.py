"""
Main Orchestrator — CMI-ASS v4
Koordinasi: fetch → TA → Quant → Decision → Telegram
"""
import os, time, logging
import pandas as pd
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    MIN_VOLUME_24H_USD, MAX_COINS_TO_SCAN,
    MIN_CONFLUENCE_SCORE, ALERT_MIN_LEVEL, ALERT_LEVELS,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)
from engine.data_fetcher       import get_all_tickers_24h, get_futures_symbols, get_coingecko_data, get_fear_greed_index, get_klines
from engine.whale_sonar        import analyze_whale
from engine.derivatives_engine import analyze_derivatives
from engine.supply_analyzer    import analyze_supply
from engine.pre_pump_detector  import analyze_pre_pump
from engine.technical_engine   import analyze_technical
from engine.quant_engine       import analyze_quant
from engine.decision_engine    import make_decision, FinalSignal
from engine.signal_validator     import validate_signal
from engine.telegram_gateway   import format_and_send_signal, send_scan_summary, send_startup_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("CMI-ASS")

# Support multiple Telegram chat IDs separated by comma
_CHAT_IDS = [c.strip() for c in str(TELEGRAM_CHAT_ID).split(",") if c.strip()]

def should_alert(level: str, min_level: str) -> bool:
    return ALERT_LEVELS.index(level) >= ALERT_LEVELS.index(min_level)


def scan_single(row: pd.Series, futures_symbols: set, cg_data: dict,
                fear_greed: dict, btc_chg_24h: float = 0.0) -> Optional[FinalSignal]:
    symbol  = row["symbol"]
    price   = float(row["lastPrice"])
    chg_24h = float(row["priceChangePercent"])
    vol_24h = float(row["quoteVolume"])
    base    = symbol.replace("USDT", "")

    if price <= 0 or vol_24h < MIN_VOLUME_24H_USD or chg_24h > 80:
        return None

    try:
        whale_res   = analyze_whale(symbol);       time.sleep(0.12)
        deriv_res   = analyze_derivatives(symbol, futures_symbols); time.sleep(0.12)
        supply_res  = analyze_supply(base, cg_data)
        prepump_res = analyze_pre_pump(symbol);    time.sleep(0.12)
        ta_res      = analyze_technical(symbol, price); time.sleep(0.12)

        # ── Quant Engine (151 Trading Strategies) ─────────────────────────
        quant_res = None
        try:
            df_1h = get_klines(symbol, interval="1h", limit=100)
            df_4h = get_klines(symbol, interval="4h", limit=100)
            # Determine signal direction for level calc (pre-estimate)
            prelim_sig = "LONG"
            quant_res = analyze_quant(symbol, price, df_1h, df_4h, btc_chg_24h, prelim_sig)
        except Exception as qe:
            logger.debug(f"quant skip {symbol}: {qe}")

        signal = make_decision(
            symbol=symbol, price=price, price_chg_24h=chg_24h, volume_24h=vol_24h,
            whale_res=whale_res, deriv_res=deriv_res, supply_res=supply_res,
            prepump_res=prepump_res, fear_greed=fear_greed,
            ta_res=ta_res, quant_res=quant_res,
        )

        # Store context for telegram display
        ctx = {}
        if ta_res is not None:
            ctx.update({
                "has_precise_entry": getattr(ta_res,"has_precise_entry",False),
                "optimal_entry_low": getattr(ta_res,"optimal_entry_low",0.0),
                "optimal_entry_high":getattr(ta_res,"optimal_entry_high",0.0),
                "structure_sl":      getattr(ta_res,"structure_sl",0.0),
                "in_discount_zone":  getattr(ta_res,"in_discount_zone",False),
                "in_premium_zone":   getattr(ta_res,"in_premium_zone",False),
                "liquidity_swept":   getattr(ta_res,"liquidity_swept",False),
                "mtf_bias":          getattr(ta_res,"mtf_bias","NEUTRAL"),
                "mtf_agree_count":   getattr(ta_res,"mtf_agree_count",0),
                "is_gated":          getattr(ta_res,"is_gated",False),
                "invalidation":      getattr(ta_res,"invalidation",[]),
            })
        if quant_res is not None:
            ctx["quant"] = quant_res
        signal.extra_context.update(ctx)

        if signal.confidence_score >= MIN_CONFLUENCE_SCORE and signal.signal_type != "NEUTRAL":
            q_info = f"Q:{quant_res.score:.0f}|C:{quant_res.confluence_count}" if quant_res else "Q:N/A"
            logger.info(
                f"SIGNAL | {symbol:16s} | {signal.signal_type:8s} | "
                f"Score {signal.confidence_score:.0f} | {signal.alert_level} | "
                f"TA:{signal.ta_bias} | {q_info}"
            )
            return signal
        else:
            logger.debug(
                f"SKIP | {symbol} | {signal.signal_type} | "
                f"Score {signal.confidence_score:.1f} | "
                f"W:{signal.whale_score:.0f} D:{signal.derivatives_score:.0f} "
                f"S:{signal.supply_score:.0f} P:{signal.pre_pump_score:.0f} "
                f"TA:{signal.ta_score:.0f} Q:{signal.quant_score:.0f}"
            )
            return signal

    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
    return None


def run():
    logger.info("=" * 60)
    logger.info("CMI-ASS v4 — Quant Engine (151 Strategies) ACTIVE")
    logger.info("=" * 60)

    if not TELEGRAM_BOT_TOKEN or not _CHAT_IDS:
        logger.critical("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tidak ada di secrets!")
        return

    send_startup_message()

    logger.info("📡 Fetching global market data...")
    df_ticker   = get_all_tickers_24h()
    futures_set = get_futures_symbols()
    fear_greed  = get_fear_greed_index()
    if df_ticker.empty:
        logger.error("Gagal ambil ticker data"); return

    logger.info(f"✅ Tickers:{len(df_ticker)} | Futures:{len(futures_set)} | F&G:{fear_greed}")

    logger.info("🦎 Fetching CoinGecko supply data...")
    top_symbols = df_ticker.nlargest(250,"quoteVolume")["symbol"].tolist()
    cg_data = get_coingecko_data(top_symbols)
    logger.info(f"✅ CoinGecko: {len(cg_data)} coins")

    # BTC 24h return as market proxy for dual momentum (§4.1.2)
    btc_row = df_ticker[df_ticker["symbol"]=="BTCUSDT"]
    btc_chg_24h = float(btc_row["priceChangePercent"].values[0]) / 100 if not btc_row.empty else 0.0

    candidates = df_ticker[
        (df_ticker["quoteVolume"] >= MIN_VOLUME_24H_USD) &
        (df_ticker["priceChangePercent"].abs() < 80)
    ].copy()
    candidates["vol_score"] = candidates["quoteVolume"] / (candidates["priceChangePercent"].abs()+1)
    candidates = candidates.nlargest(MAX_COINS_TO_SCAN,"vol_score")
    logger.info(f"🔍 Scanning {len(candidates)} coins...")

    signals: List[FinalSignal] = []
    scanned = 0
    rows = [r for _,r in candidates.iterrows()]

    with ThreadPoolExecutor(max_workers=8) as executor:
        fmap = {executor.submit(scan_single, row, futures_set, cg_data, fear_greed, btc_chg_24h): row
                for row in rows}
        for future in as_completed(fmap):
            try:
                sig = future.result()
                if sig: signals.append(sig)
            except Exception as e:
                logger.error(f"Parallel error: {e}")
            scanned += 1
            if scanned % 20 == 0:
                logger.info(f"Progress: {scanned}/{len(candidates)} | Signals: {len(signals)}")

    signals.sort(key=lambda x: x.confidence_score, reverse=True)

    # ── Validate each signal (Prosecution Case) ───────────────
    validated = []
    for sig in signals:
        try:
            vr = validate_signal(
                sig         = sig,
                ta_res      = None,   # already baked into sig.extra_context
                quant_res   = sig.extra_context.get("quant") if hasattr(sig,"extra_context") else None,
                deriv_res   = None,
                whale_res   = None,
                supply_res  = None,
                prepump_res = None,
                fear_greed  = fear_greed,
            )
            # Attach prosecution case to signal for telegram
            if hasattr(sig, "extra_context"):
                sig.extra_context["prosecution"] = vr.case
                sig.extra_context["validator_verdict"] = vr.verdict
            # Update score and alert level from validator
            sig.confidence_score = vr.final_score
            sig.alert_level      = vr.final_alert_level
            if vr.send_signal:
                validated.append(sig)
            else:
                logger.info(f"REJECTED | {sig.symbol} | {vr.case.rejection_reasons[0][:80] if vr.case.rejection_reasons else 'Quality gate'}")
        except Exception as e:
            logger.warning(f"Validator error {sig.symbol}: {e}")
            validated.append(sig)  # on error, include anyway
    signals = validated
    alerts_sent = 0
    for sig in signals:
        if should_alert(sig.alert_level, ALERT_MIN_LEVEL):
            if format_and_send_signal(sig):
                alerts_sent += 1
            time.sleep(1.5)

    send_scan_summary(scanned, signals, fear_greed)
    logger.info(f"✅ DONE | Scanned:{scanned} | Signals:{len(signals)} | Alerts:{alerts_sent}")


if __name__ == "__main__":
    run()
