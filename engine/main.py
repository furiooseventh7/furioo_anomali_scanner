"""
Main Orchestrator — mengkoordinasi semua modul:
fetch data → analyze → decide → alert Telegram.
"""
import os
import time
import logging
import pandas as pd
from typing import List, Optional

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
from engine.decision_engine    import make_decision, FinalSignal
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
                cg_data: dict, fear_greed: dict) -> Optional[FinalSignal]:
    symbol  = row["symbol"]
    price   = float(row["lastPrice"])
    chg_24h = float(row["priceChangePercent"])
    vol_24h = float(row["quoteVolume"])
    base    = symbol.replace("USDT","")

    if price <= 0 or vol_24h < MIN_VOLUME_24H_USD:
        return None
    if chg_24h > 80:
        return None

    try:
        whale_res   = analyze_whale(symbol)
        time.sleep(0.15)

        deriv_res   = analyze_derivatives(symbol, futures_symbols)
        time.sleep(0.15)

        supply_res  = analyze_supply(base, cg_data)

        prepump_res = analyze_pre_pump(symbol)
        time.sleep(0.15)

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
        )

        if signal.confidence_score >= MIN_CONFLUENCE_SCORE and signal.signal_type != "NEUTRAL":
            logger.info(
                f"SIGNAL | {symbol:16s} | {signal.signal_type:8s} | "
                f"Score {signal.confidence_score:.0f} | {signal.alert_level}"
            )
            return signal

    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")

    return None


def run():
    logger.info("=" * 60)
    logger.info("CMI-ASS — Crypto Market Intelligence System START")
    logger.info("=" * 60)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.critical("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tidak ada di secrets!")
        raise SystemExit(1)

    send_startup_message()

    # ── Fetch global data ─────────────────────────────────
    logger.info("📡 Fetching global market data...")
    df_ticker   = get_all_tickers_24h()   # auto-fallback ke CoinGecko jika Binance 451
    futures_set = get_futures_symbols()   # bisa empty set jika Binance Futures diblokir
    fear_greed  = get_fear_greed_index()

    if df_ticker.empty:
        logger.error(
            "❌ Gagal ambil ticker data dari semua sumber (Binance + CoinGecko). "
            "Cek koneksi jaringan runner."
        )
        raise SystemExit(1)   # tandai job GitHub Actions sebagai FAILED

    if not futures_set:
        logger.warning(
            "⚠️ Futures symbols kosong — scan tetap jalan tapi "
            "sinyal derivatif tidak akan akurat."
        )

    logger.info(f"✅ Tickers: {len(df_ticker)} | Futures: {len(futures_set)} | F&G: {fear_greed}")

    # ── CoinGecko supply data ─────────────────────────────
    logger.info("🦎 Fetching CoinGecko supply data...")
    top_symbols = df_ticker.nlargest(250, "quoteVolume")["symbol"].tolist()
    cg_data = get_coingecko_data(top_symbols)
    logger.info(f"✅ CoinGecko data: {len(cg_data)} coins")

    # ── Filter kandidat ───────────────────────────────────
    candidates = df_ticker[
        (df_ticker["quoteVolume"] >= MIN_VOLUME_24H_USD) &
        (df_ticker["priceChangePercent"].abs() < 80)
    ].copy()

    candidates["vol_score"] = candidates["quoteVolume"] / (candidates["priceChangePercent"].abs() + 1)
    candidates = candidates.nlargest(MAX_COINS_TO_SCAN, "vol_score")

    logger.info(f"🔍 Akan scan {len(candidates)} coins...")

    # ── Main scan loop ────────────────────────────────────
    signals: List[FinalSignal] = []
    scanned = 0

    for _, row in candidates.iterrows():
        sig = scan_single(row, futures_set, cg_data, fear_greed)
        if sig:
            signals.append(sig)

        scanned += 1
        if scanned % 25 == 0:
            logger.info(f"Progress: {scanned}/{len(candidates)} | Signals: {len(signals)}")

        time.sleep(0.1)

    # ── Kirim alert ───────────────────────────────────────
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
