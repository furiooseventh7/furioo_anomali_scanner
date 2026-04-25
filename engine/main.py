"""
Main Orchestrator — CMI-ASS v7
Whale CEX + On-Chain DEX + Volume Spike + DefiLlama + PrecisionEntry
"""
import sys
import os
from pathlib import Path

# ── Pastikan repo root ada di sys.path ─────────────────────────────────────
# GitHub Actions menjalankan: python engine/main.py (dari repo root)
# Python tidak otomatis memasukkan repo root ke sys.path,
# sehingga "from config import ..." gagal dengan ModuleNotFoundError.
# Fix ini memastikan root directory selalu bisa diimport dari manapun.
_REPO_ROOT = Path(__file__).resolve().parent.parent   # engine/../ = repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time, logging
import pandas as pd
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    MIN_VOLUME_24H_USD, MAX_COINS_TO_SCAN,
    MIN_CONFLUENCE_SCORE, ALERT_MIN_LEVEL, ALERT_LEVELS,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    ONCHAIN_SCAN_INTERVAL, MIN_ONCHAIN_USD,
)
from engine.data_fetcher          import get_all_tickers_24h, get_futures_symbols, get_coingecko_data, get_fear_greed_index, get_klines
from engine.whale_sonar           import analyze_whale
from engine.derivatives_engine    import analyze_derivatives
from engine.supply_analyzer       import analyze_supply
from engine.pre_pump_detector     import analyze_pre_pump
from engine.technical_engine      import analyze_technical
from engine.quant_engine          import analyze_quant
from engine.decision_engine       import make_decision, FinalSignal
from engine.signal_validator      import validate_signal
from engine.defillama_engine      import analyze_defillama
from engine.precision_entry_engine import analyze_precision_entry
from engine.onchain_tracker       import analyze_onchain, scan_top_dex_whales, format_standalone_onchain_alert   # NEW v7
from engine.volume_spike_detector import analyze_volume_spike, scan_volume_spikes_batch, format_standalone_spike_alert  # NEW v7
from engine.telegram_gateway      import format_and_send_signal, send_scan_summary, send_startup_message, _send

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("CMI-ASS")

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
        deriv_res   = analyze_derivatives(symbol, futures_symbols); time.sleep(0.10)
        supply_res  = analyze_supply(base, cg_data)
        prepump_res = analyze_pre_pump(symbol);    time.sleep(0.10)
        ta_res      = analyze_technical(symbol, price); time.sleep(0.10)

        # ── Quant + klines ────────────────────────────────
        quant_res = None
        df_1h = df_4h = None
        try:
            df_1h     = get_klines(symbol, interval="1h",  limit=100)
            df_4h     = get_klines(symbol, interval="4h",  limit=100)
            quant_res = analyze_quant(symbol, price, df_1h, df_4h, btc_chg_24h, "LONG")
        except Exception as qe:
            logger.debug(f"quant skip {symbol}: {qe}")

        # ── Volume Spike Detection (NEW v7) ───────────────
        vol_spike_res = None
        try:
            vol_spike_res = analyze_volume_spike(symbol, price)
        except Exception as vse:
            logger.debug(f"VolSpike skip {symbol}: {vse}")

        # ── On-Chain tracking (NEW v7, hanya jika ada indikasi) ──
        onchain_res = None
        try:
            # Jalankan on-chain HANYA jika:
            # 1. Volume spike terdeteksi (menghemat Flipside quota), ATAU
            # 2. Symbol ada di DEX juga (estimasi dari supply data), ATAU
            # 3. Whale CEX buy pressure tinggi
            should_check_onchain = (
                (vol_spike_res and vol_spike_res.has_spike and vol_spike_res.spike_magnitude >= 3.0) or
                (whale_res.buy_pressure >= 0.70) or
                (whale_res.is_accumulating and whale_res.taker_buy_ratio >= 0.65)
            )
            if should_check_onchain:
                onchain_res = analyze_onchain(symbol)
                if onchain_res and onchain_res.found_onchain:
                    # Inject ke WhaleSonarResult
                    whale_res.onchain_score  = onchain_res.score
                    whale_res.is_stealth_acc = onchain_res.is_stealth_acc
                    whale_res.signals.extend(onchain_res.signals[:3])
        except Exception as oe:
            logger.debug(f"OnChain skip {symbol}: {oe}")

        # ── DefiLlama Fundamental ─────────────────────────
        llama_res   = None
        llama_score = 0.0
        try:
            mc = supply_res.market_cap if supply_res and supply_res.market_cap > 0 else 0
            llama_res   = analyze_defillama(symbol, mc)
            llama_score = llama_res.score if llama_res else 0.0
        except Exception as le:
            logger.debug(f"DefiLlama skip {symbol}: {le}")

        # ── Decision ──────────────────────────────────────
        signal = make_decision(
            symbol=symbol, price=price, price_chg_24h=chg_24h, volume_24h=vol_24h,
            whale_res=whale_res, deriv_res=deriv_res, supply_res=supply_res,
            prepump_res=prepump_res, fear_greed=fear_greed,
            ta_res=ta_res, quant_res=quant_res,
        )

        # ── Score bonuses ─────────────────────────────────
        # DefiLlama bonus (max +15)
        if llama_score > 0:
            signal.confidence_score = min(100, signal.confidence_score + min(llama_score * 0.4, 15.0))
            signal.llama_score = llama_score
        else:
            signal.llama_score = 0.0

        # Volume spike bonus (max +12)
        if vol_spike_res and vol_spike_res.has_spike:
            vs_bonus = min(vol_spike_res.score * 0.5, 12.0)
            # Penalti jika spike adalah DUMP
            if vol_spike_res.spike_type == "DUMP":
                signal.confidence_score = max(0, signal.confidence_score - 10)
            else:
                signal.confidence_score = min(100, signal.confidence_score + vs_bonus)
            signal.vol_spike_score = vol_spike_res.score
        else:
            signal.vol_spike_score = 0.0

        # On-chain bonus (max +15)
        if onchain_res and onchain_res.found_onchain:
            oc_bonus = min(onchain_res.score * 0.5, 15.0)
            if onchain_res.is_stealth_acc:
                oc_bonus = min(oc_bonus * 1.5, 15.0)  # stealth = extra bonus
            signal.confidence_score = min(100, signal.confidence_score + oc_bonus)
            signal.onchain_score = onchain_res.score
        else:
            signal.onchain_score = 0.0

        # ── Build extra_context ───────────────────────────
        ctx = {}
        if ta_res:
            ctx.update({
                "has_precise_entry": getattr(ta_res, "has_precise_entry", False),
                "optimal_entry_low": getattr(ta_res, "optimal_entry_low", 0.0),
                "optimal_entry_high": getattr(ta_res, "optimal_entry_high", 0.0),
                "structure_sl":      getattr(ta_res, "structure_sl", 0.0),
                "in_discount_zone":  getattr(ta_res, "in_discount_zone", False),
                "in_premium_zone":   getattr(ta_res, "in_premium_zone", False),
                "liquidity_swept":   getattr(ta_res, "liquidity_swept", False),
                "mtf_bias":          getattr(ta_res, "mtf_bias", "NEUTRAL"),
                "mtf_agree_count":   getattr(ta_res, "mtf_agree_count", 0),
                "invalidation":      getattr(ta_res, "invalidation", []),
            })
        if quant_res:
            ctx["quant"] = quant_res
        ctx["llama_res"]     = llama_res
        ctx["onchain_res"]   = onchain_res
        ctx["vol_spike_res"] = vol_spike_res
        signal.extra_context.update(ctx)

        # ── Precision Entry Engine ────────────────────────
        entry_timing = None
        try:
            entry_timing = analyze_precision_entry(
                symbol=symbol, price=price,
                whale_res=whale_res, ta_res=ta_res, deriv_res=deriv_res,
                df_1h=df_1h, df_4h=df_4h,
            )
            if entry_timing.entry_timing == "AVOID":
                signal.confidence_score = max(0, signal.confidence_score - 15)
            elif entry_timing.whale_phase == "MARKUP_IMMINENT" and entry_timing.entry_timing == "NOW":
                signal.confidence_score = min(100, signal.confidence_score + 8)
        except Exception as ee:
            logger.debug(f"PrecisionEntry skip {symbol}: {ee}")

        ctx["entry_timing"] = entry_timing

        if signal.confidence_score >= MIN_CONFLUENCE_SCORE and signal.signal_type != "NEUTRAL":
            vs_info = f"VS:{vol_spike_res.spike_magnitude:.1f}x" if vol_spike_res and vol_spike_res.has_spike else ""
            oc_info = f"OC:${onchain_res.total_bought_usd/1000:.0f}K" if onchain_res and onchain_res.found_onchain else ""
            logger.info(
                f"SIGNAL | {symbol:16s} | {signal.signal_type:8s} | "
                f"Score:{signal.confidence_score:.0f} | {signal.alert_level} | "
                f"{vs_info} {oc_info}"
            )
            return signal
        else:
            logger.debug(f"SKIP | {symbol} | Score:{signal.confidence_score:.1f}")
            return signal

    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
    return None


def _send_proactive_onchain_alerts(fear_greed: dict) -> None:
    """
    Proactive: scan DEX on-chain secara mandiri untuk menemukan gem
    yang belum ada di radar CEX. Jalankan setiap ONCHAIN_SCAN_INTERVAL menit.
    """
    logger.info("🔍 Proactive on-chain scan: mencari whale DEX accumulation...")
    try:
        top_results = scan_top_dex_whales(min_usd=MIN_ONCHAIN_USD, hours=4)
        for res in top_results:
            if res.score >= 15 and res.alert_level in ("HIGH", "CRITICAL"):
                msg = format_standalone_onchain_alert(res)
                if msg:
                    # Coba ambil chart untuk token ini
                    try:
                        from engine.chart_engine import get_chart_image
                        from engine.telegram_gateway import _send_photo
                        img = get_chart_image(res.symbol, "4h")
                        if img:
                            caption = f"⛓️ ON-CHAIN WHALE ALERT — ${res.symbol.replace('USDT','')} | Score: {res.score:.0f}/30"
                            _send_photo(img, caption)
                            time.sleep(1)
                    except Exception:
                        pass
                    _send(msg)
                    time.sleep(2)
        if top_results:
            logger.info(f"On-chain scan: {len(top_results)} tokens ditemukan")
    except Exception as e:
        logger.warning(f"Proactive on-chain scan error: {e}")


def _send_standalone_spike_alerts(ticker_df: pd.DataFrame) -> None:
    """
    Kirim alert mandiri untuk volume spike CRITICAL yang terdeteksi
    bahkan sebelum analisis penuh selesai — early warning system.
    """
    quick_spikes = scan_volume_spikes_batch(ticker_df, min_spike_x=5.0, max_coins=20)
    critical = [s for s in quick_spikes if s.spike_level == "CRITICAL" and s.spike_type != "DUMP"]
    for spike in critical[:3]:
        msg = format_standalone_spike_alert(spike)
        if msg:
            try:
                from engine.chart_engine import get_chart_image
                from engine.telegram_gateway import _send_photo
                img = get_chart_image(spike.symbol, "1h")
                if img:
                    cap = (f"🚨 VOLUME SPIKE {spike.spike_magnitude:.1f}x — "
                           f"${spike.symbol.replace('USDT','')} | {spike.spike_type}")
                    _send_photo(img, cap)
                    time.sleep(1)
            except Exception:
                pass
            _send(msg)
            time.sleep(2)


def run():
    logger.info("=" * 65)
    logger.info("CMI-ASS v7 — Whale CEX+OnChain+VolumeSpike+DefiLlama ACTIVE")
    logger.info("=" * 65)

    if not TELEGRAM_BOT_TOKEN or not _CHAT_IDS:
        logger.critical("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tidak ada!")
        return

    send_startup_message()

    logger.info("📡 Fetching global market data...")
    df_ticker   = get_all_tickers_24h()
    futures_set = get_futures_symbols()
    fear_greed  = get_fear_greed_index()
    if df_ticker.empty:
        logger.error("Gagal ambil ticker data"); return

    # ── Proactive: kirim early spike alert dari ticker saja ──
    logger.info("⚡ Quick volume spike scan dari ticker data...")
    try:
        _send_standalone_spike_alerts(df_ticker)
    except Exception as e:
        logger.warning(f"Quick spike scan error: {e}")

    logger.info("🦎 Fetching CoinGecko supply data...")
    top_symbols = df_ticker.nlargest(250, "quoteVolume")["symbol"].tolist()
    cg_data     = get_coingecko_data(top_symbols)

    btc_row     = df_ticker[df_ticker["symbol"] == "BTCUSDT"]
    btc_chg_24h = float(btc_row["priceChangePercent"].values[0]) / 100 if not btc_row.empty else 0.0

    candidates = df_ticker[
        (df_ticker["quoteVolume"] >= MIN_VOLUME_24H_USD) &
        (df_ticker["priceChangePercent"].abs() < 80)
    ].copy()
    candidates["vol_score"] = candidates["quoteVolume"] / (candidates["priceChangePercent"].abs() + 1)
    candidates = candidates.nlargest(MAX_COINS_TO_SCAN, "vol_score")
    logger.info(f"🔍 Scanning {len(candidates)} coins...")

    # ── Proactive on-chain DEX scan (parallel dengan main scan) ──
    logger.info("⛓️ Starting proactive on-chain DEX scan...")
    from concurrent.futures import ThreadPoolExecutor as TPE
    onchain_future = None
    oc_executor = TPE(max_workers=1)
    try:
        onchain_future = oc_executor.submit(_send_proactive_onchain_alerts, fear_greed)
    except Exception:
        pass

    # ── Main coin scan ────────────────────────────────────
    signals: List[FinalSignal] = []
    scanned = 0
    rows    = [r for _, r in candidates.iterrows()]

    with ThreadPoolExecutor(max_workers=5) as executor:
        fmap = {executor.submit(scan_single, row, futures_set, cg_data, fear_greed, btc_chg_24h): row
                for row in rows}
        for future in as_completed(fmap):
            try:
                sig = future.result()
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"Parallel error: {e}")
            scanned += 1
            if scanned % 20 == 0:
                logger.info(f"Progress: {scanned}/{len(candidates)} | Signals: {len(signals)}")

    signals.sort(key=lambda x: x.confidence_score, reverse=True)

    # ── Validate signals ─────────────────────────────────
    validated = []
    for sig in signals:
        try:
            vr = validate_signal(
                sig=sig, ta_res=None,
                quant_res=sig.extra_context.get("quant") if hasattr(sig, "extra_context") else None,
                deriv_res=None, whale_res=None, supply_res=None, prepump_res=None,
                fear_greed=fear_greed,
            )
            if hasattr(sig, "extra_context"):
                sig.extra_context["prosecution"]       = vr.case
                sig.extra_context["validator_verdict"] = vr.verdict
            sig.confidence_score = vr.final_score
            sig.alert_level      = vr.final_alert_level
            if vr.send_signal:
                validated.append(sig)
            else:
                rej = vr.case.rejection_reasons[0][:80] if vr.case.rejection_reasons else "Quality gate"
                logger.info(f"REJECTED | {sig.symbol} | {rej}")
        except Exception as e:
            logger.warning(f"Validator error {sig.symbol}: {e}")
            validated.append(sig)
    signals = validated

    # ── Send alerts ───────────────────────────────────────
    alerts_sent = 0
    for sig in signals:
        if should_alert(sig.alert_level, ALERT_MIN_LEVEL):
            ec           = sig.extra_context if hasattr(sig, "extra_context") else {}
            llama_res    = ec.get("llama_res")
            entry_timing = ec.get("entry_timing")
            onchain_res  = ec.get("onchain_res")
            vol_spike    = ec.get("vol_spike_res")
            if format_and_send_signal(sig,
                                       vr_llama=llama_res,
                                       entry_timing=entry_timing,
                                       onchain_res=onchain_res,
                                       vol_spike_res=vol_spike):
                alerts_sent += 1
            time.sleep(2.5)

    send_scan_summary(scanned, signals, fear_greed)

    # Tunggu on-chain scan selesai
    if onchain_future:
        try:
            onchain_future.result(timeout=60)
        except Exception:
            pass
    oc_executor.shutdown(wait=False)

    logger.info(f"✅ DONE | Scanned:{scanned} | Signals:{len(signals)} | Alerts:{alerts_sent}")


if __name__ == "__main__":
    run()
