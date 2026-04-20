import os

# ── Telegram ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Filter Universe ───────────────────────────────────────
MIN_VOLUME_24H_USD   = float(os.getenv("MIN_VOLUME_24H", "150000"))
MAX_COINS_TO_SCAN    = int(os.getenv("MAX_COINS", "250"))
MIN_CONFLUENCE_SCORE = float(os.getenv("MIN_SCORE", "40"))

# ── Alert Threshold ───────────────────────────────────────
ALERT_MIN_LEVEL = os.getenv("ALERT_LEVEL", "MEDIUM")
ALERT_LEVELS    = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# ── Risk Management Default ───────────────────────────────
DEFAULT_SL_PCT  = 0.05
DEFAULT_TP1_PCT = 0.10
DEFAULT_TP2_PCT = 0.20
DEFAULT_TP3_PCT = 0.40

# ── API Base URLs ─────────────────────────────────────────
# Binance memblokir IP cloud/datacenter (GitHub Actions = Azure → HTTP 451)
# Bitget & MEXC TIDAK memblokir cloud runner → aman digunakan di GitHub Actions
#
# Arsitektur fallback:
#   PRIMARY   → Bitget   (public API, tidak butuh key, tidak blokir cloud)
#   SECONDARY → MEXC     (public API, tidak butuh key, tidak blokir cloud)
#   TERTIARY  → CoinGecko (gratis, rate-limit 30 req/menit)

# ── Bitget (Primary) ──────────────────────────────────────
BITGET_SPOT_BASE    = "https://api.bitget.com"
BITGET_FUTURES_BASE = "https://api.bitget.com"

# ── MEXC (Secondary Fallback) ─────────────────────────────
MEXC_SPOT_BASE    = "https://api.mexc.com"
MEXC_FUTURES_BASE = "https://contract.mexc.com"

# ── CoinGecko & Fear Greed ────────────────────────────────
COINGECKO      = "https://api.coingecko.com/api/v3"
FEAR_GREED_API = "https://api.alternative.me/fng/"
