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
# PRIMARY   → Bitget   (tidak blokir GitHub Actions runner)
# SECONDARY → MEXC     (fallback)
# TERTIARY  → CoinGecko

BITGET_SPOT_BASE    = "https://api.bitget.com"
BITGET_FUTURES_BASE = "https://api.bitget.com"

MEXC_SPOT_BASE    = "https://api.mexc.com"
MEXC_FUTURES_BASE = "https://contract.mexc.com"

COINGECKO      = "https://api.coingecko.com/api/v3"
FEAR_GREED_API = "https://api.alternative.me/fng/"

# ── DefiLlama (v6 NEW) ────────────────────────────────────
# Gratis, no API key needed
DEFILLAMA_BASE  = "https://api.llama.fi"
DEFILLAMA_COINS = "https://coins.llama.fi"

# ── Chart Screenshot (v6 NEW) ─────────────────────────────
# SCREENSHOTONE_KEY: optional, free tier 100 screenshots/month
# Daftar gratis di: https://screenshotone.com
# Tambahkan di GitHub Secrets sebagai: SCREENSHOTONE_KEY
# Jika kosong: sistem akan kirim link chart saja (tetap berfungsi)
SCREENSHOTONE_KEY = os.getenv("SCREENSHOTONE_KEY", "")
