import os

# ── Telegram ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Filter Universe ───────────────────────────────────────
MIN_VOLUME_24H_USD   = float(os.getenv("MIN_VOLUME_24H", "150000"))  # Min $150K/hari
MAX_COINS_TO_SCAN    = int(os.getenv("MAX_COINS", "250"))
MIN_CONFLUENCE_SCORE = float(os.getenv("MIN_SCORE", "40"))           # Min score 40/100

# ── Alert Threshold ───────────────────────────────────────
ALERT_MIN_LEVEL = os.getenv("ALERT_LEVEL", "MEDIUM")  # LOW/MEDIUM/HIGH/CRITICAL
ALERT_LEVELS    = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# ── Risk Management Default ───────────────────────────────
DEFAULT_SL_PCT  = 0.05   # Stop Loss 5%
DEFAULT_TP1_PCT = 0.10   # Take Profit 1 → 10%
DEFAULT_TP2_PCT = 0.20   # Take Profit 2 → 20%
DEFAULT_TP3_PCT = 0.40   # Take Profit 3 → 40%

# Fallback URLs jika primary diblokir (e.g. GitHub Actions / cloud IP)
BINANCE_SPOT_URLS = [
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.com",
]
BINANCE_FUTURES_URLS = [
    "https://fapi.binance.com",
]
# Tetap simpan ini untuk backward compat modul lain yang import langsung
BINANCE_SPOT    = "https://api1.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"
COINGECKO       = "https://api.coingecko.com/api/v3"
FEAR_GREED_API  = "https://api.alternative.me/fng/"
