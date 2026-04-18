"""
Technical Analysis Engine — CMI-ASS v3 (PRECISION UPGRADE)
============================================================
UPGRADE dari v2: Multi-Timeframe + Precision Confluence Gating

PERUBAHAN UTAMA vs v2:
  - Multi-Timeframe (15m, 1H, 4H, 1D): setiap TF punya bobot sendiri
  - FVG detection lebih ketat: harus impulsif, ukuran minimum, belum terisi
  - Order Block lebih presisi: validasi dengan volume spike + tidak tertembus
  - BOS/CHoCH lebih akurat: pakai swing high/low yang benar
  - Liquidity sweep detection (stop hunt sebelum reversal)
  - Premium/Discount zones (ICT concept)
  - Divergence RSI multi-swing
  - Confluence Gate: signal hanya keluar jika >=2 TF setuju arah
  - Invalidation check: jika ada sinyal bearish kuat, batalkan bullish signal
  - Entry precision: entry di dalam FVG/OB, bukan di harga pasar
  - Dynamic SL berdasarkan swing structure
"""

import numpy as np
import pandas as pd
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from engine.data_fetcher import get_klines

# Suppress numpy divide/invalid warnings — handled explicitly via np.errstate
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Bobot per timeframe
TF_WEIGHTS = {"15m": 0.10, "1h": 0.25, "4h": 0.40, "1d": 0.25}
FVG_MIN_SIZE_PCT   = 0.003
OB_MIN_BODY_PCT    = 0.008
ZONE_PROXIMITY_PCT = 0.025
SR_PROXIMITY_PCT   = 0.020
MIN_TF_CONFLUENCE  = 2
SIGNAL_GATE_SCORE  = 10.0


@dataclass
class FVGZone:
    kind         : str
    top          : float
    bottom       : float
    midpoint     : float
    size_pct     : float
    candle_idx   : int
    timeframe    : str
    is_impulsive : bool = False
    filled       : bool = False
    partially_filled: bool = False

@dataclass
class OrderBlock:
    kind         : str
    top          : float
    bottom       : float
    midpoint     : float
    candle_idx   : int
    timeframe    : str
    body_size_pct: float = 0.0
    volume_spike : bool  = False
    breached     : bool  = False
    valid        : bool  = True

@dataclass
class SwingPoint:
    kind    : str
    price   : float
    idx     : int
    timeframe: str

@dataclass
class MarketStructure:
    timeframe    : str
    trend        : str
    last_bos     : Optional[str] = None
    last_choch   : Optional[str] = None
    last_swing_h : float = 0.0
    last_swing_l : float = 0.0
    prev_swing_h : float = 0.0
    prev_swing_l : float = 0.0
    hh_hl        : bool  = False
    ll_lh        : bool  = False
    liquidity_swept: bool = False

@dataclass
class SRLevel:
    price    : float
    kind     : str
    strength : int
    timeframe: str
    is_key_level: bool = False

@dataclass
class ChartPattern:
    name       : str
    direction  : str
    confidence : float
    target_pct : float

@dataclass
class TimeframeAnalysis:
    tf          : str
    bias        : str
    score       : float
    structure   : Optional[MarketStructure] = None
    fvg_zones   : List[FVGZone]    = field(default_factory=list)
    order_blocks: List[OrderBlock] = field(default_factory=list)
    sr_supports : List[SRLevel]    = field(default_factory=list)
    sr_resists  : List[SRLevel]    = field(default_factory=list)
    rsi         : float = 50.0
    macd_bias   : str   = "NEUTRAL"
    ema_bias    : str   = "NEUTRAL"
    signals     : List[str] = field(default_factory=list)

@dataclass
class TechnicalResult:
    tf_analyses     : Dict[str, TimeframeAnalysis] = field(default_factory=dict)
    mtf_bias        : str   = "NEUTRAL"
    mtf_agree_count : int   = 0
    dominant_tf     : str   = "4h"
    rsi_14          : float = 50.0
    rsi_signal      : str   = "NEUTRAL"
    macd_line       : float = 0.0
    macd_signal     : float = 0.0
    macd_hist       : float = 0.0
    macd_signal_type: str   = "NEUTRAL"
    stoch_rsi_k     : float = 50.0
    stoch_rsi_d     : float = 50.0
    stoch_signal    : str   = "NEUTRAL"
    ema9            : float = 0.0
    ema21           : float = 0.0
    ema50           : float = 0.0
    ema200          : float = 0.0
    trend_alignment : str   = "SIDEWAYS"
    price_vs_ema200 : str   = "ABOVE"
    supertrend_dir  : str   = "NEUTRAL"
    bb_upper        : float = 0.0
    bb_middle       : float = 0.0
    bb_lower        : float = 0.0
    bb_width        : float = 0.0
    bb_position     : str   = "MIDDLE"
    atr_14          : float = 0.0
    atr_pct         : float = 0.0
    bb_squeeze      : bool  = False
    fvg_zones       : List[FVGZone]    = field(default_factory=list)
    order_blocks    : List[OrderBlock] = field(default_factory=list)
    nearest_bull_fvg: Optional[FVGZone]    = None
    nearest_bear_fvg: Optional[FVGZone]    = None
    nearest_bull_ob : Optional[OrderBlock] = None
    nearest_bear_ob : Optional[OrderBlock] = None
    price_in_fvg    : bool  = False
    price_in_ob     : bool  = False
    bos_detected    : bool  = False
    choch_detected  : bool  = False
    bos_direction   : str   = ""
    choch_direction : str   = ""
    liquidity_swept : bool  = False
    in_premium_zone : bool  = False
    in_discount_zone: bool  = False
    key_supports    : List[SRLevel] = field(default_factory=list)
    key_resists     : List[SRLevel] = field(default_factory=list)
    nearest_support : float = 0.0
    nearest_resist  : float = 0.0
    support_dist_pct: float = 0.0
    resist_dist_pct : float = 0.0
    price_at_support: bool  = False
    price_at_resist : bool  = False
    optimal_entry_low : float = 0.0
    optimal_entry_high: float = 0.0
    structure_sl      : float = 0.0
    has_precise_entry : bool  = False
    patterns        : List[ChartPattern] = field(default_factory=list)
    dominant_pattern: Optional[ChartPattern] = None
    signals         : List[str] = field(default_factory=list)
    invalidation    : List[str] = field(default_factory=list)
    ta_bias         : str   = "NEUTRAL"
    score           : float = 0.0
    is_gated        : bool  = False
    timeframe_used  : str   = "mtf"


# === HELPER MATH ===

def _rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(len(gain)); al = np.zeros(len(loss))
    if len(gain) < period:
        return np.full(len(close), np.nan)
    ag[period-1] = np.mean(gain[:period])
    al[period-1] = np.mean(loss[:period])
    for i in range(period, len(gain)):
        ag[i] = (ag[i-1]*(period-1)+gain[i])/period
        al[i] = (al[i-1]*(period-1)+loss[i])/period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(al != 0, np.divide(ag, al, where=(al != 0)), 100.0)
    rsi = 100-(100/(1+rs))
    result = np.full(len(close), np.nan)
    result[period:] = rsi[period-1:]
    return result

def _ema(data, period):
    result = np.full(len(data), np.nan)
    if len(data) < period:
        return result
    k = 2.0/(period+1)
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i]*k + result[i-1]*(1-k)
    return result

def _sma(data, period):
    result = np.full(len(data), np.nan)
    for i in range(period-1, len(data)):
        result[i] = np.mean(data[i-period+1:i+1])
    return result

def _macd(close, fast=12, slow=26, signal=9):
    ef = _ema(close, fast); es = _ema(close, slow); ml = ef-es
    vm = ~np.isnan(ml); sl = np.full(len(close), np.nan)
    if vm.sum() >= signal:
        sl[vm] = _ema(ml[vm], signal)
    return ml, sl, ml-sl

def _bollinger(close, period=20, std_dev=2.0):
    mid = _sma(close, period)
    std = np.array([np.std(close[max(0,i-period+1):i+1]) if i>=period-1 else np.nan for i in range(len(close))])
    return mid+std_dev*std, mid, mid-std_dev*std

def _atr(high, low, close, period=14):
    tr = np.maximum(high[1:]-low[1:], np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    atr = np.full(len(close), np.nan)
    if len(tr) < period:
        return atr
    atr[period] = np.mean(tr[:period])
    for i in range(period+1, len(close)):
        atr[i] = (atr[i-1]*(period-1)+tr[i-1])/period
    return atr

def _stoch_rsi(close, rsi_p=14, stoch_p=14, k_s=3, d_s=3):
    rv = _rsi(close, rsi_p); sk = np.full(len(close), np.nan)
    for i in range(stoch_p-1, len(rv)):
        w = rv[i-stoch_p+1:i+1]
        if np.any(np.isnan(w)): continue
        mn, mx = np.min(w), np.max(w)
        sk[i] = (rv[i]-mn)/(mx-mn)*100 if mx != mn else 50
    return _sma(sk, k_s), _sma(_sma(sk, k_s), d_s)

def _supertrend(high, low, close, period=10, multiplier=3.0):
    atr = _atr(high, low, close, period); hl2 = (high+low)/2
    direction = np.ones(len(close), dtype=int)
    fu = np.full(len(close), np.nan); fd = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if np.isnan(atr[i]): continue
        bu = hl2[i]-multiplier*atr[i]; bd = hl2[i]+multiplier*atr[i]
        if i == period:
            fu[i] = bu; fd[i] = bd
        else:
            fu[i] = max(bu, fu[i-1]) if (not np.isnan(fu[i-1]) and close[i-1]>fu[i-1]) else bu
            fd[i] = min(bd, fd[i-1]) if (not np.isnan(fd[i-1]) and close[i-1]<fd[i-1]) else bd
        pfd = fd[i-1] if not np.isnan(fd[i-1]) else fd[i]
        pfu = fu[i-1] if not np.isnan(fu[i-1]) else fu[i]
        if close[i] > pfd: direction[i] = 1
        elif close[i] < pfu: direction[i] = -1
        else: direction[i] = direction[i-1]
    return direction, fu, fd

def safe_last(arr):
    v = arr[~np.isnan(arr)]
    return float(v[-1]) if len(v) > 0 else 0.0

def find_swing_points(high, low, window=5, tf="1h"):
    swings = []
    n = len(high)
    for i in range(window, n-window):
        if high[i] == np.max(high[i-window:i+window+1]) and high[i] > np.mean(high[i-window:i+window+1])*1.002:
            swings.append(SwingPoint(kind='HIGH', price=high[i], idx=i, timeframe=tf))
        if low[i] == np.min(low[i-window:i+window+1]) and low[i] < np.mean(low[i-window:i+window+1])*0.998:
            swings.append(SwingPoint(kind='LOW', price=low[i], idx=i, timeframe=tf))
    return swings


# === FVG PRECISION ===

def detect_fvg_precise(df, tf, current_price, lookback=50):
    fvgs = []
    if len(df) < 3 or current_price <= 0:
        return fvgs
    high = df["high"].values; low = df["low"].values
    close = df["close"].values; open_ = df["open"].values
    vol = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    n = len(df); start = max(0, n-lookback-2)

    for i in range(start+2, n):
        mid_body = abs(close[i-1]-open_[i-1])
        mid_size = high[i-1]-low[i-1]
        if current_price <= 0: continue

        # Bullish FVG
        if high[i-2] < low[i]:
            gap_bot = high[i-2]; gap_top = low[i]
            gap_size = (gap_top-gap_bot)/current_price
            if gap_size < FVG_MIN_SIZE_PCT: continue
            is_imp = (close[i-1]>open_[i-1] and mid_size>0 and mid_body/mid_size>0.5)
            future = low[i+1:] if i+1<n else np.array([])
            filled = bool(np.any(future <= gap_bot+(gap_top-gap_bot)*0.5))
            part = bool(np.any(future <= gap_top)) and not filled
            if gap_top < current_price*1.15:
                fvgs.append(FVGZone(kind="BULLISH", top=gap_top, bottom=gap_bot,
                    midpoint=(gap_top+gap_bot)/2, size_pct=gap_size*100,
                    candle_idx=i, timeframe=tf, is_impulsive=is_imp, filled=filled, partially_filled=part))

        # Bearish FVG
        if low[i-2] > high[i]:
            gap_top = low[i-2]; gap_bot = high[i]
            gap_size = (gap_top-gap_bot)/current_price
            if gap_size < FVG_MIN_SIZE_PCT: continue
            is_imp = (close[i-1]<open_[i-1] and mid_size>0 and mid_body/mid_size>0.5)
            future = high[i+1:] if i+1<n else np.array([])
            filled = bool(np.any(future >= gap_top-(gap_top-gap_bot)*0.5))
            part = bool(np.any(future >= gap_bot)) and not filled
            if gap_bot > current_price*0.85:
                fvgs.append(FVGZone(kind="BEARISH", top=gap_top, bottom=gap_bot,
                    midpoint=(gap_top+gap_bot)/2, size_pct=gap_size*100,
                    candle_idx=i, timeframe=tf, is_impulsive=is_imp, filled=filled, partially_filled=part))
    return fvgs


# === ORDER BLOCK PRECISION ===

def detect_ob_precise(df, tf, current_price, lookback=80):
    obs = []
    if len(df) < 3 or current_price <= 0:
        return obs
    open_ = df["open"].values; high = df["high"].values
    low = df["low"].values; close = df["close"].values
    vol = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    n = len(df); start = max(0, n-lookback-1)
    avg_vol = np.mean(vol[max(0,n-50):]) if n > 5 else 1.0

    for i in range(start+1, n-1):
        pr = close[i]
        if pr <= 0: continue
        ob_body = abs(close[i]-open_[i])/pr
        if ob_body < OB_MIN_BODY_PCT: continue
        imp_size = abs(close[i+1]-open_[i+1])/pr
        vol_spike = bool(vol[i+1] > avg_vol*1.5) if avg_vol > 0 else False

        # Bullish OB
        if close[i]<open_[i] and close[i+1]>open_[i+1] and imp_size>0.010:
            fl = low[i+2:] if i+2<n else np.array([])
            breached = bool(np.any(fl < low[i]*0.998))
            if high[i] < current_price*1.10:
                obs.append(OrderBlock(kind="BULLISH", top=high[i], bottom=low[i],
                    midpoint=(high[i]+low[i])/2, candle_idx=i, timeframe=tf,
                    body_size_pct=ob_body*100, volume_spike=vol_spike,
                    breached=breached, valid=not breached))

        # Bearish OB
        if close[i]>open_[i] and close[i+1]<open_[i+1] and imp_size>0.010:
            fh = high[i+2:] if i+2<n else np.array([])
            breached = bool(np.any(fh > high[i]*1.002))
            if low[i] > current_price*0.90:
                obs.append(OrderBlock(kind="BEARISH", top=high[i], bottom=low[i],
                    midpoint=(high[i]+low[i])/2, candle_idx=i, timeframe=tf,
                    body_size_pct=ob_body*100, volume_spike=vol_spike,
                    breached=breached, valid=not breached))
    return obs


# === MARKET STRUCTURE ===

def analyze_market_structure(df, tf, current_price):
    ms = MarketStructure(timeframe=tf, trend="RANGING")
    if len(df) < 20 or current_price <= 0:
        return ms
    high = df["high"].values; low = df["low"].values; close = df["close"].values
    win = 3 if tf=="15m" else (4 if tf=="1h" else (5 if tf=="4h" else 7))
    swings = find_swing_points(high, low, window=win, tf=tf)
    if not swings: return ms
    sh = [(s.idx, s.price) for s in swings if s.kind=='HIGH']
    sl = [(s.idx, s.price) for s in swings if s.kind=='LOW']
    if not sh or not sl: return ms
    rsh = sorted(sh[-4:], key=lambda x: x[0])
    rsl = sorted(sl[-4:], key=lambda x: x[0])
    ms.last_swing_h = rsh[-1][1] if rsh else 0.0
    ms.last_swing_l = rsl[-1][1] if rsl else 0.0
    ms.prev_swing_h = rsh[-2][1] if len(rsh)>=2 else 0.0
    ms.prev_swing_l = rsl[-2][1] if len(rsl)>=2 else 0.0
    if ms.prev_swing_h > 0 and ms.prev_swing_l > 0:
        hh = ms.last_swing_h > ms.prev_swing_h*1.001
        hl = ms.last_swing_l > ms.prev_swing_l*1.001
        ll = ms.last_swing_l < ms.prev_swing_l*0.999
        lh = ms.last_swing_h < ms.prev_swing_h*0.999
        ms.hh_hl = hh and hl; ms.ll_lh = ll and lh
        if ms.hh_hl: ms.trend = "BULLISH"
        elif ms.ll_lh: ms.trend = "BEARISH"
        else: ms.trend = "RANGING"
    if ms.last_swing_h > 0 and close[-1] > ms.last_swing_h: ms.last_bos = "BULLISH"
    elif ms.last_swing_l > 0 and close[-1] < ms.last_swing_l: ms.last_bos = "BEARISH"
    if ms.trend=="BULLISH" and ms.ll_lh: ms.last_choch = "BEARISH"
    elif ms.trend=="BEARISH" and ms.hh_hl: ms.last_choch = "BULLISH"
    if len(close)>=5 and ms.last_swing_l > 0:
        rl = low[-5:]
        if np.any(rl < ms.last_swing_l*0.997) and close[-1] > ms.last_swing_l:
            ms.liquidity_swept = True
    return ms


def check_premium_discount(current_price, swing_high, swing_low):
    if swing_high <= swing_low or swing_low <= 0: return False, False
    eq = (swing_high+swing_low)/2
    return current_price > eq, current_price <= eq


def detect_sr_levels(df, current_price, tf):
    if len(df) < 20 or current_price <= 0: return [], []
    high = df["high"].values; low = df["low"].values; tol = 0.015
    pivots = []
    win = 3
    for i in range(win, len(df)-win):
        if high[i] == np.max(high[i-win:i+win+1]): pivots.append(("R", high[i]))
        if low[i] == np.min(low[i-win:i+win+1]): pivots.append(("S", low[i]))
    def cluster(prices):
        if not prices: return []
        sp = sorted(prices); clusters = []; cur = [sp[0]]
        for p in sp[1:]:
            if cur[-1]>0 and abs(p-cur[-1])/cur[-1]<tol: cur.append(p)
            else: clusters.append(cur); cur = [p]
        clusters.append(cur)
        return [(float(np.median(c)), len(c)) for c in clusters]
    s_raw = [p for k,p in pivots if k=="S" and p<current_price]
    r_raw = [p for k,p in pivots if k=="R" and p>current_price]
    supports = [SRLevel(price=p, kind="SUPPORT", strength=s, timeframe=tf, is_key_level=s>=3) for p,s in cluster(s_raw)]
    resists  = [SRLevel(price=p, kind="RESISTANCE", strength=s, timeframe=tf, is_key_level=s>=3) for p,s in cluster(r_raw)]
    return sorted(supports, key=lambda x: x.price, reverse=True)[:6], sorted(resists, key=lambda x: x.price)[:6]


def detect_patterns(df, current_price):
    patterns = []
    if len(df) < 40 or current_price <= 0: return patterns
    close = df["close"].values; high = df["high"].values; low = df["low"].values; n = len(close)
    def find_pivots(arr, is_high, win=5):
        pv = []
        for i in range(win, len(arr)-win):
            seg = arr[i-win:i+win+1]
            if is_high and arr[i]==np.max(seg) and arr[i]>np.mean(seg)*1.001: pv.append((i, arr[i]))
            elif not is_high and arr[i]==np.min(seg) and arr[i]<np.mean(seg)*0.999: pv.append((i, arr[i]))
        return pv
    ph = find_pivots(high, True); pl = find_pivots(low, False)
    if len(pl)>=2:
        b1i,b1=pl[-2]; b2i,b2=pl[-1]
        if b2i>b1i and abs(b1-b2)/max(b1,1e-10)<0.02:
            neck=max(high[b1i:b2i+1])
            if current_price > neck*0.97:
                patterns.append(ChartPattern("Double Bottom","BULLISH",0.78,(neck-min(b1,b2))/neck*100))
    if len(ph)>=2:
        t1i,t1=ph[-2]; t2i,t2=ph[-1]
        if t2i>t1i and abs(t1-t2)/max(t1,1e-10)<0.02:
            neck=min(low[t1i:t2i+1])
            if current_price < neck*1.03:
                patterns.append(ChartPattern("Double Top","BEARISH",0.75,(max(t1,t2)-neck)/neck*100))
    if len(pl)>=3:
        si,s=pl[-3]; hi,h=pl[-2]; ri,r=pl[-1]
        if h<s and h<r and abs(s-r)/max(s,1e-10)<0.035 and ri>hi>si:
            neck=max(max(high[si:hi+1]),max(high[hi:ri+1]))
            if current_price > neck*0.97:
                patterns.append(ChartPattern("Inv H&S","BULLISH",0.82,(neck-h)/neck*100))
    if len(ph)>=3:
        si,s=ph[-3]; hi,h=ph[-2]; ri,r=ph[-1]
        if h>s and h>r and abs(s-r)/max(s,1e-10)<0.035 and ri>hi>si:
            neck=min(min(low[si:hi+1]),min(low[hi:ri+1]))
            if current_price < neck*1.03:
                patterns.append(ChartPattern("H&S","BEARISH",0.82,(h-neck)/neck*100))
    if n>=20:
        pm=(np.max(close[-20:-10])-close[-20])/max(close[-20],1e-10)
        fr=(np.max(close[-10:])-np.min(close[-10:]))/max(np.max(close[-10:]),1e-10)
        if pm>0.05 and fr<0.035 and current_price>close[-20]:
            patterns.append(ChartPattern("Bull Flag","BULLISH",0.72,pm*100))
        pm2=(close[-20]-np.min(close[-20:-10]))/max(close[-20],1e-10)
        if pm2>0.05 and fr<0.035 and current_price<close[-20]:
            patterns.append(ChartPattern("Bear Flag","BEARISH",0.72,pm2*100))
    if len(ph)>=2 and len(pl)>=2:
        if abs(ph[-1][1]-ph[-2][1])/max(ph[-2][1],1e-10)<0.018 and pl[-1][1]>pl[-2][1]*1.005:
            patterns.append(ChartPattern("Ascending Triangle","BULLISH",0.68,5.0))
        if abs(pl[-1][1]-pl[-2][1])/max(pl[-2][1],1e-10)<0.018 and ph[-1][1]<ph[-2][1]*0.995:
            patterns.append(ChartPattern("Descending Triangle","BEARISH",0.68,5.0))
    return patterns


# === SINGLE TF ANALYSIS ===

def _analyze_single_tf(symbol, tf, current_price, limit=200):
    df = get_klines(symbol, interval=tf, limit=limit)
    if df.empty or len(df) < 30: return None
    tfa = TimeframeAnalysis(tf=tf, bias="NEUTRAL", score=0.0)
    score = 0.0
    close = df["close"].values; high = df["high"].values; low = df["low"].values

    # RSI
    rv = _rsi(close, 14); vr = rv[~np.isnan(rv)]
    if len(vr) >= 3:
        tfa.rsi = float(vr[-1])
        if len(close) >= 20:
            price_trend = close[-1] < np.min(close[-15:-1])
            rsi_trend = tfa.rsi > float(np.nanmin(rv[-15:-1]))
            if price_trend and rsi_trend and tfa.rsi < 50:
                tfa.signals.append(f"💚 [{tf}] RSI Bullish Divergence {tfa.rsi:.0f}"); score += 8
            elif tfa.rsi <= 25: tfa.signals.append(f"💚 [{tf}] RSI Extreme Oversold {tfa.rsi:.0f}"); score += 7
            elif tfa.rsi <= 35: tfa.signals.append(f"💚 [{tf}] RSI Oversold {tfa.rsi:.0f}"); score += 5
            elif tfa.rsi >= 75: tfa.signals.append(f"🔴 [{tf}] RSI Overbought {tfa.rsi:.0f}"); score -= 4
            elif tfa.rsi >= 65 and float(vr[-2]) > tfa.rsi:
                tfa.signals.append(f"⚠️ [{tf}] RSI Bearish Divergence {tfa.rsi:.0f}"); score -= 5

    # MACD
    ml, ms_, mh = _macd(close); vm = ~(np.isnan(ml)|np.isnan(ms_))
    if vm.sum() >= 3:
        idxs = np.where(vm)[0]; la, pr = idxs[-1], idxs[-2]
        mh_now = float(mh[la]) if not np.isnan(mh[la]) else 0.0
        mh_prev = float(mh[pr]) if not np.isnan(mh[pr]) else 0.0
        if float(ml[pr])<=float(ms_[pr]) and float(ml[la])>float(ms_[la]):
            tfa.macd_bias="BULLISH_CROSS"; tfa.signals.append(f"💚 [{tf}] MACD Golden Cross"); score+=8
        elif float(ml[pr])>=float(ms_[pr]) and float(ml[la])<float(ms_[la]):
            tfa.macd_bias="BEARISH_CROSS"; tfa.signals.append(f"🔴 [{tf}] MACD Death Cross"); score-=7
        elif mh_now>0 and mh_now>mh_prev: tfa.macd_bias="BULLISH"; score+=3
        elif mh_now<0 and mh_now<mh_prev: tfa.macd_bias="BEARISH"; score-=3
        else: tfa.macd_bias="NEUTRAL"

    # EMA
    e9=safe_last(_ema(close,9)); e21=safe_last(_ema(close,21)); e50=safe_last(_ema(close,50))
    e200=safe_last(_ema(close,min(200,len(close)-1))) if len(close)>=50 else 0.0
    bc = sum([current_price>e9>0, current_price>e21>0, current_price>e50>0, current_price>e200>0, e9>e21>0, e21>e50>0])
    if bc>=5: tfa.ema_bias="STRONG_BULL"; score+=6
    elif bc>=4: tfa.ema_bias="BULL"; score+=4
    elif bc<=1: tfa.ema_bias="STRONG_BEAR"; score-=5
    elif bc<=2: tfa.ema_bias="BEAR"; score-=3
    else: tfa.ema_bias="SIDEWAYS"

    # Supertrend
    if len(close)>=15:
        st_dir,_,_ = _supertrend(high,low,close)
        score += 3 if st_dir[-1]==1 else -2

    # FVG
    tfa.fvg_zones = detect_fvg_precise(df, tf, current_price, lookback=60)
    bull_fvgs = [f for f in tfa.fvg_zones if f.kind=="BULLISH" and not f.filled and f.top<current_price]
    if bull_fvgs:
        nearest = max(bull_fvgs, key=lambda x: x.top)
        dist = (current_price-nearest.top)/current_price*100
        if dist < 2.0 and nearest.is_impulsive:
            tfa.signals.append(f"🧲 [{tf}] Bullish FVG Impulsif {dist:.1f}% di bawah (${nearest.bottom:.5f}–${nearest.top:.5f})"); score+=7
        elif dist < 5.0:
            tfa.signals.append(f"🧲 [{tf}] Bullish FVG {dist:.1f}% di bawah"); score+=4
    bear_fvgs = [f for f in tfa.fvg_zones if f.kind=="BEARISH" and not f.filled and f.bottom>current_price]
    if bear_fvgs:
        nb = min(bear_fvgs, key=lambda x: x.bottom)
        dist_b = (nb.bottom-current_price)/current_price*100
        if dist_b < 3.0: tfa.signals.append(f"🟥 [{tf}] Bearish FVG resistance {dist_b:.1f}% di atas"); score-=3

    # OB
    tfa.order_blocks = detect_ob_precise(df, tf, current_price, lookback=80)
    vbo = [o for o in tfa.order_blocks if o.kind=="BULLISH" and o.valid and o.top<current_price]
    if vbo:
        nob = max(vbo, key=lambda x: x.top)
        dist_ob = (current_price-nob.top)/current_price*100
        if dist_ob < 2.0:
            tfa.signals.append(f"🟩 [{tf}] Bullish OB {dist_ob:.1f}% di bawah" + (" (vol spike ✅)" if nob.volume_spike else "")); score+=6+(3 if nob.volume_spike else 0)
        elif dist_ob < 5.0:
            tfa.signals.append(f"🟩 [{tf}] Bullish OB {dist_ob:.1f}% di bawah"); score+=3

    # Market Structure
    tfa.structure = analyze_market_structure(df, tf, current_price)
    ms = tfa.structure
    if ms.last_bos=="BULLISH": tfa.signals.append(f"🏗️ [{tf}] BOS Bullish!"); score+=5
    elif ms.last_bos=="BEARISH": tfa.signals.append(f"📉 [{tf}] BOS Bearish"); score-=4
    if ms.last_choch=="BULLISH": tfa.signals.append(f"🔄 [{tf}] CHoCH Bullish — reversal!"); score+=6
    elif ms.last_choch=="BEARISH": tfa.signals.append(f"🔄 [{tf}] CHoCH Bearish — waspada"); score-=5
    if ms.liquidity_swept: tfa.signals.append(f"💧 [{tf}] Liquidity Swept (stop hunt)!"); score+=7
    if ms.hh_hl: tfa.signals.append(f"📈 [{tf}] HH+HL — Uptrend"); score+=4
    elif ms.ll_lh: tfa.signals.append(f"📉 [{tf}] LL+LH — Downtrend"); score-=3

    # S&R
    tfa.sr_supports, tfa.sr_resists = detect_sr_levels(df, current_price, tf)
    if tfa.sr_supports:
        ns = tfa.sr_supports[0]; dist_s = (current_price-ns.price)/current_price*100
        if dist_s < SR_PROXIMITY_PCT*100:
            tfa.signals.append(f"🟢 [{tf}] Dekat Support{' KEY' if ns.is_key_level else ''} ${ns.price:.5f} ({dist_s:.1f}%)"); score+=4+(4 if ns.is_key_level else 0)
    if tfa.sr_resists:
        nr = tfa.sr_resists[0]; dist_r = (nr.price-current_price)/current_price*100
        if dist_r < SR_PROXIMITY_PCT*100:
            tfa.signals.append(f"🔴 [{tf}] Dekat Resistance ${nr.price:.5f} ({dist_r:.1f}%)"); score-=3

    # Bollinger
    bbu,bbm,bbl = _bollinger(close,20,2.0)
    if not np.isnan(bbl[-1]) and current_price>0:
        if current_price < bbl[-1]: tfa.signals.append(f"💚 [{tf}] Harga di bawah BB Lower"); score+=5
        elif current_price > bbu[-1]: score-=2
        if len(close)>=20 and not np.isnan(bbu[-20:]).all():
            rw = np.nanmean((bbu[-20:]-bbl[-20:])/(bbm[-20:]+1e-10))
            cw = (bbu[-1]-bbl[-1])/(bbm[-1]+1e-10) if bbm[-1]>0 else 0
            if cw < rw*0.45 and cw>0: tfa.signals.append(f"🌀 [{tf}] BB SQUEEZE!"); score+=6

    tfa.score = score
    tfa.bias = "BULLISH" if score>=15 else ("BEARISH" if score<=-8 else "NEUTRAL")
    return tfa


# === MAIN ENTRY POINT ===

def analyze_technical(symbol: str, current_price: float) -> TechnicalResult:
    res = TechnicalResult()
    if current_price <= 0: return res

    tf_configs = [("15m",100),("1h",200),("4h",150),("1d",100)]
    tf_results = {}
    for tf, limit in tf_configs:
        try:
            tfa = _analyze_single_tf(symbol, tf, current_price, limit)
            if tfa: tf_results[tf] = tfa
        except Exception as e:
            logger.warning(f"TA [{tf}] {symbol}: {e}")
    res.tf_analyses = tf_results
    if not tf_results: return res

    bull_tfs = [tf for tf,a in tf_results.items() if a.bias=="BULLISH"]
    bear_tfs = [tf for tf,a in tf_results.items() if a.bias=="BEARISH"]
    res.mtf_agree_count = max(len(bull_tfs), len(bear_tfs))
    if len(bull_tfs)>=3: res.mtf_bias="STRONG_BULLISH"
    elif len(bull_tfs)>=2: res.mtf_bias="BULLISH"
    elif len(bear_tfs)>=3: res.mtf_bias="STRONG_BEARISH"
    elif len(bear_tfs)>=2: res.mtf_bias="BEARISH"
    else: res.mtf_bias="NEUTRAL"

    # Populate 1H fields untuk kompatibilitas gateway
    if "1h" in tf_results:
        tfa1h = tf_results["1h"]
        res.rsi_14 = tfa1h.rsi
        res.macd_signal_type = tfa1h.macd_bias
        res.trend_alignment  = tfa1h.ema_bias
        df1h = get_klines(symbol, interval="1h", limit=200)
        if not df1h.empty:
            c = df1h["close"].values; h = df1h["high"].values; l = df1h["low"].values
            res.ema9=safe_last(_ema(c,9)); res.ema21=safe_last(_ema(c,21))
            res.ema50=safe_last(_ema(c,50)); res.ema200=safe_last(_ema(c,min(200,len(c)-1))) if len(c)>=50 else 0.0
            atrv = _atr(h,l,c,14); res.atr_14=safe_last(atrv)
            res.atr_pct = res.atr_14/current_price*100 if current_price>0 else 0
            bbu,bbm,bbl = _bollinger(c,20,2.0)
            if not np.isnan(bbu[-1]):
                res.bb_upper=float(bbu[-1]); res.bb_middle=float(bbm[-1]); res.bb_lower=float(bbl[-1])
                res.bb_width=(res.bb_upper-res.bb_lower)/res.bb_middle if res.bb_middle>0 else 0
                res.bb_position="BELOW_LOWER" if current_price<res.bb_lower else ("ABOVE_UPPER" if current_price>res.bb_upper else "MIDDLE")
                if len(c)>=20:
                    rw=np.nanmean((bbu[-20:]-bbl[-20:])/(bbm[-20:]+1e-10))
                    cw=(bbu[-1]-bbl[-1])/(bbm[-1]+1e-10) if bbm[-1]>0 else 0
                    res.bb_squeeze = cw<rw*0.45 and cw>0
            if len(c)>=15:
                st_dir,_,_=_supertrend(h,l,c)
                res.supertrend_dir="BULLISH" if st_dir[-1]==1 else "BEARISH"
            sk,sd=_stoch_rsi(c)
            res.stoch_rsi_k=float(sk[~np.isnan(sk)][-1]) if not np.all(np.isnan(sk)) else 50.0
            res.stoch_rsi_d=float(sd[~np.isnan(sd)][-1]) if not np.all(np.isnan(sd)) else 50.0
            res.stoch_signal="OVERSOLD" if res.stoch_rsi_k<20 else ("OVERBOUGHT" if res.stoch_rsi_k>80 else "NEUTRAL")

    # Kumpulkan FVG & OB semua TF
    all_fvgs=[]; all_obs=[]
    for tfa in tf_results.values():
        all_fvgs.extend(tfa.fvg_zones); all_obs.extend(tfa.order_blocks)
    res.fvg_zones=all_fvgs; res.order_blocks=all_obs

    tf_priority={"1d":4,"4h":3,"1h":2,"15m":1}
    bull_fvgs=[f for f in all_fvgs if f.kind=="BULLISH" and not f.filled and f.top<current_price]
    bear_fvgs=[f for f in all_fvgs if f.kind=="BEARISH" and not f.filled and f.bottom>current_price]
    if bull_fvgs:
        bull_fvgs.sort(key=lambda x:(x.is_impulsive,tf_priority.get(x.timeframe,0),x.top),reverse=True)
        res.nearest_bull_fvg=bull_fvgs[0]
        res.price_in_fvg=(current_price-res.nearest_bull_fvg.top)/current_price < ZONE_PROXIMITY_PCT
    if bear_fvgs:
        bear_fvgs.sort(key=lambda x:(x.is_impulsive,tf_priority.get(x.timeframe,0),-x.bottom),reverse=True)
        res.nearest_bear_fvg=bear_fvgs[0]

    bull_obs=[o for o in all_obs if o.kind=="BULLISH" and o.valid and o.top<current_price]
    bear_obs=[o for o in all_obs if o.kind=="BEARISH" and o.valid and o.bottom>current_price]
    if bull_obs:
        bull_obs.sort(key=lambda x:(x.volume_spike,tf_priority.get(x.timeframe,0),x.top),reverse=True)
        res.nearest_bull_ob=bull_obs[0]
        res.price_in_ob=(current_price-res.nearest_bull_ob.top)/current_price < ZONE_PROXIMITY_PCT
    if bear_obs:
        bear_obs.sort(key=lambda x:(x.volume_spike,tf_priority.get(x.timeframe,0),-x.bottom),reverse=True)
        res.nearest_bear_ob=bear_obs[0]

    # Market Structure Summary
    for tfk in ["4h","1h","1d"]:
        if tfk in tf_results and tf_results[tfk].structure:
            ms=tf_results[tfk].structure
            if ms.last_bos: res.bos_detected=True; res.bos_direction=ms.last_bos
            if ms.last_choch: res.choch_detected=True; res.choch_direction=ms.last_choch
            if ms.liquidity_swept: res.liquidity_swept=True
            if ms.last_swing_h>0 and ms.last_swing_l>0:
                res.in_premium_zone, res.in_discount_zone = check_premium_discount(current_price,ms.last_swing_h,ms.last_swing_l)
            break

    # S&R Multi-TF
    all_s=[]; all_r=[]
    for tfa in tf_results.values(): all_s.extend(tfa.sr_supports); all_r.extend(tfa.sr_resists)
    def dedup_sr(levels):
        if not levels: return []
        sl=sorted(levels,key=lambda x:x.price); merged=[sl[0]]
        for lv in sl[1:]:
            if lv.price>0 and merged[-1].price>0 and abs(lv.price-merged[-1].price)/merged[-1].price<0.015:
                if lv.strength>merged[-1].strength: merged[-1]=lv
                else: merged[-1].strength+=1
            else: merged.append(lv)
        return merged
    res.key_supports=sorted(dedup_sr([s for s in all_s if s.price<current_price]),key=lambda x:x.price,reverse=True)[:5]
    res.key_resists =sorted(dedup_sr([r for r in all_r if r.price>current_price]),key=lambda x:x.price)[:5]
    if res.key_supports:
        res.nearest_support=res.key_supports[0].price
        res.support_dist_pct=(current_price-res.nearest_support)/current_price*100
        res.price_at_support=res.support_dist_pct<SR_PROXIMITY_PCT*100
    if res.key_resists:
        res.nearest_resist=res.key_resists[0].price
        res.resist_dist_pct=(res.nearest_resist-current_price)/current_price*100
        res.price_at_resist=res.resist_dist_pct<SR_PROXIMITY_PCT*100

    # Patterns
    for tfk in ["4h","1h"]:
        if tfk in tf_results:
            df_pat=get_klines(symbol,interval=tfk,limit=150)
            if not df_pat.empty: res.patterns.extend(detect_patterns(df_pat,current_price))
    if res.patterns:
        bp=[p for p in res.patterns if p.direction=="BULLISH"]
        bep=[p for p in res.patterns if p.direction=="BEARISH"]
        if bp: res.dominant_pattern=max(bp,key=lambda x:x.confidence)
        elif bep: res.dominant_pattern=max(bep,key=lambda x:x.confidence)

    # Precise Entry Zone
    if res.nearest_bull_fvg and not res.nearest_bull_fvg.filled:
        fvg=res.nearest_bull_fvg; dist=(current_price-fvg.top)/current_price*100
        if dist<8.0: res.optimal_entry_low=fvg.bottom; res.optimal_entry_high=fvg.top; res.has_precise_entry=True
    elif res.nearest_bull_ob and res.nearest_bull_ob.valid:
        ob=res.nearest_bull_ob; dist=(current_price-ob.top)/current_price*100
        if dist<8.0: res.optimal_entry_low=ob.bottom; res.optimal_entry_high=ob.top; res.has_precise_entry=True
    for tfk in ["4h","1h"]:
        if tfk in tf_results and tf_results[tfk].structure:
            sl_p=tf_results[tfk].structure.last_swing_l
            if sl_p>0 and sl_p<current_price: res.structure_sl=sl_p*0.997; break

    # RSI signal
    if res.rsi_14<=30: res.rsi_signal="OVERSOLD"
    elif res.rsi_14>=70: res.rsi_signal="OVERBOUGHT"
    else: res.rsi_signal="NEUTRAL"

    # Weighted Score
    ws=0.0
    for tf,weight in TF_WEIGHTS.items():
        if tf in tf_results: ws+=tf_results[tf].score*weight
    if res.mtf_bias=="STRONG_BULLISH": ws+=8
    elif res.mtf_bias=="BULLISH": ws+=4
    elif res.mtf_bias=="STRONG_BEARISH": ws-=8
    elif res.mtf_bias=="BEARISH": ws-=4
    if res.liquidity_swept and res.in_discount_zone:
        ws+=6; res.signals.append("💎 Liquidity Sweep + Discount Zone = PRIME ENTRY SETUP")
    if res.price_in_fvg and res.nearest_bull_fvg and res.nearest_bull_fvg.is_impulsive:
        ws+=5; res.signals.append(f"🧲 Harga dalam Bullish FVG Impulsif [{res.nearest_bull_fvg.timeframe}]")
    if res.price_in_ob and res.nearest_bull_ob and res.nearest_bull_ob.volume_spike:
        ws+=5; res.signals.append(f"🟩 Harga dalam Bullish OB (Vol Spike) [{res.nearest_bull_ob.timeframe}]")
    if res.bos_direction=="BULLISH": ws+=4; res.signals.append("🏗️ BOS Bullish dikonfirmasi [4H]")
    if res.choch_direction=="BULLISH": ws+=5; res.signals.append("🔄 CHoCH Bullish — reversal structure!")
    if res.in_discount_zone: ws+=3; res.signals.append("💚 Harga di Discount Zone (ICT)")
    elif res.in_premium_zone: ws-=3; res.signals.append("⚠️ Harga di Premium Zone — risiko lebih tinggi")
    for pat in res.patterns:
        if pat.direction=="BULLISH": ws+=pat.confidence*5; res.signals.append(f"📐 Pattern: {pat.name} BULLISH (target +{pat.target_pct:.1f}%)")
        else: ws-=pat.confidence*4; res.signals.append(f"📐 Pattern: {pat.name} BEARISH (target -{pat.target_pct:.1f}%)")
    for tfk in ["4h","1h","1d","15m"]:
        if tfk in tf_results: res.signals.extend(tf_results[tfk].signals[:3])

    # Invalidation
    inv=[]
    if res.nearest_bear_fvg and (res.nearest_bear_fvg.bottom-current_price)/current_price<0.03:
        inv.append(f"🚫 Bearish FVG sangat dekat di atas [{res.nearest_bear_fvg.timeframe}] — resistance kuat")
    if "1d" in tf_results and tf_results["1d"].bias=="BEARISH":
        inv.append("🚫 Bias 1D BEARISH — counter trend, risiko tinggi"); ws-=5
    if res.bos_direction=="BEARISH" and res.choch_direction!="BULLISH":
        inv.append("🚫 BOS Bearish aktif — struktur pasar turun")
    if res.in_premium_zone and res.nearest_bear_ob:
        inv.append("🚫 Premium Zone + Bearish OB overhead")
    res.invalidation=inv

    # Confluence Gate
    res.is_gated = (ws>=SIGNAL_GATE_SCORE and len(bull_tfs)>=MIN_TF_CONFLUENCE and len(inv)<=1)

    if ws>=15: res.ta_bias="STRONG_BULL"
    elif ws>=8: res.ta_bias="BULL"
    elif ws<=-12: res.ta_bias="STRONG_BEAR"
    elif ws<=-5: res.ta_bias="BEAR"
    else: res.ta_bias="NEUTRAL"

    res.score=max(0.0,min(ws,30.0))
    res.timeframe_used="mtf"
    return res
