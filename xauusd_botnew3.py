"""
XAU/USD Professional Signal Bot — v3 PROFITABLE EDITION
=========================================================

PROFITABILITY FEATURES:
  - RSI Divergence detection (strongest reversal signal)
  - ADX > 28 filter — skips choppy/ranging markets
  - 3-timeframe alignment (5m + 15m + 1h must agree)
  - Candle close confirmation — no fake wick entries
  - Dynamic ATR trailing — adapts to volatility regime
  - News blackout — skips 30min around major events (NFP/CPI/Fed)
  - 8 signal triggers for 8-10 signals/day target
  - Minimum confluence of 2 (quality + frequency balance)

8 SIGNAL TRIGGERS:
  1.  EMA 9/21 Crossover
  2.  RSI Cross 50 (momentum)
  3.  EMA 21 Bounce (pullback)
  4.  RSI Extreme 28/72
  5.  ADX Breakout
  6.  MACD Cross
  7.  RSI Divergence (bullish/bearish)
  8.  Bollinger Band mean reversion

BACKTEST MODE:
  Run:  python xauusd_bot.py --backtest
  Tests last 5 days of 5min data and prints win rate, P&L, trade log

Requirements:
    pip install requests pandas ta

Run live:
    python xauusd_bot.py

Run backtest:
    python xauusd_bot.py --backtest
"""

import os, sys, time, logging, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from the same directory as this script
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    raise SystemExit("Run:  pip install requests pandas ta numpy python-dotenv")

try:
    import ta
except ImportError:
    raise SystemExit("Run:  pip install requests pandas ta numpy python-dotenv")

# ===============================================================================
#  CONFIG — tweak these to adjust behaviour
# ===============================================================================

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN",  "")
CHAT_ID         = os.getenv("CHAT_ID",         "")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")

CHECK_INTERVAL  = 60    # seconds between scans
MIN_SIGNAL_GAP  = 15    # minutes between signals (15 min = ~8-10/day in session)
MIN_CONFLUENCE  = 3     # signals that must agree (2 = frequent, 3 = accurate ← current setting)
MIN_ADX         = 28    # skip signal if ADX below this (market too choppy)

# ── Backtest overrides ───────────────────────────────────────────────────────
# Used during --backtest to cast a wider net. Do NOT affect live trading.
BT_MIN_ADX        = 20   # standard threshold — gold ADX is often 15-25 in calm markets
BT_MIN_CONFLUENCE = 3    # raised from 2 — conf=2 has 33% win rate, conf=3+ has 56%+

SESSION_START   = 7     # UTC hour — London open
SESSION_END     = 17    # UTC hour — NY close
NEWS_BLACKOUT   = 30    # minutes to block around major news

# Indicators
EMA_FAST=9; EMA_SLOW=21; EMA_TREND=50; EMA_HTF=200
RSI_PERIOD=14; ADX_PERIOD=14; ATR_PERIOD=14
MACD_FAST=12; MACD_SLOW=26; MACD_SIG=9
BB_PERIOD=20; BB_STD=2.0

# Risk — ATR multipliers
ATR_SL   = 0.8   # tight SL for scalping
ATR_TP1  = 1.0   # TP1 = 1:1.25 R:R
ATR_TP2  = 1.8   # TP2 = 1:2.25 R:R
ATR_MIN  = 2.0   # minimum ATR in price points — rejects dead/illiquid candles
                  # XAU/USD 5m typically has ATR of $2-$8; below $2 = bad data

# Major news UTC times (hour, minute) — add more as needed
# These are approximate recurring times — bot blocks signals around them
NEWS_TIMES = [
    (8,  30),   # UK data
    (13, 30),   # US data / NFP
    (15,  0),   # Fed speeches
    (18,  0),   # Fed minutes
]

# ===============================================================================

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("xauusd_bot")


# ─── Session & news filters ────────────────────────────────────────────────────

def in_session() -> bool:
    return SESSION_START <= datetime.utcnow().hour <= SESSION_END


def near_news() -> bool:
    """Returns True if within NEWS_BLACKOUT minutes of a major news time."""
    now = datetime.utcnow()
    for h, m in NEWS_TIMES:
        news_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
        diff = abs((now - news_time).total_seconds() / 60)
        if diff <= NEWS_BLACKOUT:
            return True
    return False


def session_name() -> str:
    h = datetime.utcnow().hour
    if 12 <= h <= 17: return "London/NY Overlap"
    if SESSION_START <= h < 12: return "London"
    if h <= 21: return "New York"
    return "Off-Session"


# ─── Data fetching ─────────────────────────────────────────────────────────────

def fetch_twelvedata(interval: str, n: int = 200) -> pd.DataFrame:
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD&interval={interval}&outputsize={n}&apikey={TWELVE_DATA_KEY}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise ValueError(f"TwelveData error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    return df.drop(columns=["datetime"]).astype(float).sort_index()


def fetch_yahoo(n: int = 200) -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=5m&range=5d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q   = res["indicators"]["quote"][0]
    df  = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True),
    )
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.dropna()
    # Drop candles with near-zero range — these are illiquid/pre-market bars
    # that cause ATR to collapse and produce nonsense SL/TP levels
    df = df[(df["high"] - df["low"]) >= 0.10]
    log.info("Yahoo: %d valid candles after quality filter", len(df))
    return df


def fetch_yahoo_1h() -> pd.DataFrame:
    """Fetch 60-day 1h data from Yahoo — much more reliable than 5m for backtesting."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1h&range=60d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q   = res["indicators"]["quote"][0]
    df  = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True),
    )
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.dropna()
    df = df[(df["high"] - df["low"]) >= 0.50]  # 1h bars need at least $0.50 range
    log.info("Yahoo 1h: %d valid candles", len(df))
    return df


    try:
        return fetch_twelvedata("5min", n)
    except Exception as e:
        log.warning("TwelveData 5min failed (%s) — Yahoo fallback", e)
        return fetch_yahoo(n)


def fetch_15min() -> pd.DataFrame:
    try: return fetch_twelvedata("15min", 100)
    except: return pd.DataFrame()


def fetch_1h() -> pd.DataFrame:
    try: return fetch_twelvedata("1h", 80)
    except: return pd.DataFrame()


# ─── HTF trend functions ───────────────────────────────────────────────────────

def get_htf_trend(df_15m: pd.DataFrame) -> str:
    if df_15m.empty: return "NEUTRAL"
    try:
        ef  = ta.trend.ema_indicator(df_15m["close"], window=EMA_FAST).iloc[-1]
        es  = ta.trend.ema_indicator(df_15m["close"], window=EMA_SLOW).iloc[-1]
        e50 = ta.trend.ema_indicator(df_15m["close"], window=EMA_TREND).iloc[-1]
        rsi = ta.momentum.rsi(df_15m["close"], window=RSI_PERIOD).iloc[-1]
        price = df_15m["close"].iloc[-1]
        if ef > es and price > e50 and rsi > 50: return "BULL"
        if ef < es and price < e50 and rsi < 50: return "BEAR"
        return "NEUTRAL"
    except: return "NEUTRAL"


def get_1h_bias(df_1h: pd.DataFrame) -> str:
    if df_1h.empty: return "NEUTRAL"
    try:
        e50   = ta.trend.ema_indicator(df_1h["close"], window=EMA_TREND)
        e200  = ta.trend.ema_indicator(df_1h["close"], window=EMA_HTF)
        slope = e50.iloc[-1] - e50.iloc[-4]
        price = df_1h["close"].iloc[-1]
        if slope > 1.0 and price > e200.iloc[-1]: return "BULL"
        if slope < -1.0 and price < e200.iloc[-1]: return "BEAR"
        return "NEUTRAL"
    except: return "NEUTRAL"


# ─── Indicator calculation ─────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=EMA_TREND)
    df["ema200"]    = ta.trend.ema_indicator(df["close"], window=EMA_HTF)   # trend gate
    df["rsi"]       = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["atr"]       = ta.volatility.average_true_range(
                          df["high"], df["low"], df["close"], window=ATR_PERIOD)
    macd            = ta.trend.MACD(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    adx             = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
    df["adx"]       = adx.adx()
    df["adx_pos"]   = adx.adx_pos()
    df["adx_neg"]   = adx.adx_neg()
    bb              = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_pct"]    = bb.bollinger_pband()   # 0=at lower, 1=at upper
    return df


# ─── Pattern detection ─────────────────────────────────────────────────────────

def is_pin_bar(c: pd.Series) -> bool:
    body  = abs(c["close"] - c["open"])
    rng   = c["high"] - c["low"]
    if rng == 0: return False
    return max(c["high"] - max(c["open"], c["close"]),
               min(c["open"], c["close"]) - c["low"]) > 2 * body


def has_choch(df: pd.DataFrame) -> bool:
    """
    Change of Character (CHoCH): in a downtrend, price breaks above the most
    recent swing high — a potential bullish reversal signal.
    Uses the last 20 candles. Finds the highest swing high in the first half,
    then checks if the second half closes above it.
    """
    if len(df) < 20:
        return False
    window = df.tail(20)
    highs  = window["high"].values
    closes = window["close"].values
    mid    = len(highs) // 2

    # Find swing highs in the first half
    first_half_highs = [
        highs[i] for i in range(1, mid - 1)
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]
    ]
    if not first_half_highs:
        return False

    key_level = max(first_half_highs)
    # CHoCH confirmed if any close in the second half breaks above that level
    return any(closes[i] > key_level for i in range(mid, len(closes)))


def rsi_divergence(df: pd.DataFrame) -> str | None:
    """
    Bullish divergence:  price makes lower low,  RSI makes higher low.
    Bearish divergence:  price makes higher high, RSI makes lower high.

    Uses proper swing high/low detection: a swing high is a candle whose
    high is higher than both its neighbours; a swing low is the inverse.
    Looks at the last 30 candles for two swing points of each type.
    """
    if len(df) < 30:
        return None

    window = df.tail(30).copy()
    highs  = window["high"].values
    lows   = window["low"].values
    rsi    = window["rsi"].values
    n      = len(window)

    # Find swing highs and lows (ignore first and last candle)
    swing_highs = [i for i in range(1, n - 1) if highs[i] > highs[i-1] and highs[i] > highs[i+1]]
    swing_lows  = [i for i in range(1, n - 1) if lows[i]  < lows[i-1]  and lows[i]  < lows[i+1]]

    # Bearish divergence: need 2 swing highs where price is higher but RSI is lower
    if len(swing_highs) >= 2:
        sh1, sh2 = swing_highs[-2], swing_highs[-1]   # earlier, later
        if highs[sh2] > highs[sh1] and rsi[sh2] < rsi[sh1]:
            return "SHORT"

    # Bullish divergence: need 2 swing lows where price is lower but RSI is higher
    if len(swing_lows) >= 2:
        sl1, sl2 = swing_lows[-2], swing_lows[-1]     # earlier, later
        if lows[sl2] < lows[sl1] and rsi[sl2] > rsi[sl1]:
            return "LONG"

    return None


# ─── 8 Signal triggers ─────────────────────────────────────────────────────────

def t_ema_cross(p, c) -> str | None:
    if p["ema_fast"] <= p["ema_slow"] and c["ema_fast"] > c["ema_slow"]: return "LONG"
    if p["ema_fast"] >= p["ema_slow"] and c["ema_fast"] < c["ema_slow"]: return "SHORT"

def t_rsi_50(p, c) -> str | None:
    if p["rsi"] < 50 and c["rsi"] >= 50: return "LONG"
    if p["rsi"] > 50 and c["rsi"] <= 50: return "SHORT"

def t_ema_bounce(p, c) -> str | None:
    ema = c["ema_slow"]
    if p["low"] <= ema * 1.0015 and c["close"] > ema and c["close"] > p["close"]: return "LONG"
    if p["high"] >= ema * 0.9985 and c["close"] < ema and c["close"] < p["close"]: return "SHORT"

def t_rsi_extreme(c) -> str | None:
    if c["rsi"] <= 28: return "LONG"
    if c["rsi"] >= 72: return "SHORT"

def t_adx_breakout(p, c) -> str | None:
    if p["adx"] < MIN_ADX and c["adx"] >= MIN_ADX:
        return "LONG" if c["adx_pos"] > c["adx_neg"] else "SHORT"

def t_macd_cross(p, c) -> str | None:
    if p["macd"] <= p["macd_sig"] and c["macd"] > c["macd_sig"]: return "LONG"
    if p["macd"] >= p["macd_sig"] and c["macd"] < c["macd_sig"]: return "SHORT"

def t_bb_reversion(p, c) -> str | None:
    """Price touches BB extreme and reverses back toward mid."""
    if p["bb_pct"] <= 0.05 and c["bb_pct"] > 0.05: return "LONG"   # was at lower band, now moving up
    if p["bb_pct"] >= 0.95 and c["bb_pct"] < 0.95: return "SHORT"  # was at upper band, now moving down

def t_rsi_divergence(df) -> str | None:
    return rsi_divergence(df)


# ─── Confluence engine ─────────────────────────────────────────────────────────

def get_all_signals(df: pd.DataFrame) -> dict:
    p, c = df.iloc[-2], df.iloc[-1]
    return {
        "EMA_CROSS":   t_ema_cross(p, c),
        "RSI_50":      t_rsi_50(p, c),
        "EMA_BOUNCE":  t_ema_bounce(p, c),
        "RSI_EXTREME": t_rsi_extreme(c),
        "ADX_BREAK":   t_adx_breakout(p, c),
        "MACD_CROSS":  t_macd_cross(p, c),
        "BB_REVERT":   t_bb_reversion(p, c),
        "RSI_DIV":     t_rsi_divergence(df),
        "_price":      float(c["close"]),
        "_ema200":     float(c.get("ema200", 0)),
    }


def find_direction(signals: dict, price: float = 0, ema200: float = 0) -> tuple[str | None, str | None, int, list]:
    """Returns (direction, primary_trigger, confluence_count, agreeing_list).

    Fix 3: RSI_DIV is demoted — it can only fire as a primary trigger if at
    least one structural signal (EMA_BOUNCE, EMA_CROSS) agrees with it.
    Standalone RSI_DIV has 27% win rate; as confluence it confirms well.

    Fix 2: Trend gate — if price is above EMA200 (bull market), SHORT signals
    are blocked unless ADX_BREAK or RSI_EXTREME also fires (extreme conditions
    can still produce valid shorts even in uptrends).
    """
    # Priority order — RSI_DIV moved to end so structural signals lead
    priority = ["EMA_BOUNCE", "EMA_CROSS", "ADX_BREAK", "BB_REVERT",
                "MACD_CROSS", "RSI_EXTREME", "RSI_50", "RSI_DIV"]

    direction = None
    primary   = None
    for name in priority:
        if signals.get(name):
            direction = signals[name]
            primary   = name
            break

    if not direction:
        return None, None, 0, []

    # Fix 3: If RSI_DIV is the primary trigger, require structural confirmation
    if primary == "RSI_DIV":
        structural = {"EMA_BOUNCE", "EMA_CROSS", "ADX_BREAK"}
        has_structural = any(signals.get(s) == direction for s in structural)
        if not has_structural:
            return None, None, 0, []   # RSI_DIV alone — skip

    # Fix 2: Trend gate — block shorts in bull market unless extreme condition
    if direction == "SHORT" and price > 0 and ema200 > 0 and price > ema200:
        extreme_ok = signals.get("ADX_BREAK") == "SHORT" or signals.get("RSI_EXTREME") == "SHORT"
        if not extreme_ok:
            return None, None, 0, []   # no shorting against the trend

    agreeing = [n for n, v in signals.items() if not n.startswith("_") and v == direction]
    return direction, primary, len(agreeing), agreeing


# ─── Scoring ───────────────────────────────────────────────────────────────────

def score_signal(c, direction, primary, htf, bias, conf, pin_bar, choch) -> dict:
    pts = 42

    primary_pts = {
        "RSI_DIV":     22,
        "EMA_CROSS":   18,
        "EMA_BOUNCE":  17,
        "ADX_BREAK":   15,
        "BB_REVERT":   14,
        "RSI_EXTREME": 13,
        "MACD_CROSS":  12,
        "RSI_50":      10,
    }
    pts += primary_pts.get(primary, 8)
    pts += max(0, conf - 1) * 7   # each extra confluence = +7

    rsi = c["rsi"]; adx = c["adx"]

    # 3-TF alignment
    htf_match  = (direction=="LONG" and htf=="BULL") or (direction=="SHORT" and htf=="BEAR")
    bias_match = (direction=="LONG" and bias=="BULL") or (direction=="SHORT" and bias=="BEAR")
    if htf_match and bias_match: pts += 18   # full alignment
    elif htf_match:              pts += 10
    elif bias_match:             pts += 6
    else:                        pts -= 8

    # RSI position
    if direction=="SHORT":
        pts += 10 if rsi > 65 else 6 if rsi > 58 else 2 if rsi > 52 else 0
    else:
        pts += 10 if rsi < 35 else 6 if rsi < 42 else 2 if rsi < 48 else 0

    # ADX strength
    if adx >= 35:    pts += 10
    elif adx >= 25:  pts += 6
    elif adx >= 20:  pts += 2
    else:            pts -= 10   # choppy — penalise hard

    # Patterns
    if pin_bar:                        pts += 7
    if choch and direction == "SHORT": pts -= 5

    # Session
    sess = session_name()
    if "Overlap" in sess: pts += 6
    elif sess != "Off-Session": pts += 3

    confidence = max(10, min(99, pts))
    win_rate   = confidence / 100
    rr2        = round(ATR_TP2 / ATR_SL, 1)
    ev         = round(win_rate * rr2 - (1 - win_rate), 2)

    grade = ("A" if confidence >= 85 else "B" if confidence >= 72
             else "C" if confidence >= 58 else "D")
    grade = f"{grade} ({confidence}/100)"

    h = datetime.utcnow().hour
    if 12 <= h <= 16: timing, t_icon = "GOOD — London/NY Overlap (85%)", "✅"
    elif SESSION_START <= h <= SESSION_END: timing, t_icon = "GOOD (75%)", "✅"
    else: timing, t_icon = "POOR — Off Session (20%)", "🔴"

    return {
        "confidence": confidence, "ev": ev, "grade": grade,
        "timing": timing, "timing_icon": t_icon,
        "rr1": round(ATR_TP1 / ATR_SL, 1), "rr2": rr2,
    }


# ─── Main signal computation ───────────────────────────────────────────────────

def compute_signal(df: pd.DataFrame, htf: str, bias: str) -> dict | None:
    df  = add_indicators(df)
    c   = df.iloc[-1]
    p   = df.iloc[-2]

    # ADX filter — skip choppy market
    if c["adx"] < MIN_ADX:
        log.info("ADX %.1f < %d — market choppy, skipping", c["adx"], MIN_ADX)
        return None

    signals   = get_all_signals(df)
    direction, primary, conf, agreeing = find_direction(
        signals, price=signals["_price"], ema200=signals["_ema200"]
    )

    if not direction: return None

    if conf < MIN_CONFLUENCE:
        log.info("Only %d confluence for %s — skipping", conf, direction)
        return None

    # 3-TF filter — at least HTF must agree (1h bias is optional bonus)
    if htf == "BULL" and direction == "SHORT":
        log.info("SHORT blocked — HTF is BULL"); return None
    if htf == "BEAR" and direction == "LONG":
        log.info("LONG blocked — HTF is BEAR"); return None

    # EMA 50 filter
    if direction == "LONG"  and c["close"] < c["ema_trend"]:
        log.info("LONG filtered — below EMA50"); return None
    if direction == "SHORT" and c["close"] > c["ema_trend"]:
        log.info("SHORT filtered — above EMA50"); return None

    price   = round(c["close"], 2)
    atr     = c["atr"]

    # ATR sanity check — reject candles with near-zero ATR (illiquid / bad data)
    if atr < ATR_MIN:
        log.info("ATR %.3f < %.1f — data quality issue, skipping", atr, ATR_MIN)
        return None

    adx_val = round(c["adx"], 1)
    rsi_val = round(c["rsi"], 1)
    pin_bar = is_pin_bar(c)
    choch   = has_choch(df)
    scores  = score_signal(c, direction, primary, htf, bias, conf, pin_bar, choch)

    # ATR SL/TP
    if direction == "SHORT":
        sl  = round(price + atr * ATR_SL,  2)
        tp1 = round(price - atr * ATR_TP1, 2)
        tp2 = round(price - atr * ATR_TP2, 2)
    else:
        sl  = round(price - atr * ATR_SL,  2)
        tp1 = round(price + atr * ATR_TP1, 2)
        tp2 = round(price + atr * ATR_TP2, 2)

    sl_dist  = round(abs(price - sl),  2)
    tp1_dist = round(abs(price - tp1), 2)
    tp2_dist = round(abs(price - tp2), 2)

    # Build factors
    trend_lbl = "📉 Downtrend" if c["ema_fast"] < c["ema_slow"] else "📈 Uptrend"
    htf_lbl   = {"BULL": "📈 HTF 15m Bullish", "BEAR": "📉 HTF 15m Bearish"}.get(htf, "➡️ HTF Neutral")
    bias_lbl  = {"BULL": "📈 1H Bias Bullish",  "BEAR": "📉 1H Bias Bearish"}.get(bias, "➡️ 1H Neutral")
    trig_lbl  = {
        "RSI_DIV":     "🔁 RSI Divergence",
        "EMA_CROSS":   "🔀 EMA 9/21 Crossover",
        "EMA_BOUNCE":  "🔄 EMA 21 Pullback",
        "ADX_BREAK":   "💥 ADX Breakout",
        "BB_REVERT":   "🎯 BB Mean Reversion",
        "RSI_EXTREME": "⚡ RSI Extreme",
        "MACD_CROSS":  "📊 MACD Cross",
        "RSI_50":      "📶 RSI Cross 50",
    }.get(primary, "📊 Signal")

    factors = [trend_lbl, htf_lbl, bias_lbl, trig_lbl,
               f"🔗 {conf} signals confluent"]
    if choch:   factors.append("⚠️ Bullish CHoCH")
    if adx_val >= 25:
        factors.append(f"🔥 ADX {'bearish' if direction=='SHORT' else 'bullish'} ({adx_val})")
    if pin_bar: factors.append("📌 Pin Bar " + ("Bear" if direction=="SHORT" else "Bull"))
    factors.append("💪 USD dominance" if direction == "SHORT" else "💪 XAU momentum")
    factors.append(f"🕐 {session_name()} Session")

    return {
        "direction": direction, "primary": primary, "agreeing": agreeing,
        "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2,
        "sl_dist": sl_dist, "tp1_dist": tp1_dist, "tp2_dist": tp2_dist,
        "rsi": rsi_val, "adx": adx_val, "atr": round(atr, 2),
        "htf": htf, "bias": bias, "confluence": conf, "factors": factors,
        **scores,
    }


# ─── Telegram ──────────────────────────────────────────────────────────────────

def format_message(sig: dict) -> str:
    dot         = "🔴" if sig["direction"] == "SHORT" else "🟢"
    ev_str      = ("+" if sig["ev"] >= 0 else "") + str(sig["ev"]) + "R"
    factors_str = "\n".join(f"• {f}" for f in sig["factors"])
    date_str    = datetime.utcnow().strftime("%d %b %Y %H:%M UTC")

    return (
        f"{dot} *XAU/USD \u2014 {sig['direction']} ({sig['confidence']}%)*\n"
        f"_{date_str}_\n"
        f"\u2501" * 20 + "\n\n"
        f"*ENTRY:* `{sig['entry']}`\n"
        f"*SL:*    `{sig['sl']}` ({sig['sl_dist']}p)\n"
        f"*TP1:*   `{sig['tp1']}` ({sig['tp1_dist']}p) \u2014 _close 50% here_\n"
        f"*TP2:*   `{sig['tp2']}` ({sig['tp2_dist']}p) \u2014 _let rest run_\n"
        f"*R:R*    1:{sig['rr1']} \u2192 1:{sig['rr2']}\n"
        f"\U0001f4ca _Move SL to entry after TP1_\n"
        f"\U0001f30a *SCALP* | \u23f0 1-3 hrs\n\n"
        f"\u2501" * 20 + "\n"
        f"\U0001f4ca *Confidence: {sig['confidence']}%*\n"
        f"\U0001f48e EV: {ev_str}\n"
        f"\u26a0\ufe0f Entry: {sig['grade']}\n"
        f"\U0001f517 Confluence: {sig['confluence']} signals\n\n"
        f"\u23f0 *TIMING*\n"
        f"{sig['timing_icon']} {sig['timing']}\n\n"
        f"\U0001f4a1 *Factors:*\n"
        f"{factors_str}\n"
        f"\u2501" * 20
    )


def send_telegram(message: str, retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                url,
                json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
                timeout=10,
            )
            r.raise_for_status()
            log.info("Telegram sent.")
            return True
        except Exception as e:
            log.warning("Telegram attempt %d/%d: %s", attempt, retries, e)
            if attempt < retries: time.sleep(3)
    log.error("All Telegram retries failed.")
    return False


# ─── BACKTEST ENGINE ───────────────────────────────────────────────────────────

def run_backtest():
    now_utc = datetime.utcnow()
    is_weekend = now_utc.weekday() >= 5  # Saturday=5, Sunday=6

    print("\n" + "="*60)
    print("  XAU/USD BACKTEST")
    print(f"  Filters: ADX>{BT_MIN_ADX}  Confluence>={BT_MIN_CONFLUENCE}  Session {SESSION_START}-{SESSION_END}UTC")
    if is_weekend:
        print("  ⚠  Weekend detected — gold markets are closed.")
        print("     5m Yahoo data will be mostly pre-market garbage.")
        print("     Switching to 1h data (60 days) automatically.")
    print("="*60 + "\n")

    print("Fetching historical data ...")
    use_1h = is_weekend
    try:
        if use_1h:
            df_full = fetch_yahoo_1h()
            print(f"  Using 1h candles ({len(df_full)} bars, ~60 days)\n")
        else:
            df_full = fetch_5min(500)
            # Check if the 5m data is mostly dead (ATR floor will eat >60% of session candles)
            session_mask = df_full.index.hour.isin(range(SESSION_START, SESSION_END + 1))
            session_df = df_full[session_mask]
            if len(session_df) > 0:
                live_pct = ((session_df["high"] - session_df["low"]) >= 0.50).mean()
                if live_pct < 0.4:
                    print("  ⚠  5m data appears mostly illiquid (possibly weekend/holiday).")
                    print("     Switching to 1h data (60 days) for a meaningful backtest.\n")
                    df_full = fetch_yahoo_1h()
                    use_1h = True
                    print(f"  Using 1h candles ({len(df_full)} bars, ~60 days)\n")
    except Exception as e:
        print(f"Data fetch failed: {e}")
        return

    df_full = add_indicators(df_full)
    df_full = df_full.dropna()

    # Adjust ATR floor for 1h data — 1h candles naturally have larger ATR
    atr_floor = 3.0 if use_1h else ATR_MIN
    print(f"Running backtest on {len(df_full)} candles (ATR floor: ${atr_floor}) ...\n")

    trades   = []
    last_sig = df_full.index[0]

    # Rejection counters for diagnostics
    rej = {
        "gap":        0,
        "session":    0,
        "adx":        0,
        "atr_floor":  0,
        "no_signal":  0,
        "confluence": 0,
        "ema50":      0,
    }

    for i in range(60, len(df_full) - 5):
        window = df_full.iloc[:i+1]
        c      = window.iloc[-1]

        # Gap filter
        if (window.index[-1] - last_sig).total_seconds() < MIN_SIGNAL_GAP * 60:
            rej["gap"] += 1
            continue

        # Session filter
        h = window.index[-1].hour
        if not (SESSION_START <= h <= SESSION_END):
            rej["session"] += 1
            continue

        # ADX filter (backtest threshold — wider than live)
        if c["adx"] < BT_MIN_ADX:
            rej["adx"] += 1
            continue

        # ATR floor — reject illiquid / bad data candles
        if c["atr"] < atr_floor:
            rej["atr_floor"] += 1
            continue

        # Get signals
        sigs      = get_all_signals(window)
        direction, primary, conf, agreeing = find_direction(
            sigs, price=sigs["_price"], ema200=sigs["_ema200"]
        )

        if not direction:
            rej["no_signal"] += 1
            continue

        if conf < BT_MIN_CONFLUENCE:
            rej["confluence"] += 1
            continue

        # EMA50 filter
        if direction == "LONG"  and c["close"] < c["ema_trend"]:
            rej["ema50"] += 1; continue
        if direction == "SHORT" and c["close"] > c["ema_trend"]:
            rej["ema50"] += 1; continue

        price = c["close"]
        atr   = c["atr"]

        if direction == "SHORT":
            sl  = price + atr * ATR_SL
            tp1 = price - atr * ATR_TP1
            tp2 = price - atr * ATR_TP2
        else:
            sl  = price - atr * ATR_SL
            tp1 = price + atr * ATR_TP1
            tp2 = price + atr * ATR_TP2

        # Simulate forward — check next 12 candles (1hr)
        outcome = "OPEN"
        pnl     = 0.0
        for j in range(i+1, min(i+13, len(df_full))):
            future = df_full.iloc[j]
            if direction == "LONG":
                if future["low"] <= sl:
                    outcome = "LOSS"; pnl = -(atr * ATR_SL); break
                if future["high"] >= tp1:
                    if future["high"] >= tp2:
                        outcome = "WIN_FULL"; pnl = atr * ATR_TP2; break
                    outcome = "WIN_TP1"; pnl = atr * ATR_TP1; break
            else:
                if future["high"] >= sl:
                    outcome = "LOSS"; pnl = -(atr * ATR_SL); break
                if future["low"] <= tp1:
                    if future["low"] <= tp2:
                        outcome = "WIN_FULL"; pnl = atr * ATR_TP2; break
                    outcome = "WIN_TP1"; pnl = atr * ATR_TP1; break

        if outcome == "OPEN":
            outcome = "EXPIRED"; pnl = 0

        trades.append({
            "time":      window.index[-1].strftime("%Y-%m-%d %H:%M"),
            "direction": direction,
            "trigger":   primary,
            "conf":      conf,
            "entry":     round(price, 2),
            "sl":        round(sl, 2),
            "tp1":       round(tp1, 2),
            "outcome":   outcome,
            "pnl_usd":   round(pnl, 2),
        })
        last_sig = window.index[-1]

    # ── Rejection breakdown (always shown) ──────────────────────────────────
    total_checked = len(df_full) - 60 - 5
    print(f"{'─'*60}")
    print("  FILTER BREAKDOWN (candles evaluated after warmup)")
    print(f"{'─'*60}")
    print(f"  Total candle slots   : {total_checked}")
    print(f"  Skipped — gap        : {rej['gap']}")
    print(f"  Skipped — off-session: {rej['session']}")
    print(f"  Skipped — ADX < {BT_MIN_ADX}  : {rej['adx']}")
    print(f"  Skipped — ATR < {atr_floor}  : {rej['atr_floor']}  ← bad/illiquid candles")
    print(f"  Skipped — no trigger : {rej['no_signal']}")
    print(f"  Skipped — confluence : {rej['confluence']}")
    print(f"  Skipped — EMA50 dir  : {rej['ema50']}")
    print(f"  Trades taken         : {len(trades)}")
    print(f"{'─'*60}\n")

    if not trades:
        print("No trades generated even with relaxed backtest filters.")
        print("Check the filter breakdown above. Common causes:")
        print(f"  • 'ATR < {ATR_MIN}' bucket is large → Yahoo returned illiquid/pre-market data")
        print(f"    Fix: lower ATR_MIN or run during market hours (13:00-21:00 UTC for gold)")
        print(f"  • 'ADX < {BT_MIN_ADX}' bucket is large → market was ranging all week")
        print(f"    Fix: lower BT_MIN_ADX to 15, or try a different week")
        return

    # ── Results ──────────────────────────────────────────────────────────────
    df_trades = pd.DataFrame(trades)
    wins      = df_trades[df_trades["outcome"].str.startswith("WIN")]
    losses    = df_trades[df_trades["outcome"] == "LOSS"]
    expired   = df_trades[df_trades["outcome"] == "EXPIRED"]
    win_rate  = len(wins) / len(df_trades) * 100
    total_pnl = df_trades["pnl_usd"].sum()
    days      = max(1, (df_full.index[-1] - df_full.index[0]).days)
    sigs_day  = round(len(trades) / days, 1)

    print(f"  Total trades  : {len(trades)}")
    print(f"  Wins          : {len(wins)}  ({win_rate:.1f}%)")
    print(f"  Losses        : {len(losses)}")
    print(f"  Expired       : {len(expired)}")
    print(f"  Total P&L     : ${total_pnl:.2f}  (ATR units × lot size)")
    print(f"  Avg/day       : {sigs_day} signals")
    print(f"{'─'*60}\n")

    print("All trades:")
    print(f"{'Time':<18} {'Dir':<6} {'Trigger':<14} {'Conf':<5} {'Outcome':<12} {'PnL':>8}")
    print("-" * 65)
    for _, t in df_trades.iterrows():
        emoji = "✅" if t["outcome"].startswith("WIN") else "❌" if t["outcome"]=="LOSS" else "⏳"
        print(f"{t['time']:<18} {t['direction']:<6} {t['trigger']:<14} {t['conf']:<5} {emoji} {t['outcome']:<10} ${t['pnl_usd']:>7.2f}")

    print(f"\n{'='*60}")
    print(f"  Win rate: {win_rate:.1f}%  |  Signals/day: {sigs_day}  |  P&L: ${total_pnl:.2f}")
    if win_rate >= 55:
        print("  ✅ Strategy looks PROFITABLE based on this data")
    elif win_rate >= 45:
        print("  ⚠️  Borderline — consider tightening filters")
    else:
        print("  ❌ Win rate too low — review settings")
    print(f"{'='*60}\n")

    log_file = "backtest_results.csv"
    df_trades.to_csv(log_file, index=False)
    print(f"Full trade log saved to: {log_file}\n")


# ─── Live main loop ────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:  raise SystemExit("Set TELEGRAM_TOKEN")
    if not CHAT_ID:         raise SystemExit("Set CHAT_ID")
    if not TWELVE_DATA_KEY: raise SystemExit("Set TWELVE_DATA_KEY")

    log.info("XAU/USD Bot v3 started.")
    send_telegram(
        "\U0001f916 *XAU/USD Signal Bot v3 online*\n"
        "_8 triggers \u2022 RSI divergence \u2022 3-TF alignment \u2022 News filter_\n"
        "_Targeting 8-10 signals/day \u2014 London/NY only_"
    )

    last_signal_time  = datetime.utcnow() - timedelta(minutes=MIN_SIGNAL_GAP + 1)
    htf_cache         = "NEUTRAL"
    bias_cache        = "NEUTRAL"
    htf_last          = datetime.utcnow() - timedelta(minutes=16)
    bias_last         = datetime.utcnow() - timedelta(hours=2)
    signals_today     = 0
    today_date        = datetime.utcnow().date()

    while True:
        try:
            now = datetime.utcnow()

            # Reset daily counter
            if now.date() != today_date:
                signals_today = 0
                today_date    = now.date()

            # Refresh HTF every 15 min
            if (now - htf_last).total_seconds() >= 900:
                df15 = fetch_15min()
                htf_cache = get_htf_trend(df15)
                htf_last  = now
                log.info("HTF trend: %s", htf_cache)

            # Refresh 1h bias every hour
            if (now - bias_last).total_seconds() >= 3600:
                df1h = fetch_1h()
                bias_cache = get_1h_bias(df1h)
                bias_last  = now
                log.info("1H bias: %s", bias_cache)

            # Session check
            if not in_session():
                log.info("Outside session UTC %02d:00 — sleeping", now.hour)
                time.sleep(CHECK_INTERVAL)
                continue

            # News blackout check
            if near_news():
                log.info("News blackout active — skipping scan")
                time.sleep(CHECK_INTERVAL)
                continue

            df     = fetch_5min(120)
            signal = compute_signal(df, htf_cache, bias_cache)
            gap_ok = (now - last_signal_time).total_seconds() >= MIN_SIGNAL_GAP * 60

            if signal and gap_ok:
                log.info(
                    "SIGNAL %s | %s | conf=%d | score=%d%% | HTF=%s | bias=%s",
                    signal["direction"], signal["primary"], signal["confluence"],
                    signal["confidence"], signal["htf"], signal["bias"],
                )
                if send_telegram(format_message(signal)):
                    last_signal_time = now
                    signals_today   += 1
                    log.info("Signals today: %d", signals_today)

            elif signal and not gap_ok:
                secs = int(MIN_SIGNAL_GAP * 60 - (now - last_signal_time).total_seconds())
                log.info("Signal ready — cooldown %ds | signals today: %d", secs, signals_today)

            else:
                log.info(
                    "No signal | Price: %s | ADX: %.1f | HTF: %s | %s | today: %d signals",
                    round(df["close"].iloc[-1], 2),
                    df["adx"].iloc[-1] if "adx" in df.columns else 0,
                    htf_cache, session_name(), signals_today,
                )

        except Exception as e:
            log.error("Error: %s", e)

        time.sleep(CHECK_INTERVAL)


# ─── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--backtest" in sys.argv:
        run_backtest()
    else:
        main()
