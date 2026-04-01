"""
XAU/USD Professional Signal Bot — 5 Trigger Edition
=====================================================
Fires signals from 5 different conditions:

  1. EMA 9/21 Crossover        — trend flip
  2. RSI Cross 50              — momentum shift
  3. EMA 21 Bounce             — pullback entry
  4. RSI Extreme (30/70)       — reversal zone
  5. ADX Breakout (above 25)   — new trend starting

Signal format (Valis-style):
  🔴 XAU/USD — SHORT (87%)
  ENTRY / SL / TP / R:R / Confidence / EV / Timing / Factors

Requirements:
    pip install requests pandas ta

Run:
    python xauusd_bot.py
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

try:
    import ta
except ImportError:
    raise SystemExit("Missing dependency — run:  pip install requests pandas ta")

# ===============================================================================
#  CONFIG
# ===============================================================================

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN",  "8631002566:AAEzNkCuoAO_i2h6GtrvWp4rSeaVZRr1J9s")
CHAT_ID         = os.getenv("CHAT_ID",         "5851314699")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "07d82c2484224923b517b991cf2b2442")

CHECK_INTERVAL = 60    # check every 60 seconds
MIN_SIGNAL_GAP = 5     # minimum minutes between signals

# Risk (USD distance)
TP_PIPS = 12.0
SL_PIPS =  8.0

# Indicator periods
EMA_FAST   =  9
EMA_SLOW   = 21
EMA_TREND  = 50
RSI_PERIOD = 14
ADX_PERIOD = 14

# ===============================================================================

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("xauusd_bot")


# --- Data fetching -------------------------------------------------------------

def fetch_ohlcv_twelvedata(n: int = 80) -> pd.DataFrame:
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD&interval=5min&outputsize={n}&apikey={TWELVE_DATA_KEY}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["datetime"]).astype(float)
    return df.sort_index().tail(n)


def fetch_ohlcv_yahoo(n: int = 80) -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=5m&range=1d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(result["timestamp"], unit="s"),
    )
    return df.dropna().tail(n)


def fetch_ohlcv() -> pd.DataFrame:
    try:
        log.info("Fetching via Twelve Data (XAU/USD spot) ...")
        return fetch_ohlcv_twelvedata()
    except Exception as e:
        log.warning("Twelve Data failed (%s) - falling back to Yahoo Finance ...", e)
        return fetch_ohlcv_yahoo()


# --- Indicators ----------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=EMA_TREND)
    df["rsi"]       = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    adx_ind         = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
    df["adx"]       = adx_ind.adx()
    df["adx_pos"]   = adx_ind.adx_pos()
    df["adx_neg"]   = adx_ind.adx_neg()
    return df


def is_pin_bar(candle: pd.Series) -> bool:
    body = abs(candle["close"] - candle["open"])
    rng  = candle["high"] - candle["low"]
    if rng == 0:
        return False
    upper = candle["high"] - max(candle["open"], candle["close"])
    lower = min(candle["open"], candle["close"]) - candle["low"]
    return max(upper, lower) > 2 * body


def has_choch(df: pd.DataFrame) -> bool:
    closes   = df["close"].tail(12)
    mid_high = closes.iloc[:6].max()
    recent   = closes.iloc[6:].max()
    return recent > mid_high


# --- 5 Signal triggers ---------------------------------------------------------

def check_ema_crossover(prev: pd.Series, curr: pd.Series) -> str | None:
    """Trigger 1 - EMA 9 crosses EMA 21."""
    if prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]:
        return "LONG"
    if prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]:
        return "SHORT"
    return None


def check_rsi_cross50(prev: pd.Series, curr: pd.Series) -> str | None:
    """Trigger 2 - RSI crosses the 50 line."""
    if prev["rsi"] <= 50 and curr["rsi"] > 50:
        return "LONG"
    if prev["rsi"] >= 50 and curr["rsi"] < 50:
        return "SHORT"
    return None


def check_ema_bounce(prev: pd.Series, curr: pd.Series) -> str | None:
    """Trigger 3 - Price pulls back to EMA 21 then bounces."""
    ema = curr["ema_slow"]
    if prev["low"] <= ema * 1.001 and curr["close"] > ema and curr["close"] > prev["close"]:
        return "LONG"
    if prev["high"] >= ema * 0.999 and curr["close"] < ema and curr["close"] < prev["close"]:
        return "SHORT"
    return None


def check_rsi_extreme(curr: pd.Series) -> str | None:
    """Trigger 4 - RSI at extreme reversal zones."""
    if curr["rsi"] <= 30:
        return "LONG"
    if curr["rsi"] >= 70:
        return "SHORT"
    return None


def check_adx_breakout(prev: pd.Series, curr: pd.Series) -> str | None:
    """Trigger 5 - ADX rises above 25, new trend starting."""
    if prev["adx"] < 25 and curr["adx"] >= 25:
        if curr["adx_pos"] > curr["adx_neg"]:
            return "LONG"
        else:
            return "SHORT"
    return None


# --- Scoring -------------------------------------------------------------------

def score_signal(curr: pd.Series, direction: str, trigger: str, pin_bar: bool, choch: bool) -> dict:
    pts = 45

    trigger_pts = {
        "EMA_CROSS":    20,
        "RSI_CROSS50":  12,
        "EMA_BOUNCE":   18,
        "RSI_EXTREME":  15,
        "ADX_BREAKOUT": 16,
    }
    pts += trigger_pts.get(trigger, 10)

    rsi = curr["rsi"]
    adx = curr["adx"]

    if direction == "SHORT":
        if rsi > 65:   pts += 13
        elif rsi > 55: pts += 6
    else:
        if rsi < 35:   pts += 13
        elif rsi < 45: pts += 6

    if adx >= 35:   pts += 12
    elif adx >= 25: pts += 6
    elif adx < 20:  pts -= 8

    if pin_bar:     pts += 8
    if choch and direction == "SHORT":
        pts -= 6

    confidence = max(10, min(99, pts))
    win_rate   = confidence / 100
    ev         = round(win_rate * (TP_PIPS / SL_PIPS) - (1 - win_rate), 2)

    if confidence >= 80:   grade = f"A ({confidence}/100)"
    elif confidence >= 65: grade = f"B ({confidence}/100)"
    elif confidence >= 50: grade = f"C ({confidence}/100)"
    else:                  grade = f"D ({confidence}/100)"

    hour = datetime.utcnow().hour
    if 7 <= hour <= 17:
        timing, t_icon = "GOOD (80%)", "✅"
    elif 6 <= hour <= 20:
        timing, t_icon = "OKAY (50%)", "⚠️"
    else:
        timing, t_icon = "POOR (30%)", "🔴"

    return {
        "confidence": confidence,
        "ev":          ev,
        "grade":       grade,
        "timing":      timing,
        "timing_icon": t_icon,
    }


# --- Core signal logic ---------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> dict | None:
    df   = add_indicators(df)
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Run all 5 triggers — first one that fires wins
    triggers = [
        ("EMA_CROSS",    check_ema_crossover(prev, curr)),
        ("RSI_CROSS50",  check_rsi_cross50(prev, curr)),
        ("EMA_BOUNCE",   check_ema_bounce(prev, curr)),
        ("RSI_EXTREME",  check_rsi_extreme(curr)),
        ("ADX_BREAKOUT", check_adx_breakout(prev, curr)),
    ]

    direction = None
    trigger   = None
    for name, result in triggers:
        if result:
            direction = result
            trigger   = name
            break

    if not direction:
        return None

    # EMA 50 trend filter
    if direction == "LONG"  and curr["close"] < curr["ema_trend"]:
        log.info("LONG filtered - price below EMA 50")
        return None
    if direction == "SHORT" and curr["close"] > curr["ema_trend"]:
        log.info("SHORT filtered - price above EMA 50")
        return None

    price   = round(curr["close"], 2)
    adx     = round(curr["adx"], 1)
    rsi_val = round(curr["rsi"], 1)
    pin_bar = is_pin_bar(curr)
    choch   = has_choch(df)
    scores  = score_signal(curr, direction, trigger, pin_bar, choch)

    if direction == "SHORT":
        tp = round(price - TP_PIPS, 2)
        sl = round(price + SL_PIPS, 2)
    else:
        tp = round(price + TP_PIPS, 2)
        sl = round(price - SL_PIPS, 2)

    sl_dist = round(abs(price - sl), 2)
    tp_dist = round(abs(price - tp), 2)
    rr      = round(tp_dist / sl_dist, 1)

    trend = "📉 Downtrend" if curr["ema_fast"] < curr["ema_slow"] else "📈 Uptrend"
    trigger_labels = {
        "EMA_CROSS":    "🔀 EMA 9/21 Crossover",
        "RSI_CROSS50":  "📶 RSI Momentum Cross (50)",
        "EMA_BOUNCE":   "🔄 EMA 21 Bounce",
        "RSI_EXTREME":  "⚡ RSI Extreme Level",
        "ADX_BREAKOUT": "💥 ADX Trend Breakout",
    }
    factors = [trend, trigger_labels.get(trigger, "📊 Signal")]
    if choch:
        factors.append("⚠️ Bullish CHoCH")
    if adx >= 25:
        side = "bearish" if direction == "SHORT" else "bullish"
        factors.append(f"🔥 Strong ADX {side} ({adx})")
    if pin_bar:
        factors.append("📌 Pin Bar " + ("Bear" if direction == "SHORT" else "Bull"))
    factors.append("💪 USD stronger than XAU" if direction == "SHORT" else "💪 XAU momentum vs USD")

    return {
        "direction": direction,
        "trigger":   trigger,
        "entry":     price,
        "sl":        sl,
        "tp":        tp,
        "sl_dist":   sl_dist,
        "tp_dist":   tp_dist,
        "rr":        rr,
        "rsi":       rsi_val,
        "adx":       adx,
        "factors":   factors,
        **scores,
    }


# --- Telegram ------------------------------------------------------------------

def format_message(sig: dict) -> str:
    direction   = sig["direction"]
    confidence  = sig["confidence"]
    dot         = "🔴" if direction == "SHORT" else "🟢"
    ev_str      = ("+" if sig["ev"] >= 0 else "") + str(sig["ev"]) + "R"
    date_str    = datetime.utcnow().strftime("%d %b %Y")
    factors_str = "\n".join(f"• {f}" for f in sig["factors"])

    return (
        f"{dot} *XAU/USD — {direction} ({confidence}%)*\n"
        f"_{date_str}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"*ENTRY:* `{sig['entry']}`\n"
        f"*SL:*    `{sig['sl']}` ({sig['sl_dist']}p)\n"
        f"*TP:*    `{sig['tp']}` ({sig['tp_dist']}p)\n"
        f"*R:R*    1:{sig['rr']}\n"
        f"📊 _Close 50% at TP, move SL to entry on rest_\n"
        f"🌊 *SWING* | ⏰ 5-10 hrs\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Confidence: {confidence}%*\n"
        f"💎 EV: {ev_str}\n"
        f"⚠️ Entry: {sig['grade']}\n"
        f"\n"
        f"⏰ *TIMING*\n"
        f"{sig['timing_icon']} {sig['timing']}\n"
        f"\n"
        f"💡 *Factors:*\n"
        f"{factors_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )


def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
        timeout=10,
    )
    r.raise_for_status()
    log.info("Telegram message sent.")


# --- Entry point ---------------------------------------------------------------

def validate_config() -> None:
    if "YOUR_BOT_TOKEN" in TELEGRAM_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN before running.")
    if "YOUR_CHAT_ID" in CHAT_ID:
        raise SystemExit("Set CHAT_ID before running.")


def main() -> None:
    validate_config()
    log.info("XAU/USD bot started. Interval: %ds, cooldown: %dm", CHECK_INTERVAL, MIN_SIGNAL_GAP)
    send_telegram(
        "🤖 *XAU/USD Signal Bot online*\n"
        "_5 triggers active — scanning every 60s, 5 min cooldown…_"
    )

    last_signal_time = datetime.utcnow() - timedelta(minutes=MIN_SIGNAL_GAP + 1)

    while True:
        try:
            df     = fetch_ohlcv()
            signal = compute_signal(df)

            now    = datetime.utcnow()
            gap_ok = (now - last_signal_time).total_seconds() >= MIN_SIGNAL_GAP * 60

            if signal and gap_ok:
                log.info(
                    "SIGNAL %s via %s  entry=%s  conf=%s%%",
                    signal["direction"], signal["trigger"],
                    signal["entry"], signal["confidence"],
                )
                send_telegram(format_message(signal))
                last_signal_time = now
            elif signal and not gap_ok:
                secs_left = int(MIN_SIGNAL_GAP * 60 - (now - last_signal_time).total_seconds())
                log.info("Signal suppressed - cooldown %ds remaining", secs_left)
            else:
                log.info("No signal. Price: %s", round(df["close"].iloc[-1], 2))

        except Exception as exc:
            log.error("Error: %s", exc)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
