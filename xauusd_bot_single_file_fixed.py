#!/usr/bin/env python3
"""
XAU/USD Professional Signal Bot — single-file fixed edition
============================================================

What changed in this single-file version:
- Keeps everything in one Python file
- Loads credentials from CLI args, environment variables, or a local .env file
- Removes any hardcoded secret fallbacks
- Fixes the live-loop status log that mislabeled ATR as ADX
- Makes indicator handling consistent before logging/decisioning
- Adds safer request handling and clearer startup validation
- Preserves live mode and --backtest mode

Usage:
    python xauusd_bot_single_file_fixed.py
    python xauusd_bot_single_file_fixed.py --backtest

Optional CLI overrides:
    python xauusd_bot_single_file_fixed.py \
        --telegram-token YOUR_TOKEN \
        --chat-id YOUR_CHAT_ID \
        --twelve-data-key YOUR_API_KEY

Required packages:
    pip install requests pandas numpy ta
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import ta
except ImportError as exc:
    raise SystemExit("Run: pip install requests pandas numpy ta") from exc


# ==============================================================================
# Config
# ==============================================================================

CHECK_INTERVAL = 60
MIN_SIGNAL_GAP = 12
MIN_CONFLUENCE = 2
MIN_ADX = 18
SESSION_START = 7
SESSION_END = 20
NEWS_BLACKOUT = 20

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
EMA_HTF = 200
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
BB_PERIOD = 20
BB_STD = 2.0
ATR_MA_PERIOD = 20

ATR_SL = 0.7
ATR_TP1 = 1.0
ATR_TP2 = 2.2

NEWS_TIMES = [
    (8, 30),
    (13, 30),
    (15, 0),
    (18, 0),
    (19, 0),
]

UTC = timezone.utc


@dataclass
class Settings:
    telegram_token: str
    chat_id: str
    twelve_data_key: str
    env_path: Optional[Path] = None


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("xauusd_bot")


# ==============================================================================
# Small utilities
# ==============================================================================

def utcnow() -> datetime:
    return datetime.now(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XAU/USD single-file signal bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--telegram-token", default="", help="Telegram bot token")
    parser.add_argument("--chat-id", default="", help="Telegram chat ID")
    parser.add_argument("--twelve-data-key", default="", help="Twelve Data API key")
    return parser.parse_args()


def load_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        values[key] = value
    return values


def resolve_settings(args: argparse.Namespace) -> Settings:
    env_path = Path(args.env_file).expanduser().resolve()
    dotenv_values = load_dotenv_file(env_path)

    def pick(name: str, cli_value: str) -> str:
        return (cli_value or os.environ.get(name) or dotenv_values.get(name, "")).strip()

    settings = Settings(
        telegram_token=pick("TELEGRAM_TOKEN", args.telegram_token),
        chat_id=pick("CHAT_ID", args.chat_id),
        twelve_data_key=pick("TWELVE_DATA_KEY", args.twelve_data_key),
        env_path=env_path if env_path.exists() else None,
    )
    return settings


def require_nonempty(value: str, name: str) -> str:
    if not value:
        raise SystemExit(
            f"ERROR: missing {name}. Provide it via CLI, environment variable, or .env file."
        )
    return value


def requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "xauusd-bot/1.0"})
    return session


SESSION = requests_session()


# ==============================================================================
# Filters
# ==============================================================================

def in_session(now: Optional[datetime] = None) -> bool:
    now = now or utcnow()
    return SESSION_START <= now.hour <= SESSION_END


def near_news(now: Optional[datetime] = None) -> bool:
    now = now or utcnow()
    for hour, minute in NEWS_TIMES:
        news_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        diff_minutes = abs((now - news_time).total_seconds()) / 60.0
        if diff_minutes <= NEWS_BLACKOUT:
            return True
    return False


def session_name(now: Optional[datetime] = None) -> str:
    now = now or utcnow()
    hour = now.hour
    if 12 <= hour <= 17:
        return "London/NY Overlap"
    if SESSION_START <= hour < 12:
        return "London"
    if hour <= SESSION_END:
        return "New York"
    return "Off-Session"


# ==============================================================================
# Data
# ==============================================================================

def fetch_twelvedata(api_key: str, interval: str, n: int = 200) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": n,
        "apikey": api_key,
    }
    response = SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        raise ValueError(f"Twelve Data error: {payload.get('message', payload)}")

    df = pd.DataFrame(payload["values"])
    df.index = pd.to_datetime(df["datetime"], utc=False)
    df = df.drop(columns=["datetime"]).astype(float).sort_index()
    return df[["open", "high", "low", "close"]]


def fetch_5min(api_key: str, n: int = 200) -> pd.DataFrame:
    return fetch_twelvedata(api_key, "5min", n)


def fetch_15min(api_key: str) -> pd.DataFrame:
    try:
        return fetch_twelvedata(api_key, "15min", 120)
    except Exception as exc:
        log.warning("15min fetch failed: %s", exc)
        return pd.DataFrame()


def fetch_1h(api_key: str) -> pd.DataFrame:
    try:
        return fetch_twelvedata(api_key, "1h", 120)
    except Exception as exc:
        log.warning("1h fetch failed: %s", exc)
        return pd.DataFrame()


# ==============================================================================
# Indicators and structure
# ==============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=EMA_TREND)
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=ATR_PERIOD
    )
    df["atr_ma"] = df["atr"].rolling(ATR_MA_PERIOD).median()

    macd = ta.trend.MACD(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()

    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    bb = ta.volatility.BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_pct"] = bb.bollinger_pband()
    return df


def get_htf_trend(df_15m: pd.DataFrame) -> str:
    if df_15m.empty:
        return "NEUTRAL"
    try:
        close = df_15m["close"]
        ef = ta.trend.ema_indicator(close, window=EMA_FAST).iloc[-1]
        es = ta.trend.ema_indicator(close, window=EMA_SLOW).iloc[-1]
        e50 = ta.trend.ema_indicator(close, window=EMA_TREND).iloc[-1]
        rsi = ta.momentum.rsi(close, window=RSI_PERIOD).iloc[-1]
        price = close.iloc[-1]
        if ef > es and price > e50 and rsi > 50:
            return "BULL"
        if ef < es and price < e50 and rsi < 50:
            return "BEAR"
        return "NEUTRAL"
    except Exception as exc:
        log.warning("HTF trend calc failed: %s", exc)
        return "NEUTRAL"


def get_1h_bias(df_1h: pd.DataFrame) -> str:
    if df_1h.empty:
        return "NEUTRAL"
    try:
        close = df_1h["close"]
        e50 = ta.trend.ema_indicator(close, window=EMA_TREND)
        e200 = ta.trend.ema_indicator(close, window=EMA_HTF)
        slope = e50.iloc[-1] - e50.iloc[-4]
        price = close.iloc[-1]
        if slope > 1.0 and price > e200.iloc[-1]:
            return "BULL"
        if slope < -1.0 and price < e200.iloc[-1]:
            return "BEAR"
        return "NEUTRAL"
    except Exception as exc:
        log.warning("1H bias calc failed: %s", exc)
        return "NEUTRAL"


def is_volatile_enough(candle: pd.Series) -> bool:
    atr_ma = candle.get("atr_ma", np.nan)
    if pd.isna(atr_ma):
        return True
    return float(candle["atr"]) >= float(atr_ma) * 0.9


def has_strong_body(candle: pd.Series, direction: str) -> bool:
    rng = float(candle["high"] - candle["low"])
    if rng <= 0:
        return False
    body = abs(float(candle["close"] - candle["open"]))
    if body / rng < 0.40:
        return False
    if direction == "LONG" and candle["close"] < candle["open"]:
        return False
    if direction == "SHORT" and candle["close"] > candle["open"]:
        return False
    return True


def is_pin_bar(candle: pd.Series) -> bool:
    body = abs(float(candle["close"] - candle["open"]))
    rng = float(candle["high"] - candle["low"])
    if rng <= 0:
        return False
    upper_wick = float(candle["high"] - max(candle["open"], candle["close"]))
    lower_wick = float(min(candle["open"], candle["close"]) - candle["low"])
    return max(upper_wick, lower_wick) > 2 * body


def has_choch(df: pd.DataFrame, direction: str) -> bool:
    closes = df["close"].tail(12)
    if len(closes) < 12:
        return False
    first_half = closes.iloc[:6]
    second_half = closes.iloc[6:]
    if direction == "LONG":
        return second_half.max() > first_half.max()
    if direction == "SHORT":
        return second_half.min() < first_half.min()
    return False


def rsi_divergence(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 25:
        return None

    window = df.tail(20).copy()
    price = window["close"].to_numpy()
    rsi = window["rsi"].to_numpy()
    n = len(price)

    pivot_highs = [
        i for i in range(2, n - 2)
        if price[i] > price[i - 1] and price[i] > price[i + 1]
        and price[i] > price[i - 2] and price[i] > price[i + 2]
    ]
    pivot_lows = [
        i for i in range(2, n - 2)
        if price[i] < price[i - 1] and price[i] < price[i + 1]
        and price[i] < price[i - 2] and price[i] < price[i + 2]
    ]

    if len(pivot_highs) >= 2:
        ph1, ph2 = pivot_highs[-2], pivot_highs[-1]
        if ph2 - ph1 >= 4 and price[ph2] > price[ph1] and rsi[ph2] < rsi[ph1]:
            return "SHORT"

    if len(pivot_lows) >= 2:
        pl1, pl2 = pivot_lows[-2], pivot_lows[-1]
        if pl2 - pl1 >= 4 and price[pl2] < price[pl1] and rsi[pl2] > rsi[pl1]:
            return "LONG"

    return None


# ==============================================================================
# Triggers
# ==============================================================================

def t_ema_cross(prev: pd.Series, cur: pd.Series) -> Optional[str]:
    if prev["ema_fast"] <= prev["ema_slow"] and cur["ema_fast"] > cur["ema_slow"]:
        return "LONG"
    if prev["ema_fast"] >= prev["ema_slow"] and cur["ema_fast"] < cur["ema_slow"]:
        return "SHORT"
    return None


def t_rsi_50(prev: pd.Series, cur: pd.Series) -> Optional[str]:
    if prev["rsi"] < 50 <= cur["rsi"]:
        return "LONG"
    if prev["rsi"] > 50 >= cur["rsi"]:
        return "SHORT"
    return None


def t_ema_bounce(prev: pd.Series, cur: pd.Series) -> Optional[str]:
    ema = cur["ema_slow"]
    if prev["low"] <= ema * 1.0015 and cur["close"] > ema and cur["close"] > prev["close"]:
        return "LONG"
    if prev["high"] >= ema * 0.9985 and cur["close"] < ema and cur["close"] < prev["close"]:
        return "SHORT"
    return None


def t_rsi_extreme(cur: pd.Series) -> Optional[str]:
    if cur["rsi"] <= 28:
        return "LONG"
    if cur["rsi"] >= 72:
        return "SHORT"
    return None


def t_adx_breakout(prev: pd.Series, cur: pd.Series) -> Optional[str]:
    if prev["adx"] < MIN_ADX <= cur["adx"]:
        return "LONG" if cur["adx_pos"] > cur["adx_neg"] else "SHORT"
    return None


def t_macd_cross(prev: pd.Series, cur: pd.Series) -> Optional[str]:
    if prev["macd"] <= prev["macd_sig"] and cur["macd"] > cur["macd_sig"]:
        return "LONG"
    if prev["macd"] >= prev["macd_sig"] and cur["macd"] < cur["macd_sig"]:
        return "SHORT"
    return None


def t_bb_reversion(prev: pd.Series, cur: pd.Series) -> Optional[str]:
    if prev["bb_pct"] <= 0.05 and cur["bb_pct"] > 0.05:
        return "LONG"
    if prev["bb_pct"] >= 0.95 and cur["bb_pct"] < 0.95:
        return "SHORT"
    return None


def get_all_signals(df: pd.DataFrame) -> dict[str, Optional[str]]:
    prev, cur = df.iloc[-2], df.iloc[-1]
    return {
        "EMA_CROSS": t_ema_cross(prev, cur),
        "RSI_50": t_rsi_50(prev, cur),
        "EMA_BOUNCE": t_ema_bounce(prev, cur),
        "RSI_EXTREME": t_rsi_extreme(cur),
        "ADX_BREAK": t_adx_breakout(prev, cur),
        "MACD_CROSS": t_macd_cross(prev, cur),
        "BB_REVERT": t_bb_reversion(prev, cur),
        "RSI_DIV": rsi_divergence(df),
    }


def find_direction(signals: dict[str, Optional[str]]) -> tuple[Optional[str], Optional[str], int, list[str]]:
    priority = [
        "RSI_DIV", "EMA_CROSS", "EMA_BOUNCE", "ADX_BREAK",
        "BB_REVERT", "RSI_EXTREME", "MACD_CROSS", "RSI_50",
    ]
    direction = None
    primary = None
    for name in priority:
        if signals.get(name):
            direction = signals[name]
            primary = name
            break

    if not direction:
        return None, None, 0, []

    agreeing = [name for name, value in signals.items() if value == direction]
    return direction, primary, len(agreeing), agreeing


# ==============================================================================
# Scoring and signal assembly
# ==============================================================================

def score_signal(
    candle: pd.Series,
    direction: str,
    primary: str,
    htf: str,
    bias: str,
    confluence: int,
    pin_bar: bool,
    choch: bool,
) -> dict[str, object]:
    pts = 42
    primary_pts = {
        "RSI_DIV": 22,
        "EMA_CROSS": 18,
        "EMA_BOUNCE": 17,
        "ADX_BREAK": 15,
        "BB_REVERT": 14,
        "RSI_EXTREME": 13,
        "MACD_CROSS": 12,
        "RSI_50": 10,
    }
    pts += primary_pts.get(primary, 8)
    pts += max(0, confluence - 1) * 7

    rsi = float(candle["rsi"])
    adx = float(candle["adx"])

    htf_match = (direction == "LONG" and htf == "BULL") or (direction == "SHORT" and htf == "BEAR")
    bias_match = (direction == "LONG" and bias == "BULL") or (direction == "SHORT" and bias == "BEAR")
    if htf_match and bias_match:
        pts += 18
    elif htf_match:
        pts += 10
    elif bias_match:
        pts += 6
    else:
        pts -= 12

    if direction == "SHORT":
        pts += 10 if rsi > 65 else 6 if rsi > 58 else 2 if rsi > 52 else 0
    else:
        pts += 10 if rsi < 35 else 6 if rsi < 42 else 2 if rsi < 48 else 0

    if adx >= 35:
        pts += 10
    elif adx >= 25:
        pts += 6
    elif adx >= 20:
        pts += 2
    else:
        pts -= 10

    if pin_bar:
        pts += 7
    if choch:
        pts += 8

    sess = session_name()
    if "Overlap" in sess:
        pts += 6
    elif sess != "Off-Session":
        pts += 3

    confidence = max(10, min(99, pts))
    win_rate = confidence / 100.0
    rr2 = round(ATR_TP2 / ATR_SL, 1)
    ev = round(win_rate * rr2 - (1 - win_rate), 2)
    grade = "A" if confidence >= 85 else "B" if confidence >= 72 else "C" if confidence >= 58 else "D"
    grade = f"{grade} ({confidence}/100)"

    hour = utcnow().hour
    if 12 <= hour <= 16:
        timing, timing_icon = "GOOD — London/NY Overlap (85%)", "✅"
    elif SESSION_START <= hour <= SESSION_END:
        timing, timing_icon = "GOOD (75%)", "✅"
    else:
        timing, timing_icon = "POOR — Off Session (20%)", "🔴"

    return {
        "confidence": confidence,
        "ev": ev,
        "grade": grade,
        "timing": timing,
        "timing_icon": timing_icon,
        "rr1": round(ATR_TP1 / ATR_SL, 1),
        "rr2": rr2,
    }


def compute_signal(df: pd.DataFrame, htf: str, bias: str, last_direction: Optional[str] = None) -> Optional[dict[str, object]]:
    df = add_indicators(df).dropna().copy()
    if len(df) < 30:
        return None

    candle = df.iloc[-1]

    if candle["adx"] < MIN_ADX:
        log.info("ADX %.1f < %d — choppy, skipping", candle["adx"], MIN_ADX)
        return None

    if not is_volatile_enough(candle):
        log.info("ATR %.2f below median %.2f — low vol, skipping", candle["atr"], candle.get("atr_ma", 0))
        return None

    signals = get_all_signals(df)
    direction, primary, conf, agreeing = find_direction(signals)
    if not direction:
        return None

    if conf < MIN_CONFLUENCE:
        log.info("Confluence %d < %d for %s — skipping", conf, MIN_CONFLUENCE, direction)
        return None

    if not has_strong_body(candle, direction):
        log.info("Weak candle body for %s — skipping", direction)
        return None

    if htf == "BULL" and direction == "SHORT" and primary != "RSI_DIV":
        log.info("SHORT blocked — HTF is BULL and no RSI divergence")
        return None
    if htf == "BEAR" and direction == "LONG" and primary != "RSI_DIV":
        log.info("LONG blocked — HTF is BEAR and no RSI divergence")
        return None

    if direction == "LONG" and candle["close"] < candle["ema_trend"]:
        log.info("LONG filtered — price below EMA50")
        return None
    if direction == "SHORT" and candle["close"] > candle["ema_trend"]:
        log.info("SHORT filtered — price above EMA50")
        return None

    if direction == last_direction:
        log.info("Same direction %s as last signal — skipping duplicate", direction)
        return None

    price = round(float(candle["close"]), 2)
    atr = float(candle["atr"])
    adx_val = round(float(candle["adx"]), 1)
    rsi_val = round(float(candle["rsi"]), 1)
    pin_bar = is_pin_bar(candle)
    choch = has_choch(df, direction)
    scores = score_signal(candle, direction, primary or "", htf, bias, conf, pin_bar, choch)

    if direction == "SHORT":
        sl = round(price + atr * ATR_SL, 2)
        tp1 = round(price - atr * ATR_TP1, 2)
        tp2 = round(price - atr * ATR_TP2, 2)
    else:
        sl = round(price - atr * ATR_SL, 2)
        tp1 = round(price + atr * ATR_TP1, 2)
        tp2 = round(price + atr * ATR_TP2, 2)

    trend_lbl = "📉 Downtrend" if candle["ema_fast"] < candle["ema_slow"] else "📈 Uptrend"
    htf_lbl = {"BULL": "📈 HTF 15m Bullish", "BEAR": "📉 HTF 15m Bearish"}.get(htf, "➡️ HTF Neutral")
    bias_lbl = {"BULL": "📈 1H Bias Bullish", "BEAR": "📉 1H Bias Bearish"}.get(bias, "➡️ 1H Neutral")
    trig_lbl = {
        "RSI_DIV": "🔁 RSI Divergence",
        "EMA_CROSS": "🔀 EMA 9/21 Crossover",
        "EMA_BOUNCE": "🔄 EMA 21 Pullback",
        "ADX_BREAK": "💥 ADX Breakout",
        "BB_REVERT": "🎯 BB Mean Reversion",
        "RSI_EXTREME": "⚡ RSI Extreme",
        "MACD_CROSS": "📊 MACD Cross",
        "RSI_50": "📶 RSI Cross 50",
    }.get(primary, "📊 Signal")

    factors = [
        trend_lbl,
        htf_lbl,
        bias_lbl,
        trig_lbl,
        f"🔗 {conf} signals confluent",
        f"🕐 {session_name()} Session",
    ]
    if choch:
        factors.append(f"✅ {'Bullish' if direction == 'LONG' else 'Bearish'} CHoCH")
    if adx_val >= 25:
        factors.append(f"🔥 ADX strong ({adx_val})")
    if pin_bar:
        factors.append("📌 Pin Bar " + ("Bull" if direction == "LONG" else "Bear"))

    return {
        "direction": direction,
        "primary": primary,
        "agreeing": agreeing,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "sl_dist": round(abs(price - sl), 2),
        "tp1_dist": round(abs(price - tp1), 2),
        "tp2_dist": round(abs(price - tp2), 2),
        "rsi": rsi_val,
        "adx": adx_val,
        "atr": round(atr, 2),
        "htf": htf,
        "bias": bias,
        "confluence": conf,
        "factors": factors,
        **scores,
    }


# ==============================================================================
# Telegram
# ==============================================================================

def format_message(sig: dict[str, object]) -> str:
    dot = "🔴" if sig["direction"] == "SHORT" else "🟢"
    ev_value = float(sig["ev"])
    ev_str = ("+" if ev_value >= 0 else "") + str(ev_value) + "R"
    factors = "\n".join(f"• {item}" for item in sig["factors"])
    date_str = utcnow().strftime("%d %b %Y %H:%M UTC")

    return (
        f"{dot} *XAU/USD — {sig['direction']} ({sig['confidence']}%)*\n"
        f"_{date_str}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*ENTRY:* `{sig['entry']}`\n"
        f"*SL:* `{sig['sl']}` ({sig['sl_dist']}p)\n"
        f"*TP1:* `{sig['tp1']}` ({sig['tp1_dist']}p) — _close 50% here_\n"
        f"*TP2:* `{sig['tp2']}` ({sig['tp2_dist']}p) — _trail remainder_\n"
        f"*R:R* 1:{sig['rr1']} → 1:{sig['rr2']}\n"
        f"📊 _After TP1: move SL to entry (breakeven)_\n"
        f"🌊 *SCALP* | ⏰ 1–3 hrs\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Confidence:* {sig['confidence']}%\n"
        f"💎 EV: {ev_str}\n"
        f"⚠️ Entry: {sig['grade']}\n"
        f"🔗 Confluence: {sig['confluence']} signals\n\n"
        f"⏰ *TIMING*\n"
        f"{sig['timing_icon']} {sig['timing']}\n\n"
        f"💡 *Factors:*\n"
        f"{factors}"
    )


def send_telegram(settings: Settings, message: str, retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"
    payload = {
        "chat_id": settings.chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    for attempt in range(1, retries + 1):
        try:
            response = SESSION.post(url, json=payload, timeout=15)
            response.raise_for_status()
            body = response.json()
            if not body.get("ok", False):
                raise RuntimeError(body)
            log.info("Telegram sent.")
            return True
        except Exception as exc:
            log.warning("Telegram attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(3)
    log.error("All Telegram retries failed.")
    return False


# ==============================================================================
# Backtest
# ==============================================================================

def run_backtest(settings: Settings) -> None:
    require_nonempty(settings.twelve_data_key, "TWELVE_DATA_KEY")

    print("\n" + "=" * 62)
    print("  XAU/USD BACKTEST — single-file fixed edition")
    print("=" * 62 + "\n")

    print("Fetching historical data...")
    try:
        raw = fetch_5min(settings.twelve_data_key, 1500)
    except Exception as exc:
        print(f"Data fetch failed: {exc}")
        return

    raw = raw[raw.index.hour.isin(range(SESSION_START, SESSION_END + 1))]
    raw = raw.dropna(subset=["open", "high", "low", "close"])

    trades: list[dict[str, object]] = []
    last_sig_time = raw.index[0]
    last_direction: Optional[str] = None
    warmup = 80

    print(f"Running backtest on {len(raw)} session candles...\n")

    for i in range(warmup, len(raw) - 12):
        window_raw = raw.iloc[: i + 1]
        window = add_indicators(window_raw).dropna()
        if len(window) < 30:
            continue

        candle = window.iloc[-1]
        if (window.index[-1] - last_sig_time).total_seconds() < MIN_SIGNAL_GAP * 60:
            continue
        if candle["adx"] < MIN_ADX:
            continue
        if not is_volatile_enough(candle):
            continue

        signals = get_all_signals(window)
        direction, primary, conf, agreeing = find_direction(signals)
        if not direction or conf < MIN_CONFLUENCE:
            continue
        if not has_strong_body(candle, direction):
            continue

        htf_proxy = "BULL" if candle["close"] > candle["ema_trend"] else "BEAR" if candle["close"] < candle["ema_trend"] else "NEUTRAL"
        if htf_proxy == "BULL" and direction == "SHORT" and primary != "RSI_DIV":
            continue
        if htf_proxy == "BEAR" and direction == "LONG" and primary != "RSI_DIV":
            continue
        if direction == "LONG" and candle["close"] < candle["ema_trend"]:
            continue
        if direction == "SHORT" and candle["close"] > candle["ema_trend"]:
            continue
        if direction == last_direction:
            continue

        price = float(candle["close"])
        atr = float(candle["atr"])
        if direction == "SHORT":
            sl = price + atr * ATR_SL
            tp1 = price - atr * ATR_TP1
            tp2 = price - atr * ATR_TP2
        else:
            sl = price - atr * ATR_SL
            tp1 = price + atr * ATR_TP1
            tp2 = price + atr * ATR_TP2

        outcome = "OPEN"
        pnl = 0.0
        for j in range(i + 1, min(i + 19, len(raw))):
            future = raw.iloc[j]
            if direction == "LONG":
                if future["low"] <= sl:
                    outcome, pnl = "LOSS", -(atr * ATR_SL)
                    break
                if future["high"] >= tp2:
                    outcome, pnl = "WIN_FULL", atr * ATR_TP2
                    break
                if future["high"] >= tp1:
                    outcome, pnl = "WIN_TP1", atr * ATR_TP1
                    break
            else:
                if future["high"] >= sl:
                    outcome, pnl = "LOSS", -(atr * ATR_SL)
                    break
                if future["low"] <= tp2:
                    outcome, pnl = "WIN_FULL", atr * ATR_TP2
                    break
                if future["low"] <= tp1:
                    outcome, pnl = "WIN_TP1", atr * ATR_TP1
                    break

        if outcome == "OPEN":
            outcome, pnl = "EXPIRED", 0.0

        trades.append({
            "time": window.index[-1].strftime("%Y-%m-%d %H:%M"),
            "direction": direction,
            "trigger": primary,
            "conf": conf,
            "entry": round(price, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "outcome": outcome,
            "pnl_usd": round(pnl, 2),
        })

        last_sig_time = window.index[-1]
        last_direction = direction

    if not trades:
        print("No trades generated. Try lowering MIN_CONFLUENCE or MIN_ADX.")
        return

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades["outcome"].str.startswith("WIN")]
    full_wins = df_trades[df_trades["outcome"] == "WIN_FULL"]
    tp1_wins = df_trades[df_trades["outcome"] == "WIN_TP1"]
    losses = df_trades[df_trades["outcome"] == "LOSS"]
    expired = df_trades[df_trades["outcome"] == "EXPIRED"]
    win_rate = len(wins) / len(df_trades) * 100
    total_pnl = float(df_trades["pnl_usd"].sum())
    days = max(1, (raw.index[-1] - raw.index[0]).days)
    sigs_day = round(len(trades) / days, 1)
    avg_win = float(wins["pnl_usd"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl_usd"].mean()) if len(losses) else 0.0

    print("─" * 62)
    print(f"  Total trades   : {len(trades)}")
    print(f"  Wins (TP1)     : {len(tp1_wins)}")
    print(f"  Wins (TP2)     : {len(full_wins)}")
    print(f"  Total Wins     : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses         : {len(losses)}")
    print(f"  Expired        : {len(expired)}")
    print(f"  Avg win        : {avg_win:.2f}")
    print(f"  Avg loss       : {avg_loss:.2f}")
    print(f"  Total P&L      : ${total_pnl:.2f}")
    print(f"  Avg/day        : {sigs_day} signals")
    print("─" * 62 + "\n")

    print("Performance by trigger:")
    for trigger, grp in df_trades.groupby("trigger"):
        wr = len(grp[grp["outcome"].str.startswith("WIN")]) / len(grp) * 100
        print(f"  {trigger:<14} {len(grp):>3} trades {wr:>5.1f}% win pnl={grp['pnl_usd'].sum():.2f}")

    print("\nLast 25 trades:")
    print(f"{'Time':<18} {'Dir':<6} {'Trigger':<14} {'C':<3} {'Outcome':<12} {'PnL':>8}")
    print("-" * 65)
    for _, trade in df_trades.tail(25).iterrows():
        emoji = "✅" if str(trade["outcome"]).startswith("WIN") else "❌" if trade["outcome"] == "LOSS" else "⏳"
        print(
            f"{trade['time']:<18} {trade['direction']:<6} {trade['trigger']:<14} {trade['conf']:<3} "
            f"{emoji} {trade['outcome']:<10} ${trade['pnl_usd']:>7.2f}"
        )

    rr2 = round(ATR_TP2 / ATR_SL, 1)
    breakeven_wr = 1 / (1 + rr2) * 100
    print("\n" + "=" * 62)
    print(f"  Win rate: {win_rate:.1f}% | Signals/day: {sigs_day} | P&L: ${total_pnl:.2f}")
    print(f"  Breakeven win rate at 1:{rr2} = {breakeven_wr:.1f}%")
    if win_rate >= breakeven_wr + 5:
        print("  ✅ Strategy is above breakeven with margin")
    elif win_rate >= breakeven_wr:
        print("  ⚠️ Borderline — above breakeven with thin margin")
    else:
        print("  ❌ Below breakeven — review filters")
    print("=" * 62 + "\n")

    out_path = Path("/mnt/data/backtest_results_single_file.csv")
    df_trades.to_csv(out_path, index=False)
    print(f"Trade log saved to: {out_path}")


# ==============================================================================
# Live loop
# ==============================================================================

def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)

    require_nonempty(settings.twelve_data_key, "TWELVE_DATA_KEY")
    if not args.backtest:
        require_nonempty(settings.telegram_token, "TELEGRAM_TOKEN")
        require_nonempty(settings.chat_id, "CHAT_ID")

    if args.backtest:
        run_backtest(settings)
        return

    log.info("XAU/USD Bot single-file edition started.")
    if settings.env_path:
        log.info("Loaded credentials from %s", settings.env_path)

    send_telegram(
        settings,
        "🤖 *XAU/USD Signal Bot online*\n"
        "_Single-file fixed edition_\n"
        "_Min confluence: 2 • ATR SL: 0.7 • TP2: 2.2R_",
    )

    last_signal_time = utcnow() - timedelta(minutes=MIN_SIGNAL_GAP + 1)
    last_direction: Optional[str] = None
    htf_cache = "NEUTRAL"
    bias_cache = "NEUTRAL"
    htf_last = utcnow() - timedelta(minutes=16)
    bias_last = utcnow() - timedelta(hours=2)
    signals_today = 0
    today_date = utcnow().date()

    while True:
        try:
            now = utcnow()

            if now.date() != today_date:
                signals_today = 0
                last_direction = None
                today_date = now.date()

            if (now - htf_last).total_seconds() >= 900:
                htf_cache = get_htf_trend(fetch_15min(settings.twelve_data_key))
                htf_last = now
                log.info("HTF trend: %s", htf_cache)

            if (now - bias_last).total_seconds() >= 3600:
                bias_cache = get_1h_bias(fetch_1h(settings.twelve_data_key))
                bias_last = now
                log.info("1H bias: %s", bias_cache)

            if not in_session(now):
                log.info("Outside session UTC %02d:00 — sleeping", now.hour)
                time.sleep(CHECK_INTERVAL)
                continue

            if near_news(now):
                log.info("News blackout active — skipping scan")
                time.sleep(CHECK_INTERVAL)
                continue

            raw = fetch_5min(settings.twelve_data_key, 120)
            enriched = add_indicators(raw)
            signal = compute_signal(raw, htf_cache, bias_cache, last_direction)
            gap_ok = (now - last_signal_time).total_seconds() >= MIN_SIGNAL_GAP * 60

            if signal and gap_ok:
                log.info(
                    "SIGNAL %s | %s | conf=%d | score=%d%% | HTF=%s | bias=%s",
                    signal["direction"], signal["primary"], signal["confluence"],
                    signal["confidence"], signal["htf"], signal["bias"],
                )
                if send_telegram(settings, format_message(signal)):
                    last_signal_time = now
                    last_direction = str(signal["direction"])
                    signals_today += 1
                    log.info("Signals today: %d", signals_today)
            elif signal and not gap_ok:
                remaining = int(MIN_SIGNAL_GAP * 60 - (now - last_signal_time).total_seconds())
                log.info("Signal ready — cooldown %ds | today: %d signals", max(0, remaining), signals_today)
            else:
                latest = enriched.dropna().iloc[-1] if not enriched.dropna().empty else None
                if latest is not None:
                    log.info(
                        "No signal | Price: %.2f | ADX: %.1f | ATR: %.2f | HTF: %s | %s | today: %d signals",
                        latest["close"], latest["adx"], latest["atr"], htf_cache, session_name(now), signals_today,
                    )
                else:
                    log.info("No signal | indicators warming up | HTF: %s | %s", htf_cache, session_name(now))

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            return
        except Exception as exc:
            log.error("Error in main loop: %s", exc, exc_info=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
