"""
lib.py — shared detection engine for the Candlestick Pattern Scanner app.

Pure logic only (data fetching, indicators, candlestick pattern detection,
multi-bar chart pattern detection, intraday setup signals, and the
composite Setup Score). No Streamlit UI code lives here — both the
single-ticker Home page and the Market Scanner page import from this
module so detection logic never has to be duplicated or drift out of sync.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

PATTERN_INFO = {
    "Bullish Marubozu": ("Bullish", "Very strong buying. Trend likely to continue up."),
    "Bearish Marubozu": ("Bearish", "Very strong selling. Trend likely to continue down."),
    "Hammer": ("Bullish", "Rejection of lower prices after a downtrend. Potential bullish reversal."),
    "Inverted Hammer": ("Bullish", "Early sign of reversal after a downtrend. Bullish signal."),
    "Bullish Pin Bar": ("Bullish", "Rejection of lower prices. Potential bullish reversal."),
    "Shooting Star": ("Bearish", "Rejection of higher prices. Potential bearish reversal."),
    "Hanging Man": ("Bearish", "Weak buying pressure at the top. Potential bearish reversal."),
    "Dragonfly Doji": ("Bullish", "Big rejection of lower prices. Potential bullish reversal."),
    "Gravestone Doji": ("Bearish", "Big rejection of higher prices. Potential bearish reversal."),
    "Doji": ("Neutral", "Indecision between buyers and sellers."),
    "Long-Legged Doji": ("Neutral", "High volatility, strong indecision. Reversal possible."),
    "Spinning Top": ("Neutral", "Market indecision. Wait for confirmation."),
    "Bullish Engulfing": ("Bullish", "Bullish candle engulfs the prior bearish candle. Strong reversal signal."),
    "Bearish Engulfing": ("Bearish", "Bearish candle engulfs the prior bullish candle. Strong reversal signal."),
    "Piercing Line": ("Bullish", "Bullish candle closes above 50% of the prior bearish candle. Reversal signal."),
    "Dark Cloud Cover": ("Bearish", "Bearish candle closes below 50% of the prior bullish candle. Reversal signal."),
    "Bullish Harami": ("Bullish", "Small bullish candle inside the prior large bearish candle. Reversal signal."),
    "Bearish Harami": ("Bearish", "Small bearish candle inside the prior large bullish candle. Reversal signal."),
    "Tweezer Bottom": ("Bullish", "Rejection at support across two candles. Potential reversal up."),
    "Tweezer Top": ("Bearish", "Rejection at resistance across two candles. Potential reversal down."),
    "Morning Star": ("Bullish", "Three-candle pattern signalling the end of a downtrend."),
    "Evening Star": ("Bearish", "Three-candle pattern signalling the end of an uptrend."),
    "Three White Soldiers": ("Bullish", "Three strong bullish candles. Confirms uptrend continuation."),
    "Three Black Crows": ("Bearish", "Three strong bearish candles. Confirms downtrend continuation."),
    "No clear pattern": ("Neutral", "No recognizable candlestick pattern at this timestamp."),
}

# Multi-bar chart pattern descriptions (from the chart-pattern cheat sheet)
CHART_PATTERN_DESC = {
    "Double Top": "Uptrend forms two comparable highs at resistance. Break below the neckline confirms bearish reversal.",
    "Double Bottom": "Downtrend forms two comparable lows at support. Break above the neckline confirms bullish reversal.",
    "Head & Shoulders": "Lower peak (head) between two similar shoulders. Break below the neckline confirms bearish reversal.",
    "Inverse Head & Shoulders": "Higher trough (head) between two similar shoulders. Break above the neckline confirms bullish reversal.",
    "Ascending Triangle": "Flat resistance with rising support (higher lows). Breakout can go either way; the rising lows favor the upside.",
    "Descending Triangle": "Flat support with falling resistance (lower highs). Breakout can go either way; the falling highs favor the downside.",
    "Symmetrical Triangle": "Highs falling and lows rising, converging. Wait for a confirmed breakout direction.",
    "Rising Wedge": "Price contracts inside an upward-sloping wedge. Breakdown to the downside signals reversal.",
    "Falling Wedge": "Price contracts inside a downward-sloping wedge. Breakout to the upside signals reversal.",
    "Rectangle": "Price moves sideways between support and resistance. Breakout tends to continue the prior trend.",
    "Bullish Flag/Pennant": "Strong upward move (flagpole) followed by tight consolidation. Breakout continues the uptrend.",
    "Bearish Flag/Pennant": "Strong downward move (flagpole) followed by tight consolidation. Breakdown continues the downtrend.",
}

# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_candles(ticker: str, period_days: int) -> pd.DataFrame:
    # Explicit start/end instead of period="Nd" — the period-based intraday
    # request has a known Yahoo/yfinance quirk where "today" can get dropped
    # right around session boundaries. end = tomorrow guarantees today's
    # partial session is included regardless of local/exchange timezone.
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=period_days + 2)
    df = yf.download(ticker, start=start, end=end, interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_chunk_raw(tickers: tuple, period_days: int) -> pd.DataFrame:
    """Single multi-symbol download for one chunk of tickers, cached for
    10 minutes. No client-side timeout — an earlier version of this added
    one to guard against hung connections, but a timeout that's too short
    for a slower connection or a larger batch ends up killing requests
    that would have succeeded given more time, which does more harm than
    good. Let yfinance/requests use their own defaults instead.

    Caching this per-chunk (not the whole scan) means re-running a scan
    for the same universe/date within the TTL window skips straight to
    cache for any chunk that already succeeded — so retrying after a
    partial failure doesn't re-fetch everything from scratch.
    """
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=period_days + 2)
    return yf.download(list(tickers), start=start, end=end, interval="5m",
                        group_by="ticker", threads=True, progress=False, auto_adjust=False)


def parse_chunk(data: pd.DataFrame, chunk_tickers: list) -> dict:
    """Split a fetch_chunk_raw() result into {ticker: DataFrame}, applying
    the same column/index normalization as the single-ticker fetch."""
    out = {}
    if data is None or data.empty:
        return out
    if len(chunk_tickers) == 1:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title).dropna(how="all")
        df.index = pd.to_datetime(df.index)
        if not df.empty and df["Close"].notna().any():
            out[chunk_tickers[0]] = df
    else:
        for t in chunk_tickers:
            try:
                sub = data[t].copy()
            except (KeyError, IndexError):
                continue
            sub = sub.rename(columns=str.title).dropna(how="all")
            sub.index = pd.to_datetime(sub.index)
            if not sub.empty and sub["Close"].notna().any():
                out[t] = sub
    return out


def chunk_list(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # session VWAP: resets every trading day
    df["Date"] = df.index.date
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = typical * df["Volume"]
    df["VWAP"] = tpv.groupby(df["Date"]).cumsum() / df["Volume"].groupby(df["Date"]).cumsum()

    # RSI(14) — Wilder's smoothing (equivalent to an EWM with alpha=1/14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # zero-loss edge case: replace(0, nan) above turns this into NaN, but it
    # should be RSI=100 (all gains) or RSI=50 (perfectly flat), not neutral-50
    rsi = np.where(avg_loss == 0, np.where(avg_gain == 0, 50, 100), rsi)
    df["RSI"] = pd.Series(rsi, index=df.index).fillna(50)  # neutral before enough bars exist

    # 20-bar volume moving average, for relative-volume comparisons
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=5).mean()

    return df


def smooth_edges(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Average the OHLC of the first n and last n candles to reduce
    boundary noise from the fetched window."""
    if len(df) < 2 * n:
        return df
    df = df.copy()
    cols = ["Open", "High", "Low", "Close"]

    first_avg = df.iloc[:n][cols].mean()
    last_avg = df.iloc[-n:][cols].mean()

    for c in cols:
        df.iloc[:n, df.columns.get_loc(c)] = first_avg[c]
        df.iloc[-n:, df.columns.get_loc(c)] = last_avg[c]
    return df


# --------------------------------------------------------------------------
# Pattern detection
# --------------------------------------------------------------------------
def _body(row):
    return abs(row["Close"] - row["Open"])


def _range(row):
    return row["High"] - row["Low"] or 1e-9


def _upper_wick(row):
    return row["High"] - max(row["Open"], row["Close"])


def _lower_wick(row):
    return min(row["Open"], row["Close"]) - row["Low"]


def _is_bull(row):
    return row["Close"] > row["Open"]


def detect_pattern(df: pd.DataFrame, idx: int) -> str:
    """Look at the candle at position idx (and up to 2 prior candles) and
    return the best-matching pattern name."""
    c0 = df.iloc[idx]
    body0, range0 = _body(c0), _range(c0)
    up0, low0 = _upper_wick(c0), _lower_wick(c0)

    # --- 3-candle patterns ---
    if idx >= 2:
        c1, c2 = df.iloc[idx - 1], df.iloc[idx - 2]
        if (not _is_bull(c2) and _body(c2) > _range(c2) * 0.5
                and _body(c1) < _range(c1) * 0.3
                and _is_bull(c0) and c0["Close"] > (c2["Open"] + c2["Close"]) / 2):
            return "Morning Star"
        if (_is_bull(c2) and _body(c2) > _range(c2) * 0.5
                and _body(c1) < _range(c1) * 0.3
                and not _is_bull(c0) and c0["Close"] < (c2["Open"] + c2["Close"]) / 2):
            return "Evening Star"
        if all(_is_bull(x) for x in (c2, c1, c0)) and c0["Close"] > c1["Close"] > c2["Close"] \
                and all(_body(x) > _range(x) * 0.5 for x in (c2, c1, c0)):
            return "Three White Soldiers"
        if all(not _is_bull(x) for x in (c2, c1, c0)) and c0["Close"] < c1["Close"] < c2["Close"] \
                and all(_body(x) > _range(x) * 0.5 for x in (c2, c1, c0)):
            return "Three Black Crows"

    # --- 2-candle patterns ---
    if idx >= 1:
        c1 = df.iloc[idx - 1]
        body1, range1 = _body(c1), _range(c1)
        if not _is_bull(c1) and _is_bull(c0) and c0["Close"] >= c1["Open"] and c0["Open"] <= c1["Close"]:
            return "Bullish Engulfing"
        if _is_bull(c1) and not _is_bull(c0) and c0["Open"] >= c1["Close"] and c0["Close"] <= c1["Open"]:
            return "Bearish Engulfing"
        if not _is_bull(c1) and _is_bull(c0) and c0["Open"] < c1["Close"] and c0["Close"] > (c1["Open"] + c1["Close"]) / 2 and c0["Close"] < c1["Open"]:
            return "Piercing Line"
        if _is_bull(c1) and not _is_bull(c0) and c0["Open"] > c1["Close"] and c0["Close"] < (c1["Open"] + c1["Close"]) / 2 and c0["Close"] > c1["Open"]:
            return "Dark Cloud Cover"
        if not _is_bull(c1) and body1 > range1 * 0.5 and body0 < body1 * 0.5 \
                and max(c0["Open"], c0["Close"]) < c1["Open"] and min(c0["Open"], c0["Close"]) > c1["Close"]:
            return "Bullish Harami"
        if _is_bull(c1) and body1 > range1 * 0.5 and body0 < body1 * 0.5 \
                and max(c0["Open"], c0["Close"]) < c1["Close"] and min(c0["Open"], c0["Close"]) > c1["Open"]:
            return "Bearish Harami"
        if abs(c0["Low"] - c1["Low"]) < range0 * 0.15 and not _is_bull(c1) and _is_bull(c0):
            return "Tweezer Bottom"
        if abs(c0["High"] - c1["High"]) < range0 * 0.15 and _is_bull(c1) and not _is_bull(c0):
            return "Tweezer Top"

    # --- 1-candle patterns ---
    if body0 < range0 * 0.05:
        if low0 > range0 * 0.6 and up0 < range0 * 0.1:
            return "Dragonfly Doji"
        if up0 > range0 * 0.6 and low0 < range0 * 0.1:
            return "Gravestone Doji"
        if up0 > range0 * 0.35 and low0 > range0 * 0.35:
            return "Long-Legged Doji"
        return "Doji"

    if body0 > range0 * 0.9:
        return "Bullish Marubozu" if _is_bull(c0) else "Bearish Marubozu"

    if low0 > body0 * 2 and up0 < body0 * 0.5:
        return "Hammer" if _is_bull(c0) else "Hanging Man"

    if up0 > body0 * 2 and low0 < body0 * 0.5:
        return "Inverted Hammer" if _is_bull(c0) else "Shooting Star"

    if body0 < range0 * 0.3 and up0 < range0 * 0.35 and low0 < range0 * 0.35:
        return "Spinning Top"

    return "No clear pattern"


# --------------------------------------------------------------------------
# Multi-bar chart pattern detection (double top/bottom, H&S, triangles,
# wedges, flags/pennants, rectangles) — swing-point + trendline heuristics.
# This is approximate: real chart patterns are fuzzy, so treat matches as
# "worth a closer look," not certainties.
# --------------------------------------------------------------------------
def find_swings(highs: np.ndarray, lows: np.ndarray, order: int):
    """Local extrema swing highs/lows, then merge points that sit within
    `order` bars of each other (keep the more extreme one)."""
    n = len(highs)
    raw_hi, raw_lo = [], []
    for i in range(order, n - order):
        if highs[i] == highs[i - order:i + order + 1].max():
            raw_hi.append(i)
        if lows[i] == lows[i - order:i + order + 1].min():
            raw_lo.append(i)

    def dedupe(idx_list, values, keep_max):
        if not idx_list:
            return []
        merged = [idx_list[0]]
        for i in idx_list[1:]:
            if i - merged[-1] <= order:
                better = values[i] > values[merged[-1]] if keep_max else values[i] < values[merged[-1]]
                if better:
                    merged[-1] = i
            else:
                merged.append(i)
        return merged

    return dedupe(raw_hi, highs, True), dedupe(raw_lo, lows, False)


def detect_chart_pattern(window: pd.DataFrame):
    """window: reset-index DataFrame (Open/High/Low/Close). Returns a dict
    with name / bias / entry / sl / tp / lines, or None."""
    n = len(window)
    if n < 15:
        return None

    highs, lows, closes = window["High"].values, window["Low"].values, window["Close"].values
    order = max(2, n // 15)
    sh, sl = find_swings(highs, lows, order)
    last_price = closes[-1]

    # ---- Double Top / Bottom ----
    if len(sh) >= 2:
        i1, i2 = sh[-2], sh[-1]
        h1, h2 = highs[i1], highs[i2]
        if i2 > i1 + order and abs(h1 - h2) / max(h1, h2) < 0.006:
            neckline = lows[i1:i2 + 1].min()
            height = max(h1, h2) - neckline
            if last_price < neckline * 1.001:
                return {"name": "Double Top", "bias": "Bearish", "entry": neckline,
                        "sl": max(h1, h2), "tp": neckline - height,
                        "lines": [("Neckline", neckline)]}
    if len(sl) >= 2:
        i1, i2 = sl[-2], sl[-1]
        l1, l2 = lows[i1], lows[i2]
        if i2 > i1 + order and abs(l1 - l2) / max(l1, l2) < 0.006:
            neckline = highs[i1:i2 + 1].max()
            height = neckline - min(l1, l2)
            if last_price > neckline * 0.999:
                return {"name": "Double Bottom", "bias": "Bullish", "entry": neckline,
                        "sl": min(l1, l2), "tp": neckline + height,
                        "lines": [("Neckline", neckline)]}

    # ---- Head & Shoulders / Inverse ----
    if len(sh) >= 3:
        i1, i2, i3 = sh[-3], sh[-2], sh[-1]
        h1, h2, h3 = highs[i1], highs[i2], highs[i3]
        if h2 > h1 and h2 > h3 and abs(h1 - h3) / max(h1, h3) < 0.012:
            neckline = (lows[i1:i2 + 1].min() + lows[i2:i3 + 1].min()) / 2
            height = h2 - neckline
            if last_price < neckline * 1.001:
                return {"name": "Head & Shoulders", "bias": "Bearish", "entry": neckline,
                        "sl": h2, "tp": neckline - height, "lines": [("Neckline", neckline)]}
    if len(sl) >= 3:
        i1, i2, i3 = sl[-3], sl[-2], sl[-1]
        l1, l2, l3 = lows[i1], lows[i2], lows[i3]
        if l2 < l1 and l2 < l3 and abs(l1 - l3) / max(l1, l3) < 0.012:
            neckline = (highs[i1:i2 + 1].max() + highs[i2:i3 + 1].max()) / 2
            height = neckline - l2
            if last_price > neckline * 0.999:
                return {"name": "Inverse Head & Shoulders", "bias": "Bullish", "entry": neckline,
                        "sl": l2, "tp": neckline + height, "lines": [("Neckline", neckline)]}

    # ---- Triangles / Wedges (need >=2 recent swing highs AND lows) ----
    if len(sh) >= 2 and len(sl) >= 2:
        rh, rl = sh[-3:] if len(sh) >= 3 else sh[-2:], sl[-3:] if len(sl) >= 3 else sl[-2:]
        hs_slope = np.polyfit(rh, highs[rh], 1)[0]
        ls_slope = np.polyfit(rl, lows[rl], 1)[0]
        flat = closes[-20:].mean() * 0.00012
        res, sup = highs[rh].max(), lows[rl].min()
        h_flat, l_flat = abs(hs_slope) < flat, abs(ls_slope) < flat

        if h_flat and ls_slope > flat:
            return {"name": "Ascending Triangle", "bias": "Bullish bias (flat top, rising bottom)",
                    "entry": res, "sl": lows[rl[-1]], "tp": res + (res - sup),
                    "lines": [("Resistance", res)]}
        if l_flat and hs_slope < -flat:
            return {"name": "Descending Triangle", "bias": "Bearish bias (flat bottom, falling top)",
                    "entry": sup, "sl": highs[rh[-1]], "tp": sup - (res - sup),
                    "lines": [("Support", sup)]}
        if hs_slope < -flat and ls_slope > flat:
            return {"name": "Symmetrical Triangle", "bias": "Neutral — wait for breakout",
                    "entry": last_price, "sl": None, "tp": None, "lines": []}
        if hs_slope > flat and ls_slope > flat and ls_slope > hs_slope * 1.3:
            return {"name": "Rising Wedge", "bias": "Bearish", "entry": lows[rl[-1]],
                    "sl": res, "tp": lows[rl[-1]] - (res - sup), "lines": []}
        if hs_slope < -flat and ls_slope < -flat and hs_slope < ls_slope * 1.3:
            return {"name": "Falling Wedge", "bias": "Bullish", "entry": highs[rh[-1]],
                    "sl": sup, "tp": highs[rh[-1]] + (res - sup), "lines": []}
        if h_flat and l_flat:
            prior_up = closes[0] < closes[n // 3]
            bias = "Bullish (continuation)" if prior_up else "Bearish (continuation)"
            return {"name": "Rectangle", "bias": bias,
                    "entry": res if prior_up else sup, "sl": sup if prior_up else res,
                    "tp": None, "lines": [("Resistance", res), ("Support", sup)]}

    # ---- Flag / Pennant: sharp move (pole) then tight consolidation ----
    pole_len = max(6, n // 3)
    pole = closes[:pole_len]
    pole_move = (pole[-1] - pole[0]) / pole[0]
    consolidation = window.iloc[pole_len:]
    if len(consolidation) >= 6:
        cons_range = consolidation["High"].max() - consolidation["Low"].min()
        pole_range = abs(pole[-1] - pole[0])
        if pole_range > 0 and cons_range < pole_range * 0.6:
            if pole_move > 0.008:
                return {"name": "Bullish Flag/Pennant", "bias": "Bullish",
                        "entry": consolidation["High"].max(), "sl": consolidation["Low"].min(),
                        "tp": consolidation["High"].max() + pole_range, "lines": []}
            if pole_move < -0.008:
                return {"name": "Bearish Flag/Pennant", "bias": "Bearish",
                        "entry": consolidation["Low"].min(), "sl": consolidation["High"].max(),
                        "tp": consolidation["Low"].min() - pole_range, "lines": []}

    return None


# --------------------------------------------------------------------------
# Intraday setup signals (VWAP fade/breakout, MA pullback, trend-shift,
# opening range breakout, high-of-day, bull/bear trap, red-to-green,
# whole/half-dollar levels, momentum streaks) — from the day-trading
# study-guide reference. Unlike the pattern engines above, several of
# these can be true at the same time, so this returns a LIST.
# --------------------------------------------------------------------------
def get_prev_day_close(df: pd.DataFrame, sel_date):
    dates = sorted(set(df.index.date))
    if sel_date not in dates:
        return None
    pos = dates.index(sel_date)
    if pos == 0:
        return None
    prev_day = df[df.index.date == dates[pos - 1]]
    return prev_day["Close"].iloc[-1] if not prev_day.empty else None


def detect_setups(day_df: pd.DataFrame, idx: int, prev_day_close):
    signals = []
    row = day_df.iloc[idx]
    prev = day_df.iloc[idx - 1] if idx >= 1 else None

    # ---- Opening Range Breakout (first 15 min = 3 x 5-min bars) ----
    OR_N = 3
    if len(day_df) > OR_N and idx >= OR_N and prev is not None:
        or_high = day_df.iloc[:OR_N]["High"].max()
        or_low = day_df.iloc[:OR_N]["Low"].min()
        if row["Close"] > or_high and prev["Close"] <= or_high:
            signals.append(("Opening Range Breakout", "Bullish",
                             f"Price broke above the opening-range high (~{or_high:.2f})."))
        if row["Close"] < or_low and prev["Close"] >= or_low:
            signals.append(("Opening Range Breakdown", "Bearish",
                             f"Price broke below the opening-range low (~{or_low:.2f})."))

    # ---- High of Day / Low of Day breakout ----
    if idx >= 1:
        prior_high = day_df.iloc[:idx]["High"].max()
        prior_low = day_df.iloc[:idx]["Low"].min()
        if row["High"] > prior_high and row["Close"] > prior_high:
            signals.append(("High of Day Breakout", "Bullish",
                             "New high of day with a strong close."))
        if row["Low"] < prior_low and row["Close"] < prior_low:
            signals.append(("Low of Day Breakdown", "Bearish",
                             "New low of day with a weak close."))

    # ---- VWAP breakout / breakdown / fade ----
    if prev is not None:
        vwap = row["VWAP"]
        if prev["Close"] <= prev["VWAP"] and row["Close"] > vwap:
            signals.append(("VWAP Breakout", "Bullish", "Price crossed above VWAP."))
        if prev["Close"] >= prev["VWAP"] and row["Close"] < vwap:
            signals.append(("VWAP Breakdown", "Bearish", "Price crossed below VWAP."))

        recent = day_df.iloc[max(0, idx - 5):idx + 1]
        ext_up = (recent["High"].max() - vwap) / vwap
        ext_dn = (vwap - recent["Low"].min()) / vwap
        if ext_up > 0.01 and row["Close"] < recent["High"].max() * 0.995 and row["Close"] < row["Open"]:
            signals.append(("VWAP Fade (Short)", "Bearish",
                             "Price extended well above VWAP and is fading back down."))
        if ext_dn > 0.01 and row["Close"] > recent["Low"].min() * 1.005 and row["Close"] > row["Open"]:
            signals.append(("VWAP Fade (Long)", "Bullish",
                             "Price extended well below VWAP and is bouncing back up."))

    # ---- EMA9/EMA20 pullback + trend-shift cross ----
    if idx >= 2:
        near_ema9 = min(abs(row["Low"] - row["EMA9"]), abs(row["Close"] - row["EMA9"])) / row["EMA9"] < 0.0015
        if row["EMA9"] > row["EMA20"] and near_ema9 and row["Close"] > row["Open"]:
            signals.append(("Moving Average Pullback", "Bullish",
                             "Uptrend pulling back to EMA9 and bouncing."))
        if row["EMA9"] < row["EMA20"] and near_ema9 and row["Close"] < row["Open"]:
            signals.append(("Moving Average Pop (Short)", "Bearish",
                             "Downtrend popping into EMA9 and rejecting."))

        prevrow = day_df.iloc[idx - 1]
        if prevrow["EMA9"] >= prevrow["EMA20"] and row["EMA9"] < row["EMA20"]:
            signals.append(("Trend Shift — Bearish Cross", "Bearish", "EMA9 crossed below EMA20."))
        if prevrow["EMA9"] <= prevrow["EMA20"] and row["EMA9"] > row["EMA20"]:
            signals.append(("Trend Shift — Bullish Cross", "Bullish", "EMA9 crossed above EMA20."))

    # ---- Red-to-green / green-to-red vs prior day's close ----
    if prev_day_close is not None and prev is not None:
        if prev["Close"] <= prev_day_close and row["Close"] > prev_day_close:
            signals.append(("Red to Green Move", "Bullish", "Price crossed above yesterday's close."))
        if prev["Close"] >= prev_day_close and row["Close"] < prev_day_close:
            signals.append(("Green to Red Move", "Bearish", "Price crossed below yesterday's close."))

    # ---- Whole-dollar / half-dollar level ----
    px = row["Close"]
    if abs(px - round(px)) < 0.05:
        signals.append(("Whole Dollar Level", "Neutral",
                         "Price is sitting right at a whole-dollar level — common magnet/barrier."))
    elif abs(px - (round(px * 2) / 2)) < 0.03:
        signals.append(("Half Dollar Level", "Neutral", "Price is sitting right at a half-dollar level."))

    # ---- Bull trap / bear trap (false breakout within last ~8 bars) ----
    if idx >= 4:
        lookback = day_df.iloc[max(0, idx - 8):idx]
        recent_high, recent_low = lookback["High"].max(), lookback["Low"].min()
        broke_up = any(day_df.iloc[max(0, idx - 2):idx]["Close"] > recent_high)
        broke_dn = any(day_df.iloc[max(0, idx - 2):idx]["Close"] < recent_low)
        if broke_up and row["Close"] < recent_high:
            signals.append(("Bull Trap", "Bearish",
                             "Price broke a recent high then failed back below it — possible false breakout."))
        if broke_dn and row["Close"] > recent_low:
            signals.append(("Bear Trap", "Bullish",
                             "Price broke a recent low then failed back above it — possible false breakdown."))

    # ---- Consecutive same-direction candle streak ----
    direction = row["Close"] > row["Open"]
    streak, j = 1, idx
    while j > 0 and (day_df.iloc[j - 1]["Close"] > day_df.iloc[j - 1]["Open"]) == direction:
        streak += 1
        j -= 1
    if streak >= 5:
        tag = "Bullish" if direction else "Bearish"
        signals.append((f"{streak} Consecutive {tag} Candles", tag,
                         f"{streak} same-direction candles in a row — strong momentum, but also stretched."))

    # ---- RSI overbought / oversold ----
    if "RSI" in day_df.columns:
        rsi_val = row["RSI"]
        if rsi_val >= 70:
            signals.append(("RSI Overbought", "Bearish",
                             f"RSI at {rsi_val:.0f} — stretched to the upside, watch for a pullback."))
        elif rsi_val <= 30:
            signals.append(("RSI Oversold", "Bullish",
                             f"RSI at {rsi_val:.0f} — stretched to the downside, watch for a bounce."))

    # ---- RSI divergence (price vs RSI swing points over the last ~30 bars) ----
    if "RSI" in day_df.columns and idx >= 12:
        sub = day_df.iloc[max(0, idx - 30):idx + 1]
        if len(sub) >= 14:
            highs_, lows_, rsis_ = sub["High"].values, sub["Low"].values, sub["RSI"].values
            div_order = max(2, len(sub) // 10)
            sh_, sl_ = find_swings(highs_, lows_, div_order)
            if len(sh_) >= 2:
                i1, i2 = sh_[-2], sh_[-1]
                if highs_[i2] > highs_[i1] and rsis_[i2] < rsis_[i1] - 2:
                    signals.append(("RSI Bearish Divergence", "Bearish",
                                     "Price made a higher high while RSI made a lower high — momentum weakening."))
            if len(sl_) >= 2:
                i1, i2 = sl_[-2], sl_[-1]
                if lows_[i2] < lows_[i1] and rsis_[i2] > rsis_[i1] + 2:
                    signals.append(("RSI Bullish Divergence", "Bullish",
                                     "Price made a lower low while RSI made a higher low — momentum strengthening."))

    # ---- Volume confirmation tag on breakout/trap-style signals ----
    vol_ma = row.get("VolMA20", np.nan)
    if pd.notna(vol_ma) and vol_ma > 0:
        vol_ratio = row["Volume"] / vol_ma
        tagged_names = {
            "Opening Range Breakout", "Opening Range Breakdown", "High of Day Breakout",
            "Low of Day Breakdown", "VWAP Breakout", "VWAP Breakdown", "Bull Trap", "Bear Trap",
        }
        for i, (name, sbias, sdesc) in enumerate(signals):
            if name in tagged_names:
                if vol_ratio >= 1.3:
                    sdesc += f" Backed by strong volume ({vol_ratio:.1f}x average)."
                elif vol_ratio < 0.7:
                    sdesc += f" Volume is light ({vol_ratio:.1f}x average) — treat with extra caution."
                signals[i] = (name, sbias, sdesc)

    return signals


# --------------------------------------------------------------------------
# Composite Setup Score (0-100) — five weighted categories built on top of
# the signals already detected above. Direction-symmetric: works identically
# whether the anchor direction is Bullish or Bearish. The "anchor" is the
# immediate candlestick pattern's bias; if that candle is neutral, the
# short-term EMA9/EMA20 relationship is used instead.
# --------------------------------------------------------------------------
CANDLE_STRENGTH_TIERS = {
    "Bullish Marubozu": 25, "Bearish Marubozu": 25,
    "Bullish Engulfing": 22, "Bearish Engulfing": 22,
    "Morning Star": 22, "Evening Star": 22,
    "Three White Soldiers": 22, "Three Black Crows": 22,
    "Hammer": 15, "Hanging Man": 15, "Shooting Star": 15, "Inverted Hammer": 15,
    "Piercing Line": 15, "Dark Cloud Cover": 15,
    "Dragonfly Doji": 12, "Gravestone Doji": 12,
    "Bullish Harami": 10, "Bearish Harami": 10,
    "Tweezer Bottom": 10, "Tweezer Top": 10,
    "Spinning Top": 5, "Doji": 3, "Long-Legged Doji": 3,
    "No clear pattern": 0,
}

STRONG_CHART_PATTERNS = {"Double Top", "Double Bottom", "Head & Shoulders", "Inverse Head & Shoulders"}
MID_CHART_PATTERNS = {"Rising Wedge", "Falling Wedge", "Bullish Flag/Pennant", "Bearish Flag/Pennant"}


def lerp_score(value: float, points: list) -> float:
    """Piecewise-linear interpolation through (x, y) points sorted by x.
    Used so score categories grade smoothly instead of jumping between
    fixed tiers — e.g. VWAP distance earns proportionally more credit
    the further price pushes away, rather than a flat cliff."""
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            frac = (value - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + frac * (y1 - y0)
    return points[-1][1]


def compute_setup_score(pattern: str, pattern_bias: str, cp, row, setups: list):
    if pattern_bias in ("Bullish", "Bearish"):
        anchor = pattern_bias
    elif row["EMA9"] > row["EMA20"]:
        anchor = "Bullish"
    elif row["EMA9"] < row["EMA20"]:
        anchor = "Bearish"
    else:
        anchor = "Neutral"

    if anchor == "Neutral":
        return {"anchor": "Neutral", "total": 0, "label": "No directional setup", "breakdown": []}

    breakdown = []

    # -- 1. Candle strength (0-25) — categorical: a pattern either is a
    #    given shape or it isn't, so tiers (not a continuous scale) fit here.
    strength = CANDLE_STRENGTH_TIERS.get(pattern, 0)
    if pattern_bias == anchor:
        c1, c1_note = strength, f"{pattern} supports the {anchor.lower()} case."
    else:
        c1 = 0
        c1_note = "No candle confirmation at this bar." if pattern == "No clear pattern" \
            else f"{pattern} doesn't confirm the {anchor.lower()} case."
    breakdown.append(("Candle Strength", c1, 25, c1_note))

    # -- 2. Multi-bar structure (0-25) — also categorical (which named
    #    chart pattern, if any, is currently forming).
    if cp is None:
        c2, c2_note = 0, "No structural chart pattern currently forming."
    elif anchor in cp["bias"]:
        if cp["name"] in STRONG_CHART_PATTERNS:
            c2 = 25
        elif cp["name"] in MID_CHART_PATTERNS:
            c2 = 20
        elif cp["name"] == "Rectangle":
            c2 = 15
        else:
            c2 = 12
        c2_note = f"{cp['name']} aligns with the {anchor.lower()} case."
    elif "Neutral" in cp["bias"] or "wait" in cp["bias"].lower():
        c2, c2_note = 8, f"{cp['name']} is unresolved — no confirmed direction yet."
    else:
        c2, c2_note = 0, f"{cp['name']} points the other way."
    breakdown.append(("Multi-bar Structure", c2, 25, c2_note))

    # -- 3. Trend baseline (0-20) — CONTINUOUS. Scores the size of the gap
    #    between price and VWAP (0-10) plus how cleanly the EMAs are
    #    stacked apart (0-10), both scaling smoothly with distance rather
    #    than jumping at fixed thresholds. An overstretch penalty then
    #    fades the score back down if price has run too far from EMA9.
    price, ema9, ema20, vwap = row["Close"], row["EMA9"], row["EMA20"], row["VWAP"]
    sign = 1 if anchor == "Bullish" else -1
    vwap_dist_pct = sign * (price - vwap) / vwap if vwap else 0
    ema_spread_pct = sign * (ema9 - ema20) / ema20 if ema20 else 0

    # 0% distance -> 0 pts, 0.6% favorable distance -> full 10 pts (clamped)
    vwap_pts = lerp_score(vwap_dist_pct, [(0.0, 0), (0.006, 10)])
    # EMA9 at/through EMA20 -> 0 pts, 0.4% favorable spread -> full 10 pts
    ema_pts = lerp_score(ema_spread_pct, [(0.0, 0), (0.004, 10)])
    c3_raw = vwap_pts + ema_pts

    ext = sign * (price - ema9) / ema9 if ema9 else 0
    # fades score to ~30% once price is >3% extended beyond EMA9
    penalty = lerp_score(ext, [(0.012, 1.0), (0.03, 0.3)])
    c3 = round(c3_raw * penalty)

    vwap_desc = f"{abs(vwap_dist_pct) * 100:.2f}% {'above' if vwap_dist_pct >= 0 else 'below'} VWAP (favorable)" \
        if vwap_dist_pct >= 0 else f"{abs(vwap_dist_pct) * 100:.2f}% on the wrong side of VWAP"
    c3_note = f"{vwap_desc}; EMA9/EMA20 spread {ema_spread_pct * 100:.2f}%."
    if penalty < 0.95:
        c3_note += " Reduced — price overstretched from EMA9 (mean-reversion risk)."
    breakdown.append(("Trend Baseline", c3, 20, c3_note))

    # -- 4. Volume conviction (0-15) — CONTINUOUS, direction-agnostic.
    vol_ma = row.get("VolMA20", float("nan"))
    if pd.notna(vol_ma) and vol_ma > 0:
        ratio = row["Volume"] / vol_ma
        c4 = round(lerp_score(ratio, [(0.5, 0), (1.0, 4), (1.5, 8), (3.0, 12), (5.0, 15)]))
        c4_note = f"Volume {ratio:.1f}x average."
    else:
        c4, c4_note = 0, "Not enough bars yet for a volume baseline."
    breakdown.append(("Volume Conviction", c4, 15, c4_note))

    # -- 5. Momentum / RSI state (0-15) — CONTINUOUS "lifecycle" curve that
    #    peaks in the zone with the most room left to run in the anchor's
    #    direction, and tapers off both toward exhaustion and toward
    #    "hasn't turned yet." Bullish is just the mirror image of bearish
    #    (reflected around RSI 50), so one curve serves both directions.
    rsi = row["RSI"]
    div_conflict = any(
        (anchor == "Bullish" and name == "RSI Bearish Divergence") or
        (anchor == "Bearish" and name == "RSI Bullish Divergence")
        for name, _, _ in setups
    )
    bearish_curve = [(0, 2), (30, 2), (35, 5), (40, 8), (55, 15), (65, 15), (70, 10), (78, 3), (100, 0)]
    rsi_for_curve = rsi if anchor == "Bearish" else (100 - rsi)
    c5 = round(lerp_score(rsi_for_curve, bearish_curve))
    if div_conflict:
        c5, c5_note = 0, "RSI divergence is working against this direction."
    else:
        c5_note = f"RSI {rsi:.0f}."
    breakdown.append(("Momentum (RSI)", c5, 15, c5_note))

    total = sum(b[1] for b in breakdown)
    if total >= 80:
        tag = "High-Conviction"
    elif total >= 60:
        tag = "Solid"
    elif total >= 40:
        tag = "Mixed"
    elif total >= 20:
        tag = "Weak"
    else:
        tag = "Very Weak"

    return {"anchor": anchor, "total": total, "label": f"{anchor} Setup — {tag}", "breakdown": breakdown}
