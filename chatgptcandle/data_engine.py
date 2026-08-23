"""Market-data and indicator layer for the Chart Reading Machine."""

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
def get_data(ticker, interval):
    period = "60d" if interval in ["1m","2m","5m","15m","30m"] else "6mo"
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open","High","Low","Close","Volume"]].dropna()

def add_indicators(df):
    d = df.copy()
    d["EMA9"] = d.Close.ewm(span=9, adjust=False).mean()
    d["EMA20"] = d.Close.ewm(span=20, adjust=False).mean()

    typical = (d.High + d.Low + d.Close) / 3
    day = pd.Series(d.index.date, index=d.index)
    pv = (typical * d.Volume).groupby(day).cumsum()
    vv = d.Volume.groupby(day).cumsum().replace(0, np.nan)
    d["VWAP"] = pv / vv
    d["VolAvg20"] = d.Volume.rolling(20).mean()
    return d

def swings(df, n=3):
    sh = df.High[(df.High.shift(n)<df.High)&(df.High.shift(-n)<df.High)].dropna()
    sl = df.Low[(df.Low.shift(n)>df.Low)&(df.Low.shift(-n)>df.Low)].dropna()
    return sh, sl

def level_detail(df):
    """Build simple support/resistance zones from confirmed fractal swings."""
    sh, sl = swings(df, n=3)
    price = float(df.Close.iloc[-1])

    # Keep the zone rule simple: nearby fractal prices belong to one zone.
    # The threshold adapts slightly to the instrument price and recent range.
    threshold = max(
        price * 0.0015,
        float(df.High.tail(20).max() - df.Low.tail(20).min()) * 0.02
    )

    highs = list(sh.tail(20).astype(float)) if len(sh) else []
    lows = list(sl.tail(20).astype(float)) if len(sl) else []

    def cluster(values):
        clusters = []
        for value in sorted(values):
            if not clusters or value - clusters[-1]["high"] > threshold:
                clusters.append({
                    "low": value,
                    "high": value,
                    "values": [value]
                })
            else:
                clusters[-1]["values"].append(value)
                clusters[-1]["low"] = min(clusters[-1]["low"], value)
                clusters[-1]["high"] = max(clusters[-1]["high"], value)

        for c in clusters:
            c["level"] = float(np.mean(c["values"]))
            c["touches"] = len(c["values"])
        return clusters

    resistance_zones = [c for c in cluster(highs) if c["level"] >= price]
    support_zones = [c for c in cluster(lows) if c["level"] <= price]

    # If price is already inside a zone, allow that zone to be selected.
    if not resistance_zones and highs:
        resistance_zones = cluster(highs)
    if not support_zones and lows:
        support_zones = cluster(lows)

    if resistance_zones:
        rc = min(resistance_zones, key=lambda c: abs(c["level"] - price))
    else:
        rc = {"level": float(df.High.tail(20).max()),
              "low": float(df.High.tail(20).max()),
              "high": float(df.High.tail(20).max()),
              "touches": 0}

    if support_zones:
        sc = min(support_zones, key=lambda c: abs(c["level"] - price))
    else:
        sc = {"level": float(df.Low.tail(20).min()),
              "low": float(df.Low.tail(20).min()),
              "high": float(df.Low.tail(20).min()),
              "touches": 0}

    r = float(rc["level"])
    s = float(sc["level"])

    rlow, rhigh = float(rc["low"]), float(rc["high"])
    slow, shigh = float(sc["low"]), float(sc["high"])

    # A small buffer lets a zone behave like a zone rather than a single line.
    buffer = threshold * 0.35

    if slow - buffer <= price <= shigh + buffer:
        location = "Inside support zone"
        location_bias = "Neutral"
    elif rlow - buffer <= price <= rhigh + buffer:
        location = "Inside resistance zone"
        location_bias = "Neutral"
    elif price > rhigh + buffer:
        location = "Above resistance"
        location_bias = "Bullish"
    elif price < slow - buffer:
        location = "Below support"
        location_bias = "Bearish"
    else:
        location = "Between key zones"
        location_bias = "Neutral"

    def strength(touches):
        if touches >= 4:
            return "Strong"
        if touches >= 2:
            return "Moderate"
        return "Weak"

    return {
        "support": s,
        "resistance": r,
        "support_low": slow,
        "support_high": shigh,
        "resistance_low": rlow,
        "resistance_high": rhigh,
        "zone": threshold,
        "support_touches": int(sc["touches"]),
        "resistance_touches": int(rc["touches"]),
        "support_strength": strength(int(sc["touches"])),
        "resistance_strength": strength(int(rc["touches"])),
        "location": location,
        "location_bias": location_bias,
    }

def levels(df):
    d = level_detail(df)
    return d["support"], d["resistance"], d["zone"]

