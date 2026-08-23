"""Chart-reading logic for the Chart Reading Machine."""

import numpy as np
import pandas as pd

from data_engine import level_detail, swings

# ============================================================
def candle(df):
    """Recognize common single- and multi-candle patterns."""
    if len(df) < 3:
        return "Not enough candles","Neutral","Low"

    a, p, p2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    def parts(x):
        body = abs(float(x.Open) - float(x.Close))
        rng = max(float(x.High) - float(x.Low), 1e-9)
        upper = float(x.High) - max(float(x.Open), float(x.Close))
        lower = min(float(x.Open), float(x.Close)) - float(x.Low)
        return body, rng, upper, lower

    body, rng, upper, lower = parts(a)
    pbody, prng, pupper, plower = parts(p)
    p2body, p2rng, p2upper, p2lower = parts(p2)

    # Three-candle reversal/continuation patterns.
    if (p2.Close < p2.Open and
        pbody <= p2body * 0.6 and
        a.Close > a.Open and
        a.Close >= (p2.Open + p2.Close) / 2):
        return "Morning Star","Bullish","High"

    if (p2.Close > p2.Open and
        pbody <= p2body * 0.6 and
        a.Close < a.Open and
        a.Close <= (p2.Open + p2.Close) / 2):
        return "Evening Star","Bearish","High"

    if (all(x.Close > x.Open for x in [p2, p, a]) and
        p2.Close < p.Close < a.Close and
        all(parts(x)[2] < parts(x)[0] * 0.8 for x in [p2, p, a])):
        return "Three White Soldiers","Bullish","High"

    if (all(x.Close < x.Open for x in [p2, p, a]) and
        p2.Close > p.Close > a.Close and
        all(parts(x)[3] < parts(x)[0] * 0.8 for x in [p2, p, a])):
        return "Three Black Crows","Bearish","High"

    # Two-candle patterns.
    if p.Close < p.Open and a.Close > a.Open and a.Open <= p.Close and a.Close >= p.Open:
        return "Bullish Engulfing","Bullish","High"

    if p.Close > p.Open and a.Close < a.Open and a.Open >= p.Close and a.Close <= p.Open:
        return "Bearish Engulfing","Bearish","High"

    if p.Close < p.Open and a.Close > a.Open and a.Close > (p.Open + p.Close) / 2:
        return "Piercing Line","Bullish","Medium"

    if p.Close > p.Open and a.Close < a.Open and a.Close < (p.Open + p.Close) / 2:
        return "Dark Cloud Cover","Bearish","Medium"

    # Single-candle patterns.
    if body / rng < 0.10:
        return "Doji","Neutral","Medium"

    if lower > body * 2 and upper < body * 0.75:
        return ("Hammer" if a.Close >= a.Open else "Hanging Man"),                ("Bullish" if a.Close >= a.Open else "Bearish"), "Medium"

    if upper > body * 2 and lower < body * 0.75:
        return ("Inverted Hammer" if a.Close >= a.Open else "Shooting Star"),                ("Bullish" if a.Close >= a.Open else "Bearish"), "Medium"

    if a.Close > a.Open:
        return "Bullish Candle","Bullish","Low"
    if a.Close < a.Open:
        return "Bearish Candle","Bearish","Low"
    return "Neutral Candle","Neutral","Low"

def structure(df):
    """Read market structure from confirmed swing highs/lows.

    EMA alignment remains a supporting input, but no longer defines
    the trend by itself. The machine first asks: HH/HL or LH/LL?
    """
    if len(df) < 25:
        return "Developing","Neutral",50

    sh, sl = swings(df, n=3)

    if len(sh) < 2 or len(sl) < 2:
        c=float(df.Close.iloc[-1])
        e9=float(df.EMA9.iloc[-1])
        e20=float(df.EMA20.iloc[-1])
        if c > e20 and e9 > e20:
            return "Bullish — developing structure","Bullish",65
        if c < e20 and e9 < e20:
            return "Bearish — developing structure","Bearish",35
        return "Structure developing","Neutral",50

    last_h, prev_h = float(sh.iloc[-1]), float(sh.iloc[-2])
    last_l, prev_l = float(sl.iloc[-1]), float(sl.iloc[-2])

    higher_high = last_h > prev_h
    higher_low = last_l > prev_l
    lower_high = last_h < prev_h
    lower_low = last_l < prev_l

    # Price structure gets priority.
    if higher_high and higher_low:
        return "Bullish Structure (HH + HL)","Bullish",85
    if lower_high and lower_low:
        return "Bearish Structure (LH + LL)","Bearish",15

    # One side improving while the other is not = transition.
    if higher_low and not lower_high:
        return "Bullish pressure — structure forming","Bullish",68
    if lower_high and not higher_low:
        return "Bearish pressure — structure forming","Bearish",32

    return "Range / mixed structure","Neutral",50

def structure_detail(df):
    """Return the latest confirmed swing relationship for the UI."""
    sh, sl = swings(df, n=3)

    if len(sh) < 2 or len(sl) < 2:
        return "Not enough confirmed swings", "Neutral"

    hh = float(sh.iloc[-1]) > float(sh.iloc[-2])
    hl = float(sl.iloc[-1]) > float(sl.iloc[-2])
    lh = float(sh.iloc[-1]) < float(sh.iloc[-2])
    ll = float(sl.iloc[-1]) < float(sl.iloc[-2])

    if hh and hl:
        return "Higher High + Higher Low (HH + HL)", "Bullish"
    if lh and ll:
        return "Lower High + Lower Low (LH + LL)", "Bearish"
    if hl:
        return "Higher Low forming", "Bullish"
    if lh:
        return "Lower High forming", "Bearish"
    return "Mixed / range structure", "Neutral"

def pattern(df):
    """Detect the initial V1.1 pattern set, including H&S structures."""
    if len(df) < 35:
        return "No clear pattern","Developing","Neutral",40

    sh, sl = swings(df, n=3)
    price = float(df.Close.iloc[-1])

    # --------------------------------------------------------------
    # Head & Shoulders
    # Three swing highs: left shoulder, head, right shoulder.
    # The two intervening swing lows form the neckline.
    # --------------------------------------------------------------
    if len(sh) >= 3 and len(sl) >= 2:
        hs = [float(x) for x in sh.tail(3)]
        ls = [float(x) for x in sl.tail(2)]

        left_shoulder, head, right_shoulder = hs
        shoulder_diff = abs(left_shoulder - right_shoulder) / max(
            abs(left_shoulder), 1e-9
        )

        # Head should stand clearly above both shoulders.
        head_above = (
            head > left_shoulder * 1.003 and
            head > right_shoulder * 1.003
        )

        if shoulder_diff <= 0.015 and head_above:
            neckline = float(np.mean(ls))
            neckline_range = max(ls) - min(ls)

            # The troughs should be reasonably close so this is one
            # neckline rather than two unrelated lows.
            neckline_ok = neckline_range / max(abs(neckline), 1e-9) <= 0.02

            if neckline_ok:
                stage = "Breakdown" if price < neckline else "Developing"
                strength = 88 if stage == "Breakdown" else 76
                return "Head & Shoulders", stage, "Bearish", strength

    # --------------------------------------------------------------
    # Inverse Head & Shoulders
    # Three swing lows: left shoulder, head, right shoulder.
    # The two intervening swing highs form the neckline.
    # --------------------------------------------------------------
    if len(sl) >= 3 and len(sh) >= 2:
        is_ = [float(x) for x in sl.tail(3)]
        ih_left, ih_head, ih_right = is_
        shoulder_diff = abs(ih_left - ih_right) / max(abs(ih_left), 1e-9)

        head_below = (
            ih_head < ih_left * 0.997 and
            ih_head < ih_right * 0.997
        )

        if shoulder_diff <= 0.015 and head_below:
            neckline = float(np.mean([float(x) for x in sh.tail(2)]))
            highs = [float(x) for x in sh.tail(2)]
            neckline_range = max(highs) - min(highs)
            neckline_ok = neckline_range / max(abs(neckline), 1e-9) <= 0.02

            if neckline_ok:
                stage = "Breakout" if price > neckline else "Developing"
                strength = 88 if stage == "Breakout" else 76
                return "Inverse Head & Shoulders", stage, "Bullish", strength

    if len(sh) >= 2 and len(sl) >= 2:
        h1, h2 = float(sh.iloc[-2]), float(sh.iloc[-1])
        l1, l2 = float(sl.iloc[-2]), float(sl.iloc[-1])

        high_diff = abs(h2 - h1) / max(abs(h1), 1e-9)
        low_diff = abs(l2 - l1) / max(abs(l1), 1e-9)

        # Double bottom/top: comparable extremes with a meaningful reaction.
        if low_diff < 0.012:
            intervening_high = float(df.loc[sl.index[-2]:sl.index[-1], "High"].max())
            if intervening_high > max(l1, l2) * 1.004:
                neckline = intervening_high
                stage = "Breakout" if price > neckline else "Developing"
                strength = 85 if stage == "Breakout" else 72
                return "Double Bottom", stage, "Bullish", strength

        if high_diff < 0.012:
            intervening_low = float(df.loc[sh.index[-2]:sh.index[-1], "Low"].min())
            if intervening_low < min(h1, h2) * 0.996:
                neckline = intervening_low
                stage = "Breakdown" if price < neckline else "Developing"
                strength = 85 if stage == "Breakdown" else 72
                return "Double Top", stage, "Bearish", strength

        # Triangles: converging range with directional pressure.
        recent_highs = [float(x) for x in sh.tail(3)]
        recent_lows = [float(x) for x in sl.tail(3)]

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            highs_falling = recent_highs[-1] < recent_highs[-2]
            highs_flat = abs(recent_highs[-1] - recent_highs[-2]) / recent_highs[-2] < 0.006
            lows_rising = recent_lows[-1] > recent_lows[-2]
            lows_flat = abs(recent_lows[-1] - recent_lows[-2]) / recent_lows[-2] < 0.006

            if highs_flat and lows_rising:
                return "Ascending Triangle", "Developing", "Bullish", 70
            if lows_flat and highs_falling:
                return "Descending Triangle", "Developing", "Bearish", 70
            if highs_falling and lows_rising:
                return "Symmetrical Triangle", "Developing", "Neutral", 65

    return "No clear pattern","No strong structure","Neutral",40

def volume(df):
    if len(df)<20: return "Developing","Neutral",50
    avg=float(df.VolAvg20.iloc[-1])
    if not np.isfinite(avg) or avg<=0: return "Unavailable","Neutral",50
    ratio=float(df.Volume.iloc[-1])/avg
    if ratio>=1.5: return "Volume expansion","Bullish",80
    if ratio<=.7: return "Volume contraction","Neutral",45
    return "Normal volume","Neutral",60

def event(df,s,r,z):
    c=float(df.Close.iloc[-1]); p=float(df.Close.iloc[-2])
    cn, cb, cs = candle(df)
    ld = level_detail(df)

    # Work with the actual zone boundaries rather than a single line.
    if c > ld["resistance_high"] + z and p <= ld["resistance_high"] + z:
        return "Breakout attempt","Bullish"
    if c < ld["support_low"] - z and p >= ld["support_low"] - z:
        return "Breakdown attempt","Bearish"

    if ld["resistance_low"] - z <= c <= ld["resistance_high"] + z:
        if cb == "Bearish":
            return "Resistance zone + rejection","Bearish"
        return "Testing resistance zone","Neutral"

    if ld["support_low"] - z <= c <= ld["support_high"] + z:
        if cb == "Bullish":
            return "Support zone + rejection","Bullish"
        return "Testing support zone","Neutral"

    if cb=="Bearish" and c<p:
        return "Bearish rejection developing","Bearish"
    if cb=="Bullish" and c>p:
        return "Bullish recovery","Bullish"
    return "Consolidating / no immediate event","Neutral"

def score(df,ts,ps):
    c=float(df.Close.iloc[-1]); last=df.iloc[-1]

    # Structure is the backbone of the reading.
    x=ts*.34

    # VWAP + EMA = alignment/confirmation.
    if c > float(last.VWAP):
        x += 8
    if float(last.EMA9) > float(last.EMA20):
        x += 8

    _,_,vs=volume(df)
    x += vs*.10

    cn,cb,sig=candle(df)
    x += {"High":14,"Medium":10,"Low":7}.get(sig,7)

    # Pattern is useful, but cannot overpower the structure.
    x += ps*.12

    ld=level_detail(df)
    distance=min(abs(c-ld["support"]),abs(c-ld["resistance"]))
    x += 9 if distance <= ld["zone"]*2 else 5

    # Reward coherent alignment; penalize obvious conflict.
    structure_bias = "Bullish" if ts >= 65 else "Bearish" if ts <= 35 else "Neutral"
    if structure_bias == cb and cb != "Neutral":
        x += 5
    elif structure_bias != "Neutral" and cb != "Neutral" and structure_bias != cb:
        x -= 5

    return int(max(0,min(100,round(x))))

