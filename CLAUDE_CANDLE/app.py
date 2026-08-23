"""
Home — Candlestick Pattern Scanner
-----------------------------------
Single-ticker, live-tracking view: pulls 5-min candles from Yahoo Finance,
overlays EMA9/EMA20/VWAP, and surfaces candlestick patterns, multi-bar
chart patterns, intraday setup signals, and a composite Setup Score.

For scanning many tickers at once, see the "Market Scanner" page in the
sidebar navigation.

Run locally with:
    pip install streamlit yfinance pandas plotly pytz streamlit-autorefresh
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

from lib import (
    PATTERN_INFO, CHART_PATTERN_DESC,
    fetch_candles, add_indicators, smooth_edges,
    detect_pattern, find_swings, detect_chart_pattern,
    get_prev_day_close, detect_setups, compute_setup_score,
)

st.set_page_config(page_title="Candlestick Pattern Scanner", layout="wide")

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📊 Candlestick Pattern Scanner")

with st.sidebar:
    st.header("Live Mode")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🔄 Refresh now", key="refresh_button"):
            fetch_candles.clear()
            st.rerun()
    with col_b:
        if AUTOREFRESH_AVAILABLE:
            auto_refresh = st.checkbox("Auto-refresh", value=False, key="auto_refresh_toggle")
        else:
            auto_refresh = False
            st.caption("Auto-refresh unavailable (package not installed) — use Refresh now instead.")

    if auto_refresh and AUTOREFRESH_AVAILABLE:
        refresh_secs = st.slider("Every (seconds)", 15, 300, 60, step=15, key="refresh_secs")
        st_autorefresh(interval=refresh_secs * 1000, key="live_autorefresh")

    follow_latest = st.checkbox("Follow latest bar (stay live)", value=True, key="follow_latest")

    st.header("Watchlist")
    tickers_raw = st.text_input(
        "Tickers (comma-separated)", value="AAPL,MSFT,TSLA,NVDA,AMZN", key="tickers_raw"
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    if not tickers:
        st.warning("Enter at least one ticker.")
        st.stop()

    # Key includes the watchlist content itself, not just a fixed name.
    # Streamlit can keep a stale internal copy of a widget's option list
    # when its value is overridden programmatically under a reused key —
    # baking the ticker list into the key forces a genuinely fresh widget
    # (not a stale cached one) whenever the watchlist text changes.
    ticker_widget_key = "ticker_select__" + "_".join(tickers)
    ticker = st.selectbox("Active ticker", tickers, key=ticker_widget_key)

    period_days = st.slider("Days of history to fetch", 1, 59, 5, key="period_days")
    smooth = st.checkbox("Smooth first/last 3 candles (edge noise)", value=True, key="smooth_edges")

    raw = fetch_candles(ticker, period_days)
    if raw.empty:
        st.error(f"No data returned for {ticker}. Try a different ticker or fewer days.")
        st.stop()

    if smooth:
        raw = smooth_edges(raw, n=3)

    df = add_indicators(raw)

    st.header("Date / Time")
    available_dates = sorted(set(df.index.date))
    latest_date = available_dates[-1]

    if follow_latest:
        sel_date = latest_date
        st.caption(f"Date: **{sel_date}** (following latest)")
    else:
        sel_date = st.selectbox("Date", available_dates, index=len(available_dates) - 1, key="sel_date")

    day_df = df[df.index.date == sel_date]
    times = [t.strftime("%H:%M") for t in day_df.index]

    if follow_latest:
        sel_time = times[-1]
        st.caption(f"Time: **{sel_time}** (following latest)")
    else:
        sel_time = st.selectbox("Time (5-min bar)", times, index=len(times) - 1, key="sel_time")

    st.header("Multi-bar chart patterns")
    intraday_only = st.checkbox("Intraday only (stay within selected day)", value=True, key="intraday_only")
    chart_lookback = st.slider("Lookback bars for chart patterns", 20, 150, 60, key="chart_lookback")

sel_ts = day_df.index[times.index(sel_time)]
sel_pos_in_day = times.index(sel_time)

# Detect everything the score needs BEFORE building the chart, so the score
# badge can be drawn directly onto the chart (top-right corner).
pattern = detect_pattern(day_df.reset_index(drop=True), sel_pos_in_day)
bias, desc = PATTERN_INFO.get(pattern, ("Neutral", ""))
color = {"Bullish": "green", "Bearish": "red", "Neutral": "gray"}[bias]

prev_close = get_prev_day_close(df, sel_date)
setups = detect_setups(day_df, sel_pos_in_day, prev_close)

if intraday_only:
    source = day_df[day_df.index <= sel_ts]
else:
    source = df[df.index <= sel_ts]
window = source.tail(chart_lookback).copy()
cp = detect_chart_pattern(window.reset_index(drop=True))

score_row = day_df.iloc[sel_pos_in_day]
score = compute_setup_score(pattern, bias, cp, score_row, setups)

# fig ---------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=day_df.index, open=day_df["Open"], high=day_df["High"],
    low=day_df["Low"], close=day_df["Close"], name=ticker
))
fig.add_trace(go.Scatter(x=day_df.index, y=day_df["EMA9"], line=dict(width=1.5), name="EMA9"))
fig.add_trace(go.Scatter(x=day_df.index, y=day_df["EMA20"], line=dict(width=1.5), name="EMA20"))
fig.add_trace(go.Scatter(x=day_df.index, y=day_df["VWAP"], line=dict(width=1.5, dash="dot"), name="VWAP"))

fig.add_vline(x=sel_ts, line_width=1, line_dash="dash", line_color="gray")

# Setup Score badge, top-right corner of the chart
if score["anchor"] != "Neutral":
    badge_color = "#26a69a" if score["anchor"] == "Bullish" else "#ef5350"
    fig.add_annotation(
        xref="paper", yref="paper", x=1.0, y=1.18, showarrow=False,
        align="right", xanchor="right",
        text=f"<b>{score['total']}/100</b>&nbsp;&nbsp;{score['label']}",
        font=dict(size=20, color="white"),
        bgcolor=badge_color, bordercolor="white", borderwidth=1.5, borderpad=10, opacity=0.92,
    )

fig.update_layout(
    height=600, xaxis_rangeslider_visible=False,
    title=f"{ticker} — 5m — {sel_date}" + (" 🔴 LIVE" if follow_latest else ""),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=110),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# RSI / Volume compact readout box
# --------------------------------------------------------------------------
def rsi_reading(rsi_val: float):
    if rsi_val >= 70:
        return "Overbought", "red", "Stretched to the upside — momentum extended, watch for a pullback."
    if rsi_val >= 55:
        return "Bullish", "green", "Momentum favors the upside."
    if rsi_val <= 30:
        return "Oversold", "green", "Stretched to the downside — momentum extended, watch for a bounce."
    if rsi_val <= 45:
        return "Bearish", "red", "Momentum favors the downside."
    return "Neutral", "gray", "No strong momentum bias either way."


def volume_reading(ratio):
    if ratio is None:
        return "n/a", "gray", "Not enough bars yet to compute a volume average."
    if ratio >= 1.5:
        return "Very strong activity", "orange", "Well above average participation."
    if ratio >= 1.1:
        return "Above average", "orange", "More participation than a typical bar."
    if ratio >= 0.7:
        return "Average activity", "gray", "Typical participation for this ticker."
    return "Light activity", "gray", "Below-average participation — moves here carry less weight."


live_row = day_df.iloc[sel_pos_in_day]
live_vol_ma = live_row.get("VolMA20", float("nan"))
live_vol_ratio = (live_row["Volume"] / live_vol_ma) if pd.notna(live_vol_ma) and live_vol_ma > 0 else None

rsi_label, rsi_color, rsi_note = rsi_reading(live_row["RSI"])
vol_label, vol_color, vol_note = volume_reading(live_vol_ratio)

with st.container(border=True):
    rb1, rb2 = st.columns(2)
    with rb1:
        st.markdown(
            f"**RSI (14): {live_row['RSI']:.0f} — "
            f"<span style='color:{rsi_color}'>{rsi_label}</span>**",
            unsafe_allow_html=True,
        )
        st.caption(rsi_note)
    with rb2:
        ratio_str = f"{live_vol_ratio:.1f}x avg" if live_vol_ratio is not None else "n/a"
        st.markdown(
            f"**Volume: {live_row['Volume']:,.0f} ({ratio_str}) — "
            f"<span style='color:{vol_color}'>{vol_label}</span>**",
            unsafe_allow_html=True,
        )
        st.caption(vol_note)

col_pat, col_setup, col_multibar = st.columns(3)

with col_pat:
    st.markdown(
        f"### Pattern at {sel_time}: "
        f"<span style='color:{color}'>{pattern}</span> ({bias})",
        unsafe_allow_html=True,
    )
    st.write(desc)

    row = day_df.iloc[sel_pos_in_day]
    st.caption(
        f"O {row['Open']:.2f}  H {row['High']:.2f}  L {row['Low']:.2f}  C {row['Close']:.2f}  "
        f"| EMA9 {row['EMA9']:.2f}  EMA20 {row['EMA20']:.2f}  VWAP {row['VWAP']:.2f}"
    )

with col_setup:
    st.markdown("### 🎯 Intraday Setup Signals")
    st.caption("Several can be true at once — that's normal, not a conflict.")
    if setups:
        for name, sbias, sdesc in setups:
            color3 = {"Bullish": "green", "Bearish": "red", "Neutral": "gray"}[sbias]
            st.markdown(f"**{name}** — <span style='color:{color3}'>{sbias}</span>  \n{sdesc}",
                        unsafe_allow_html=True)
    else:
        st.write("No intraday setup signals at this timestamp.")

with col_multibar:
    st.markdown("### 📐 Multi-bar Chart Pattern")
    if cp:
        color2 = "green" if "Bullish" in cp["bias"] else ("red" if "Bearish" in cp["bias"] else "gray")
        st.markdown(f"**{cp['name']}: <span style='color:{color2}'>{cp['bias']}</span>**",
                    unsafe_allow_html=True)
        st.caption(CHART_PATTERN_DESC.get(cp["name"], ""))
        levels = []
        if cp.get("entry") is not None:
            levels.append(f"Entry ≈ {cp['entry']:.2f}")
        if cp.get("sl") is not None:
            levels.append(f"SL ≈ {cp['sl']:.2f}")
        if cp.get("tp") is not None:
            levels.append(f"TP ≈ {cp['tp']:.2f}")
        if levels:
            st.caption(" | ".join(levels))
    else:
        st.write("No clear multi-bar pattern right now.")
    st.caption("Full chart with swing points below ⬇️")

if score["anchor"] != "Neutral":
    with st.expander(f"📊 Setup Score breakdown — {score['total']}/100 ({score['label']})"):
        for cat_name, pts, max_pts, note in score["breakdown"]:
            st.markdown(f"**{cat_name}: {pts}/{max_pts}** — {note}")
        st.caption(
            "Heuristic confluence score across candle pattern, chart structure, trend baseline, "
            "volume, and RSI — not backtested, treat as a quick sanity-check, not a signal on its own."
        )

st.divider()

# --------------------------------------------------------------------------
# Multi-bar chart pattern section
# --------------------------------------------------------------------------
st.subheader("📐 Multi-bar Chart Pattern")
mode_label = "intraday only" if intraday_only else "spanning multiple days"
st.caption(f"Scanning {mode_label} — requested lookback: {chart_lookback} bars.")

wh, wl = window["High"].values, window["Low"].values
w_order = max(2, len(window) // 15)
sh_idx, sl_idx = find_swings(wh, wl, w_order)

fig2 = go.Figure()
fig2.add_trace(go.Candlestick(
    x=window.index, open=window["Open"], high=window["High"],
    low=window["Low"], close=window["Close"], name=ticker, showlegend=False
))
if sh_idx:
    fig2.add_trace(go.Scatter(
        x=window.index[sh_idx], y=wh[sh_idx], mode="markers",
        marker=dict(symbol="triangle-down", size=9, color="red"), name="Swing High"
    ))
if sl_idx:
    fig2.add_trace(go.Scatter(
        x=window.index[sl_idx], y=wl[sl_idx], mode="markers",
        marker=dict(symbol="triangle-up", size=9, color="green"), name="Swing Low"
    ))

if cp:
    for label, level in cp.get("lines", []):
        fig2.add_hline(y=level, line_dash="dash", line_color="orange",
                        annotation_text=label, annotation_position="top left")
    if cp.get("entry") is not None:
        fig2.add_hline(y=cp["entry"], line_dash="dot", line_color="gold",
                        annotation_text="Entry", annotation_position="bottom right")
    if cp.get("sl") is not None:
        fig2.add_hline(y=cp["sl"], line_dash="dot", line_color="red",
                        annotation_text="SL", annotation_position="bottom right")
    if cp.get("tp") is not None:
        fig2.add_hline(y=cp["tp"], line_dash="dot", line_color="limegreen",
                        annotation_text="TP", annotation_position="bottom right")

fig2.update_layout(
    height=500, xaxis_rangeslider_visible=False,
    title=f"{ticker} — last {len(window)} bars ending {sel_time}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

if cp:
    color2 = "green" if "Bullish" in cp["bias"] else ("red" if "Bearish" in cp["bias"] else "gray")
    st.markdown(
        f"### {cp['name']}: <span style='color:{color2}'>{cp['bias']}</span>",
        unsafe_allow_html=True,
    )
    st.write(CHART_PATTERN_DESC.get(cp["name"], ""))
    levels = []
    if cp.get("entry") is not None:
        levels.append(f"Entry ≈ {cp['entry']:.2f}")
    if cp.get("sl") is not None:
        levels.append(f"SL ≈ {cp['sl']:.2f}")
    if cp.get("tp") is not None:
        levels.append(f"TP ≈ {cp['tp']:.2f}")
    if levels:
        st.caption(" | ".join(levels))
else:
    st.write("No clear multi-bar chart pattern detected in this window. Try widening or narrowing the lookback slider.")

if intraday_only and len(window) < chart_lookback:
    st.caption(
        f"Only {len(window)} bars available so far today (before {sel_time}) — "
        f"fewer than the requested {chart_lookback}. Pattern detection works on what's available."
    )

st.caption(
    "Multi-bar patterns are geometry-based approximations (swing points + trendline slope), "
    "not exact detections — use them as a prompt to look closer, not a signal on their own."
)
st.caption("All patterns here are probabilistic signals, not guarantees. Confirm with volume, S/R, and higher timeframes.")
