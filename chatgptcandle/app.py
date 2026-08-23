import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, time, timedelta

from data_engine import get_data, add_indicators, level_detail, levels
from reader_engine import candle, structure, structure_detail, pattern, volume, event, score
from ui_components import metric, box
from universe_scanner import scan_universe, filter_and_rank

st.set_page_config(page_title="Chart Reading Machine", layout="wide")

# Apply radar -> single-chart navigation BEFORE any keyed widgets are created.
if "pending_single_ticker" in st.session_state:
    st.session_state["single_ticker"] = st.session_state.pop("pending_single_ticker")
    st.session_state["view_mode"] = "Single Chart"
if "pending_single_date" in st.session_state:
    st.session_state["pending_date_value"] = st.session_state.pop("pending_single_date")
if "pending_single_time" in st.session_state:
    st.session_state["pending_time_value"] = st.session_state.pop("pending_single_time")

st.title("🧠 Chart Reading Machine")
st.caption("V1.2 — selected-session chart context reader; not a buy/sell engine.")

with st.sidebar:
    mode=st.radio("View", ["Single Chart", "Universe Radar"], key="view_mode", index=0)
    ticker=st.text_input("Yahoo ticker", value=st.session_state.get("single_ticker", "RELIANCE.NS"), key="single_ticker").upper().strip()

    st.markdown("### ⏱ Candle Timeframe")
    interval=st.selectbox(
        "Candle interval",
        ["5m","15m","30m","1h","1d"],
        index=0,
        help="5m = one candle every 5 minutes. For the intraday view, the selected session is shown across the trading day."
    )

    refresh_live = st.button(
        "🔄 Refresh Live",
        use_container_width=True,
        help="Reload Yahoo Finance data and move the current-session chart to the latest available candle."
    )

    st.markdown("---")
    st.markdown("### 📅 Test Session")

    st.caption("Select a date and a market time. The machine will read the chart only up to that point.")

    st.markdown("**Trading Date**")

all_df=get_data(ticker,interval)
if all_df.empty:
    st.error("No Yahoo Finance data returned. Check ticker/timeframe.")
    st.stop()

if refresh_live:
    st.cache_data.clear() if hasattr(st, "cache_data") else None
    st.rerun()

all_df=add_indicators(all_df)
dates=sorted(set(all_df.index.date))

# Date selector.
date_default = st.session_state.pop("pending_date_value", dates[-1])
if date_default not in dates:
    date_default = dates[-1]
selected_date=st.sidebar.date_input(
    "Select date",
    value=date_default,
    min_value=dates[0],
    max_value=dates[-1],
    key="selected_date"
)

# Build explicit NSE-style 5-minute market slots.
# 09:15, 09:20 ... 15:25, 15:30.
def market_slots(step_minutes=5):
    start = datetime.combine(date.today(), time(9,15))
    end = datetime.combine(date.today(), time(15,30))
    slots=[]
    cur=start
    while cur <= end:
        slots.append(cur.time())
        cur += timedelta(minutes=step_minutes)
    return slots

if interval == "5m":
    time_options = market_slots(5)
elif interval == "15m":
    time_options = market_slots(15)
elif interval == "30m":
    time_options = market_slots(30)
elif interval == "1h":
    time_options = [time(9,15), time(10,15), time(11,15), time(12,15),
                    time(13,15), time(14,15), time(15,15)]
else:
    time_options = [time(15,30)]

# Default to the latest available market time on the selected date.
selected_day_data = all_df[all_df.index.date == selected_date]
available_intraday = selected_day_data.index

if len(available_intraday):
    latest_clock = available_intraday[-1].time().replace(second=0, microsecond=0)
    # Choose the last selector slot not later than the latest available candle.
    valid_slots = [t for t in time_options if t <= latest_clock]
    default_time = valid_slots[-1] if valid_slots else time_options[0]
else:
    default_time = time_options[0]

time_labels = [t.strftime("%H:%M") for t in time_options]
default_index = time_labels.index(default_time.strftime("%H:%M"))

pending_time = st.session_state.pop("pending_time_value", None)
if pending_time:
    pending_label = pending_time.strftime("%H:%M") if hasattr(pending_time, "strftime") else str(pending_time)
    if pending_label in time_labels:
        default_index = time_labels.index(pending_label)
selected_time_label = st.sidebar.selectbox(
    "Select market time",
    time_labels,
    index=default_index,
    key="selected_time_label",
    help="For 5m mode this gives 09:15, 09:20, 09:25 ... through the trading session."
)
selected_time = datetime.strptime(selected_time_label, "%H:%M").time()

st.sidebar.caption(
    f"⏱ Reading point: **{selected_time.strftime('%H:%M')}**"
)

# Machine context includes all data up to the selected date/time.
# Yahoo Finance intraday indexes are timezone-aware.
# Build the test-point timestamp in the SAME timezone as the data index.
selected_timestamp = pd.Timestamp(
    datetime.combine(selected_date, selected_time)
)

if getattr(all_df.index, "tz", None) is not None:
    selected_timestamp = selected_timestamp.tz_localize(all_df.index.tz)

context=all_df[all_df.index <= selected_timestamp].copy()

if mode == "Universe Radar":
    st.markdown("## 🌐 Universe Radar")
    st.caption("Applies the same chart-reading eyes to a ticker universe at the selected date/time. It is a chart observer, not a buy/sell engine.")

    # Universe source: either the user's own ticker list or the supplied NSE 500 JSON.
    universe_source = st.selectbox(
        "Universe source",
        ["My Tickers", "NSE 500"],
        key="universe_source",
        help="Choose a small custom universe for testing, or load the supplied nifty500.json directly."
    )

    if universe_source == "My Tickers":
        universe_text=st.text_area(
            "Universe tickers",
            "RELIANCE.NS, INFY.NS, TCS.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, LT.NS, ITC.NS, AXISBANK.NS, MARUTI.NS",
            key="custom_universe_text"
        )
        tickers=[x.strip() for x in universe_text.replace("\n",",").replace(" ",",").split(",") if x.strip()]
        st.caption(f"Custom universe: **{len(tickers)} tickers**")
    else:
        nse500_path = Path(__file__).with_name("nifty500.json")
        try:
            with nse500_path.open("r", encoding="utf-8") as f:
                nse500_data=json.load(f)
            tickers=[x.strip().upper() for x in nse500_data.get("tickers", []) if str(x).strip()]
            st.caption(f"NSE 500 universe loaded from JSON: **{len(tickers)} tickers**")
        except Exception as exc:
            st.error(f"Could not load nifty500.json: {exc}")
            tickers=[]

    f1,f2,f3=st.columns(3)
    with f1: min_score=st.slider("Minimum context score",0,100,0,5, key="radar_min_score")
    with f2: min_consensus=st.selectbox("Minimum consensus",[0,3,4,5,6],index=0, key="radar_min_consensus")
    with f3: direction_filter=st.selectbox("Direction",["All","Bullish","Bearish"],index=0, key="radar_direction")

    # Scan is deliberately separate from filtering. Once scanned, filters operate
    # on the stored dataframe and never call Yahoo again.
    batch_size = 25 if len(tickers) > 25 else max(1, len(tickers))
    st.caption(f"Batch scanning: **{batch_size} tickers per Yahoo request**")

    if st.button("🔎 Scan Universe",type="primary",use_container_width=True):
        progress = st.progress(0, text="Preparing batch scan…")
        status = st.empty()

        def _progress(done, total):
            pct = 1.0 if total == 0 else done / total
            progress.progress(pct, text=f"Reading batch: {done}/{total} charts…")
            status.caption(f"Completed {done} of {total} tickers")

        raw = scan_universe(
            tickers, interval, selected_timestamp,
            batch_size=batch_size, progress_callback=_progress
        )
        progress.progress(1.0, text="Batch scan complete")
        st.session_state["radar_raw"] = raw
        st.session_state["radar_timestamp"] = selected_timestamp
        st.session_state["radar_universe_count"] = len(tickers)

    raw=st.session_state.get("radar_raw")
    if raw is None:
        st.info("Set the date/time and click **Scan Universe**. The scan will then remain available while you change filters.")
        st.stop()

    radar=filter_and_rank(raw,min_score,min_consensus,direction_filter)
    if radar.empty:
        st.warning("No charts passed the selected filters.")
        st.stop()

    cols=["Rank","Ticker","Score","Direction","Consensus","Structure","Event","Pattern","Candle","Latest"]
    st.dataframe(radar[cols],use_container_width=True,hide_index=True)
    st.caption("Ranking is consensus-first, then context score. Filters act on the completed scan and do not re-scan Yahoo data.")

    st.markdown("### 🔍 Strongest chart agreements")
    for _,row in radar.head(5).iterrows():
        metric(f"#{int(row['Rank'])} {row['Ticker']}",f"{row['Score']}/100 • {row['Consensus']}",row["Direction"])
        st.write(f"{row['Structure']} • {row['Event']} • {row['Pattern']} • {row['Candle']}")

    # Candidate bridge: do NOT write to the selectbox's session-state key after
    # instantiation. The button stores pending navigation, then reruns; the top
    # of the script consumes it before the single-chart widgets are created.
    st.markdown("### 🎯 Detailed analysis")
    candidates=radar["Ticker"].tolist()
    selected_candidate=st.selectbox(
        "Select a shortlisted candidate",
        candidates,
        key="radar_candidate"
    )
    candidate_row=radar[radar["Ticker"]==selected_candidate].iloc[0]
    st.caption(f"{selected_candidate} • {candidate_row['Direction']} • {candidate_row['Consensus']} consensus • {candidate_row['Score']}/100 • {candidate_row['Pattern']}")

    if st.button("📊 Open Detailed Analysis",type="primary",use_container_width=True):
        radar_ts=st.session_state.get("radar_timestamp", selected_timestamp)
        # Preserve the exact radar timestamp; the single chart will use the
        # nearest valid selector slot for the selected interval.
        st.session_state["pending_single_ticker"] = selected_candidate
        st.session_state["pending_single_date"] = radar_ts.date()
        st.session_state["pending_single_time"] = radar_ts.time().replace(second=0,microsecond=0)
        st.rerun()
    st.stop()

# Visible chart is ONLY the selected date, and ONLY up to the selected time.
chart=context[context.index.date==selected_date].copy()

if chart.empty:
    st.warning("No market data is available up to the selected date/time.")
    st.stop()

# Recalculate selected-session VWAP for the visible portion.
chart=add_indicators(chart)

# Live/current-session status.
is_latest_session = selected_date == dates[-1]
latest_candle_time = chart.index[-1] if not chart.empty else None

if is_latest_session and interval != "1d":
    st.caption(
        f"🟢 Current-session test point • {interval} candles • "
        f"reading at **{selected_time.strftime('%H:%M')}** • "
        f"latest available candle: "
        f"{latest_candle_time.strftime('%H:%M:%S') if latest_candle_time is not None else '—'} • "
        f"Click **Refresh Live** to update."
    )
else:
    st.caption(
        f"📅 Historical test point • {interval} candles • "
        f"{selected_date.strftime('%d %b %Y')} at **{selected_time.strftime('%H:%M')}**"
    )

# The last candle of the selected session drives the reading.
analysis=context
s,r,z=levels(analysis)
ld=level_detail(analysis)
tn,tb,ts=structure(analysis)
pn,stage,pb,ps=pattern(analysis)
cn,cb,cs=candle(analysis)
vn,vb,vs=volume(analysis)
ev,eb=event(analysis,s,r,z)
sc=score(analysis,ts,ps)

st.subheader(f"{ticker} • {interval} • {selected_date.strftime('%d %b %Y')} • {selected_time.strftime('%H:%M')}")

h1,h2,h3,h4=st.columns(4)
with h1: metric("Chart Context",f"{sc}/100",tb)
with h2: metric("Market State",tn,tb)
with h3: metric("Current Event",ev,eb)
with h4: metric("Pattern",pn,pb)

# Selected-day chart only.
fig=go.Figure(go.Candlestick(x=chart.index,open=chart.Open,high=chart.High,
                             low=chart.Low,close=chart.Close,name="Price"))
for col,name in [("EMA9","EMA 9"),("EMA20","EMA 20"),("VWAP","VWAP")]:
    fig.add_trace(go.Scatter(x=chart.index,y=chart[col],name=name,mode="lines"))
# Make key levels visually distinct from the dark chart/grid.
fig.add_hrect(
    y0=ld["support_low"], y1=ld["support_high"],
    fillcolor="rgba(74,222,128,0.10)", line_width=0,
    annotation_text="Support zone", annotation_position="bottom right",
    annotation_font_color="#4ADE80"
)
fig.add_hrect(
    y0=ld["resistance_low"], y1=ld["resistance_high"],
    fillcolor="rgba(255,107,107,0.10)", line_width=0,
    annotation_text="Resistance zone", annotation_position="top right",
    annotation_font_color="#FF6B6B"
)
fig.add_hline(
    y=s,
    annotation_text="Support",
    line_color="#4ADE80",
    line_width=2,
    line_dash="dash",
    annotation_position="bottom right",
    annotation_font_color="#4ADE80"
)
fig.add_hline(
    y=r,
    annotation_text="Resistance",
    line_color="#FF6B6B",
    line_width=2,
    line_dash="dash",
    annotation_position="top right",
    annotation_font_color="#FF6B6B"
)
fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10,r=10,t=20,b=10),
    xaxis=dict(
        title="Time",
        tickformat="%H:%M",
        dtick=60*60*1000 if interval in ["5m","15m","30m"] else None,
        showgrid=True,
        gridcolor="rgba(130,130,130,0.16)"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(130,130,130,0.16)"
    )
)
st.plotly_chart(fig,use_container_width=True)

# Compact boxes directly under live chart.
x,y,w=st.columns(3)
with x:
    box("🕯 Candle Context",
        [f"<b>{cn}</b>",f"{'Bullish' if cb=='Bullish' else 'Bearish' if cb=='Bearish' else 'Neutral'} behaviour",
         f"Significance: <b>{cs}</b>"],cb)
with y:
    box("📐 Pattern Recognition",
        [f"<b>{pn}</b>",stage,f"Pattern strength: <b>{ps}/100</b>"],pb)
with w:
    ld=level_detail(analysis)
    box("🎯 Key Levels",
        [f"Resistance: <b>{ld['resistance_low']:.2f}–{ld['resistance_high']:.2f}</b> • {ld['resistance_strength']}",
         f"Support: <b>{ld['support_low']:.2f}–{ld['support_high']:.2f}</b> • {ld['support_strength']}",
         f"Current: <b>{float(analysis.Close.iloc[-1]):.2f}</b>"],eb)

# Alignment is explicitly green/red where directional; neutral remains grey.
st.markdown("### 🔗 Alignment")
a,b,c,d=st.columns(4)
last=analysis.iloc[-1]
price=float(last.Close); vwap=float(last.VWAP); ema9=float(last.EMA9); ema20=float(last.EMA20)

with a: metric("Price vs VWAP","Above VWAP" if price>vwap else "Below VWAP",
               "Bullish" if price>vwap else "Bearish")
with b: metric("EMA Alignment","EMA 9 > EMA 20" if ema9>ema20 else "EMA 9 < EMA 20",
               "Bullish" if ema9>ema20 else "Bearish")
with c: metric("Candle Alignment",cn,cb)
with d: metric("Trend Alignment",tn,tb)

q1,q2,q3=st.columns(3)
with q1:
    st.markdown("### 📈 Structure")
    metric("Trend",tn,tb)
    st.write(f"EMA 9: {ema9:.2f}")
    st.write(f"EMA 20: {ema20:.2f}")
with q2:
    st.markdown("### VWAP / Volume")
    metric("VWAP Position","Above VWAP" if price>vwap else "Below VWAP",
           "Bullish" if price>vwap else "Bearish")
    metric("Volume",vn,vb)
with q3:
    st.markdown("### 👀 Machine Watching")
    sd, sdb = structure_detail(analysis)
    metric("Structure", sd, sdb)
    st.write(f"• {ev}")
    st.write(f"• {cn} ({cs} significance)")
    st.write(f"• {pn} — {stage}")
    st.write(f"• S/R zones: {ld['support_strength']} support / {ld['resistance_strength']} resistance")
    st.write("• VWAP / EMA relationship")

st.markdown("---")
st.markdown("### 🧠 Machine Reading")
ld = level_detail(analysis)
location = ld["location"]
sd, sdb = structure_detail(analysis)

# A compact narrative assembled from the three major reading modules:
# 1) levels/structure, 2) candle context, 3) chart pattern.
level_sentence = (
    f"Support zone is **{ld['support_low']:.2f}–{ld['support_high']:.2f} "
    f"({ld['support_strength'].lower()}, {ld['support_touches']} touch"
    f"{'es' if ld['support_touches'] != 1 else ''})** and resistance zone is "
    f"**{ld['resistance_low']:.2f}–{ld['resistance_high']:.2f} "
    f"({ld['resistance_strength'].lower()}, {ld['resistance_touches']} touch"
    f"{'es' if ld['resistance_touches'] != 1 else ''})**."
)

candle_sentence = f"The latest candle reads **{cn.lower()}** with **{cs.lower()} significance**."
pattern_sentence = (
    f"The chart pattern is **{pn.lower()}** ({stage.lower()})"
    if pn != "No clear pattern"
    else "No strong chart pattern is confirmed yet"
)

st.info(
    f"**Chart reading:** The market is showing **{tn.lower()}**, with price **{location.lower()}**. "
    f"The confirmed swing structure is **{sd.lower()}**. {level_sentence} "
    f"{candle_sentence} {pattern_sentence}. "
    f"VWAP, EMA alignment and volume are used as confirmation. "
    f"Current event: **{ev.lower()}**."
)
