"""
Market Scanner — Candlestick Pattern Scanner
---------------------------------------------
Batch-scans a universe of tickers (NSE 500 by default) at a chosen
date/time, running the same candlestick pattern / multi-bar chart pattern /
intraday setup / Setup Score engine as the Home page against every ticker.

Scan once, then filter instantly — filtering never re-fetches data, it
just slices the cached results table in memory.
"""

import json
import os
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd

from lib import (
    fetch_chunk_raw, parse_chunk, chunk_list, add_indicators, smooth_edges,
    detect_pattern, PATTERN_INFO, detect_chart_pattern,
    get_prev_day_close, detect_setups, compute_setup_score,
)

st.set_page_config(page_title="Market Scanner", layout="wide")
st.title("🔍 Market Scanner")
st.caption("Batch-scan a ticker universe, then filter the results instantly — no re-scanning needed.")

# --------------------------------------------------------------------------
# Ticker universe
# --------------------------------------------------------------------------
UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), "..", "nifty500.json")


@st.cache_data
def load_universe():
    try:
        with open(UNIVERSE_PATH) as f:
            data = json.load(f)
        return data.get("tickers", [])
    except FileNotFoundError:
        return []


bundled_universe = load_universe()

with st.sidebar:
    st.header("Universe")
    if bundled_universe:
        universe_choice = st.radio(
            "Ticker list", ["NSE 500 (bundled)", "Custom list"], key="universe_choice"
        )
    else:
        st.warning("nifty500.json not found next to app.py — using custom list only.")
        universe_choice = "Custom list"

    if universe_choice == "NSE 500 (bundled)":
        full_list = bundled_universe
    else:
        custom_raw = st.text_area(
            "Tickers (comma-separated)", value="RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS",
            key="custom_universe", height=100,
        )
        full_list = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]

    st.caption(f"{len(full_list)} tickers available in this list.")
    scan_limit = st.slider(
        "How many to scan (from the top of the list)",
        min_value=min(10, max(len(full_list), 1)),
        max_value=max(len(full_list), 1),
        value=min(100, len(full_list)) if full_list else 1,
        key="scan_limit",
        help="Yahoo's free feed has no true batch endpoint — each ticker is still one HTTP "
             "request under the hood. Start smaller (e.g. 100) to test before scanning all 500.",
    )
    tickers_to_scan = full_list[:scan_limit]

    st.header("Date / Time")
    sel_date = st.date_input("Date", value=date.today(), key="scan_date")

    def nse_time_options():
        t = datetime.strptime("09:15", "%H:%M")
        end = datetime.strptime("15:30", "%H:%M")
        out = []
        while t <= end:
            out.append(t.strftime("%H:%M"))
            t += timedelta(minutes=5)
        return out

    time_options = nse_time_options()
    now = datetime.now()
    if sel_date == date.today() and "09:15" <= now.strftime("%H:%M") <= "15:30":
        rounded = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        default_time = rounded.strftime("%H:%M")
        default_idx = time_options.index(default_time) if default_time in time_options else len(time_options) - 1
    else:
        default_idx = len(time_options) - 1
    sel_time = st.selectbox("Time (NSE session, 5-min bars)", time_options, index=default_idx, key="scan_time")

    st.header("Data settings")
    period_days = st.slider("Days of history to fetch", 1, 20, 5, key="scan_period_days",
                             help="Needs enough history behind the selected date for the multi-bar "
                                  "pattern lookback to have bars to work with.")
    smooth = st.checkbox("Smooth first/last 3 candles (edge noise)", value=True, key="scan_smooth")
    chart_lookback = st.slider("Multi-bar pattern lookback (bars)", 20, 150, 60, key="scan_lookback")

    with st.expander("⚙️ Advanced (tune if scans disconnect)"):
        batch_size = st.slider(
            "Chunk size (tickers per request)", 5, 50, 15, key="scan_batch_size",
            help="Each chunk is one yf.download() call for that many tickers. Smaller chunks "
                 "mean more requests but each one is lighter — try smaller if scans are failing.",
        )
        st.caption("Each chunk is cached for 10 minutes, and results are saved as each chunk "
                    "completes — so a failure partway through won't lose everything already "
                    "fetched, and re-running soon after skips chunks that already succeeded.")

    scan_clicked = st.button("🔍 Scan Market", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Run the scan
# --------------------------------------------------------------------------
if scan_clicked:
    if not tickers_to_scan:
        st.warning("No tickers to scan — check your universe selection.")
    else:
        st.session_state["scan_rows"] = []
        progress_bar = st.progress(0.0)
        status = st.empty()
        warn_area = st.empty()

        def process_ticker(ticker, raw):
            """Run the full detector pipeline for one ticker. Returns a result
            row dict, or None if there's no data at the selected date/time or
            detection fails for any reason (never lets one bad ticker abort
            the scan)."""
            try:
                df = smooth_edges(raw, n=3) if smooth else raw
                df = add_indicators(df)

                day_df = df[df.index.date == sel_date]
                if day_df.empty:
                    return None

                times = [t.strftime("%H:%M") for t in day_df.index]
                candidates = [i for i, tm in enumerate(times) if tm <= sel_time]
                if not candidates:
                    return None
                idx = candidates[-1]
                sel_ts = day_df.index[idx]
                row = day_df.iloc[idx]

                pattern = detect_pattern(day_df.reset_index(drop=True), idx)
                bias, _ = PATTERN_INFO.get(pattern, ("Neutral", ""))

                prev_close = get_prev_day_close(df, sel_date)
                setups = detect_setups(day_df, idx, prev_close)

                window = day_df[day_df.index <= sel_ts].tail(chart_lookback)
                cp = detect_chart_pattern(window.reset_index(drop=True)) if len(window) >= 15 else None

                score = compute_setup_score(pattern, bias, cp, row, setups)
                bd = {b[0]: b[1] for b in score["breakdown"]}

                vol_ma = row.get("VolMA20", float("nan"))
                vol_ratio = row["Volume"] / vol_ma if pd.notna(vol_ma) and vol_ma > 0 else None

                return {
                    "Ticker": ticker.replace(".NS", ""),
                    "Close": round(row["Close"], 2),
                    "Anchor": score["anchor"],
                    "Total Score": score["total"],
                    "Score Label": score["label"] if score["anchor"] != "Neutral" else "No directional setup",
                    "Candle Pattern": pattern,
                    "Candle Score": bd.get("Candle Strength", 0),
                    "Chart Pattern": cp["name"] if cp else "None",
                    "Chart Score": bd.get("Multi-bar Structure", 0),
                    "Setup Count": len(setups),
                    "Setups": ", ".join(s[0] for s in setups) if setups else "",
                    "RSI": round(row["RSI"], 0),
                    "Vol x Avg": round(vol_ratio, 1) if vol_ratio is not None else None,
                    "Time": sel_time,
                }
            except Exception:
                return None

        chunks = list(chunk_list(tickers_to_scan, batch_size))
        total = len(tickers_to_scan)
        done = 0

        for ch in chunks:
            try:
                raw_data = fetch_chunk_raw(tuple(ch), period_days)
                batch_raw = parse_chunk(raw_data, ch)
                note = None
            except Exception as e:
                batch_raw = {}
                note = str(e)

            # Score whatever this chunk fetched immediately, and persist to
            # session_state right away — so if a later chunk fails or the
            # connection drops, everything up to this point is not lost.
            for t, raw in batch_raw.items():
                row = process_ticker(t, raw)
                if row:
                    st.session_state["scan_rows"].append(row)

            done += len(ch)
            scored_so_far = len(st.session_state["scan_rows"])
            if st.session_state["scan_rows"]:
                st.session_state["scan_results"] = pd.DataFrame(st.session_state["scan_rows"])
                st.session_state["scan_meta"] = {
                    "date": str(sel_date), "time": sel_time, "requested": total,
                    "fetched": done, "scored": scored_so_far, "partial": done < total,
                }

            if note:
                warn_area.warning(f"Chunk issue ({', '.join(ch[:3])}{'…' if len(ch) > 3 else ''}): {note}")
            status.text(f"Fetched {done}/{total} tickers, {scored_so_far} scored so far…")
            progress_bar.progress(done / total)

        progress_bar.empty()
        status.empty()

        if st.session_state["scan_rows"]:
            meta = st.session_state.get("scan_meta", {})
            meta["partial"] = False
            st.session_state["scan_meta"] = meta
            st.success(f"Scanned {len(st.session_state['scan_rows'])} tickers with data at {sel_time} on {sel_date}.")
        else:
            st.error("No tickers produced results — check the date (market holiday?) and time selected.")

# --------------------------------------------------------------------------
# Results + filters (operate on cached results, no re-scan needed)
# --------------------------------------------------------------------------
if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    meta = st.session_state["scan_meta"]

    st.divider()
    st.subheader(f"Results — {meta['date']} at {meta['time']}")
    if meta.get("partial"):
        st.warning(
            f"⚠️ Scan didn't finish — only {meta['fetched']} of {meta['requested']} tickers were "
            f"reached before it stopped. Results below are from what completed; rerun the scan "
            f"(maybe with a smaller batch size in Advanced) to cover the rest."
        )
    st.caption(f"{meta['scored']} of {meta['requested']} requested tickers produced a result "
               f"({meta['fetched']} had data fetched at all).")

    st.markdown("#### Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        direction_filter = st.multiselect(
            "Direction", ["Bullish", "Bearish", "Neutral"], default=["Bullish", "Bearish"],
            key="filter_direction",
        )
    with f2:
        min_candle = st.slider("Min Candle Score", 0, 25, 0, key="filter_candle")
    with f3:
        min_chart = st.slider("Min Multi-bar Score", 0, 25, 0, key="filter_chart")
    with f4:
        min_setups = st.slider("Min Setup Count", 0, int(results["Setup Count"].max() or 0), 0, key="filter_setups")

    all_labels = ["High-Conviction", "Solid", "Mixed", "Weak", "Very Weak", "No directional setup"]
    score_filter = st.multiselect(
        "Score category", options=all_labels, default=[l for l in all_labels if l != "No directional setup"],
        key="filter_score_label",
    )

    filtered = results[
        results["Anchor"].isin(direction_filter)
        & (results["Candle Score"] >= min_candle)
        & (results["Chart Score"] >= min_chart)
        & (results["Setup Count"] >= min_setups)
        & results["Score Label"].apply(lambda lbl: any(cat in lbl for cat in score_filter))
    ].sort_values("Total Score", ascending=False).reset_index(drop=True)

    st.markdown(f"#### {len(filtered)} of {len(results)} tickers pass the current filters")
    st.dataframe(filtered, use_container_width=True, height=min(600, 60 + 35 * max(len(filtered), 1)))

    if not filtered.empty:
        st.markdown("#### Jump to a ticker's full chart")
        jc1, jc2 = st.columns([3, 1])
        with jc1:
            jump_ticker = st.selectbox("Ticker", filtered["Ticker"].tolist(), key="jump_ticker")
        with jc2:
            st.write("")
            st.write("")
            if st.button("Open in Home →", use_container_width=True):
                st.session_state["tickers_raw"] = f"{jump_ticker}.NS"
                st.session_state["follow_latest"] = False
                st.switch_page("app.py")
else:
    st.info("Set your scan parameters in the sidebar and click **🔍 Scan Market** to get started.")
