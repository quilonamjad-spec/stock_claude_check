"""Universe scanning/ranking layer.

The scanner deliberately reuses the existing chart-reader rules.  Yahoo data
is fetched in batches so a large universe (such as NSE 500) does not make 500
sequential network requests.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from data_engine import add_indicators, get_data, level_detail
from reader_engine import candle, event, pattern, score, structure, volume


def _context_for_time(df, selected_timestamp):
    # Yahoo's timezone and the app selector can differ slightly.  Normalize the
    # comparison when necessary, without changing the actual candle timestamps.
    ts = selected_timestamp
    try:
        if getattr(df.index, "tz", None) is not None and getattr(ts, "tzinfo", None) is None:
            ts = pd.Timestamp(ts).tz_localize(df.index.tz)
        elif getattr(df.index, "tz", None) is None and getattr(ts, "tzinfo", None) is not None:
            ts = pd.Timestamp(ts).tz_localize(None)
        elif getattr(df.index, "tz", None) is not None and getattr(ts, "tzinfo", None) is not None:
            ts = pd.Timestamp(ts).tz_convert(df.index.tz)
    except Exception:
        ts = selected_timestamp
    return df[df.index <= ts].copy()


def _votes(df):
    last = df.iloc[-1]
    price = float(last.Close); vwap = float(last.VWAP)
    ema9 = float(last.EMA9); ema20 = float(last.EMA20)
    tn, tb, ts = structure(df); pn, stage, pb, ps = pattern(df); cn, cb, cs = candle(df)
    ld = level_detail(df); ev, eb = event(df, ld["support"], ld["resistance"], ld["zone"])
    signals = {
        "Structure": tb,
        "VWAP": "Bullish" if price > vwap else "Bearish",
        "EMA": "Bullish" if ema9 > ema20 else "Bearish",
        "Candle": cb,
        "Pattern": pb,
        "Event": eb,
    }
    bullish = sum(v == "Bullish" for v in signals.values())
    bearish = sum(v == "Bearish" for v in signals.values())
    neutral = len(signals) - bullish - bearish
    consensus = max(bullish, bearish)
    direction = "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Neutral"
    return signals, bullish, bearish, neutral, consensus, direction


def analyze_ticker(ticker, interval, selected_timestamp, df=None):
    if df is None:
        df = get_data(ticker, interval)
    if df.empty:
        return None
    df = add_indicators(df)
    context = _context_for_time(df, selected_timestamp)
    if len(context) < 25:
        return None
    ld = level_detail(context)
    tn, tb, ts = structure(context); pn, stage, pb, ps = pattern(context)
    cn, cb, cs = candle(context); vn, vb, vs = volume(context)
    ev, eb = event(context, ld["support"], ld["resistance"], ld["zone"])
    sc = score(context, ts, ps)
    signals, bullish, bearish, neutral, consensus, direction = _votes(context)
    return {
        "Ticker": ticker, "Score": sc, "Direction": direction, "Consensus": f"{consensus}/6",
        "Bullish Votes": bullish, "Bearish Votes": bearish, "Neutral": neutral,
        "Structure": tn, "Event": ev, "Pattern": pn, "Pattern Stage": stage,
        "Candle": cn, "Volume": vn, "S/R Location": ld["location"],
        "Support": ld["support"], "Resistance": ld["resistance"],
        "Latest": context.index[-1].strftime("%H:%M"), "Signals": signals,
    }


def _clean_tickers(tickers):
    cleaned = []
    seen = set()
    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        if not ticker:
            continue
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker += ".NS"
        if ticker not in seen:
            cleaned.append(ticker); seen.add(ticker)
    return cleaned


def _period_for_interval(interval):
    return "60d" if interval in ["1m", "2m", "5m", "15m", "30m"] else "6mo"


def _extract_ticker_frame(batch_df, ticker):
    """Extract one ticker's OHLCV frame from a yfinance multi-ticker result."""
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()

    try:
        if not isinstance(batch_df.columns, pd.MultiIndex):
            # Only possible for a one-ticker batch.
            frame = batch_df.copy()
        else:
            level0 = list(batch_df.columns.get_level_values(0))
            level1 = list(batch_df.columns.get_level_values(1))
            if ticker in level0:
                frame = batch_df[ticker].copy()
            elif ticker in level1:
                frame = batch_df.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in frame.columns for c in required):
        return pd.DataFrame()
    return frame[required].dropna()


def _download_batch(tickers, interval):
    """Fetch one batch concurrently through yfinance."""
    try:
        data = yf.download(
            tickers=tickers,
            period=_period_for_interval(interval),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return {}

    result = {}
    for ticker in tickers:
        frame = _extract_ticker_frame(data, ticker)
        if not frame.empty:
            result[ticker] = frame
    return result


def scan_universe(tickers, interval, selected_timestamp, min_score=0,
                  min_consensus=0, direction_filter="All", batch_size=25,
                  progress_callback=None):
    """Scan a universe in network batches, then rank the completed results.

    ``progress_callback(done, total)`` is optional and is used by Streamlit to
    show genuine batch progress without rescanning when filters change.
    """
    tickers = _clean_tickers(tickers)
    rows = []
    total = len(tickers)

    for start in range(0, total, max(1, int(batch_size))):
        batch = tickers[start:start + max(1, int(batch_size))]
        frames = _download_batch(batch, interval)

        for ticker in batch:
            try:
                row = analyze_ticker(ticker, interval, selected_timestamp, frames.get(ticker))
                if row is not None:
                    rows.append(row)
            except Exception as exc:
                rows.append({"Ticker": ticker, "Error": str(exc)})

        if progress_callback:
            progress_callback(min(start + len(batch), total), total)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "Score" in df.columns:
        df["ConsensusNum"] = df["Consensus"].str.split("/").str[0].astype(int)
        df = df.sort_values(["ConsensusNum", "Score"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def filter_and_rank(radar, min_score=0, min_consensus=0, direction_filter="All"):
    """Filter and rank an already-scanned radar without re-reading Yahoo data."""
    if radar is None or radar.empty:
        return pd.DataFrame()
    df = radar.copy()
    if min_score:
        df = df[df["Score"] >= min_score]
    if min_consensus:
        df = df[df["ConsensusNum"] >= min_consensus]
    if direction_filter != "All":
        df = df[df["Direction"] == direction_filter]
    if df.empty:
        return df
    df = df.sort_values(["ConsensusNum", "Score"], ascending=[False, False]).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    cols = ["Rank"] + [c for c in df.columns if c != "Rank"]
    return df[cols]
