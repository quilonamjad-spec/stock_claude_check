# Chart Reading Machine V0.6

## Next step: improve the machine's eyes

The interface and date/time replay mechanism are intentionally unchanged.

### What changed
The trend reader now gives **price structure** priority over EMA alignment.

It looks for:
- Higher High + Higher Low = bullish structure
- Lower High + Lower Low = bearish structure
- Higher Low forming = bullish pressure
- Lower High forming = bearish pressure
- Mixed swings = range / transition

EMA 9/20 and VWAP remain supporting alignment inputs.

### Why
Previously the machine could call something "Bullish Trend" mainly because:
`price > EMA20` and `EMA9 > EMA20`.

That is useful, but it is not really "reading the chart."

Now the machine first asks what the price is actually doing through its swings.

The score has also been adjusted so structure has more influence while VWAP/EMA remain confirmation rather than the main driver.

### Testing
Use the date + market-time selectors to replay a session:
09:15 → 09:20 → 09:25 → ... → 15:30.

This version is deliberately a small step. We should test the structural reading across several sessions before adding more pattern intelligence.


### V0.7 — Key-level visibility
- Support is now a clearly visible **green dashed line**.
- Resistance is now a clearly visible **red dashed line**.
- Labels use the same colors as their levels.
- Grid lines are softened so the key levels stand out.
- No chart-reading logic was changed; this release is visual only.


# V0.8 — The first three "eyes"

This release combines the first three major intelligence modules.

## 1. Swing Levels / Support & Resistance
- Clusters recent confirmed swing highs/lows into practical levels.
- Estimates support/resistance strength from repeated touches.
- Reports whether price is testing, above, below, or between key levels.
- Keeps the visible support/resistance lines already introduced in V0.7.

## 2. Candle Intelligence
Adds contextual recognition for:
- Bullish/Bearish Engulfing
- Morning Star / Evening Star
- Three White Soldiers / Three Black Crows
- Piercing Line / Dark Cloud Cover
- Doji
- Hammer / Hanging Man
- Inverted Hammer / Shooting Star
- Basic bullish/bearish candles

## 3. Chart Pattern Intelligence
Improves pattern recognition for:
- Double Bottom / Double Top
- Ascending / Descending Triangle
- Symmetrical Triangle

The machine now distinguishes "Developing", "Breakout" and "Breakdown" for double-top/bottom structures where the data supports it.

## Wholesome narrative
The final Machine Reading now combines:
**structure + key levels + candle context + chart pattern + VWAP/EMA + volume + current event**

The goal is not to make a buy/sell call. It is to make the machine explain what it sees and whether the different pieces agree.

The interface and date/time replay controls remain unchanged.


# V1.1 — Fractal S/R Zones + Head & Shoulders

This release keeps V1.0's overall reading approach and adds two connected improvements:

## 1. Fractal-based Support/Resistance Zones
- Uses confirmed rolling local highs/lows (fractals) as S/R candidates.
- Groups nearby fractal prices using a simple adaptive distance threshold.
- Represents support and resistance as zones rather than isolated single-price lines.
- The chart shows subtle green/red shaded zones plus the central support/resistance line.
- Event detection now uses the zone boundaries.

## 2. Head & Shoulders patterns
Added:
- Head & Shoulders
- Inverse Head & Shoulders

The detector uses the same confirmed swing highs/lows used by the S/R reader, so the two modules are connected rather than using separate definitions of chart structure.

The pattern can be reported as:
- Developing
- Breakdown (Head & Shoulders)
- Breakout (Inverse Head & Shoulders)

V1.0 scoring, replay controls, VWAP/EMA/volume context and the overall narrative remain intact.


# V1.2 — Universe Radar + Modular Structure

The individual V1.1 reader rules are preserved. The code is split into:
- `app.py` — Streamlit UI and session/time controls
- `data_engine.py` — Yahoo data, indicators, swings and S/R zones
- `reader_engine.py` — candle, structure, pattern, volume, event and score
- `ui_components.py` — presentation helpers
- `universe_scanner.py` — multi-ticker analysis, filters and ranking

Universe Radar uses the same selected date/time and applies filters for minimum score, minimum six-component consensus, and direction. Results are ranked consensus-first, then by context score.
