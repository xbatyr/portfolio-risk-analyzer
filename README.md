# Advanced VaR & Expected Shortfall Engine

A Streamlit dashboard that measures the market risk of a single stock from its own price history, and shows the upside side of the distribution too. Point it at a ticker, set your position size, and it tells you how much you can lose on a normal bad day, how much on a crisis day, and the mirror image for gains.

Everything runs on plain NumPy. No black box models, no assumed bell curve. The numbers come straight from the real return distribution, so they are easy to explain and easy to defend.

## What it does

- **1-Day VaR** and **Expected Shortfall** for the downside, computed by historical simulation.
- **Upside Potential** and **Expected Upside**, the same idea applied to the right tail.
- Headline stats: current price, expected annual return, annual volatility.
- An interactive Plotly histogram with the risk tail shaded red and the profit tail shaded green.
- A plain-language "what it means" panel that turns the percentages into dollars.
- Full English / Russian toggle. Flip the language and nothing reloads, the data stays cached.

![Dashboard](docs/screenshot.png)

## How the math works

The whole pipeline is four steps, all vectorized.

**1. Log returns.** For each day we take `r = ln(price_today / price_yesterday)`. Log returns add up cleanly over time and their distribution sits closer to normal, which makes them the standard choice for this kind of work.

**2. VaR is a quantile.** Value at Risk at 95 percent confidence is just the 5th percentile of the return distribution. `np.percentile` sorts the array in `O(n log n)` and reads off the element at position `alpha * n`. Nothing is assumed about the shape of the curve, we use the real data.

**3. Expected Shortfall is a tail average.** VaR tells you the threshold, but not how bad things get past it. So we take a boolean mask of every day worse than VaR and average those days. That is the mean loss once VaR is breached. Same trick on the right side gives Upside Potential (the 95th percentile) and Expected Upside (the average of the best days).

**4. Back to dollars.** A log return maps to a real amount exactly, no small-number approximation:

```
loss = V * (1 - e^r)      for the downside
gain = V * (e^r - 1)      for the upside
```

The sign works out on its own: a negative return gives a positive loss, a positive return gives a positive gain.

## How the code is organized

Everything lives in one file, split into clear layers:

| Part | What it does |
|------|--------------|
| `load_price_data` | Pulls Adjusted Close from yfinance. Cached on `(ticker, years)`, so moving the risk sliders or switching language never hits the network again. |
| `compute_log_returns` | Turns prices into a NumPy array of daily log returns. |
| `compute_risk_metrics` | Both tails: VaR, ES, Upside, Expected Upside, plus the dollar fractions. |
| `build_distribution_chart` | The Plotly histogram with the red and green tail zones. |
| `render_sidebar` / `main` | Inputs and layout. All visible text comes from the `TEXTS` dictionary. |

The language switch is the reason the app feels instant. Data caching is keyed only on the ticker and the history window, so the language is a pure display concern. Flipping EN and RU is a rerun over data that is already in memory.

## Tech stack

Streamlit for the UI, yfinance for market data, NumPy and pandas for the math, Plotly for the chart.

## Notes

This is a demo and a teaching tool, not investment advice. It computes a one day horizon and reads history at face value, so it inherits every bias in that history. Past returns do not predict future returns.
