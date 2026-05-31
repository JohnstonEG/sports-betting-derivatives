# Shiny (Python) dashboard

A Shiny for Python dashboard that runs on top of the same C# / .NET risk
engine as the [Blazor dashboard](../src/RiskEngine.Dashboard/). The Blazor UI
proves the C# side; this Shiny UI proves the Python side. Both consume the
same JSON contract emitted by `RiskEngine.Cli`, so the engine has exactly
one source of truth and the dashboards are thin presentation layers.

> *Shiny for Python is Posit's Python port of R Shiny — same reactive idea,
> pure Python, no R involved.*

## Architecture

```
  data/portfolio.json + data/calibration.json
                │
                ▼
  ┌─────────────────────────────┐
  │   RiskEngine.Cli  (C# / .NET) │   <-- the engine you already built
  └─────────────────────────────┘
                │
                ▼ writes risk-report.json
  ┌─────────────────────────────┐
  │   engine.py  (subprocess +  │
  │   JSON load)                │
  └─────────────────────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │   app.py  (Shiny Core UI)   │   Plotly via shinywidgets
  └─────────────────────────────┘
```

`engine.py` prefers the built executable (`bin/Debug/net10.0/RiskEngine.Cli.exe`
or the Release equivalent); if that's missing it falls back to `dotnet run`,
which works but adds 2–3 seconds of startup overhead per interaction.

## Build & run

```bash
# 1. Build the C# engine once, so the Shiny app can run it fast
cd risk-engine
dotnet build

# 2. Install the Python packages (Python 3.10+)
cd dashboard-shiny
pip install -r requirements.txt

# 3. Launch the dashboard
shiny run --reload app.py
```

Open the URL Shiny prints (usually `http://127.0.0.1:8000`). Use the
**Pricing model** and **Paths** dropdowns on the Overview tab to recompute;
each change re-runs the C# engine in the background and re-renders the
Plotly charts.

## What it shows

Three tabs that mirror the Blazor dashboard scope:

| Tab | Content |
|---|---|
| **Overview** | Headline risk cards (Mean PnL, VaR, CVaR, P(loss), Sharpe-like), live model toggle, P&L distribution histogram (central 98%, clipped for readability) |
| **Scenarios** | Per-regime risk table, regime-switching mixture, stress test battery (table + horizontal-bar comparison), volatility scenario sweep (line chart) |
| **Positions** | Position book table with per-leg diagnostics, payoff diagram per instrument (Plotly subplot grid) |

## Files

| File | Purpose |
|---|---|
| `engine.py` | Subprocess wrapper for `RiskEngine.Cli`; returns the parsed risk report dict |
| `charts.py` | Plotly figure builders (histogram, vol curve, stress bars, payoff grid) |
| `app.py` | Shiny UI + server (Shiny Core / R-Shiny-style API) |
| `requirements.txt` | Python dependencies |

## Notes

- The `.cache/risk-report.json` is recreated every time the engine runs; it's
  gitignored.
- This dashboard expects `dotnet` (the .NET 10 SDK) to be on PATH, or a built
  `RiskEngine.Cli` executable already on disk. Pure-Python deployment isn't
  the goal here — the point is showcasing the Python ↔ C# integration.
