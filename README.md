# Synthetic Derivatives on Sports Betting Markets

**Pricing and backtesting derivative-like instruments on bookmaker implied probabilities using 2.52 million observations across 14 sports (2013–2024).**

## Overview

Sports betting markets offer only spot-market positions — straight bets on outcomes. This project asks: what if derivative instruments existed on the *line-movement process* itself? I construct synthetic options, spreads, variance swaps, and other instruments on the implied probability evolution from opening to closing lines, price them using three methods — nonparametric (KDE Monte Carlo), Bachelier normal, and Student-t Bachelier — and test whether derivative-augmented portfolios dominate spot-only strategies.

The core underlying asset is the **line movement** — the change in implied probability between opening and closing odds:

```
Δp = p_close − p_open = (1/odds_close) − (1/odds_open)
```

This represents market-implied belief updating: the aggregation of information (injury news, weather, sharp money) between when a line opens and when it closes. Derivatives written on Δp allow expressing views on the *dynamics of price discovery* rather than on match outcomes.

## Key Findings

### 1. The line-movement distribution rejects normality decisively

| Statistic | Value |
|-----------|-------|
| N | 2,522,019 |
| Mean Δp | −0.0013 |
| Std Dev | 0.0526 |
| Skewness | 0.171 |
| Excess Kurtosis | **5.46** |
| Jarque-Bera | 3,140,434 (p = 0) |

Fat tails (kurtosis 5.46) mean extreme line movements occur far more often than a normal distribution predicts. The slight negative mean reflects systematic correction away from home favorites — consistent with public money initially inflating home probabilities, followed by sharp-money correction.

### 2. Three-model pricing comparison reveals structural mispricing

The Bachelier (normal) model is the correct BSM analog for this setting since Δp can be negative — but it systematically misprices. The Student-t Bachelier (fitted df = 2.70) dramatically improves near-the-money pricing but overcorrects in the deep tails, revealing that the true distribution lies between normal and power-law.

| Strike | Empirical | Bachelier Error | Student-t Error |
|--------|-----------|----------------|-----------------|
| K = 0.02 (call) | 0.0102 | −17.6% | **−0.8%** |
| K = 0.03 (call) | 0.0079 | −13.0% | **−0.3%** |
| K = 0.05 (call) | 0.0047 | +2.6% | −5.4% |
| K = 0.07 (call) | 0.0028 | +24.7% | −17.8% |
| K = 0.10 (call) | 0.0014 | **+59.9%** | −50.4% |

Near the money (K ≤ 0.03): Student-t reduces mispricing from 13–18% to under 1%. Deep OTM (K ≥ 0.07): Bachelier underprices by up to 60% (misses fat tails), Student-t overprices by up to 50% (too much tail weight from df < 3). The crossover at K ≈ 0.05 is where the normal density intersects the empirical leptokurtic density.

### 3. Implied volatility smile mirrors equity options

| Region | Implied σ | Ratio to Historical |
|--------|-----------|-------------------|
| ATM | 0.0445 | 0.85x |
| OTM (K ≥ 0.10) | 0.0667 | **1.50x** |

Deep OTM options require 50% higher implied volatility to match empirical prices — upward-sloping wings, the same structural pattern as equity vol smiles. This means these betting markets exhibit the same fat-tail pricing dynamics as real options markets.

### 4. PASPA repeal reshaped market microstructure

The 2018 repeal of the Professional and Amateur Sports Protection Act increased line-movement volatility by 36.5% while reducing tail thickness. More market participants create higher baseline information flow but more efficient price discovery with fewer extreme mispricings.

| Regime | σ | Kurtosis | Straddle Price | Bachelier Error (K=0.05) |
|--------|---|----------|----------------|--------------------------|
| Pre-PASPA | 0.041 | 6.72 | 0.0264 | +18.1% |
| Post-PASPA (ex COVID) | 0.057 | 4.77 | 0.0396 | +4.3% |
| COVID | 0.067 | 3.95 | 0.0501 | +0.4% |

The Bachelier model becomes more accurate as markets become more efficient — mispricing at K=0.05 drops from 18% to 4% post-PASPA. As kurtosis approaches normality, parametric pricing converges to nonparametric.

### 5. Sharp vs. recreational bookmakers show distinct microstructure

| Bookmaker Type | Avg σ | Avg Kurtosis | Mean Drift |
|----------------|-------|-------------|------------|
| Sharp (Pinnacle, Bet-in-Asia) | 0.058 | 4.25 | −0.004 |
| Recreational (888sport, bet365, bet-at-home) | 0.051 | 6.11 | −0.001 |

Sharp books have higher volatility but thinner tails (smoother price discovery from informed flow). Pinnacle lines drift toward away teams (sharp money fading home bias), while 888sport lines drift toward home teams (public money). Recreational books show fatter tails — occasional extreme jumps when accumulated sharp-money pressure forces corrections on stale lines.

### 6. Line movements are unpredictable but volatility clusters

After deduplicating to one observation per match (Pinnacle, N = 606,276):

| Test | Result | Interpretation |
|------|--------|----------------|
| Lag-1 autocorrelation | 0.007 (t = 5.35) | Statistically significant, economically negligible |
| Volatility clustering β | 0.040 (t = 31.45) | Significant — periods of high vol cluster together |
| Ljung-Box Q(20) | 310.9 (p = 0) | Rejects i.i.d., but driven by sample size |

Line movements are essentially unpredictable at the individual event level (efficient market). But volatility clusters — high-vol periods follow high-vol periods — mirroring the ARCH/GARCH structure of financial returns.

### 7. Derivatives improve risk-adjusted portfolio returns

No synthetic derivative generates standalone alpha (all negative Sharpe ratios) — consistent with market efficiency, though also attributable to implicit transaction costs (vig), model misspecification, and nonstationarity. The derivative-augmented portfolio achieves a **29% Sharpe improvement** (0.279 → 0.361) through hedging.

| Strategy | Sharpe | Hit Rate |
|----------|--------|----------|
| Call K=0.02 | −0.129 | 42.1% |
| Straddle ATM | −0.510 | 27.7% |
| Spot Benchmark (Favorite) | −0.455 | 30.8% |

The optimal portfolio shorts straddles/strangles (selling volatility) and goes long calls and underdog spot positions. The value of derivatives is not standalone alpha but risk management.

**Note on pricing measure**: Empirical (KDE) prices use the physical measure (P), not a risk-neutral measure (Q). Prices represent fair value under historical expectations, not arbitrage-free prices. A risk premium λ would bridge P → Q: E^Q[·] = E^P[·] + λ.

## Data

**2,524,200** bookmaker-match observations from OddsPortal covering:
- **14 sports**: football (soccer), hockey, handball, basketball, rugby, baseball, cricket, and others
- **7 bookmakers**: Pinnacle, 888sport, bet-at-home, Bet-in-Asia, bet365, BC.Game, BetMGM
- **943,134 unique matches** from March 2013 to December 2024
- Opening and closing decimal odds with timestamps

Source files (not included — too large). Place in `data/` at project root:
```
data/cleaned_data.csv
data/analysis_data.csv
```

Or set the `DATA_DIR` environment variable:
```bash
export DATA_DIR="/path/to/your/odds/data"
```

## Instruments

**Tier 1 — Vanilla Options**: Calls/puts on line movement, digital (binary) options

**Tier 2 — Spread Strategies**: Bull/bear spreads, straddles, strangles, butterflies

**Tier 3 — Volatility Instruments**: Variance swaps (realized vs. implied vol), margin swaps

25 instruments in the full catalog, 6 key instruments backtested across 5,039 rolling evaluation windows.

## Methodology

### Pricing (Three Models)
- **Empirical (KDE MC)**: Kernel density estimation (Silverman bandwidth) of the Δp distribution, priced via Monte Carlo (50,000 draws). Model-free baseline under the physical measure.
- **Bachelier (normal)**: The correct parametric analog for options on arithmetic processes (Δp can be negative, so BSM's lognormal assumption fails). Assumes normality.
- **Student-t Bachelier**: Replaces normal innovations with Student-t (fitted df = 2.70 via MLE on trimmed data). Captures fat tails parametrically. Where empirical and parametric prices diverge reveals the pricing impact of non-normality.

### Implied Volatility
For each strike, invert the Bachelier formula to extract the implied σ that matches the empirical price. Deviations from flat = fat tails. The resulting smile (1.50x at OTM) mirrors the structure seen in equity options.

### Backtesting
Rolling window protocol (train: 5,000 obs, step: 500 obs) with fair pricing from the training period. P&L = realized payoff − fair price. Performance metrics: Sharpe, Sortino, max drawdown, hit rate. Bootstrap confidence intervals (5,000 resamples).

### Portfolio Optimization
Mean-variance (Markowitz) and CVaR optimization comparing spot-only vs. derivative-augmented efficient frontiers.

### Predictive Analysis
Autocorrelation structure, predictive regressions (level and absolute), Ljung-Box test, and volatility clustering — all computed on deduplicated single-bookmaker data to avoid spurious cross-bookmaker correlation.

## Project Structure

```
sports-derivatives/
├── config.py                    # Paths, constants, parameters
├── src/
│   ├── data_loader.py           # Load/validate data, compute implied probs
│   ├── implied_process.py       # Extract stochastic process, regime analysis
│   ├── derivatives.py           # 25 synthetic instruments with payoff functions
│   ├── pricing.py               # Empirical, Bachelier, and Student-t pricing
│   ├── backtester.py            # Rolling-window backtesting framework
│   ├── portfolio.py             # MV and CVaR portfolio optimization
│   ├── predictive.py            # Autocorrelation, momentum/reversion tests
│   └── visualization.py         # 15 publication-quality figures
├── notebooks/
│   └── 01_exploration.py        # Full analysis pipeline (10 steps)
├── tests/
│   └── test_derivatives.py      # 28 unit tests for payoff correctness
├── output/
│   ├── figures/                 # 15 PNG figures
│   ├── tables/                  # 12 CSV result tables
│   └── results/                 # JSON summary
├── docs/
│   └── methodology.md           # Full mathematical methodology
└── risk-engine/                 # C# / .NET 10 risk engine — see risk-engine/README.md
    ├── src/RiskEngine.Core/        # Monte Carlo pricing & risk library
    ├── src/RiskEngine.Cli/         # console risk report
    ├── src/RiskEngine.Dashboard/   # Blazor risk dashboard
    └── tests/                      # xUnit test suite
```

## Setup & Usage

```bash
cd sports-derivatives
pip install -e .              # install in dev mode (resolves imports)
python notebooks/01_exploration.py   # run full analysis (~5 min)
python -m pytest tests/ -v           # run unit tests (28 tests)
```

Update data paths in `config.py` if your files are in a different location.

## Outputs

**15 figures**: line movement distribution (with QQ plot), sport-level KDE comparison, derivative payoff diagrams, two-model and three-model pricing curves, implied volatility smile, mispricing improvement bar chart, backtest cumulative P&L, strategy summary table, autocorrelation function, efficient frontier comparison, PASPA regime distributions, regime pricing table, rolling volatility over time, bookmaker microstructure comparison.

**12 tables**: process summary statistics, process by sport/bookmaker/regime, three-model pricing comparison, implied vol smile, catalog prices, backtest summary, optimal portfolio weights, regime-specific derivative pricing, autocorrelation structure, predictive regressions.

## C# Risk Engine

The [`risk-engine/`](risk-engine/) folder contains a C# / .NET 10 companion to this research: a quantitative risk and pricing engine that consumes the Python-generated calibration of the line-movement process and runs a production-style risk pipeline on a portfolio of synthetic instruments.

It mirrors how quant desks are organised — Python for research and calibration, a separate compiled engine for production risk. The Python pipeline exports the calibrated Δp distribution per regime to `risk-engine/data/calibration.json` (via `risk-engine/tools/export_calibration.py`); the C# engine consumes that file and runs:

- Monte Carlo scenario generation with Normal and fat-tailed Student-t samplers
- payoff distributions for a loaded portfolio (JSON/CSV ingestion)
- VaR / CVaR monitoring, with a closed-form Bachelier pricing benchmark
- a stress-test battery and a regime-switching mixture model
- volatility scenario analysis

It ships as a .NET solution with an engine library, a console runner, an interactive Blazor dashboard, and an xUnit test suite. See [`risk-engine/README.md`](risk-engine/README.md) for build and run instructions.

## Connection to Related Work

This project extends my research on sports betting market microstructure following the PASPA repeal (presented at MEA 2025). The PASPA spillover paper examines how US legalization affected international betting margins; this project examines whether the line-movement dynamics that drive those margins can be exploited through derivative-like instruments.

The Bachelier model connection links sports betting microstructure to the earliest option pricing literature (Bachelier, 1900) — and the finding that Bachelier misprices while Student-t overcorrects suggests the true data-generating process lies between arithmetic Brownian motion and a power-law jump process. The implied volatility smile at 1.50x confirms these markets share structural properties with equity options markets despite having no formal derivatives exchange.

## Future Directions

- **Mixture-of-normals pricing**: A two-component normal mixture (routine moves + information shocks) may bridge the gap between Bachelier and Student-t in the deep tails
- **Risk-neutral measure**: Estimate the risk premium λ that bridges P → Q pricing, potentially from the overround structure
- **Path-dependent derivatives**: With intraday line data (currently unavailable from OddsPortal), Asian options and barrier options become tractable
- **Cross-sport hedging**: Can volatility positions in one sport hedge directional risk in another?

## Requirements

Python ≥ 3.10, pandas, numpy, scipy, matplotlib, seaborn, pyarrow, scikit-learn.

The C# risk engine additionally requires the .NET 10 SDK — see [`risk-engine/README.md`](risk-engine/README.md).

## License

Academic use. Data not redistributable per OddsPortal terms of service.
