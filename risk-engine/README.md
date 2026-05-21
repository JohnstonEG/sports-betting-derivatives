# Synthetic Derivatives Risk Engine

**A C# / .NET quantitative risk and pricing engine that consumes Python-generated
derivative calibrations and runs Monte Carlo risk on a book of synthetic
sports-betting instruments.**

This is the C# companion to the [Synthetic Derivatives on Sports Betting Markets](../README.md)
research project. The Python project *calibrates* the line-movement process and
prices derivatives; this engine *consumes* those calibrations and runs a
production-style risk pipeline — Monte Carlo scenario generation, VaR/CVaR
monitoring, stress testing, regime switching and payoff visualisation.

That split — Python for research and calibration, C# for the risk engine — is
how many quant desks are actually organised, which is the point of this component.

## What it does

The underlying asset is the **line movement** `Δp = p_close − p_open`, the change
in bookmaker-implied probability between the opening and closing line. The Python
study showed this process is fat-tailed (excess kurtosis 5.46) and regime-dependent
(pre-/post-PASPA, COVID). This engine takes those calibrated parameters and:

| Capability | Where |
|---|---|
| Load portfolio positions (JSON / CSV) | `PortfolioLoader` |
| Compute payoff distributions | `MonteCarloEngine`, `Histogram` |
| Monte Carlo scenario generation | `MonteCarloEngine`, `Normal`/`StudentT` samplers |
| VaR / CVaR monitoring | `RiskMetrics`, `RiskSummary` |
| Stress testing | `StressTester` (6-scenario battery) |
| Regime switching | `RegimeSwitchingSampler`, `ScenarioAnalyzer` |
| Volatility scenario analysis | `ScenarioAnalyzer.VolatilityCurve` |
| Payoff visualisation / dashboard | `RiskEngine.Dashboard` (Blazor) |
| Analytic benchmark pricing | `Bachelier` (closed-form + implied vol) |

## Architecture

```
  Python research pipeline                  C# risk engine
  ────────────────────────                  ──────────────────────────────
  data → calibrate Δp process   ──JSON──►   RiskEngine.Core   (engine library)
  tools/export_calibration.py               ├─ RiskEngine.Cli       (console)
        writes calibration.json             └─ RiskEngine.Dashboard (Blazor UI)
```

The bridge is a single JSON file (`data/calibration.json`) holding the calibrated
distribution parameters per regime. `tools/export_calibration.py` regenerates it
from the source odds data; the C# side never touches Python.

### Projects

```
risk-engine/
├── RiskEngine.sln
├── data/
│   ├── calibration.json        # Python-generated calibration (the bridge)
│   ├── portfolio.json          # sample portfolio of synthetic derivatives
│   └── portfolio.csv           # same book, CSV form (ingestion demo)
├── tools/
│   └── export_calibration.py   # Python → JSON exporter
├── src/
│   ├── RiskEngine.Core/        # the engine — pure .NET, no dependencies
│   │   ├── Models/             # Instrument, Position, Portfolio, Calibration
│   │   ├── Pricing/            # payoff functions + analytic Bachelier pricing
│   │   ├── Simulation/         # RNG, Normal/Student-t/regime samplers, MC engine
│   │   ├── Risk/               # VaR/CVaR metrics, histogram, stress, scenarios
│   │   ├── Io/                 # JSON/CSV loaders
│   │   └── RiskEngineService.cs # orchestration → RiskReport
│   ├── RiskEngine.Cli/         # console runner — formatted report + JSON output
│   └── RiskEngine.Dashboard/   # Blazor dashboard with inline-SVG charts
└── tests/
    └── RiskEngine.Tests/       # xUnit suite (payoffs, metrics, MC, I/O)
```

## Quick start

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
cd risk-engine

# Build everything
dotnet build

# Run the console risk report
dotnet run --project src/RiskEngine.Cli

# Run the same engine behind the Blazor dashboard, then open the URL it prints
dotnet run --project src/RiskEngine.Dashboard

# Run the test suite
dotnet test
```

The CLI and dashboard ship with a sample `portfolio.json` and `calibration.json`,
so they run with no extra setup.

### CLI options

```
dotnet run --project src/RiskEngine.Cli -- [options]

  -p, --portfolio <path>    Portfolio file (.json or .csv)
  -c, --calibration <path>  Calibration file (.json)
  -m, --model <name>        Normal | StudentT
  -n, --paths <int>         Monte Carlo paths (default 50000)
  -o, --out <path>          JSON report output
```

### Regenerating the calibration from data

```bash
# From the project's analysis data (Δp already a column):
python tools/export_calibration.py --data ../data/analysis_data.csv \
    --delta-col delta_p --date-col match_date

# From decimal odds (computes Δp = 1/close − 1/open):
python tools/export_calibration.py --data ../data/cleaned_data.csv \
    --open-col odds_open --close-col odds_close --date-col match_date

# Or write the documented sample values:
python tools/export_calibration.py --use-defaults
```

## Methodology notes

**Pricing models.** The engine simulates `Δp` under either a Normal (Bachelier)
or a scaled Student-t distribution. Bachelier — not Black-Scholes — is the correct
analogue because `Δp` can be negative. The Student-t sampler is rescaled so its
standard deviation matches the calibrated `σ`, isolating the effect of fat tails;
its degrees of freedom come from the Python MLE fit (df ≈ 2.70 full-sample).

**Monte Carlo.** Each run is seeded, so results are reproducible and scenario
comparisons use *common random numbers* — a standard variance-reduction technique
that makes a parameter change, not simulation noise, the only difference between
two scenarios.

**VaR / CVaR.** VaR at confidence *c* is the loss not exceeded with probability
*c* (interpolated quantile of the P&L distribution). CVaR (expected shortfall) is
the mean loss conditional on being in the worst *(1 − c)* tail, so by construction
CVaR ≥ VaR — a property the test suite checks.

**Regime switching.** The regime-switching sampler is a mixture: each simulated
path first draws a regime by its historical weight, then draws `Δp` from that
regime's distribution. This reproduces the volatility-clustered, fat-tailed
behaviour documented across the pre-/post-PASPA and COVID regimes.

**Analytic benchmark.** `Bachelier` provides closed-form call/put/digital prices
and a bisection implied-vol solver, used both as a Monte Carlo convergence check
in the tests and as a reference inside the engine.

## Validation

The xUnit suite (`dotnet test`) covers:

- payoff correctness for all ten instrument types, and put-call parity;
- closed-form Bachelier pricing, put-call parity, and implied-vol round-tripping;
- risk metrics against hand-computed values, and the CVaR ≥ VaR invariant;
- Monte Carlo determinism, and convergence of the simulated call price to the
  analytic Bachelier value;
- Student-t sampler rescaling, Gamma/Gaussian variate moments;
- JSON and CSV portfolio loading, and the full `RiskEngineService` pipeline.

## Resume framing

> Developed a C# quantitative risk and pricing engine integrating Python-generated
> derivative valuations, portfolio exposures and scenario simulations for synthetic
> sports-betting instruments — Monte Carlo scenario generation, VaR/CVaR monitoring,
> stress testing and payoff visualisation across a .NET solution (engine library,
> console runner, Blazor dashboard, xUnit suite).

## License

Academic use, consistent with the parent project. Data not redistributable per
OddsPortal terms of service.
