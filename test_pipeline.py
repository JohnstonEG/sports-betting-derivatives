"""
Quick pipeline test using synthetic data.
Verifies the full analysis chain works before running on real data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

# ============================================================
# Generate synthetic betting data
# ============================================================
print("Generating synthetic test data...")
np.random.seed(42)
N = 20_000

# Simulate opening implied probs (favorites around 0.55-0.75)
open_ip_home = np.random.beta(5, 3, N)  # right-skewed, centered ~0.62
open_ip_away = 1 - open_ip_home

# Add overround
margin = np.random.uniform(0.02, 0.08, N)
open_ip_home_raw = open_ip_home * (1 + margin / 2)
open_ip_away_raw = open_ip_away * (1 + margin / 2)

# Line movements: fat-tailed (mixture of normal)
# 90% small moves, 10% big moves (news/injury)
small_moves = np.random.normal(0, 0.015, N)
big_moves = np.random.normal(0, 0.06, N)
is_big = np.random.random(N) < 0.10
delta = np.where(is_big, big_moves, small_moves)

close_ip_home_raw = open_ip_home_raw + delta
close_ip_away_raw = open_ip_away_raw - delta * 0.8  # not perfectly correlated

# Clip to valid range
close_ip_home_raw = np.clip(close_ip_home_raw, 0.05, 0.98)
close_ip_away_raw = np.clip(close_ip_away_raw, 0.05, 0.98)

# Convert back to odds
open_home_odds = 1 / open_ip_home_raw
open_away_odds = 1 / open_ip_away_raw
close_home_odds = 1 / close_ip_home_raw
close_away_odds = 1 / close_ip_away_raw

# Outcomes
true_prob = open_ip_home + delta * 2  # noisy signal
true_prob = np.clip(true_prob, 0.1, 0.9)
home_win = np.random.random(N) < true_prob

# Build DataFrame
dates = pd.date_range("2015-01-01", periods=N, freq="3h")
sports = np.random.choice(["soccer", "basketball", "baseball", "tennis"], N,
                           p=[0.35, 0.25, 0.25, 0.15])

df = pd.DataFrame({
    "match_id": [f"M{i:06d}" for i in range(N)],
    "sport": sports,
    "league": [f"{s}_league_1" for s in sports],
    "country": "synthetic",
    "sportsbook": np.random.choice(["3", "18", "27"], N),
    "bookmaker_name": np.random.choice(["Pinnacle", "bet365", "1xBet"], N),
    "start_time": dates,
    "match_date": dates.normalize(),
    "open_home_odds": open_home_odds,
    "open_away_odds": open_away_odds,
    "open_draw_odds": np.where(sports == "soccer", np.random.uniform(3, 5, N), np.nan),
    "close_home_odds": close_home_odds,
    "close_away_odds": close_away_odds,
    "close_draw_odds": np.where(sports == "soccer", np.random.uniform(3, 5, N), np.nan),
    "close_margin": np.clip(close_ip_home_raw + close_ip_away_raw - 1, 0.01, 0.15),
    "open_margin": margin,
    "home_score": np.where(home_win, np.random.poisson(2, N), np.random.poisson(1, N)),
    "away_score": np.where(~home_win, np.random.poisson(2, N), np.random.poisson(1, N)),
    "home_win": home_win,
    "is_draw": False,
})

print(f"Synthetic data: {len(df):,} rows, {df['sport'].nunique()} sports")
print(f"Date range: {df['match_date'].min().date()} to {df['match_date'].max().date()}")

# ============================================================
# Run pipeline modules
# ============================================================
from src.data_loader import _add_implied_probabilities, _add_line_movements
from src.implied_process import extract_implied_process, compare_processes
from src.derivatives import build_instrument_catalog, VanillaOption, Straddle, VarianceSwap
from src.pricing import EmpiricalPricer, BSMPricer, compare_pricing_methods
from src.backtester import Backtester, BacktestResult
from src.portfolio import PortfolioOptimizer

# Step 1: Add implied probs and line movements
print("\n--- Adding implied probabilities ---")
df = _add_implied_probabilities(df)
df = _add_line_movements(df)
print(f"  delta_ip_home: mean={df['delta_ip_home'].mean():.5f}, "
      f"std={df['delta_ip_home'].std():.5f}")

# Step 2: Extract process
print("\n--- Extracting implied process ---")
process = extract_implied_process(df, outcome="home")
print(f"  N={process.n_obs}, mean={process.mean:.5f}, std={process.std:.5f}")
print(f"  Skew={process.skew:.3f}, Kurt={process.kurtosis:.3f}")
print(f"  JB p-value={process.normality_test['jarque_bera']['pvalue']:.2e}")

# By sport
processes = extract_implied_process(df, outcome="home", segment_by="sport")
comparison = compare_processes(processes)
print(f"\n  Process by sport:")
print(comparison[["segment", "n_obs", "std", "kurtosis"]].to_string(index=False))

# Step 3: Price derivatives
print("\n--- Pricing derivatives ---")
catalog = build_instrument_catalog()
emp = EmpiricalPricer(process)

for name in ["call_K0.02", "straddle_ATM", "strangle_2pp"]:
    result = emp.price(catalog[name])
    print(f"  {name:20s} price={result['price']:.5f} ITM={result['pct_itm']:.1%}")

# BSM comparison
print("\n--- BSM vs Empirical ---")
pricing_df = compare_pricing_methods(process, strikes=[0.01, 0.02, 0.05])
for _, row in pricing_df.iterrows():
    print(f"  {row['type']:4s} K={row['strike']:+.2f}: "
          f"emp={row['emp_price']:.5f} bsm={row['bsm_price']:.5f} "
          f"diff={row['pct_diff']:+.1f}%")

# Step 4: Backtest
print("\n--- Backtesting (small window for speed) ---")
bt = Backtester(df, train_window=2000, step=500)

bench = bt.run_spot_benchmark()
print(f"  Benchmark: SR={bench.sharpe:.3f}, hit={bench.hit_rate:.1%}")

straddle_result = bt.run_strategy(catalog["straddle_ATM"], "straddle_ATM")
print(f"  Straddle:  SR={straddle_result.sharpe:.3f}, hit={straddle_result.hit_rate:.1%}")

call_result = bt.run_strategy(catalog["call_K0.02"], "call_K0.02")
print(f"  Call K=2%: SR={call_result.sharpe:.3f}, hit={call_result.hit_rate:.1%}")

# Bootstrap CI
boot = bt.bootstrap_sharpe_ci(straddle_result)
print(f"\n  Straddle Sharpe bootstrap: {boot['sharpe']:.3f} "
      f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

# Step 5: Portfolio optimization
print("\n--- Portfolio optimization ---")
key_instruments = {
    "call_K0.02": catalog["call_K0.02"],
    "straddle_ATM": catalog["straddle_ATM"],
    "strangle_2pp": catalog["strangle_2pp"],
}
opt = PortfolioOptimizer(df, process, catalog=key_instruments, n_sims=5000)

spot_opt = opt.optimize_sharpe(include_derivatives=False)
deriv_opt = opt.optimize_sharpe(include_derivatives=True)
print(f"  Spot-only Sharpe:  {spot_opt.sharpe:.4f}")
print(f"  With derivatives:  {deriv_opt.sharpe:.4f}")
print(f"  Improvement:       {deriv_opt.sharpe - spot_opt.sharpe:+.4f}")

# Step 6: Figures (just test they don't crash)
print("\n--- Testing figure generation ---")
from src.visualization import (
    plot_line_movement_distribution,
    plot_payoff_diagrams,
    plot_pricing_comparison,
)

# Override output dir for test
import config
config.FIGURES_DIR = Path("/home/claude/sports-derivatives/output/figures")
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plot_line_movement_distribution(process, filename="test_dist")
plot_payoff_diagrams(filename="test_payoffs")
plot_pricing_comparison(pricing_df, filename="test_pricing")

print("\n" + "=" * 60)
print("ALL PIPELINE TESTS PASSED")
print("=" * 60)
print("\nReady to run on your real data at:")
print("  D:\\Data\\Odds\\Output\\processed\\cleaned_data.parquet")
print("\nUsage: python notebooks/01_exploration.py")
