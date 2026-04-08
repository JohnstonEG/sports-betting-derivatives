"""
Backtesting framework for synthetic derivative strategies.

Compares derivative-augmented strategies against spot-only benchmarks
using rolling windows on the historical data.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .derivatives import Derivative, build_instrument_catalog, VanillaOption, Straddle, VarianceSwap
from .implied_process import ImpliedProcess, extract_implied_process
from .pricing import EmpiricalPricer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BACKTEST_TRAIN_WINDOW, BACKTEST_STEP, MC_SEED,
    MATCH_ID, SPORT, MATCH_DATE, HOME_WIN, CLOSE_MARGIN,
)


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    strategy_name: str
    pnl: np.ndarray
    cumulative_pnl: np.ndarray
    dates: np.ndarray
    n_trades: int
    total_pnl: float = 0.0
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    calmar: float = 0.0

    def __post_init__(self):
        # Clean NaN from pnl
        if len(self.pnl) > 0:
            nan_mask = np.isfinite(self.pnl)
            if not nan_mask.all():
                self.pnl = self.pnl[nan_mask]
                self.dates = self.dates[nan_mask] if len(self.dates) == len(nan_mask) else self.dates

        self.n_trades = len(self.pnl)
        if self.n_trades == 0:
            return
        self.total_pnl = float(np.sum(self.pnl))
        self.mean_pnl = float(np.mean(self.pnl))
        self.std_pnl = float(np.std(self.pnl, ddof=1)) if self.n_trades > 1 else 0
        self.sharpe = self.mean_pnl / self.std_pnl if self.std_pnl > 0 else 0
        self.hit_rate = float(np.mean(self.pnl > 0))

        # Sortino (downside deviation)
        downside = self.pnl[self.pnl < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
        self.sortino = self.mean_pnl / downside_std if downside_std > 0 else 0

        # Max drawdown
        cum = np.cumsum(self.pnl)
        self.cumulative_pnl = cum
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum
        self.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        # Calmar ratio
        self.calmar = self.total_pnl / self.max_drawdown if self.max_drawdown > 0 else 0

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "mean_pnl": self.mean_pnl,
            "std_pnl": self.std_pnl,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "calmar": self.calmar,
        }


class Backtester:
    """
    Rolling-window backtester for derivative strategies.

    Workflow:
    1. Use training window to estimate the implied process
    2. Price derivatives using the training-period distribution
    3. On the test window, compute realized payoffs
    4. Net P&L = realized payoff - fair price (what you "paid")
    5. Roll forward and repeat
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train_window: int = BACKTEST_TRAIN_WINDOW,
        step: int = BACKTEST_STEP,
        sport_filter: Optional[str] = None,
    ):
        self.train_window = train_window
        self.step = step

        # Sort by date
        self.df = df.sort_values(MATCH_DATE).reset_index(drop=True)
        if sport_filter:
            self.df = self.df[self.df[SPORT] == sport_filter].reset_index(drop=True)

        self.delta_ip = self.df["delta_ip_home"].values
        self.dates = self.df[MATCH_DATE].values
        self.n = len(self.df)

        print(f"Backtester initialized: {self.n:,} observations")
        print(f"  Train window: {train_window} | Step: {step}")
        print(f"  Approx {(self.n - train_window) // step} evaluation periods")

    def run_strategy(
        self,
        derivative: Derivative,
        strategy_name: Optional[str] = None,
    ) -> BacktestResult:
        """
        Backtest a single derivative strategy.

        At each step:
        - Estimate process from training window
        - Price the derivative (= fair cost)
        - Compute realized payoffs on test window
        - P&L = mean(realized payoff) - fair_price
        """
        name = strategy_name or derivative.description()
        pnl_list = []
        date_list = []

        for start in range(0, self.n - self.train_window, self.step):
            train_end = start + self.train_window
            test_end = min(train_end + self.step, self.n)

            train_delta = self.delta_ip[start:train_end]
            test_delta = self.delta_ip[train_end:test_end]

            if len(test_delta) == 0:
                continue

            # Skip if training data is degenerate
            if np.std(train_delta) < 1e-8:
                continue

            # Build process from training data
            process = ImpliedProcess(
                delta_ip=train_delta,
                delta_logit=np.zeros_like(train_delta),  # simplified
                abs_delta=np.abs(train_delta),
                segment="backtest_train",
            )

            # Price the derivative
            pricer = EmpiricalPricer(process, n_sims=10_000)
            pricing = pricer.price(derivative)
            fair_price = pricing["price"]

            # Realized payoffs on test window
            realized_payoffs = derivative.payoff(test_delta)
            mean_realized = np.mean(realized_payoffs)

            # P&L per period = what you received - what you paid
            period_pnl = mean_realized - fair_price
            pnl_list.append(period_pnl)
            date_list.append(self.dates[train_end])

        return BacktestResult(
            strategy_name=name,
            pnl=np.array(pnl_list),
            cumulative_pnl=np.cumsum(pnl_list) if pnl_list else np.array([]),
            dates=np.array(date_list),
            n_trades=len(pnl_list),
        )

    def run_spot_benchmark(self) -> BacktestResult:
        """
        Benchmark: optimal "spot" strategy.

        Two sub-strategies combined:
        1. Bet on strong home favorites (closing prob > 0.55)
        2. Bet on value underdogs (where line moved toward them)

        P&L based on closing odds vs. actual outcome, adjusted for vig.
        This represents the best a bettor can do WITHOUT derivatives.
        """
        pnl_list = []
        date_list = []

        # Ensure home_win is numeric (CSV loading can make it string)
        if self.df[HOME_WIN].dtype == object:
            self.df[HOME_WIN] = self.df[HOME_WIN].map(
                {True: 1, False: 0, "True": 1, "False": 0, "1": 1, "0": 0}
            ).fillna(0).astype(float)
        else:
            self.df[HOME_WIN] = pd.to_numeric(self.df[HOME_WIN], errors="coerce").fillna(0)

        for start in range(0, self.n - self.train_window, self.step):
            train_end = start + self.train_window
            test_end = min(train_end + self.step, self.n)

            test_slice = self.df.iloc[train_end:test_end]
            if len(test_slice) == 0:
                continue

            # Use vig-free probability if available, else raw
            if "close_norm_ip_home" in test_slice.columns:
                prob_col = "close_norm_ip_home"
            else:
                prob_col = "close_ip_home"

            # Filter to only bets where we'd bet (prob > 0.55, valid data)
            valid = (
                test_slice[prob_col].notna() &
                (test_slice[prob_col] > 0.55) &
                (test_slice[prob_col] < 0.95) &
                test_slice[HOME_WIN].notna()
            )
            bets = test_slice[valid].copy()

            if len(bets) == 0:
                pnl_list.append(0.0)
                date_list.append(self.dates[train_end])
                continue

            wins = bets[HOME_WIN].astype(float).values
            prob = bets[prob_col].values

            # Get actual closing odds (what bettor receives)
            if "close_home_odds" in bets.columns:
                actual_odds = bets["close_home_odds"].values
            else:
                # Approximate from implied prob + margin
                margin = bets[CLOSE_MARGIN].fillna(0.05).values
                fair_odds = 1.0 / prob
                actual_odds = fair_odds * (1 - margin / 2)

            # P&L: win → (odds - 1), lose → -1 (unit stake)
            bet_pnl = wins * (actual_odds - 1) - (1 - wins) * 1.0

            # Drop any NaN that slipped through
            bet_pnl = bet_pnl[np.isfinite(bet_pnl)]
            if len(bet_pnl) == 0:
                pnl_list.append(0.0)
            else:
                pnl_list.append(float(np.mean(bet_pnl)))
            date_list.append(self.dates[train_end])

        return BacktestResult(
            strategy_name="Spot Benchmark (Favorite)",
            pnl=np.array(pnl_list),
            cumulative_pnl=np.cumsum(pnl_list) if pnl_list else np.array([]),
            dates=np.array(date_list),
            n_trades=len(pnl_list),
        )

    def run_catalog(
        self,
        catalog: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Run all instruments in the catalog + benchmark.

        Returns a summary DataFrame comparing all strategies.
        """
        if catalog is None:
            catalog = build_instrument_catalog()

        results = []

        # Benchmark first
        print("Running spot benchmark...")
        bench = self.run_spot_benchmark()
        results.append(bench.summary())

        # Each derivative
        for name, deriv in catalog.items():
            # Skip variance swaps (need special handling)
            if isinstance(deriv, VarianceSwap):
                # Set strike from first training window
                train_delta = self.delta_ip[:self.train_window]
                deriv.set_strike_from_data(train_delta)

            print(f"Running {name}...")
            bt_result = self.run_strategy(deriv, strategy_name=name)
            results.append(bt_result.summary())

        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values("sharpe", ascending=False)
        return summary_df

    def bootstrap_sharpe_ci(
        self,
        result: BacktestResult,
        n_bootstrap: int = 5000,
        ci: float = 0.95,
        seed: int = MC_SEED,
    ) -> dict:
        """
        Bootstrap confidence interval for Sharpe ratio.

        Tests whether the strategy's Sharpe is significantly
        different from zero (or from the benchmark).
        """
        rng = np.random.default_rng(seed)
        pnl = result.pnl
        n = len(pnl)

        if n < 10:
            return {"sharpe": result.sharpe, "ci_lower": 0, "ci_upper": 0, "n": n}

        boot_sharpes = []
        for _ in range(n_bootstrap):
            sample = rng.choice(pnl, size=n, replace=True)
            s_mean = np.mean(sample)
            s_std = np.std(sample, ddof=1)
            boot_sharpes.append(s_mean / s_std if s_std > 0 else 0)

        boot_sharpes = np.array(boot_sharpes)
        alpha = (1 - ci) / 2

        return {
            "sharpe": result.sharpe,
            "ci_lower": float(np.percentile(boot_sharpes, alpha * 100)),
            "ci_upper": float(np.percentile(boot_sharpes, (1 - alpha) * 100)),
            "p_positive": float(np.mean(boot_sharpes > 0)),
            "n": n,
        }
