"""
Portfolio optimization: spot-only vs. derivative-augmented.

Compares efficient frontiers to test whether adding synthetic derivatives
expands the opportunity set for sports bettors.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional

from .derivatives import Derivative, build_instrument_catalog
from .implied_process import ImpliedProcess
from .pricing import EmpiricalPricer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIN_WEIGHT, MAX_WEIGHT, RISK_FREE_RATE, CVAR_ALPHA, MC_N_SIMS


@dataclass
class PortfolioResult:
    """Results from portfolio optimization."""
    weights: np.ndarray
    asset_names: list
    expected_return: float
    volatility: float
    sharpe: float
    cvar: float


class PortfolioOptimizer:
    """
    Portfolio optimization for betting strategies.

    Constructs the return matrix for:
    - Spot positions (bet on outcome)
    - Derivative positions (synthetic instruments)

    Then solves for optimal weights under mean-variance and CVaR.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        process: ImpliedProcess,
        catalog: Optional[dict] = None,
        n_sims: int = MC_N_SIMS,
    ):
        self.df = df
        self.process = process
        self.catalog = catalog or build_instrument_catalog()
        self.n_sims = n_sims

        # Build return matrix
        self._build_returns()

    def _build_returns(self):
        """
        Construct the return matrix.

        Each column is an "asset" (strategy), each row is a time period.
        """
        delta_ip = self.df["delta_ip_home"].dropna().values
        n = len(delta_ip)

        returns_dict = {}

        # Spot strategies
        # 1. Bet on home favorite (simple directional)
        returns_dict["spot_home_fav"] = np.where(
            self.df["close_ip_home"].values[:n] > 0.5,
            delta_ip,  # aligned with our position
            0,
        )[:n]

        # 2. Bet on underdog
        returns_dict["spot_underdog"] = np.where(
            self.df["close_ip_home"].values[:n] < 0.5,
            -delta_ip,  # we're on the other side
            0,
        )[:n]

        # 3. Market-neutral (no position)
        returns_dict["cash"] = np.zeros(n)

        # Derivative strategies
        for name, deriv in self.catalog.items():
            payoffs = deriv.payoff(delta_ip)

            # Estimate fair price from the full sample (simplified)
            pricer = EmpiricalPricer(self.process, n_sims=10_000)
            fair_price = pricer.price(deriv)["price"]

            # "Return" = realized payoff - fair price
            if fair_price > 0:
                returns_dict[f"deriv_{name}"] = (payoffs - fair_price) / max(fair_price, 1e-6)
            else:
                returns_dict[f"deriv_{name}"] = payoffs

        # Align lengths
        min_len = min(len(v) for v in returns_dict.values())
        self.returns = pd.DataFrame({
            k: v[:min_len] for k, v in returns_dict.items()
        })

        self.asset_names = list(self.returns.columns)
        self.n_assets = len(self.asset_names)
        self.mu = self.returns.mean().values
        self.cov = self.returns.cov().values

        print(f"Portfolio optimizer: {self.n_assets} assets, {min_len} periods")

    def optimize_sharpe(
        self,
        include_derivatives: bool = True,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
    ) -> PortfolioResult:
        """
        Find the maximum-Sharpe-ratio portfolio.
        """
        if include_derivatives:
            mu = self.mu
            cov = self.cov
            names = self.asset_names
        else:
            # Spot-only
            spot_mask = [not n.startswith("deriv_") for n in self.asset_names]
            idx = [i for i, m in enumerate(spot_mask) if m]
            mu = self.mu[idx]
            cov = self.cov[np.ix_(idx, idx)]
            names = [self.asset_names[i] for i in idx]

        n = len(mu)

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - RISK_FREE_RATE) / vol if vol > 1e-10 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(min_weight, max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(neg_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000})

        w = result.x
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-10 else 0

        # CVaR
        portfolio_returns = self.returns[names].values @ w
        cvar = self._compute_cvar(portfolio_returns)

        return PortfolioResult(
            weights=w,
            asset_names=names,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe=float(sharpe),
            cvar=float(cvar),
        )

    def optimize_cvar(
        self,
        include_derivatives: bool = True,
        target_return: Optional[float] = None,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
    ) -> PortfolioResult:
        """
        Minimize CVaR (Conditional Value-at-Risk) subject to return target.
        """
        if include_derivatives:
            returns_matrix = self.returns.values
            mu = self.mu
            names = self.asset_names
        else:
            spot_mask = [not n.startswith("deriv_") for n in self.asset_names]
            idx = [i for i, m in enumerate(spot_mask) if m]
            returns_matrix = self.returns.iloc[:, idx].values
            mu = self.mu[idx]
            names = [self.asset_names[i] for i in idx]

        n = len(mu)
        if target_return is None:
            target_return = np.mean(mu) * 0.5  # modest target

        def cvar_objective(w):
            port_ret = returns_matrix @ w
            return self._compute_cvar(port_ret)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: w @ mu - target_return},
        ]
        bounds = [(min_weight, max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(cvar_objective, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000})

        w = result.x
        ret = w @ mu
        vol = np.sqrt(w @ np.cov(returns_matrix.T) @ w)
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-10 else 0
        cvar = cvar_objective(w)

        return PortfolioResult(
            weights=w,
            asset_names=names,
            expected_return=float(ret),
            volatility=float(vol),
            sharpe=float(sharpe),
            cvar=float(cvar),
        )

    def efficient_frontier(
        self,
        include_derivatives: bool = True,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Trace the efficient frontier.

        Returns DataFrame with columns: target_return, volatility, sharpe, cvar.
        """
        if include_derivatives:
            mu = self.mu
            cov = self.cov
            names = self.asset_names
            returns_matrix = self.returns.values
        else:
            spot_mask = [not n.startswith("deriv_") for n in self.asset_names]
            idx = [i for i, m in enumerate(spot_mask) if m]
            mu = self.mu[idx]
            cov = self.cov[np.ix_(idx, idx)]
            names = [self.asset_names[i] for i in idx]
            returns_matrix = self.returns.iloc[:, idx].values

        n = len(mu)
        target_returns = np.linspace(mu.min(), mu.max(), n_points)

        frontier = []
        for target in target_returns:
            def portfolio_vol(w):
                return np.sqrt(w @ cov @ w)

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: w @ mu - target},
            ]
            bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n
            x0 = np.ones(n) / n

            result = minimize(portfolio_vol, x0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 500})

            if result.success:
                vol = result.fun
                sharpe = (target - RISK_FREE_RATE) / vol if vol > 1e-10 else 0
                port_ret = returns_matrix @ result.x
                cvar = self._compute_cvar(port_ret)
                frontier.append({
                    "target_return": target,
                    "volatility": vol,
                    "sharpe": sharpe,
                    "cvar": cvar,
                })

        return pd.DataFrame(frontier)

    @staticmethod
    def _compute_cvar(returns: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
        """Compute CVaR (Expected Shortfall) at given confidence level."""
        var = np.percentile(returns, alpha * 100)
        return -float(np.mean(returns[returns <= var]))

    def compare_frontiers(self, n_points: int = 50) -> dict:
        """
        Compare efficient frontiers: spot-only vs. derivative-augmented.

        The key research question: does the derivative frontier dominate?
        """
        print("Computing spot-only frontier...")
        spot_frontier = self.efficient_frontier(include_derivatives=False, n_points=n_points)
        spot_frontier["portfolio_type"] = "spot_only"

        print("Computing derivative-augmented frontier...")
        deriv_frontier = self.efficient_frontier(include_derivatives=True, n_points=n_points)
        deriv_frontier["portfolio_type"] = "with_derivatives"

        # Optimal portfolios
        spot_opt = self.optimize_sharpe(include_derivatives=False)
        deriv_opt = self.optimize_sharpe(include_derivatives=True)

        return {
            "frontiers": pd.concat([spot_frontier, deriv_frontier], ignore_index=True),
            "spot_optimal": spot_opt,
            "deriv_optimal": deriv_opt,
            "sharpe_improvement": deriv_opt.sharpe - spot_opt.sharpe,
            "cvar_improvement": spot_opt.cvar - deriv_opt.cvar,  # lower CVaR is better
        }
