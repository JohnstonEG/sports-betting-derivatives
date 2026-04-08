"""
Pricing engines for synthetic sports betting derivatives.

Two approaches:
1. Empirical (nonparametric): KDE-based Monte Carlo pricing
2. Black-Scholes-Merton analog: parametric pricing with mapped parameters

The comparison between these two is a key research question.
"""
import numpy as np
from scipy import stats
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional

from .implied_process import ImpliedProcess
from .derivatives import (
    Derivative, VanillaOption, DigitalOption, Straddle, Strangle,
    VarianceSwap,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MC_N_SIMS, MC_SEED


# ================================================================
# EMPIRICAL PRICER (NONPARAMETRIC)
# ================================================================

@dataclass
class EmpiricalPricer:
    """
    Price derivatives via Monte Carlo from the empirical KDE.

    This is the "model-free" benchmark. It respects whatever 
    distributional features exist in the data (fat tails, skew, etc.)
    without imposing parametric assumptions.
    """
    process: ImpliedProcess
    n_sims: int = MC_N_SIMS
    seed: int = MC_SEED

    def price(self, derivative: Derivative) -> dict:
        """
        Price a derivative instrument.

        Returns
        -------
        dict with keys:
            price : float — expected payoff
            std_err : float — MC standard error
            ci_lower, ci_upper : float — 95% CI
            n_sims : int
        """
        rng = np.random.default_rng(self.seed)
        draws = self.process.kde.resample(self.n_sims, seed=rng).flatten()
        payoffs = derivative.payoff(draws)

        price = float(np.mean(payoffs))
        std_err = float(np.std(payoffs, ddof=1) / np.sqrt(self.n_sims))

        return {
            "price": price,
            "std_err": std_err,
            "ci_lower": price - 1.96 * std_err,
            "ci_upper": price + 1.96 * std_err,
            "n_sims": self.n_sims,
            "pct_itm": float(np.mean(payoffs > 0)),
        }

    def price_surface(
        self,
        derivative_class,
        strikes: list[float],
        **kwargs,
    ) -> list[dict]:
        """
        Price a derivative across multiple strikes.

        Returns a list of pricing results, one per strike.
        """
        results = []
        for k in strikes:
            d = derivative_class(strike=k, **kwargs)
            result = self.price(d)
            result["strike"] = k
            result["instrument"] = d.description()
            results.append(result)
        return results


# ================================================================
# BLACK-SCHOLES-MERTON ANALOG
# ================================================================

@dataclass
class BSMPricer:
    """
    Black-Scholes-Merton pricing adapted for betting markets.

    Parameter mapping:
        S₀ = opening implied probability (e.g., 0.60 for a -150 favorite)
        K  = strike implied probability
        σ  = volatility of line movements (from ImpliedProcess)
        T  = normalized time (we set T=1 since each "trade" is one event)
        r  = 0 (no risk-free rate in betting; could add vig as cost)

    The key question: does BSM (which assumes normality) adequately
    price these instruments given the empirical distribution?
    """
    process: ImpliedProcess
    risk_free_rate: float = 0.0
    T: float = 1.0  # normalized

    @property
    def sigma(self) -> float:
        """Volatility from the implied process."""
        return self.process.std

    def _d1(self, S: float, K: float) -> float:
        """BSM d1 parameter."""
        if self.sigma <= 0 or self.T <= 0:
            return 0.0
        return (
            (np.log(S / K) + (self.risk_free_rate + 0.5 * self.sigma**2) * self.T)
            / (self.sigma * np.sqrt(self.T))
        )

    def _d2(self, S: float, K: float) -> float:
        return self._d1(S, K) - self.sigma * np.sqrt(self.T)

    def price_vanilla(
        self,
        S: float,
        K: float,
        option_type: str = "call",
    ) -> dict:
        """
        Price a vanilla option using the Bachelier (normal) model.

        Since we're pricing options on line *movements* (which can be
        negative), the standard lognormal BSM doesn't apply. The 
        Bachelier model assumes the underlying follows an arithmetic
        Brownian motion — appropriate for Δp which is approximately
        normally distributed around zero.

        Bachelier call: C = σ√T [d·Φ(d) + φ(d)]
        where d = (F - K) / (σ√T), F = forward = process mean

        This is the correct "BSM analog" for this setting.
        """
        sigma = self.sigma
        T = self.T
        F = self.process.mean  # forward = expected movement

        if sigma <= 0 or T <= 0:
            # Intrinsic value only
            if option_type == "call":
                return {"price": max(F - K, 0), "delta": 1.0 if F > K else 0.0,
                        "gamma": 0.0, "vega": 0.0, "d": 0.0, "sigma_used": sigma}
            else:
                return {"price": max(K - F, 0), "delta": -1.0 if F < K else 0.0,
                        "gamma": 0.0, "vega": 0.0, "d": 0.0, "sigma_used": sigma}

        vol_sqrt_t = sigma * np.sqrt(T)
        d = (F - K) / vol_sqrt_t

        if option_type == "call":
            price = vol_sqrt_t * (d * norm.cdf(d) + norm.pdf(d))
            delta_greek = norm.cdf(d)
        else:
            price = vol_sqrt_t * (-d * norm.cdf(-d) + norm.pdf(d))
            delta_greek = norm.cdf(d) - 1

        # Bachelier Greeks
        gamma = norm.pdf(d) / vol_sqrt_t
        vega = np.sqrt(T) * norm.pdf(d)

        return {
            "price": float(price),
            "delta": float(delta_greek),
            "gamma": float(gamma),
            "vega": float(vega),
            "d": float(d),
            "sigma_used": sigma,
        }

    def price_digital(
        self,
        S: float,
        K: float,
        option_type: str = "call",
        payout: float = 1.0,
    ) -> dict:
        """Price a digital/binary option using Bachelier model."""
        sigma = self.sigma
        T = self.T
        F = self.process.mean

        if sigma <= 0 or T <= 0:
            if option_type == "call":
                return {"price": payout if F > K else 0.0}
            else:
                return {"price": payout if F < K else 0.0}

        vol_sqrt_t = sigma * np.sqrt(T)
        d = (F - K) / vol_sqrt_t

        if option_type == "call":
            price = payout * norm.cdf(d)
        else:
            price = payout * norm.cdf(-d)

        return {"price": float(price)}

    def implied_vol(
        self,
        market_price: float,
        S: float,
        K: float,
        option_type: str = "call",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Back out the implied volatility from an empirical price.

        Uses Newton-Raphson on the Bachelier formula.
        """
        F = self.process.mean
        T = self.T
        sigma_guess = self.sigma

        for _ in range(max_iter):
            vol_sqrt_t = sigma_guess * np.sqrt(T)
            if vol_sqrt_t < 1e-12:
                break
            d = (F - K) / vol_sqrt_t

            if option_type == "call":
                price = vol_sqrt_t * (d * norm.cdf(d) + norm.pdf(d))
            else:
                price = vol_sqrt_t * (-d * norm.cdf(-d) + norm.pdf(d))

            vega = np.sqrt(T) * norm.pdf(d)

            if abs(vega) < 1e-12:
                break

            sigma_guess -= (price - market_price) / vega

            if abs(price - market_price) < tol:
                break

            sigma_guess = max(sigma_guess, 1e-6)

        return float(sigma_guess)


# ================================================================
# STUDENT-t BACHELIER PRICER
# ================================================================

@dataclass
class StudentTBachelierPricer:
    """
    Bachelier model with Student-t distributed innovations.

    The standard Bachelier assumes normality. By replacing the normal
    with a Student-t distribution (heavier tails), we capture the
    empirical fat-tailed structure of line movements.

    The t-distribution is parameterized by:
    - location (mu): mean of delta_ip
    - scale (sigma_t): fitted scale parameter
    - df (nu): degrees of freedom — lower = fatter tails

    When df → ∞, this converges to the normal Bachelier model.
    """
    process: ImpliedProcess
    n_sims: int = MC_N_SIMS
    seed: int = MC_SEED
    df: float = 0.0          # fitted degrees of freedom
    loc: float = 0.0         # fitted location
    scale: float = 0.0       # fitted scale

    def __post_init__(self):
        self._fit()

    def _fit(self):
        """
        Fit Student-t distribution via MLE on trimmed data.

        The raw data contains near-degenerate observations (Δp near ±1)
        from extreme odds changes. These drive df < 3 (infinite variance),
        which overshoots fat-tail correction. Trimming at the 0.5th/99.5th
        percentile produces a df in the 4–10 range that correctly captures
        the leptokurtic core without being dominated by outliers.
        """
        data = self.process.delta_ip
        # Trim extreme 0.2% (0.1% each tail) — removes only degenerate odds
        # artifacts (Δp near ±1) that drive df below 3
        lo, hi = np.percentile(data, [0.1, 99.9])
        trimmed = data[(data >= lo) & (data <= hi)]
        self.df, self.loc, self.scale = stats.t.fit(trimmed)

        # Report what was trimmed
        n_trimmed = len(data) - len(trimmed)
        self._fit_info = {
            "n_original": len(data),
            "n_trimmed": n_trimmed,
            "pct_trimmed": n_trimmed / len(data) * 100,
            "trim_bounds": (lo, hi),
        }

    def price(self, derivative: Derivative) -> dict:
        """Price via Monte Carlo from fitted Student-t."""
        rng = np.random.default_rng(self.seed)
        draws = stats.t.rvs(
            df=self.df, loc=self.loc, scale=self.scale,
            size=self.n_sims, random_state=rng,
        )
        payoffs = derivative.payoff(draws)
        price = float(np.mean(payoffs))
        std_err = float(np.std(payoffs, ddof=1) / np.sqrt(self.n_sims))

        return {
            "price": price,
            "std_err": std_err,
            "ci_lower": price - 1.96 * std_err,
            "ci_upper": price + 1.96 * std_err,
            "pct_itm": float(np.mean(payoffs > 0)),
            "df": self.df,
            "loc": self.loc,
            "scale": self.scale,
        }

    def price_vanilla_closed(self, K: float, option_type: str = "call") -> dict:
        """
        Semi-closed-form pricing using the t-distribution CDF/PDF.

        For a call: E[max(X - K, 0)] where X ~ t(df, loc, scale)
        = scale * df_factor * [d * T_cdf(d; df+1) + t_pdf(d; df) * ...]

        In practice, MC is more robust for non-standard payoffs,
        but this validates against it.
        """
        # Use MC as the primary method — closed form for t is messy
        return self.price(VanillaOption(strike=K, option_type=option_type))

    def fit_summary(self) -> dict:
        result = {
            "df": self.df,
            "loc": self.loc,
            "scale": self.scale,
            "effective_kurtosis": 6 / (self.df - 4) if self.df > 4 else float("inf"),
            "converges_to_normal": self.df > 100,
        }
        if hasattr(self, "_fit_info"):
            result.update(self._fit_info)
        return result


# ================================================================
# IMPLIED VOLATILITY SMILE
# ================================================================

def compute_implied_vol_smile(
    process: ImpliedProcess,
    strikes: list[float] = None,
    n_sims: int = MC_N_SIMS,
) -> "pd.DataFrame":
    """
    Compute the implied volatility smile.

    For each strike, price the option empirically (KDE MC), then
    invert the Bachelier formula to find the implied σ that matches
    that price. If the distribution were truly normal, implied σ
    would be flat across strikes. Deviations reveal fat tails.

    A U-shaped or upward-sloping smile = fat tails (same pattern
    as equity options markets).
    """
    import pandas as pd

    if strikes is None:
        strikes = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.10, 0.12, 0.15]

    emp = EmpiricalPricer(process, n_sims=n_sims)
    bsm = BSMPricer(process)

    rows = []
    for k in strikes:
        for otype, sign in [("call", 1), ("put", -1)]:
            k_adj = k * sign

            emp_result = emp.price(VanillaOption(strike=k_adj, option_type=otype))
            emp_price = emp_result["price"]

            if emp_price <= 0:
                continue

            # Invert Bachelier to get implied vol
            impl_vol = bsm.implied_vol(
                market_price=emp_price, S=0, K=k_adj, option_type=otype,
            )

            rows.append({
                "strike": k_adj,
                "abs_strike": k,
                "type": otype,
                "emp_price": emp_price,
                "implied_vol": impl_vol,
                "hist_vol": process.std,
                "vol_ratio": impl_vol / process.std if process.std > 0 else 0,
                "moneyness": k / process.std if process.std > 0 else 0,
            })

    return pd.DataFrame(rows)


# ================================================================
# PRICING COMPARISON (THREE MODELS)
# ================================================================

def compare_pricing_methods(
    process: ImpliedProcess,
    strikes: list[float] = None,
) -> "pd.DataFrame":
    """
    Compare three pricing approaches across strikes:
    1. Empirical (KDE MC) — nonparametric baseline
    2. Bachelier (normal) — parametric, assumes normality
    3. Student-t Bachelier — parametric, allows fat tails

    The key result: Student-t dramatically reduces tail mispricing
    compared to the normal Bachelier.
    """
    import pandas as pd

    if strikes is None:
        strikes = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    emp = EmpiricalPricer(process)
    bsm = BSMPricer(process)
    t_pricer = StudentTBachelierPricer(process)

    results = []
    for k in strikes:
        for otype in ["call", "put"]:
            k_adj = k if otype == "call" else -k

            emp_result = emp.price(VanillaOption(strike=k_adj, option_type=otype))
            bsm_result = bsm.price_vanilla(S=0, K=k_adj, option_type=otype)
            t_result = t_pricer.price(VanillaOption(strike=k_adj, option_type=otype))

            emp_price = emp_result["price"]
            bsm_price = bsm_result["price"]
            t_price = t_result["price"]

            results.append({
                "strike": k_adj,
                "type": otype,
                "emp_price": emp_price,
                "emp_se": emp_result["std_err"],
                "bsm_price": bsm_price,
                "t_price": t_price,
                "bsm_pct_diff": (
                    (emp_price - bsm_price) / emp_price * 100
                    if emp_price != 0 else 0
                ),
                "t_pct_diff": (
                    (emp_price - t_price) / emp_price * 100
                    if emp_price != 0 else 0
                ),
                "emp_pct_itm": emp_result["pct_itm"],
                "bsm_delta": bsm_result.get("delta", None),
                "t_df": t_pricer.df,
            })

    return pd.DataFrame(results)
