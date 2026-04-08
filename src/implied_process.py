"""
Extract and characterize the implied probability process.

This module treats the evolution of implied probabilities (opening → closing)
as a stochastic process and estimates its distributional properties — the
foundation for all derivative pricing.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SPORT, LEAGUE, SPORTSBOOK, KDE_BANDWIDTH


@dataclass
class ImpliedProcess:
    """
    Characterization of the line-movement process for a market segment.

    Attributes
    ----------
    delta_ip : np.ndarray
        Array of line movements (close_ip - open_ip) in probability space.
    delta_logit : np.ndarray
        Same movements in logit space (more symmetric).
    abs_delta : np.ndarray
        Absolute line movements.
    segment : str
        Description of the segment (e.g., "soccer / Pinnacle").
    n_obs : int
        Number of observations.
    mean : float
        Mean line movement.
    std : float
        Standard deviation of line movements.
    skew : float
        Skewness.
    kurtosis : float
        Excess kurtosis.
    kde : gaussian_kde
        Fitted kernel density estimator on delta_ip.
    kde_logit : gaussian_kde
        Fitted KDE on delta_logit.
    percentiles : dict
        Key percentiles of the distribution.
    normality_test : dict
        Jarque-Bera and Shapiro-Wilk test results.
    """
    delta_ip: np.ndarray
    delta_logit: np.ndarray
    abs_delta: np.ndarray
    segment: str
    n_obs: int = 0
    mean: float = 0.0
    std: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    kde: Optional[gaussian_kde] = None
    kde_logit: Optional[gaussian_kde] = None
    percentiles: dict = field(default_factory=dict)
    normality_test: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_obs = len(self.delta_ip)
        self.mean = np.mean(self.delta_ip)
        self.std = np.std(self.delta_ip, ddof=1)
        self.skew = float(stats.skew(self.delta_ip))
        self.kurtosis = float(stats.kurtosis(self.delta_ip))

        # Percentiles
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.percentiles = {
            p: np.percentile(self.delta_ip, p) for p in pcts
        }

        # KDE
        bw = KDE_BANDWIDTH if isinstance(KDE_BANDWIDTH, str) else KDE_BANDWIDTH
        self.kde = gaussian_kde(self.delta_ip, bw_method=bw)
        
        valid_logit = self.delta_logit[np.isfinite(self.delta_logit)]
        if len(valid_logit) > 10 and np.std(valid_logit) > 1e-10:
            try:
                self.kde_logit = gaussian_kde(valid_logit, bw_method=bw)
            except np.linalg.LinAlgError:
                self.kde_logit = None

        # Normality tests
        self._test_normality()

    def _test_normality(self):
        """Test whether line movements follow a normal distribution."""
        # Jarque-Bera (works for any n)
        jb_stat, jb_p = stats.jarque_bera(self.delta_ip)

        # Shapiro-Wilk (subsample if n > 5000)
        sample = self.delta_ip
        if len(sample) > 5000:
            rng = np.random.default_rng(42)
            sample = rng.choice(sample, 5000, replace=False)
        sw_stat, sw_p = stats.shapiro(sample)

        # Anderson-Darling
        ad_result = stats.anderson(self.delta_ip, dist="norm")

        self.normality_test = {
            "jarque_bera": {"statistic": jb_stat, "pvalue": jb_p},
            "shapiro_wilk": {"statistic": sw_stat, "pvalue": sw_p},
            "anderson_darling": {
                "statistic": ad_result.statistic,
                "critical_values": dict(zip(
                    [f"{cv}%" for cv in ad_result.significance_level],
                    ad_result.critical_values,
                )),
            },
        }

    def price_empirical(self, payoff_func, n_sims: int = 50_000, seed: int = 42) -> float:
        """
        Price a derivative by Monte Carlo simulation from the empirical KDE.

        Parameters
        ----------
        payoff_func : callable
            Function mapping delta_ip → payoff.
        n_sims : int
            Number of simulation draws.
        seed : int
            Random seed.

        Returns
        -------
        float
            Expected payoff (empirical price).
        """
        rng = np.random.default_rng(seed)
        draws = self.kde.resample(n_sims, seed=rng).flatten()
        payoffs = payoff_func(draws)
        return float(np.mean(payoffs))

    def vol_surface(
        self,
        df: pd.DataFrame,
        time_col: str = "line_duration_hours",
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Estimate a volatility surface: sigma as a function of time-to-expiry.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain delta_ip_home and a time column.
        time_col : str
            Column measuring time the line was open.
        n_bins : int
            Number of duration bins.

        Returns
        -------
        pd.DataFrame
            Columns: time_bin, mean_duration, sigma, n_obs
        """
        if time_col not in df.columns:
            raise ValueError(f"Column {time_col} not found. "
                             "Compute line duration from timestamps first.")

        df = df.copy()
        df["time_bin"] = pd.qcut(df[time_col], n_bins, duplicates="drop")
        
        vol_surface = df.groupby("time_bin", observed=True).agg(
            mean_duration=(time_col, "mean"),
            sigma=("delta_ip_home", "std"),
            n_obs=("delta_ip_home", "count"),
        ).reset_index()

        return vol_surface

    def summary_table(self) -> pd.DataFrame:
        """Return a summary statistics table."""
        rows = {
            "N": self.n_obs,
            "Mean": self.mean,
            "Std Dev": self.std,
            "Skewness": self.skew,
            "Excess Kurtosis": self.kurtosis,
            "Min": np.min(self.delta_ip),
            "P1": self.percentiles[1],
            "P5": self.percentiles[5],
            "Median": self.percentiles[50],
            "P95": self.percentiles[95],
            "P99": self.percentiles[99],
            "Max": np.max(self.delta_ip),
            "JB stat": self.normality_test["jarque_bera"]["statistic"],
            "JB p-value": self.normality_test["jarque_bera"]["pvalue"],
        }
        return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])


def extract_implied_process(
    df: pd.DataFrame,
    outcome: str = "home",
    segment_by: Optional[str] = None,
) -> dict | ImpliedProcess:
    """
    Extract the implied probability process from betting data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have delta_ip_{outcome} and delta_logit_{outcome} columns
        (created by data_loader).
    outcome : str
        Which outcome to analyze: "home", "away", or "draw".
    segment_by : str, optional
        Column to segment by (e.g., "sport", "bookmaker_name").
        If None, returns a single ImpliedProcess for the whole dataset.

    Returns
    -------
    ImpliedProcess or dict[str, ImpliedProcess]
    """
    delta_col = f"delta_ip_{outcome}"
    logit_col = f"delta_logit_{outcome}"
    abs_col = f"abs_delta_ip_{outcome}" if f"abs_delta_ip_{outcome}" in df.columns else None

    # Drop NaN/inf
    mask = df[delta_col].notna() & np.isfinite(df[delta_col])
    if logit_col in df.columns:
        mask &= df[logit_col].notna()
    df_valid = df[mask].copy()

    if segment_by is None:
        abs_vals = df_valid[abs_col].values if abs_col else np.abs(df_valid[delta_col].values)
        return ImpliedProcess(
            delta_ip=df_valid[delta_col].values,
            delta_logit=df_valid[logit_col].values if logit_col in df_valid.columns else np.array([]),
            abs_delta=abs_vals,
            segment="all",
        )

    # Segmented
    processes = {}
    for name, group in df_valid.groupby(segment_by):
        if len(group) < 50:  # skip tiny segments
            continue
        abs_vals = group[abs_col].values if abs_col else np.abs(group[delta_col].values)
        processes[name] = ImpliedProcess(
            delta_ip=group[delta_col].values,
            delta_logit=group[logit_col].values if logit_col in group.columns else np.array([]),
            abs_delta=abs_vals,
            segment=str(name),
        )

    return processes


def compare_processes(processes: dict) -> pd.DataFrame:
    """
    Compare ImpliedProcess objects across segments.

    Returns a DataFrame with summary statistics for each segment.
    """
    rows = []
    for name, proc in processes.items():
        rows.append({
            "segment": name,
            "n_obs": proc.n_obs,
            "mean": proc.mean,
            "std": proc.std,
            "skew": proc.skew,
            "kurtosis": proc.kurtosis,
            "p5": proc.percentiles[5],
            "median": proc.percentiles[50],
            "p95": proc.percentiles[95],
            "jb_pvalue": proc.normality_test["jarque_bera"]["pvalue"],
        })
    return pd.DataFrame(rows).sort_values("n_obs", ascending=False)


def extract_regime_processes(
    df: pd.DataFrame,
    outcome: str = "home",
    paspa_date: str = "2018-05-14",
    covid_start: str = "2020-03-11",
    covid_end: str = "2020-07-23",
) -> dict:
    """
    Extract implied processes split by PASPA regime.

    Returns dict with keys: pre_paspa, post_paspa, post_paspa_ex_covid,
    covid, and optionally by year.
    """
    delta_col = f"delta_ip_{outcome}"
    logit_col = f"delta_logit_{outcome}"

    df = df.copy()
    date_col = "match_date"
    df[date_col] = pd.to_datetime(df[date_col])

    paspa = pd.Timestamp(paspa_date)
    covid_s = pd.Timestamp(covid_start)
    covid_e = pd.Timestamp(covid_end)

    regimes = {
        "pre_paspa": df[df[date_col] < paspa],
        "post_paspa": df[df[date_col] >= paspa],
        "post_paspa_ex_covid": df[
            (df[date_col] >= paspa) &
            ~((df[date_col] >= covid_s) & (df[date_col] <= covid_e))
        ],
        "covid": df[(df[date_col] >= covid_s) & (df[date_col] <= covid_e)],
    }

    processes = {}
    for name, subset in regimes.items():
        mask = subset[delta_col].notna() & np.isfinite(subset[delta_col])
        valid = subset[mask]
        if len(valid) < 50:
            continue

        logit_vals = valid[logit_col].values if logit_col in valid.columns else np.array([])

        processes[name] = ImpliedProcess(
            delta_ip=valid[delta_col].values,
            delta_logit=logit_vals,
            abs_delta=np.abs(valid[delta_col].values),
            segment=name,
        )

    return processes


def regime_pricing_comparison(
    regime_processes: dict,
    strikes: list = None,
) -> pd.DataFrame:
    """
    Compare derivative prices across PASPA regimes.

    Tests whether the same derivative costs different amounts
    pre vs. post PASPA — i.e., did the regulatory change affect
    the pricing of line-movement risk?
    """
    from .pricing import EmpiricalPricer, BSMPricer
    from .derivatives import VanillaOption, Straddle

    if strikes is None:
        strikes = [0.02, 0.05]

    rows = []
    for regime_name, proc in regime_processes.items():
        emp = EmpiricalPricer(proc, n_sims=30_000)
        bsm = BSMPricer(proc)

        # ATM straddle price (pure vol measure)
        straddle = Straddle(strike=0.0)
        straddle_result = emp.price(straddle)

        row = {
            "regime": regime_name,
            "n_obs": proc.n_obs,
            "sigma": proc.std,
            "skew": proc.skew,
            "kurtosis": proc.kurtosis,
            "straddle_price": straddle_result["price"],
        }

        # Vanilla options at each strike
        for k in strikes:
            call = VanillaOption(strike=k, option_type="call")
            put = VanillaOption(strike=-k, option_type="put")
            call_r = emp.price(call)
            put_r = emp.price(put)
            bsm_call = bsm.price_vanilla(S=0, K=k, option_type="call")

            row[f"call_K{k:.2f}"] = call_r["price"]
            row[f"put_K{k:.2f}"] = put_r["price"]
            row[f"bsm_call_K{k:.2f}"] = bsm_call["price"]
            row[f"mispricing_K{k:.2f}"] = (
                (call_r["price"] - bsm_call["price"]) / call_r["price"] * 100
                if call_r["price"] != 0 else 0
            )

        rows.append(row)

    return pd.DataFrame(rows)
