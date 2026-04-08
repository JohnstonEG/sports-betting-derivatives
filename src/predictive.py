"""
Predictive analysis: does current line movement predict future movement?

This module tests for autocorrelation, momentum, and mean-reversion
in line movements — providing microstructure evidence on information
flow dynamics.

Even null results are informative:
- No autocorrelation → efficient market (consistent with semi-strong EMH)
- Positive autocorrelation → momentum (information cascades / herding)
- Negative autocorrelation → mean-reversion (overreaction correction)
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MATCH_ID, SPORT, SPORTSBOOK, MATCH_DATE


def compute_autocorrelation(
    df: pd.DataFrame,
    max_lag: int = 20,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute autocorrelation of line movements at various lags.

    Within each group (e.g., sport-bookmaker pair), sort by date
    and compute the correlation between Δp_t and Δp_{t+k}.

    Parameters
    ----------
    df : pd.DataFrame
        Must have delta_ip_home and match_date.
    max_lag : int
        Maximum lag to compute.
    group_by : str, optional
        Column to group by (e.g., "sport"). If None, compute overall.

    Returns
    -------
    pd.DataFrame with columns: lag, autocorr, se, t_stat, p_value, [group]
    """
    def _acf_for_series(delta: np.ndarray, max_lag: int) -> list:
        n = len(delta)
        rows = []
        for lag in range(1, max_lag + 1):
            if n - lag < 30:
                break
            x = delta[:-lag]
            y = delta[lag:]
            corr, p = stats.pearsonr(x, y)
            se = 1.0 / np.sqrt(n - lag)  # Bartlett SE
            rows.append({
                "lag": lag,
                "autocorr": corr,
                "se": se,
                "t_stat": corr / se,
                "p_value": p,
                "n": n - lag,
            })
        return rows

    df = df.sort_values(MATCH_DATE).copy()
    delta = df["delta_ip_home"].dropna().values

    if group_by is None:
        rows = _acf_for_series(delta, max_lag)
        return pd.DataFrame(rows)

    all_rows = []
    for name, group in df.groupby(group_by):
        group = group.sort_values(MATCH_DATE)
        d = group["delta_ip_home"].dropna().values
        if len(d) < 100:
            continue
        group_rows = _acf_for_series(d, max_lag)
        for r in group_rows:
            r[group_by] = name
        all_rows.extend(group_rows)

    return pd.DataFrame(all_rows)


def predictive_regression(
    df: pd.DataFrame,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Run simple predictive regressions:
        Δp_{t+1} = α + β * Δp_t + ε

    Tests whether current line movement predicts the next.

    Also tests:
        |Δp_{t+1}| = α + β * |Δp_t| + ε
    (volatility clustering)

    Returns
    -------
    pd.DataFrame with regression results.
    """
    if lags is None:
        lags = [1, 2, 5, 10]

    df = df.sort_values(MATCH_DATE).copy()
    delta = df["delta_ip_home"].dropna().values
    abs_delta = np.abs(delta)
    n = len(delta)

    results = []
    for lag in lags:
        if n - lag < 50:
            continue

        # Level prediction
        x = delta[:-lag]
        y = delta[lag:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        results.append({
            "target": "Δp",
            "lag": lag,
            "beta": slope,
            "beta_se": std_err,
            "t_stat": slope / std_err if std_err > 0 else 0,
            "p_value": p_value,
            "r_squared": r_value ** 2,
            "n": n - lag,
        })

        # Volatility clustering
        ax = abs_delta[:-lag]
        ay = abs_delta[lag:]
        slope_v, intercept_v, r_v, p_v, se_v = stats.linregress(ax, ay)
        results.append({
            "target": "|Δp|",
            "lag": lag,
            "beta": slope_v,
            "beta_se": se_v,
            "t_stat": slope_v / se_v if se_v > 0 else 0,
            "p_value": p_v,
            "r_squared": r_v ** 2,
            "n": n - lag,
        })

    return pd.DataFrame(results)


def ljung_box_test(
    df: pd.DataFrame,
    max_lag: int = 20,
) -> dict:
    """
    Ljung-Box test for joint significance of autocorrelations.

    H0: No autocorrelation up to lag k.
    """
    delta = df.sort_values(MATCH_DATE)["delta_ip_home"].dropna().values
    n = len(delta)

    # Compute autocorrelations
    acf_vals = []
    for k in range(1, max_lag + 1):
        if n - k < 30:
            break
        corr = np.corrcoef(delta[:-k], delta[k:])[0, 1]
        acf_vals.append(corr)

    # Ljung-Box Q statistic
    q_stat = n * (n + 2) * sum(
        r**2 / (n - k) for k, r in enumerate(acf_vals, 1)
    )
    df_chi2 = len(acf_vals)
    p_value = 1 - stats.chi2.cdf(q_stat, df_chi2)

    return {
        "Q_stat": q_stat,
        "df": df_chi2,
        "p_value": p_value,
        "reject_H0": p_value < 0.05,
        "interpretation": (
            "Significant autocorrelation detected"
            if p_value < 0.05
            else "No significant autocorrelation (consistent with efficiency)"
        ),
    }


def run_predictive_analysis(
    df: pd.DataFrame,
) -> dict:
    """
    Run all predictive analyses and return results dict.

    IMPORTANT: The raw data has multiple bookmakers per match.
    Adjacent rows sorted by date may be the same match from different
    bookmakers, creating spurious autocorrelation. We deduplicate to
    one observation per match (using Pinnacle as the reference, or
    the first available bookmaker) before computing autocorrelation.
    """
    # Deduplicate: one observation per match
    # Prefer Pinnacle (sharpest book), fall back to first available
    df_sorted = df.sort_values(MATCH_DATE).copy()

    if "bookmaker_name" in df_sorted.columns:
        # Try Pinnacle first
        pinnacle = df_sorted[df_sorted["bookmaker_name"] == "Pinnacle"]
        if len(pinnacle) > 10000:
            df_dedup = pinnacle.drop_duplicates(subset=[MATCH_ID], keep="first")
            book_used = "Pinnacle"
        else:
            df_dedup = df_sorted.drop_duplicates(subset=[MATCH_ID], keep="first")
            book_used = "first available"
    else:
        df_dedup = df_sorted.drop_duplicates(subset=[MATCH_ID], keep="first")
        book_used = "first available"

    print(f"  Deduplicating to one obs per match ({book_used}): "
          f"{len(df_sorted):,} → {len(df_dedup):,}")

    print("  Computing autocorrelation structure...")
    acf_df = compute_autocorrelation(df_dedup, max_lag=20)

    print("  Running predictive regressions...")
    reg_df = predictive_regression(df_dedup, lags=[1, 2, 5, 10, 20])

    print("  Ljung-Box test...")
    lb = ljung_box_test(df_dedup)

    # Interpretation
    print(f"\n  Ljung-Box Q = {lb['Q_stat']:.1f} (p = {lb['p_value']:.4f})")
    print(f"  → {lb['interpretation']}")

    if len(acf_df) > 0:
        lag1 = acf_df[acf_df["lag"] == 1].iloc[0]
        print(f"\n  Lag-1 autocorrelation: {lag1['autocorr']:.4f} "
              f"(t = {lag1['t_stat']:.2f}, p = {lag1['p_value']:.4f})")

        if abs(lag1["autocorr"]) > 0.05 and lag1["p_value"] < 0.01:
            if lag1["autocorr"] > 0:
                print(f"  → MOMENTUM in line movements (information cascades)")
            else:
                print(f"  → MEAN-REVERSION in line movements (overreaction correction)")
        elif abs(lag1["autocorr"]) > 0.01 and lag1["p_value"] < 0.01:
            if lag1["autocorr"] > 0:
                print(f"  → Weak momentum (statistically significant, economically small)")
            else:
                print(f"  → Weak mean-reversion (statistically significant, economically small)")
        else:
            print(f"  → No economically significant predictability (efficient market)")

    # Volatility clustering
    vol_results = reg_df[reg_df["target"] == "|Δp|"]
    if len(vol_results) > 0:
        lag1_vol = vol_results[vol_results["lag"] == 1].iloc[0]
        print(f"\n  Volatility clustering (lag-1 |Δp| → |Δp|):")
        print(f"    β = {lag1_vol['beta']:.4f} (t = {lag1_vol['t_stat']:.2f})")
        if lag1_vol["p_value"] < 0.01 and lag1_vol["beta"] > 0.05:
            print(f"    → Significant volatility clustering (like financial markets)")
            print(f"    → Periods of high line-movement volatility cluster together")
        elif lag1_vol["p_value"] < 0.01:
            print(f"    → Statistically significant but economically weak clustering")
        else:
            print(f"    → No significant volatility clustering")

    return {
        "autocorrelation": acf_df,
        "regressions": reg_df,
        "ljung_box": lb,
    }
