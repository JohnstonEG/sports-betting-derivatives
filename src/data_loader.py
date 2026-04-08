"""
Data loading and validation for the sports betting dataset.

Loads cleaned_data.parquet and prepares the implied probability fields
needed for derivative construction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLEANED_DATA, ANALYSIS_DATA, FILTERED_SAMPLE,
    CLOSE_ODDS, OPEN_ODDS, CLOSE_MARGIN, OPEN_MARGIN,
    MATCH_ID, SPORT, LEAGUE, SPORTSBOOK, BOOKMAKER,
    MATCH_DATE, HOME_WIN, IS_DRAW,
    TWO_WAY_SPORTS, THREE_WAY_SPORTS,
    odds_to_prob,
)


def load_data(
    path: Optional[Path] = None,
    sports: Optional[list] = None,
    bookmakers: Optional[list] = None,
    require_opening: bool = True,
    min_matches_per_sport: int = 100,
) -> pd.DataFrame:
    """
    Load and validate the cleaned betting data.

    Parameters
    ----------
    path : Path, optional
        Path to parquet file. Defaults to CLEANED_DATA from config.
    sports : list, optional
        Filter to these sports only.
    bookmakers : list, optional
        Filter to these bookmaker IDs only.
    require_opening : bool
        If True, drop rows missing opening odds (needed for derivatives).
    min_matches_per_sport : int
        Drop sports with fewer unique matches than this.

    Returns
    -------
    pd.DataFrame
        Validated data with implied probability columns added.
    """
    path = path or CLEANED_DATA
    print(f"Loading data from {path}...")
    if str(path).endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_parquet(path)
    n_raw = len(df)
    print(f"  Raw: {n_raw:,} rows, {df[MATCH_ID].nunique():,} matches")

    # --- Ensure datetime ---
    if MATCH_DATE in df.columns:
        df[MATCH_DATE] = pd.to_datetime(df[MATCH_DATE])
    
    # --- Filters ---
    if sports:
        df = df[df[SPORT].isin(sports)]
        print(f"  After sport filter: {len(df):,}")

    if bookmakers:
        df = df[df[SPORTSBOOK].astype(str).isin([str(b) for b in bookmakers])]
        print(f"  After bookmaker filter: {len(df):,}")

    if require_opening:
        has_open = df[OPEN_ODDS["home"]].notna() & df[OPEN_ODDS["away"]].notna()
        df = df[has_open].copy()
        print(f"  After requiring opening odds: {len(df):,}")

    # --- Drop thin sports ---
    sport_counts = df.groupby(SPORT)[MATCH_ID].nunique()
    valid_sports = sport_counts[sport_counts >= min_matches_per_sport].index
    df = df[df[SPORT].isin(valid_sports)].copy()

    # --- Compute implied probabilities ---
    df = _add_implied_probabilities(df)

    # --- Compute line movements ---
    df = _add_line_movements(df)

    # --- Summary ---
    print(f"\n  Final: {len(df):,} rows, {df[MATCH_ID].nunique():,} matches")
    print(f"  Sports: {df[SPORT].nunique()} | "
          f"Date range: {df[MATCH_DATE].min().date()} to {df[MATCH_DATE].max().date()}")
    if "bookmaker_name" in df.columns:
        print(f"  Bookmakers: {df['bookmaker_name'].nunique()}")

    return df


def _add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert decimal odds to implied probabilities (raw and normalized).

    For 2-way sports: home + away
    For 3-way sports: home + draw + away
    """
    # Raw implied probs
    df["close_ip_home"] = odds_to_prob(df[CLOSE_ODDS["home"]])
    df["close_ip_away"] = odds_to_prob(df[CLOSE_ODDS["away"]])
    df["open_ip_home"] = odds_to_prob(df[OPEN_ODDS["home"]])
    df["open_ip_away"] = odds_to_prob(df[OPEN_ODDS["away"]])

    # Draw (may be NaN for 2-way sports)
    if CLOSE_ODDS["draw"] in df.columns:
        df["close_ip_draw"] = odds_to_prob(
            df[CLOSE_ODDS["draw"]].where(df[CLOSE_ODDS["draw"]].notna(), np.nan)
        )
        df["open_ip_draw"] = odds_to_prob(
            df[OPEN_ODDS["draw"]].where(df[OPEN_ODDS["draw"]].notna(), np.nan)
        )
    else:
        df["close_ip_draw"] = np.nan
        df["open_ip_draw"] = np.nan

    # Overround (sum of raw implied probs)
    df["close_overround"] = df["close_ip_home"] + df["close_ip_away"]
    df["open_overround"] = df["open_ip_home"] + df["open_ip_away"]

    has_draw = df["close_ip_draw"].notna()
    df.loc[has_draw, "close_overround"] += df.loc[has_draw, "close_ip_draw"]
    df.loc[df["open_ip_draw"].notna(), "open_overround"] += df.loc[
        df["open_ip_draw"].notna(), "open_ip_draw"
    ]

    # Normalized (vig-free) implied probs
    for prefix in ["close", "open"]:
        overround = df[f"{prefix}_overround"]
        for outcome in ["home", "away", "draw"]:
            raw_col = f"{prefix}_ip_{outcome}"
            norm_col = f"{prefix}_norm_ip_{outcome}"
            df[norm_col] = df[raw_col] / overround

    return df


def _add_line_movements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute line movement metrics in probability space.
    
    These are the "returns" on the underlying asset — the foundation 
    for all derivative pricing.
    """
    # Raw movement in implied probability
    df["delta_ip_home"] = df["close_ip_home"] - df["open_ip_home"]
    df["delta_ip_away"] = df["close_ip_away"] - df["open_ip_away"]
    df["delta_ip_draw"] = df["close_ip_draw"] - df["open_ip_draw"]

    # Normalized movement
    df["delta_norm_ip_home"] = df["close_norm_ip_home"] - df["open_norm_ip_home"]
    df["delta_norm_ip_away"] = df["close_norm_ip_away"] - df["open_norm_ip_away"]

    # Absolute movement (useful for straddle pricing)
    df["abs_delta_ip_home"] = df["delta_ip_home"].abs()

    # Log-odds movement (more symmetric)
    df["close_logit_home"] = np.log(df["close_ip_home"] / (1 - df["close_ip_home"]))
    df["open_logit_home"] = np.log(df["open_ip_home"] / (1 - df["open_ip_home"]))
    df["delta_logit_home"] = df["close_logit_home"] - df["open_logit_home"]

    # Margin change
    df["delta_margin"] = df[CLOSE_MARGIN] - df[OPEN_MARGIN]

    return df


def load_analysis_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the monthly-aggregated analysis data."""
    path = path or ANALYSIS_DATA
    if str(path).endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for reporting."""
    summary = {
        "n_rows": len(df),
        "n_matches": df[MATCH_ID].nunique(),
        "n_sports": df[SPORT].nunique(),
        "sports": sorted(df[SPORT].unique()),
        "date_min": df[MATCH_DATE].min(),
        "date_max": df[MATCH_DATE].max(),
        "mean_close_margin": df[CLOSE_MARGIN].mean(),
        "mean_open_margin": df[OPEN_MARGIN].mean() if OPEN_MARGIN in df.columns else None,
        "pct_with_opening": df[OPEN_ODDS["home"]].notna().mean(),
        "mean_delta_ip_home": df["delta_ip_home"].mean() if "delta_ip_home" in df.columns else None,
        "std_delta_ip_home": df["delta_ip_home"].std() if "delta_ip_home" in df.columns else None,
    }
    return summary


if __name__ == "__main__":
    df = load_data()
    s = data_summary(df)
    print("\n=== DATA SUMMARY ===")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: {', '.join(v)}")
        else:
            print(f"  {k}: {v}")
