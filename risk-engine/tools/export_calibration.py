#!/usr/bin/env python3
"""
export_calibration.py
=====================
Bridge from the Python research pipeline to the C# risk engine.

This script reads the line-movement series  delta_p = p_close - p_open  from the
project's data, computes the distributional parameters of delta_p (full sample
and per market regime), and writes ``risk-engine/data/calibration.json`` -- the
file consumed by RiskEngine.Core (CalibrationLoader / CalibrationSet).

It is the concrete realisation of the resume claim "integrating Python-generated
derivative valuations": Python calibrates, C# consumes and runs the risk engine.

Usage
-----
    # If delta_p is already a column in your data:
    python export_calibration.py --data ../../data/analysis_data.csv \\
        --delta-col delta_p --date-col match_date

    # If you only have decimal odds, compute delta_p = 1/close - 1/open:
    python export_calibration.py --data ../../data/cleaned_data.csv \\
        --open-col odds_open --close-col odds_close --date-col match_date

    # No source data handy? Write the documented sample calibration:
    python export_calibration.py --use-defaults

Output JSON keys are camelCase to match the C# CalibrationSet model exactly.

Requires: numpy, scipy, pandas  (already dependencies of the parent project).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# --- Market-history boundaries -------------------------------------------------
PASPA_REPEAL = dt.date(2018, 5, 14)   # Murphy v. NCAA struck down PASPA
COVID_START = dt.date(2020, 3, 1)
COVID_END = dt.date(2021, 6, 30)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "calibration.json"


# --- Documented fallback (full-sample statistics from the project README) ------
def default_calibration() -> dict:
    """The documented sample calibration, used when no source data is supplied."""
    return {
        "_comment": "Documented sample calibration written by export_calibration.py "
                    "--use-defaults. Re-run against the source data to refresh.",
        "source": "sports-derivatives Python pipeline - documented full-sample statistics",
        "generatedUtc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sampleSize": 2522019,
        "distributionModel": "StudentT",
        "base": {
            "name": "Full Sample", "mean": -0.0013, "sigma": 0.0526,
            "skewness": 0.171, "excessKurtosis": 5.46, "studentTDof": 2.70,
            "weight": 1.0,
            "notes": "Full-sample distribution of delta_p across all sports and bookmakers.",
        },
        "regimes": [
            {"name": "Pre-PASPA", "mean": -0.0010, "sigma": 0.041, "skewness": 0.205,
             "excessKurtosis": 6.72, "studentTDof": 2.55, "weight": 0.40,
             "notes": "Before the May 2018 PASPA repeal."},
            {"name": "Post-PASPA", "mean": -0.0013, "sigma": 0.057, "skewness": 0.158,
             "excessKurtosis": 4.77, "studentTDof": 2.85, "weight": 0.42,
             "notes": "After the PASPA repeal, excluding COVID."},
            {"name": "COVID", "mean": -0.0015, "sigma": 0.067, "skewness": 0.121,
             "excessKurtosis": 3.95, "studentTDof": 3.05, "weight": 0.18,
             "notes": "2020-2021 disrupted sporting calendar."},
        ],
    }


# --- Statistics ---------------------------------------------------------------
def fit_student_t_dof(x) -> float:
    """Fit Student-t degrees of freedom by MLE on lightly trimmed data."""
    import numpy as np
    from scipy import stats

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return 5.0
    try:
        lo, hi = np.percentile(x, [0.5, 99.5])
        trimmed = x[(x >= lo) & (x <= hi)]
        df, _, _ = stats.t.fit(trimmed)
        # Keep df in a band that supports a finite-variance sampler in C#.
        return float(min(max(df, 2.1), 30.0))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Student-t fit failed ({exc}); using dof = 5.0")
        return 5.0


def describe(x, name: str, weight: float, notes: str) -> dict:
    """Compute a RegimeParameters record for one delta_p sample."""
    import numpy as np
    from scipy import stats

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"Regime '{name}' has no finite observations.")
    return {
        "name": name,
        "mean": round(float(np.mean(x)), 8),
        "sigma": round(float(np.std(x, ddof=1)), 8),
        "skewness": round(float(stats.skew(x)), 6),
        "excessKurtosis": round(float(stats.kurtosis(x, fisher=True)), 6),
        "studentTDof": round(fit_student_t_dof(x), 4),
        "weight": round(float(weight), 6),
        "notes": notes,
    }


# --- Data loading -------------------------------------------------------------
def load_delta_p(args):
    """Return a DataFrame with a 'delta_p' column and an optional 'date' column."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(args.data)
    print(f"  loaded {len(df):,} rows from {args.data}")

    if args.delta_col and args.delta_col in df.columns:
        delta = pd.to_numeric(df[args.delta_col], errors="coerce")
    elif args.open_col and args.close_col:
        if args.open_col not in df.columns or args.close_col not in df.columns:
            sys.exit(f"  [error] odds columns '{args.open_col}'/'{args.close_col}' not found. "
                     f"Available: {list(df.columns)}")
        odds_open = pd.to_numeric(df[args.open_col], errors="coerce")
        odds_close = pd.to_numeric(df[args.close_col], errors="coerce")
        # Implied probability from decimal odds, then the line move.
        delta = (1.0 / odds_close) - (1.0 / odds_open)
    else:
        sys.exit("  [error] supply --delta-col, or both --open-col and --close-col.")

    out = pd.DataFrame({"delta_p": delta})
    if args.date_col and args.date_col in df.columns:
        out["date"] = pd.to_datetime(df[args.date_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_p"])
    print(f"  usable delta_p observations: {len(out):,}")
    return out


def split_regimes(df) -> list[dict]:
    """Split observations into Pre-PASPA / Post-PASPA / COVID regimes by date."""
    if "date" not in df.columns:
        print("  [info] no date column - skipping regime split (base regime only).")
        return []

    d = df.dropna(subset=["date"]).copy()
    if d.empty:
        return []

    dates = d["date"].dt.date
    n = len(d)
    pre = d[dates < PASPA_REPEAL]["delta_p"]
    covid = d[(dates >= COVID_START) & (dates <= COVID_END)]["delta_p"]
    post = d[(dates >= PASPA_REPEAL) & ~((dates >= COVID_START) & (dates <= COVID_END))]["delta_p"]

    regimes = []
    for name, sample, notes in [
        ("Pre-PASPA", pre, "Before the May 2018 PASPA repeal."),
        ("Post-PASPA", post, "After the PASPA repeal, excluding the COVID window."),
        ("COVID", covid, "2020-2021 disrupted sporting calendar."),
    ]:
        if len(sample) >= 100:
            regimes.append(describe(sample, name, len(sample) / n, notes))
        else:
            print(f"  [info] regime '{name}' has too few rows ({len(sample)}) - skipped.")
    return regimes


# --- Main ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="Path to a CSV containing delta_p or decimal odds.")
    parser.add_argument("--delta-col", help="Column already holding delta_p.")
    parser.add_argument("--open-col", help="Decimal opening-odds column.")
    parser.add_argument("--close-col", help="Decimal closing-odds column.")
    parser.add_argument("--date-col", help="Date/timestamp column (enables the regime split).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output path (default: {DEFAULT_OUT}).")
    parser.add_argument("--use-defaults", action="store_true",
                        help="Write the documented sample calibration without reading data.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_defaults or not args.data:
        if not args.use_defaults:
            print("No --data supplied; writing the documented sample calibration.")
        payload = default_calibration()
    else:
        print("Computing calibration from source data...")
        df = load_delta_p(args)
        base = describe(df["delta_p"], "Full Sample", 1.0,
                        "Full-sample distribution of delta_p.")
        regimes = split_regimes(df)
        payload = {
            "_comment": "Generated by tools/export_calibration.py from source data.",
            "source": f"sports-derivatives Python pipeline - exported from {Path(args.data).name}",
            "generatedUtc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sampleSize": int(len(df)),
            "distributionModel": "StudentT",
            "base": base,
            "regimes": regimes if regimes else [base],
        }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nWrote calibration -> {out_path}")
    print(f"  base: mean={payload['base']['mean']}, sigma={payload['base']['sigma']}, "
          f"dof={payload['base']['studentTDof']}, regimes={len(payload['regimes'])}")
    print("The C# engine will pick this up automatically (RiskEngine.Core.CalibrationLoader).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
