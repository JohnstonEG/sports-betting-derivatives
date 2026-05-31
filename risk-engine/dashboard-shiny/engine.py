"""
engine.py
=========
Python <-> C# bridge for the Shiny dashboard.

Shells out to the existing RiskEngine.Cli (C# / .NET 10), tells it to write a
JSON risk report, and returns the parsed report as a dict. The Shiny app
treats this as its data source -- exactly the same JSON contract the Blazor
dashboard consumes via shared in-process types.

Architecture:
    Shiny (Python)  ->  subprocess  ->  RiskEngine.Cli (C#)  ->  JSON  ->  Shiny
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# --- Project layout -----------------------------------------------------------
# This file lives at risk-engine/dashboard-shiny/engine.py
HERE = Path(__file__).resolve().parent
RISK_ENGINE_ROOT = HERE.parent
CLI_PROJECT = RISK_ENGINE_ROOT / "src" / "RiskEngine.Cli" / "RiskEngine.Cli.csproj"
DATA_DIR = RISK_ENGINE_ROOT / "data"

_EXE_NAME = "RiskEngine.Cli.exe" if sys.platform == "win32" else "RiskEngine.Cli"
CLI_BIN_RELEASE = RISK_ENGINE_ROOT / "src" / "RiskEngine.Cli" / "bin" / "Release" / "net10.0" / _EXE_NAME
CLI_BIN_DEBUG = RISK_ENGINE_ROOT / "src" / "RiskEngine.Cli" / "bin" / "Debug" / "net10.0" / _EXE_NAME

CACHE_DIR = HERE / ".cache"
DEFAULT_REPORT = CACHE_DIR / "risk-report.json"


class EngineError(RuntimeError):
    """Raised when the C# engine cannot be invoked or fails."""


# --- Public API ---------------------------------------------------------------
def run_engine(
    model: str = "StudentT",
    paths: int = 50_000,
    seed: int = 20240517,
    portfolio_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    timeout: int = 120,
) -> dict:
    """Run the C# risk engine and return the parsed risk report."""
    portfolio_path = Path(portfolio_path) if portfolio_path else DATA_DIR / "portfolio.json"
    calibration_path = Path(calibration_path) if calibration_path else DATA_DIR / "calibration.json"
    out_path = Path(out_path) if out_path else DEFAULT_REPORT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(model, paths, seed, portfolio_path, calibration_path, out_path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as e:
        raise EngineError(
            f"Could not execute the C# engine ({e.filename}). "
            "Ensure either a built RiskEngine.Cli exists under bin/, or "
            "the .NET SDK is on PATH (`dotnet --info`)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise EngineError(
            f"RiskEngine.Cli timed out after {timeout}s. "
            "Try reducing the Monte Carlo path count."
        ) from e

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        # The CLI prints a friendly error to stderr; surface it.
        detail = stderr or stdout[-400:] or "(no output)"
        raise EngineError(
            f"RiskEngine.Cli failed (exit {result.returncode}).\n{detail}"
        )

    if not out_path.exists():
        raise EngineError(
            f"RiskEngine.Cli reported success but did not produce {out_path}."
        )

    try:
        with open(out_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise EngineError(f"Engine output is not valid JSON: {e}") from e


def load_portfolio(path: Optional[Path] = None) -> dict:
    """Load the portfolio JSON used by the engine (needed for payoff diagrams)."""
    path = Path(path) if path else DATA_DIR / "portfolio.json"
    if not path.exists():
        raise EngineError(f"Portfolio file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def engine_status() -> str:
    """Return a short string describing how the engine will be invoked."""
    exe = _find_built_exe()
    if exe is not None:
        return f"built executable: {exe}"
    if shutil.which("dotnet") is not None:
        return "dotnet run (slower; build the engine for faster startup)"
    return "engine NOT available - run `dotnet build` in risk-engine/"


# --- Helpers ------------------------------------------------------------------
def _build_command(
    model: str, paths: int, seed: int,
    portfolio: Path, calibration: Path, out_path: Path,
) -> list[str]:
    """Build the CLI command. Prefer the built exe; fall back to `dotnet run`."""
    exe = _find_built_exe()
    base_args = [
        "--portfolio", str(portfolio),
        "--calibration", str(calibration),
        "--model", model,
        "--paths", str(paths),
        "--seed", str(seed),
        "--out", str(out_path),
    ]
    if exe is not None:
        return [str(exe), *base_args]

    if shutil.which("dotnet") is None:
        raise EngineError(
            "No built RiskEngine.Cli found and 'dotnet' is not on PATH. "
            "From the risk-engine folder, run `dotnet build` and try again."
        )
    return [
        "dotnet", "run",
        "--project", str(CLI_PROJECT),
        "--no-launch-profile",
        "--",
        *base_args,
    ]


def _find_built_exe() -> Optional[Path]:
    for candidate in (CLI_BIN_RELEASE, CLI_BIN_DEBUG):
        if candidate.exists():
            return candidate
    return None
