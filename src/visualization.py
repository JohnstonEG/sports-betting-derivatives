"""
Publication-quality visualizations for the sports derivatives project.

Generates figures for:
1. Line movement distribution (empirical vs. normal)
2. Derivative payoff diagrams
3. Pricing comparison (empirical vs. BSM)
4. Backtest performance (cumulative P&L, drawdowns)
5. Efficient frontier comparison
6. Volatility surface
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import Optional

from .implied_process import ImpliedProcess
from .derivatives import Derivative, VanillaOption, Straddle, Strangle, Butterfly
from .backtester import BacktestResult

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT, COLOR_PALETTE

# Style setup
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def save_fig(fig, name: str, dpi: int = FIGURE_DPI):
    """Save figure to output directory."""
    path = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ================================================================
# 1. LINE MOVEMENT DISTRIBUTION
# ================================================================

def plot_line_movement_distribution(
    process: ImpliedProcess,
    title: str = "Distribution of Line Movements (Implied Probability)",
    filename: str = "line_movement_dist",
) -> None:
    """
    Plot the empirical distribution of line movements vs. fitted normal.

    This is the foundational figure — shows the "underlying asset" distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Histogram + KDE + Normal ---
    ax = axes[0]
    data = process.delta_ip

    # Histogram
    ax.hist(data, bins=120, density=True, alpha=0.5,
            color=COLOR_PALETTE["secondary"], edgecolor="white", linewidth=0.3,
            label="Empirical")

    # KDE
    x = np.linspace(np.percentile(data, 0.5), np.percentile(data, 99.5), 500)
    ax.plot(x, process.kde(x), color=COLOR_PALETTE["accent"], lw=2.5,
            label="KDE (Silverman)")

    # Normal overlay
    from scipy.stats import norm
    normal_pdf = norm.pdf(x, loc=process.mean, scale=process.std)
    ax.plot(x, normal_pdf, color=COLOR_PALETTE["primary"], lw=2, ls="--",
            label=f"Normal(μ={process.mean:.4f}, σ={process.std:.4f})")

    ax.set_xlabel("Δ Implied Probability (close − open)")
    ax.set_ylabel("Density")
    ax.set_title("Empirical vs. Normal Distribution")
    ax.legend(loc="upper right", framealpha=0.9)

    # Stats annotation
    stats_text = (
        f"N = {process.n_obs:,}\n"
        f"Skew = {process.skew:.3f}\n"
        f"Kurt = {process.kurtosis:.3f}\n"
        f"JB p = {process.normality_test['jarque_bera']['pvalue']:.2e}"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # --- Right: QQ Plot ---
    ax = axes[1]
    from scipy.stats import probplot
    probplot(data, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot")
    ax.get_lines()[0].set_color(COLOR_PALETTE["secondary"])
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[1].set_color(COLOR_PALETTE["accent"])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, filename)


def plot_distribution_by_segment(
    processes: dict,
    segment_label: str = "Sport",
    filename: str = "dist_by_segment",
) -> None:
    """Plot overlaid KDEs for different segments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(processes)))

    for (name, proc), color in zip(processes.items(), colors):
        x = np.linspace(
            np.percentile(proc.delta_ip, 1),
            np.percentile(proc.delta_ip, 99),
            300,
        )
        ax.plot(x, proc.kde(x), lw=2, label=f"{name} (n={proc.n_obs:,})",
                color=color)

    ax.set_xlabel("Δ Implied Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Line Movement Distribution by {segment_label}")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# 2. DERIVATIVE PAYOFF DIAGRAMS
# ================================================================

def plot_payoff_diagrams(
    filename: str = "payoff_diagrams",
) -> None:
    """Plot payoff diagrams for all derivative types."""
    delta_range = np.linspace(-0.10, 0.10, 500)

    instruments = {
        "Call (K=0.02)": VanillaOption(strike=0.02, option_type="call"),
        "Put (K=-0.02)": VanillaOption(strike=-0.02, option_type="put"),
        "Straddle (K=0)": Straddle(strike=0.0),
        "Strangle (±0.03)": Strangle(k_call=0.03, k_put=-0.03),
        "Butterfly (±0.02)": Butterfly(k_low=-0.02, k_mid=0.0, k_high=0.02),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (name, deriv) in enumerate(instruments.items()):
        ax = axes[idx]
        payoffs = deriv.payoff(delta_range)

        ax.fill_between(delta_range, payoffs, 0,
                        where=payoffs > 0, alpha=0.3, color=COLOR_PALETTE["profit"])
        ax.fill_between(delta_range, payoffs, 0,
                        where=payoffs < 0, alpha=0.3, color=COLOR_PALETTE["loss"])
        ax.plot(delta_range, payoffs, lw=2, color=COLOR_PALETTE["primary"])
        ax.axhline(0, color="gray", lw=0.8, ls="-")
        ax.axvline(0, color="gray", lw=0.8, ls=":")

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Δ Implied Prob")
        ax.set_ylabel("Payoff")

    # Remove extra subplot
    axes[-1].set_visible(False)

    fig.suptitle("Synthetic Derivative Payoff Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# 3. PRICING COMPARISON
# ================================================================

def plot_pricing_comparison(
    pricing_df: pd.DataFrame,
    filename: str = "pricing_comparison",
) -> None:
    """Plot empirical vs. BSM prices across strikes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, otype in enumerate(["call", "put"]):
        ax = axes[idx]
        subset = pricing_df[pricing_df["type"] == otype]

        ax.plot(subset["strike"], subset["emp_price"], "o-",
                color=COLOR_PALETTE["accent"], lw=2, markersize=6,
                label="Empirical (KDE MC)")
        ax.fill_between(
            subset["strike"],
            subset["emp_price"] - 1.96 * subset["emp_se"],
            subset["emp_price"] + 1.96 * subset["emp_se"],
            alpha=0.2, color=COLOR_PALETTE["accent"],
        )
        ax.plot(subset["strike"], subset["bsm_price"], "s--",
                color=COLOR_PALETTE["primary"], lw=2, markersize=6,
                label="BSM Analog")

        ax.set_xlabel("Strike (Δ probability)")
        ax.set_ylabel("Price")
        ax.set_title(f"{otype.title()} Option Prices")
        ax.legend()

    fig.suptitle("Empirical vs. Black-Scholes Pricing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, filename)


def plot_three_model_pricing(
    pricing_df: pd.DataFrame,
    filename: str = "three_model_pricing",
) -> None:
    """
    Plot three-model pricing comparison:
    Empirical vs Bachelier vs Student-t Bachelier.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, otype in enumerate(["call", "put"]):
        ax = axes[idx]
        subset = pricing_df[pricing_df["type"] == otype]

        ax.plot(subset["strike"], subset["emp_price"], "o-",
                color=COLOR_PALETTE["accent"], lw=2, markersize=6,
                label="Empirical (KDE MC)")
        ax.fill_between(
            subset["strike"],
            subset["emp_price"] - 1.96 * subset["emp_se"],
            subset["emp_price"] + 1.96 * subset["emp_se"],
            alpha=0.15, color=COLOR_PALETTE["accent"],
        )
        ax.plot(subset["strike"], subset["bsm_price"], "s--",
                color=COLOR_PALETTE["primary"], lw=2, markersize=6,
                label="Bachelier (normal)")

        if "t_price" in subset.columns:
            ax.plot(subset["strike"], subset["t_price"], "D:",
                    color="#2ecc71", lw=2, markersize=6,
                    label="Student-t Bachelier")

        ax.set_xlabel("Strike (Δ probability)")
        ax.set_ylabel("Price")
        ax.set_title(f"{otype.title()} Option Prices")
        ax.legend(fontsize=9)

    t_df = pricing_df["t_df"].iloc[0] if "t_df" in pricing_df.columns else "?"
    fig.suptitle(f"Three-Model Pricing Comparison (t df={t_df:.1f})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, filename)


def plot_implied_vol_smile(
    smile_df: pd.DataFrame,
    filename: str = "implied_vol_smile",
) -> None:
    """
    Plot the implied volatility smile.

    If flat → normal distribution is correct.
    If U-shaped / upward wings → fat tails (like equity markets).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Implied vol vs strike ---
    ax = axes[0]
    for otype, color, marker in [("call", COLOR_PALETTE["accent"], "o"),
                                  ("put", COLOR_PALETTE["secondary"], "s")]:
        subset = smile_df[smile_df["type"] == otype]
        ax.plot(subset["abs_strike"], subset["implied_vol"],
                f"{marker}-", color=color, lw=2, markersize=6,
                label=f"{otype.title()}s")

    # Reference line at historical vol
    hist_vol = smile_df["hist_vol"].iloc[0]
    ax.axhline(hist_vol, color="gray", ls="--", lw=1.5,
               label=f"Historical σ = {hist_vol:.4f}")

    ax.set_xlabel("|Strike| (Δ probability)")
    ax.set_ylabel("Implied Volatility (σ)")
    ax.set_title("Implied Volatility Smile")
    ax.legend(fontsize=9)

    # --- Right: Vol ratio (implied/historical) ---
    ax = axes[1]
    for otype, color, marker in [("call", COLOR_PALETTE["accent"], "o"),
                                  ("put", COLOR_PALETTE["secondary"], "s")]:
        subset = smile_df[smile_df["type"] == otype]
        ax.plot(subset["moneyness"], subset["vol_ratio"],
                f"{marker}-", color=color, lw=2, markersize=6,
                label=f"{otype.title()}s")

    ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Ratio = 1 (normal)")
    ax.set_xlabel("Moneyness (|K| / σ)")
    ax.set_ylabel("Implied σ / Historical σ")
    ax.set_title("Volatility Smile Ratio")
    ax.legend(fontsize=9)

    fig.suptitle("Implied Volatility Smile — Evidence of Fat Tails",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, filename)


def plot_autocorrelation(
    acf_df: pd.DataFrame,
    filename: str = "autocorrelation",
) -> None:
    """Plot autocorrelation function with significance bands."""
    fig, ax = plt.subplots(figsize=(10, 5))

    lags = acf_df["lag"].values
    acf_vals = acf_df["autocorr"].values
    se = acf_df["se"].values

    colors = [COLOR_PALETTE["accent"] if abs(a) > 2 * s
              else COLOR_PALETTE["secondary"]
              for a, s in zip(acf_vals, se)]

    ax.bar(lags, acf_vals, color=colors, alpha=0.8, edgecolor="white")

    # 95% confidence bands
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(lags, -1.96 * se, 1.96 * se,
                    alpha=0.15, color="gray", label="95% CI (Bartlett)")

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of Line Movements (Δp)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, filename)


def plot_mispricing_improvement(
    pricing_df: pd.DataFrame,
    filename: str = "mispricing_improvement",
) -> None:
    """
    Bar chart showing mispricing reduction: Bachelier vs Student-t.
    The headline result: "Tail mispricing reduced from X% to Y%".
    """
    if "t_pct_diff" not in pricing_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    calls = pricing_df[pricing_df["type"] == "call"].copy()
    strikes = calls["strike"].values
    bsm_err = calls["bsm_pct_diff"].abs().values
    t_err = calls["t_pct_diff"].abs().values

    x = np.arange(len(strikes))
    width = 0.35

    bars1 = ax.bar(x - width/2, bsm_err, width,
                   label="Bachelier (normal)", color=COLOR_PALETTE["primary"], alpha=0.8)
    bars2 = ax.bar(x + width/2, t_err, width,
                   label="Student-t Bachelier", color="#2ecc71", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k:.2f}" for k in strikes])
    ax.set_xlabel("Strike")
    ax.set_ylabel("|Mispricing| (%)")
    ax.set_title("Pricing Error Reduction: Normal vs. Student-t Bachelier",
                 fontweight="bold")
    ax.legend()

    # Annotate max improvement
    max_idx = np.argmax(bsm_err - t_err)
    improvement = bsm_err[max_idx] - t_err[max_idx]
    if improvement > 0:
        ax.annotate(
            f"−{improvement:.0f}pp",
            xy=(max_idx + width/2, t_err[max_idx]),
            xytext=(max_idx + 1, t_err[max_idx] + 10),
            arrowprops=dict(arrowstyle="->", color=COLOR_PALETTE["accent"]),
            fontsize=12, fontweight="bold", color=COLOR_PALETTE["accent"],
        )

    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# 4. BACKTEST PERFORMANCE
# ================================================================

def plot_backtest_cumulative(
    results: list[BacktestResult],
    filename: str = "backtest_cumulative",
) -> None:
    """Plot cumulative P&L for multiple strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    # --- Top: Cumulative P&L ---
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for res, color in zip(results, colors):
        if len(res.cumulative_pnl) == 0:
            continue
        ax.plot(res.dates, res.cumulative_pnl, lw=2, color=color,
                label=f"{res.strategy_name} (SR={res.sharpe:.2f})")

    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel("Cumulative P&L")
    ax.set_title("Strategy Performance Comparison", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # --- Bottom: Drawdowns for best derivative strategy ---
    ax = axes[1]
    # Sort by Sharpe, pick best non-benchmark
    deriv_results = [r for r in results if "Spot" not in r.strategy_name]
    if deriv_results:
        best = max(deriv_results, key=lambda r: r.sharpe)
        if len(best.cumulative_pnl) > 0:
            running_max = np.maximum.accumulate(best.cumulative_pnl)
            drawdown = best.cumulative_pnl - running_max
            ax.fill_between(best.dates, drawdown, 0,
                            color=COLOR_PALETTE["loss"], alpha=0.5)
            ax.set_ylabel("Drawdown")
            ax.set_title(f"Drawdowns: {best.strategy_name}", fontsize=11)

    plt.tight_layout()
    save_fig(fig, filename)


def plot_strategy_summary_table(
    summary_df: pd.DataFrame,
    filename: str = "strategy_summary",
) -> None:
    """Create a formatted table of strategy performance metrics."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(summary_df) * 0.4 + 2)))
    ax.axis("off")

    cols = ["strategy", "n_trades", "sharpe", "sortino", "max_drawdown",
            "hit_rate", "total_pnl"]
    
    display_df = summary_df[cols].copy()
    display_df.columns = ["Strategy", "N Trades", "Sharpe", "Sortino",
                          "Max DD", "Hit Rate", "Total P&L"]

    # Format
    for col in ["Sharpe", "Sortino", "Total P&L"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")
    display_df["Max DD"] = display_df["Max DD"].map(lambda x: f"{x:.4f}")
    display_df["Hit Rate"] = display_df["Hit Rate"].map(lambda x: f"{x:.1%}")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(display_df.columns))))

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLOR_PALETTE["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    ax.set_title("Strategy Performance Summary", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# 5. EFFICIENT FRONTIER
# ================================================================

def plot_efficient_frontiers(
    frontiers_df: pd.DataFrame,
    spot_opt=None,
    deriv_opt=None,
    filename: str = "efficient_frontier",
) -> None:
    """Plot overlaid efficient frontiers: spot vs. derivative-augmented."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for ptype, group in frontiers_df.groupby("portfolio_type"):
        color = COLOR_PALETTE["secondary"] if "spot" in ptype else COLOR_PALETTE["accent"]
        label = "Spot Only" if "spot" in ptype else "With Derivatives"
        ax.plot(group["volatility"], group["target_return"],
                lw=2.5, color=color, label=label)

    # Mark optimal portfolios
    if spot_opt:
        ax.scatter(spot_opt.volatility, spot_opt.expected_return,
                   s=100, marker="^", color=COLOR_PALETTE["secondary"],
                   zorder=5, edgecolor="black",
                   label=f"Spot Optimal (SR={spot_opt.sharpe:.2f})")

    if deriv_opt:
        ax.scatter(deriv_opt.volatility, deriv_opt.expected_return,
                   s=100, marker="*", color=COLOR_PALETTE["accent"],
                   zorder=5, edgecolor="black",
                   label=f"Deriv Optimal (SR={deriv_opt.sharpe:.2f})")

    ax.set_xlabel("Portfolio Volatility (σ)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier: Spot Only vs. Derivative-Augmented",
                 fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# 6. VOLATILITY SURFACE
# ================================================================

def plot_volatility_surface(
    vol_surface: pd.DataFrame,
    filename: str = "vol_surface",
) -> None:
    """Plot the volatility term structure of line movements."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(range(len(vol_surface)), vol_surface["sigma"],
           color=COLOR_PALETTE["secondary"], alpha=0.7, edgecolor="white")

    ax.set_xticks(range(len(vol_surface)))
    ax.set_xticklabels(
        [f"{d:.0f}h" for d in vol_surface["mean_duration"]],
        rotation=45,
    )

    ax.set_xlabel("Mean Duration (hours line is open)")
    ax.set_ylabel("σ (std of Δ implied probability)")
    ax.set_title("Volatility Term Structure of Line Movements",
                 fontweight="bold")

    # Annotate N
    for i, row in vol_surface.iterrows():
        ax.text(i, row["sigma"] + 0.001, f"n={row['n_obs']:,}",
                ha="center", fontsize=7)

    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# 7. PASPA REGIME ANALYSIS
# ================================================================

def plot_regime_distributions(
    regime_processes: dict,
    filename: str = "regime_distributions",
) -> None:
    """Overlay KDEs for pre/post-PASPA regimes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Full KDE overlay ---
    ax = axes[0]
    regime_colors = {
        "pre_paspa": COLOR_PALETTE["secondary"],
        "post_paspa": COLOR_PALETTE["accent"],
        "covid": "#e67e22",
        "post_paspa_ex_covid": "#2ecc71",
    }
    regime_labels = {
        "pre_paspa": "Pre-PASPA (before May 2018)",
        "post_paspa": "Post-PASPA (all)",
        "covid": "COVID period",
        "post_paspa_ex_covid": "Post-PASPA (ex COVID)",
    }

    for name, proc in regime_processes.items():
        color = regime_colors.get(name, "gray")
        label = regime_labels.get(name, name)
        x = np.linspace(
            np.percentile(proc.delta_ip, 1),
            np.percentile(proc.delta_ip, 99),
            300,
        )
        ax.plot(x, proc.kde(x), lw=2.5, color=color,
                label=f"{label} (n={proc.n_obs:,}, σ={proc.std:.4f})")

    ax.set_xlabel("Δ Implied Probability")
    ax.set_ylabel("Density")
    ax.set_title("Line Movement Distribution by Regime")
    ax.legend(loc="upper right", fontsize=8)

    # --- Right: Tail comparison (zoomed) ---
    ax = axes[1]
    for name in ["pre_paspa", "post_paspa_ex_covid"]:
        if name not in regime_processes:
            continue
        proc = regime_processes[name]
        color = regime_colors[name]
        label = regime_labels[name]

        x_right = np.linspace(0.03, np.percentile(proc.delta_ip, 99.5), 200)
        x_left = np.linspace(np.percentile(proc.delta_ip, 0.5), -0.03, 200)

        ax.plot(x_right, proc.kde(x_right), lw=2.5, color=color, label=f"{label} (right tail)")
        ax.plot(x_left, proc.kde(x_left), lw=2.5, color=color, ls="--", alpha=0.6)

    ax.set_xlabel("Δ Implied Probability")
    ax.set_ylabel("Density")
    ax.set_title("Tail Comparison: Pre vs. Post-PASPA")
    ax.legend(fontsize=8)

    fig.suptitle("PASPA Regime Effects on Line Movement Dynamics",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, filename)


def plot_regime_pricing_table(
    regime_pricing_df: pd.DataFrame,
    filename: str = "regime_pricing",
) -> None:
    """Table showing how derivative prices change across regimes."""
    fig, ax = plt.subplots(figsize=(14, max(3, len(regime_pricing_df) * 0.6 + 2)))
    ax.axis("off")

    display_cols = [c for c in regime_pricing_df.columns if c != "n_obs"]
    display = regime_pricing_df[display_cols].copy()

    # Format numeric columns
    for col in display.columns:
        if col == "regime":
            continue
        if "mispricing" in col:
            display[col] = display[col].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
        elif display[col].dtype in [float, np.float64]:
            display[col] = display[col].map(lambda x: f"{x:.5f}" if pd.notna(x) else "")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(display.columns))))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLOR_PALETTE["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    ax.set_title("Derivative Pricing Across PASPA Regimes",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    save_fig(fig, filename)


def plot_rolling_volatility(
    df: pd.DataFrame,
    window: int = 5000,
    filename: str = "rolling_volatility",
) -> None:
    """
    Plot rolling volatility of line movements over time.
    Shows how the "implied vol" of the underlying process evolves.
    """
    df = df.sort_values("match_date").copy()
    delta = df["delta_ip_home"].values
    dates = pd.to_datetime(df["match_date"].values)

    # Rolling std
    roll_std = pd.Series(delta).rolling(window, min_periods=window // 2).std().values
    # Rolling kurtosis
    roll_kurt = pd.Series(delta).rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).kurtosis()), raw=False
    ).values

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Volatility ---
    ax = axes[0]
    ax.plot(dates, roll_std, lw=1.2, color=COLOR_PALETTE["secondary"], alpha=0.8)
    ax.axvline(pd.Timestamp("2018-05-14"), color=COLOR_PALETTE["accent"],
               ls="--", lw=2, label="PASPA Repeal")
    ax.axvspan(pd.Timestamp("2020-03-11"), pd.Timestamp("2020-07-23"),
               alpha=0.15, color="#e67e22", label="COVID")
    ax.set_ylabel("Rolling σ (Δ implied prob)")
    ax.set_title(f"Rolling Volatility of Line Movements (window={window:,})",
                 fontweight="bold")
    ax.legend(loc="upper right")

    # --- Kurtosis ---
    ax = axes[1]
    ax.plot(dates, roll_kurt, lw=1.2, color=COLOR_PALETTE["primary"], alpha=0.8)
    ax.axvline(pd.Timestamp("2018-05-14"), color=COLOR_PALETTE["accent"],
               ls="--", lw=2)
    ax.axvspan(pd.Timestamp("2020-03-11"), pd.Timestamp("2020-07-23"),
               alpha=0.15, color="#e67e22")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Rolling Excess Kurtosis")
    ax.set_xlabel("Date")
    ax.set_title("Rolling Tail Thickness")

    plt.tight_layout()
    save_fig(fig, filename)


def plot_bookmaker_comparison(
    processes_by_book: dict,
    filename: str = "bookmaker_comparison",
) -> None:
    """
    Compare line movement properties across bookmakers.
    Sharp books (Pinnacle) vs recreational (888sport, bet365).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Sort by N for consistent ordering
    sorted_books = sorted(processes_by_book.items(),
                          key=lambda x: x[1].n_obs, reverse=True)

    # --- Left: KDE overlay ---
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_books)))
    for (name, proc), color in zip(sorted_books, colors):
        x = np.linspace(np.percentile(proc.delta_ip, 1),
                        np.percentile(proc.delta_ip, 99), 300)
        ax.plot(x, proc.kde(x), lw=2, color=color,
                label=f"{name} (σ={proc.std:.4f})")

    ax.set_xlabel("Δ Implied Probability")
    ax.set_ylabel("Density")
    ax.set_title("Line Movement Distribution by Bookmaker")
    ax.legend(fontsize=8)

    # --- Right: Bar chart of key stats ---
    ax = axes[1]
    names = [n for n, _ in sorted_books]
    stds = [p.std for _, p in sorted_books]
    kurts = [p.kurtosis for _, p in sorted_books]

    x_pos = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, stds, width,
                   label="σ (volatility)", color=COLOR_PALETTE["secondary"])
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, kurts, width,
                    label="Excess Kurtosis", color=COLOR_PALETTE["accent"], alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("σ (std of Δ implied prob)")
    ax2.set_ylabel("Excess Kurtosis")
    ax.set_title("Bookmaker Microstructure Comparison")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    save_fig(fig, filename)


# ================================================================
# MASTER PLOT FUNCTION
# ================================================================

def generate_all_figures(
    process: ImpliedProcess,
    processes_by_sport: dict,
    pricing_df: pd.DataFrame,
    backtest_results: list,
    summary_df: pd.DataFrame,
    frontiers: dict,
    vol_surface: pd.DataFrame = None,
) -> None:
    """Generate all publication figures."""
    print("\n=== Generating Figures ===")

    print("1. Line movement distribution...")
    plot_line_movement_distribution(process)

    print("2. Distribution by sport...")
    plot_distribution_by_segment(processes_by_sport)

    print("3. Payoff diagrams...")
    plot_payoff_diagrams()

    print("4. Pricing comparison...")
    plot_pricing_comparison(pricing_df)

    print("5. Backtest cumulative P&L...")
    plot_backtest_cumulative(backtest_results)

    print("6. Strategy summary table...")
    plot_strategy_summary_table(summary_df)

    print("7. Efficient frontier comparison...")
    plot_efficient_frontiers(
        frontiers["frontiers"],
        frontiers.get("spot_optimal"),
        frontiers.get("deriv_optimal"),
    )

    if vol_surface is not None:
        print("8. Volatility surface...")
        plot_volatility_surface(vol_surface)

    print("\nAll figures saved to:", FIGURES_DIR)
