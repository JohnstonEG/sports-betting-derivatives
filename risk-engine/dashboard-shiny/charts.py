"""
charts.py
=========
Plotly chart builders for the Shiny dashboard.

Every chart takes data straight from the JSON shape produced by RiskEngine.Cli
(see RiskReport / HistogramData / VolScenarioPoint / RiskSummary on the C# side)
so the dashboard is a thin presentation layer over the engine output.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Palette ------------------------------------------------------------------
POS = "#2f9e6b"      # profitable / "good"
NEG = "#d1495b"      # losses / "bad"
ACCENT = "#3a6ea5"   # neutral accent (payoff, VaR series)
GRID = "#e6e9ef"
INK_SOFT = "#5b6472"


# --- Overview: P&L histogram --------------------------------------------------
def pnl_histogram(hist: dict) -> go.Figure:
    """Bar histogram of the portfolio P&L distribution (clipped to central 98%)."""
    centers = hist.get("binCenters", [])
    counts = hist.get("counts", [])
    if not centers or not counts:
        return empty_figure("No histogram data")

    colors = [POS if c >= 0 else NEG for c in centers]
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            marker_color=colors,
            marker_line_width=0,
            opacity=0.86,
            hovertemplate="P&L: %{x:.4f}<br>Paths: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=56, r=20, t=20, b=44),
        height=320,
        bargap=0.04,
        xaxis_title="Portfolio P&L (per path) - central 98% shown",
        yaxis_title="Paths",
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=True, zerolinecolor="#33404f", zerolinewidth=1.4)
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero")
    return fig


# --- Scenarios: volatility curve ---------------------------------------------
def vol_curve(points: list) -> go.Figure:
    """95% VaR and CVaR across the volatility sweep."""
    if not points:
        return empty_figure("No volatility-curve data")

    x = [p["volMultiplier"] for p in points]
    var = [p["var95"] for p in points]
    cvar = [p["cVar95"] for p in points]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=var, mode="lines+markers", name="95% VaR",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=6),
        hovertemplate="Vol x%{x:.2f}<br>VaR %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cvar, mode="lines+markers", name="95% CVaR (ES)",
        line=dict(color=NEG, width=2.5),
        marker=dict(size=6),
        hovertemplate="Vol x%{x:.2f}<br>CVaR %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=58, r=20, t=44, b=50),
        height=340,
        xaxis_title="Volatility multiplier",
        yaxis_title="Loss (positive)",
        plot_bgcolor="white",
        legend=dict(orientation="h", x=0.02, y=1.10, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, tickformat=".2f")
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero")
    return fig


# --- Scenarios: stress comparison --------------------------------------------
def stress_bars(stress: list) -> go.Figure:
    """Horizontal bar chart comparing 95% VaR across stress scenarios."""
    if not stress:
        return empty_figure("No stress scenarios")

    labels = [s["label"] for s in stress]
    values = [s["var95"] for s in stress]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=NEG,
            opacity=0.82,
            hovertemplate="%{y}: 95%% VaR %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=140, r=40, t=20, b=44),
        height=44 + 32 * max(len(labels), 1),
        xaxis_title="95% VaR (loss)",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return fig


# --- Positions: payoff diagrams grid -----------------------------------------
def payoffs_grid(portfolio: dict) -> go.Figure:
    """Per-position payoff and PnL curves vs Δp, as a Plotly subplot grid."""
    positions = portfolio.get("positions", [])
    if not positions:
        return empty_figure("No positions in portfolio")

    cols = 2
    rows = (len(positions) + cols - 1) // cols
    titles = [
        f"{p['instrument']['id']} - {p['instrument']['type']} (qty {p['quantity']:g})"
        for p in positions
    ]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    dp_grid = np.linspace(-0.15, 0.15, 121)
    for i, pos in enumerate(positions):
        r = i // cols + 1
        c = i % cols + 1
        inst = pos["instrument"]
        entry = float(pos["entryPrice"])
        payoff_fn = _payoff_fn(inst)
        gross = np.array([payoff_fn(x) for x in dp_grid])
        pnl = gross - entry

        fig.add_trace(
            go.Scatter(
                x=dp_grid, y=gross, mode="lines", name="Payoff",
                line=dict(color=ACCENT, width=2),
                legendgroup="payoff", showlegend=(i == 0),
                hovertemplate="Δp %{x:.3f}<br>Payoff %{y:.4f}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=dp_grid, y=pnl, mode="lines", name="PnL (net of premium)",
                line=dict(color=POS, width=2),
                legendgroup="pnl", showlegend=(i == 0),
                hovertemplate="Δp %{x:.3f}<br>PnL %{y:.4f}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.add_hline(y=0, line_width=1, line_color=GRID, row=r, col=c)
        fig.add_vline(x=0, line_width=1, line_color=GRID, row=r, col=c)

    fig.update_layout(
        height=260 * rows + 40,
        margin=dict(l=50, r=20, t=60, b=40),
        plot_bgcolor="white",
        legend=dict(orientation="h", x=0, y=1.05, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, tickformat=".2f")
    fig.update_yaxes(gridcolor=GRID)
    # Trim subplot title font size.
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=INK_SOFT)
    return fig


# --- Helpers ------------------------------------------------------------------
def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=13, color=INK_SOFT))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _payoff_fn(instrument: dict):
    """Mirror of the C# IPayoff factory -- pure Python, identical formulas."""
    t = instrument["type"]
    K = float(instrument.get("strike", 0.0))
    K2 = float(instrument.get("secondStrike", 0.0))
    K3 = float(instrument.get("thirdStrike", 0.0))

    if t == "Call":
        return lambda dp: max(dp - K, 0.0)
    if t == "Put":
        return lambda dp: max(K - dp, 0.0)
    if t == "DigitalCall":
        return lambda dp: 1.0 if dp > K else 0.0
    if t == "DigitalPut":
        return lambda dp: 1.0 if dp < K else 0.0
    if t == "Straddle":
        return lambda dp: abs(dp - K)
    if t == "Strangle":
        return lambda dp: max(K - dp, 0.0) + max(dp - K2, 0.0)
    if t == "BullCallSpread":
        return lambda dp: max(dp - K, 0.0) - max(dp - K2, 0.0)
    if t == "BearPutSpread":
        return lambda dp: max(K2 - dp, 0.0) - max(K - dp, 0.0)
    if t == "Butterfly":
        return lambda dp: max(dp - K, 0.0) - 2.0 * max(dp - K2, 0.0) + max(dp - K3, 0.0)
    if t == "VarianceSwap":
        return lambda dp: dp * dp - K
    raise ValueError(f"Unknown instrument type: {t}")
