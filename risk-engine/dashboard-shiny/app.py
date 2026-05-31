"""
app.py
======
Shiny (Python) dashboard for the Synthetic Derivatives Risk Engine.

Sits on top of the same C# / .NET engine that powers the Blazor dashboard;
RiskEngine.Cli is invoked as a subprocess, the resulting JSON report is
rendered into three tabs (Overview, Scenarios, Positions) using Plotly.

Run with:
    shiny run --reload app.py
"""
from __future__ import annotations

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

from engine import EngineError, load_portfolio, run_engine
from charts import (
    empty_figure,
    payoffs_grid,
    pnl_histogram,
    stress_bars,
    vol_curve,
)

# --- Load portfolio once at startup ------------------------------------------
PORTFOLIO = load_portfolio()


# --- Formatting helpers ------------------------------------------------------
def _is_nan(v) -> bool:
    return isinstance(v, float) and v != v


def fmt_num(v, dp: int = 4) -> str:
    if v is None or _is_nan(v):
        return "-"
    return f"{v:.{dp}f}"


def fmt_signed(v, dp: int = 4) -> str:
    if v is None or _is_nan(v):
        return "-"
    return f"{v:+.{dp}f}"


def fmt_pct(v, dp: int = 1) -> str:
    if v is None or _is_nan(v):
        return "-"
    return f"{v * 100:.{dp}f}%"


# --- Stat card (plain HTML, no theme dependencies) ---------------------------
def stat_card(label: str, value: str, tone: str = "") -> ui.Tag:
    if tone == "pos":
        val_color = "color:#2f9e6b;"
        border = "border-left:3px solid #2f9e6b;"
    elif tone == "neg":
        val_color = "color:#d1495b;"
        border = "border-left:3px solid #d1495b;"
    else:
        val_color = ""
        border = "border-left:3px solid #3a6ea5;"

    return ui.div(
        ui.div(
            label,
            style=(
                "font-size:11px;text-transform:uppercase;letter-spacing:.4px;"
                "color:#5b6472;font-weight:600;"
            ),
        ),
        ui.div(
            value,
            style=(
                "font-size:22px;font-weight:650;margin-top:4px;"
                "font-variant-numeric:tabular-nums;" + val_color
            ),
        ),
        class_="bg-white rounded p-3 shadow-sm h-100",
        style=border,
    )


# ====== UI ===================================================================
app_ui = ui.page_navbar(
    # ---- Overview -----------------------------------------------------------
    ui.nav_panel(
        "Overview",
        ui.layout_columns(
            ui.input_select(
                "model",
                "Pricing model",
                {
                    "StudentT": "Student-t Bachelier (fat tails)",
                    "Normal": "Normal (Bachelier)",
                },
                selected="StudentT",
            ),
            ui.input_select(
                "paths",
                "Monte Carlo paths",
                {"10000": "10,000", "50000": "50,000", "200000": "200,000"},
                selected="50000",
            ),
            ui.output_ui("meta"),
            col_widths=[3, 3, 6],
        ),
        ui.h4("Headline risk - base regime", class_="mt-3"),
        ui.output_ui("stat_cards"),
        ui.h4("PnL distribution - base regime", class_="mt-4"),
        ui.p(
            "Chart shows the central 98% of outcomes so the shape stays "
            "readable; the deep tails are captured in the VaR and CVaR cards "
            "above. Green bars are profitable paths, red are losses.",
            class_="text-muted small",
        ),
        output_widget("hist_chart"),
    ),
    # ---- Scenarios ----------------------------------------------------------
    ui.nav_panel(
        "Scenarios",
        ui.h4("Risk by market regime"),
        ui.output_data_frame("regime_table"),
        ui.p(
            "The regime-switching row is the weighted mixture of all regimes "
            "- each simulated path first draws a regime by its historical "
            "frequency, then draws the line move from that regime's "
            "distribution.",
            class_="text-muted small mt-2",
        ),
        ui.h4("Stress test battery - 95% VaR", class_="mt-4"),
        output_widget("stress_chart"),
        ui.output_data_frame("stress_table"),
        ui.h4("Volatility scenario sweep", class_="mt-4"),
        ui.p(
            "95% VaR and CVaR as the calibrated base volatility is scaled "
            "from 0.5x to 2.0x. Tail risk grows faster than linearly - the "
            "convexity a short-volatility book is exposed to.",
            class_="text-muted small",
        ),
        output_widget("vol_chart"),
    ),
    # ---- Positions ----------------------------------------------------------
    ui.nav_panel(
        "Positions",
        ui.h4("Position book"),
        ui.output_data_frame("positions_table"),
        ui.p(
            "Model value is the mean Monte Carlo payoff per contract under "
            "the base regime. Where it sits above the entry price the "
            "position carries positive expected value before correlation "
            "with the rest of the book.",
            class_="text-muted small mt-2",
        ),
        ui.h4("Payoff diagrams", class_="mt-4"),
        ui.p(
            "Per-contract payoff (blue) and PnL net of premium (green) "
            "against the realised line move.",
            class_="text-muted small",
        ),
        output_widget("payoff_chart"),
    ),
    title="Synthetic Derivatives Risk Engine - Shiny",
    id="navbar",
    header=ui.output_ui("error_banner"),
    fillable=False,
)


# ====== Server ===============================================================
def server(input: Inputs, output: Outputs, session: Session):

    error_msg = reactive.value(None)

    # The single source of truth: invoke the C# engine and cache the report.
    @reactive.calc
    def report():
        try:
            r = run_engine(model=input.model(), paths=int(input.paths()))
            error_msg.set(None)
            return r
        except EngineError as exc:
            error_msg.set(str(exc))
            return None

    # ---- Global error banner -----------------------------------------------
    @render.ui
    def error_banner():
        err = error_msg()
        if err is None:
            return ui.div()
        return ui.div(
            ui.tags.strong("Engine error: "),
            ui.tags.pre(
                err,
                style="white-space:pre-wrap;margin:0;font-size:12px;",
            ),
            class_="alert alert-danger mx-3 mt-2 mb-0",
        )

    # ---- Overview ----------------------------------------------------------
    @render.ui
    def meta():
        r = report()
        if r is None:
            return ui.div(
                ui.tags.em("Computing..."),
                class_="text-end text-muted",
            )
        return ui.div(
            ui.div(
                ui.tags.strong(r["portfolioName"]),
                f"  -  {r['positionCount']} positions",
            ),
            ui.div(
                f"{r['distributionModel']} model  -  "
                f"{r['paths']:,} paths  -  "
                f"net premium {fmt_signed(r['netPremium'])}",
                class_="small",
            ),
            class_="text-end text-muted",
        )

    @render.ui
    def stat_cards():
        r = report()
        if r is None:
            return ui.p("Computing...", class_="text-muted")
        h = r["headline"]
        cards = [
            stat_card(
                "Mean PnL",
                fmt_signed(h["meanPnl"]),
                "pos" if h["meanPnl"] >= 0 else "neg",
            ),
            stat_card("Std deviation", fmt_num(h["stdDevPnl"])),
            stat_card("95% VaR", fmt_num(h["var95"])),
            stat_card("99% VaR", fmt_num(h["var99"])),
            stat_card("95% CVaR", fmt_num(h["cVar95"])),
            stat_card("99% CVaR", fmt_num(h["cVar99"])),
            stat_card("P(loss)", fmt_pct(h["probabilityOfLoss"])),
            stat_card(
                "Sharpe-like",
                fmt_signed(h["sharpeLike"], 3),
                "pos" if h["sharpeLike"] >= 0 else "neg",
            ),
        ]
        return ui.layout_columns(*cards, col_widths=[3, 3, 3, 3, 3, 3, 3, 3])

    @render_widget
    def hist_chart():
        r = report()
        if r is None:
            return empty_figure("Computing...")
        return pnl_histogram(r["pnlHistogram"])

    # ---- Scenarios ---------------------------------------------------------
    @render.data_frame
    def regime_table():
        r = report()
        if r is None:
            return pd.DataFrame()
        rows = list(r["regimes"]) + [r["regimeSwitching"]]
        df = pd.DataFrame(
            [
                {
                    "Regime": s["label"],
                    "Mean PnL": fmt_signed(s["meanPnl"]),
                    "Std dev": fmt_num(s["stdDevPnl"]),
                    "95% VaR": fmt_num(s["var95"]),
                    "95% CVaR": fmt_num(s["cVar95"]),
                    "99% VaR": fmt_num(s["var99"]),
                    "P(loss)": fmt_pct(s["probabilityOfLoss"]),
                }
                for s in rows
            ]
        )
        return render.DataGrid(df, width="100%")

    @render_widget
    def stress_chart():
        r = report()
        if r is None:
            return empty_figure("Computing...")
        return stress_bars(r["stressScenarios"])

    @render.data_frame
    def stress_table():
        r = report()
        if r is None:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "Scenario": s["label"],
                    "Mean PnL": fmt_signed(s["meanPnl"]),
                    "95% VaR": fmt_num(s["var95"]),
                    "95% CVaR": fmt_num(s["cVar95"]),
                    "99% CVaR": fmt_num(s["cVar99"]),
                    "Excess kurtosis": fmt_signed(s["excessKurtosis"], 2),
                    "P(loss)": fmt_pct(s["probabilityOfLoss"]),
                }
                for s in r["stressScenarios"]
            ]
        )
        return render.DataGrid(df, width="100%")

    @render_widget
    def vol_chart():
        r = report()
        if r is None:
            return empty_figure("Computing...")
        return vol_curve(r["volatilityCurve"])

    # ---- Positions ---------------------------------------------------------
    @render.data_frame
    def positions_table():
        r = report()
        if r is None:
            return pd.DataFrame()
        rows = []
        for p in r["positions"]:
            q = p["quantity"]
            q_disp = f"{int(q)}" if abs(q - int(q)) < 1e-9 else f"{q:g}"
            rows.append(
                {
                    "Instrument": p["instrumentId"],
                    "Type": p["instrumentType"],
                    "Qty": q_disp,
                    "Entry": fmt_num(p["entryPrice"]),
                    "Model": fmt_num(p["modelValue"]),
                    "Mean PnL": fmt_signed(p["meanPnl"]),
                    "PnL std dev": fmt_num(p["pnlStdDev"]),
                    "95% VaR": fmt_num(p["var95"]),
                    "PnL share": fmt_pct(p["pnlShare"]),
                }
            )
        return render.DataGrid(pd.DataFrame(rows), width="100%")

    @render_widget
    def payoff_chart():
        return payoffs_grid(PORTFOLIO)


app = App(app_ui, server)
