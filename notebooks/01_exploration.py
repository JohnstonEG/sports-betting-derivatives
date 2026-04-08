"""
01_exploration.py — Full analysis pipeline for sports derivatives.

Run this script to:
1. Load and validate data
2. Extract the implied probability process
3. Test distributional properties
4. Price all synthetic derivatives
5. Backtest strategies
6. Compare portfolios
7. PASPA regime analysis
8. Bookmaker microstructure
9. Generate all figures

Usage:
    cd sports-derivatives
    python notebooks/01_exploration.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import CLEANED_DATA, RESULTS_DIR, TABLES_DIR, PASPA_REPEAL, COVID_START, COVID_END
from src.data_loader import load_data, data_summary
from src.implied_process import (
    extract_implied_process, compare_processes,
    extract_regime_processes, regime_pricing_comparison,
)
from src.derivatives import build_instrument_catalog, VanillaOption, Straddle
from src.pricing import (
    EmpiricalPricer, BSMPricer, StudentTBachelierPricer,
    compare_pricing_methods, compute_implied_vol_smile,
)
from src.backtester import Backtester
from src.portfolio import PortfolioOptimizer
from src.predictive import run_predictive_analysis
from src.visualization import (
    plot_line_movement_distribution,
    plot_distribution_by_segment,
    plot_payoff_diagrams,
    plot_pricing_comparison,
    plot_three_model_pricing,
    plot_implied_vol_smile,
    plot_autocorrelation,
    plot_mispricing_improvement,
    plot_backtest_cumulative,
    plot_strategy_summary_table,
    plot_efficient_frontiers,
    plot_volatility_surface,
    plot_regime_distributions,
    plot_regime_pricing_table,
    plot_rolling_volatility,
    plot_bookmaker_comparison,
)


def main():
    print("=" * 60)
    print("SYNTHETIC DERIVATIVES ON SPORTS BETTING MARKETS")
    print("=" * 60)

    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n--- STEP 1: Loading Data ---")
    df = load_data(require_opening=True)
    summary = data_summary(df)
    
    print("\nData Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: {', '.join(str(s) for s in v[:10])}{'...' if len(v) > 10 else ''}")
        else:
            print(f"  {k}: {v}")

    # ============================================================
    # STEP 2: EXTRACT IMPLIED PROCESS
    # ============================================================
    print("\n--- STEP 2: Extracting Implied Process ---")
    
    # Overall process
    process = extract_implied_process(df, outcome="home")
    print("\nOverall Process:")
    print(process.summary_table())
    process.summary_table().to_csv(TABLES_DIR / "process_summary.csv")

    # By sport
    processes_by_sport = extract_implied_process(df, outcome="home", segment_by="sport")
    comparison = compare_processes(processes_by_sport)
    print("\nProcess by Sport:")
    print(comparison.to_string(index=False))
    comparison.to_csv(TABLES_DIR / "process_by_sport.csv", index=False)

    # By bookmaker
    processes_by_book = {}
    if "bookmaker_name" in df.columns:
        processes_by_book = extract_implied_process(
            df, outcome="home", segment_by="bookmaker_name"
        )
        book_comparison = compare_processes(processes_by_book)
        print("\nProcess by Bookmaker:")
        print(book_comparison.to_string(index=False))
        book_comparison.to_csv(TABLES_DIR / "process_by_bookmaker.csv", index=False)

    # ============================================================
    # STEP 3: DISTRIBUTIONAL TESTS
    # ============================================================
    print("\n--- STEP 3: Distributional Analysis ---")
    
    print(f"\nNormality Tests (overall):")
    for test_name, result in process.normality_test.items():
        if "pvalue" in result:
            print(f"  {test_name}: stat={result['statistic']:.2f}, p={result['pvalue']:.2e}")
        else:
            print(f"  {test_name}: stat={result['statistic']:.2f}")
            for crit, val in result.get("critical_values", {}).items():
                print(f"    {crit}: {val:.2f}")

    print(f"\n  Excess kurtosis = {process.kurtosis:.3f} "
          f"({'fat-tailed' if process.kurtosis > 0 else 'thin-tailed'})")
    print(f"  Skewness = {process.skew:.3f} "
          f"({'right-skewed' if process.skew > 0 else 'left-skewed'})")

    # ============================================================
    # STEP 4: PRICE DERIVATIVES (Three Models)
    # ============================================================
    print("\n--- STEP 4: Pricing Derivatives (Three Models) ---")

    # Fit Student-t to the data
    t_pricer = StudentTBachelierPricer(process)
    t_fit = t_pricer.fit_summary()
    print(f"\n  Student-t fit: df={t_fit['df']:.2f}, "
          f"loc={t_fit['loc']:.5f}, scale={t_fit['scale']:.5f}")

    # Three-model comparison
    pricing_df = compare_pricing_methods(process)
    print("\nPricing Comparison (Empirical vs Bachelier vs Student-t):")
    cols_to_show = ["strike", "type", "emp_price", "bsm_price", "t_price",
                    "bsm_pct_diff", "t_pct_diff"]
    print(pricing_df[cols_to_show].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    pricing_df.to_csv(TABLES_DIR / "pricing_comparison.csv", index=False)

    # Headline result: tail mispricing improvement
    calls_otm = pricing_df[
        (pricing_df["type"] == "call") & (pricing_df["strike"] >= 0.07)
    ]
    if len(calls_otm) > 0:
        max_bsm_err = calls_otm["bsm_pct_diff"].abs().max()
        max_t_err = calls_otm["t_pct_diff"].abs().max()
        print(f"\n  *** HEADLINE: Deep OTM tail mispricing ***")
        print(f"  Bachelier (normal): up to {max_bsm_err:.1f}%")
        print(f"  Student-t Bachelier: up to {max_t_err:.1f}%")
        print(f"  → Reduction: {max_bsm_err:.1f}% → {max_t_err:.1f}%")

    # Implied volatility smile
    print("\n  Computing implied volatility smile...")
    smile_df = compute_implied_vol_smile(process)
    smile_df.to_csv(TABLES_DIR / "implied_vol_smile.csv", index=False)

    call_smile = smile_df[smile_df["type"] == "call"]
    if len(call_smile) > 1:
        atm_vol = call_smile[call_smile["abs_strike"] <= 0.005]["implied_vol"]
        otm_vol = call_smile[call_smile["abs_strike"] >= 0.10]["implied_vol"]
        if len(atm_vol) > 0 and len(otm_vol) > 0:
            print(f"\n  Implied Vol Smile:")
            print(f"    ATM implied σ:  {atm_vol.mean():.5f}")
            print(f"    OTM implied σ:  {otm_vol.mean():.5f}")
            print(f"    Smile ratio:    {otm_vol.mean() / atm_vol.mean():.2f}x")
            print(f"    → {'Upward-sloping wings (fat tails confirmed)' if otm_vol.mean() > atm_vol.mean() else 'Flat (near-normal)'}")

    # Full catalog pricing
    catalog = build_instrument_catalog()
    emp_pricer = EmpiricalPricer(process)

    print(f"\nInstrument Catalog ({len(catalog)} instruments):")
    catalog_prices = []
    for name, deriv in catalog.items():
        result = emp_pricer.price(deriv)
        catalog_prices.append({
            "instrument": name,
            "description": deriv.description(),
            "emp_price": result["price"],
            "pct_itm": result["pct_itm"],
            "std_err": result["std_err"],
        })
        print(f"  {name:25s} | price={result['price']:.5f} | "
              f"ITM={result['pct_itm']:.1%} | SE={result['std_err']:.6f}")

    pd.DataFrame(catalog_prices).to_csv(TABLES_DIR / "catalog_prices.csv", index=False)

    # ============================================================
    # STEP 5: BACKTEST STRATEGIES
    # ============================================================
    print("\n--- STEP 5: Backtesting ---")

    key_instruments = {
        "call_K0.02": catalog["call_K0.02"],
        "put_K0.02": catalog["put_K0.02"],
        "straddle_ATM": catalog["straddle_ATM"],
        "strangle_2pp": catalog["strangle_2pp"],
        "bull_spread_1_5": catalog["bull_spread_1_5"],
        "var_swap_50": catalog["var_swap_50"],
    }

    bt = Backtester(df, train_window=5000, step=500)
    
    print("\nRunning spot benchmark...")
    bench_result = bt.run_spot_benchmark()
    all_results = [bench_result]

    for name, deriv in key_instruments.items():
        print(f"Running {name}...")
        from src.derivatives import VarianceSwap
        if isinstance(deriv, VarianceSwap):
            deriv.set_strike_from_data(bt.delta_ip[:5000])
        result = bt.run_strategy(deriv, strategy_name=name)
        all_results.append(result)

    summary_rows = [r.summary() for r in all_results]
    summary_df = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
    print("\nBacktest Summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_df.to_csv(TABLES_DIR / "backtest_summary.csv", index=False)

    # Bootstrap CIs
    deriv_only = [r for r in all_results if "Spot" not in r.strategy_name]
    best_deriv = None
    if deriv_only:
        best_deriv = max(deriv_only, key=lambda r: r.sharpe)
        boot_ci = bt.bootstrap_sharpe_ci(best_deriv)
        print(f"\nBest derivative strategy: {best_deriv.strategy_name}")
        print(f"  Sharpe: {boot_ci['sharpe']:.3f} "
              f"[{boot_ci['ci_lower']:.3f}, {boot_ci['ci_upper']:.3f}]")
        print(f"  P(Sharpe > 0): {boot_ci['p_positive']:.3f}")

    if bench_result.n_trades > 10 and bench_result.std_pnl > 0:
        bench_boot = bt.bootstrap_sharpe_ci(bench_result)
        print(f"\nBenchmark (Spot Favorite):")
        print(f"  Sharpe: {bench_boot['sharpe']:.3f} "
              f"[{bench_boot['ci_lower']:.3f}, {bench_boot['ci_upper']:.3f}]")

    # Nuanced interpretation
    print(f"\n  Interpretation: Negative Sharpe ratios are consistent with")
    print(f"  market efficiency, but not proof of it. Alternative explanations")
    print(f"  include implicit transaction costs (the vig), model misspecification,")
    print(f"  and nonstationarity in the line-movement process.")
    print(f"  The key finding is relative: derivatives lose less than spot,")
    print(f"  and improve risk-adjusted returns when combined in portfolios.")
    print(f"\n  Note on pricing measure: Empirical (KDE) prices use the physical")
    print(f"  measure (P), not a risk-neutral measure (Q). This means prices")
    print(f"  represent fair value under historical expectations, not arbitrage-free")
    print(f"  prices. A risk premium λ would bridge P → Q: E^Q[·] = E^P[·] + λ.")

    # ============================================================
    # STEP 5b: PREDICTIVE ANALYSIS
    # ============================================================
    print("\n--- STEP 5b: Predictive Analysis ---")
    print("  (Does current Δp predict future Δp?)")

    predictive_results = run_predictive_analysis(df)

    acf_df = predictive_results["autocorrelation"]
    reg_df = predictive_results["regressions"]
    acf_df.to_csv(TABLES_DIR / "autocorrelation.csv", index=False)
    reg_df.to_csv(TABLES_DIR / "predictive_regressions.csv", index=False)

    # ============================================================
    # STEP 6: PORTFOLIO OPTIMIZATION
    # ============================================================
    print("\n--- STEP 6: Portfolio Optimization ---")

    opt = PortfolioOptimizer(df, process, catalog=key_instruments)
    frontiers = opt.compare_frontiers(n_points=30)

    print(f"\nSharpe improvement: {frontiers['sharpe_improvement']:.4f}")
    print(f"CVaR improvement: {frontiers['cvar_improvement']:.4f}")

    spot_opt = frontiers["spot_optimal"]
    deriv_opt = frontiers["deriv_optimal"]
    print(f"\nSpot-only optimal: SR={spot_opt.sharpe:.3f}, "
          f"vol={spot_opt.volatility:.4f}, ret={spot_opt.expected_return:.4f}")
    print(f"Derivative-augmented optimal: SR={deriv_opt.sharpe:.3f}, "
          f"vol={deriv_opt.volatility:.4f}, ret={deriv_opt.expected_return:.4f}")

    weights_df = pd.DataFrame({
        "asset": deriv_opt.asset_names,
        "weight": deriv_opt.weights,
    }).sort_values("weight", ascending=False)
    print("\nTop portfolio weights (derivative-augmented):")
    print(weights_df[weights_df["weight"].abs() > 0.01].to_string(index=False))
    weights_df.to_csv(TABLES_DIR / "optimal_weights.csv", index=False)

    # ============================================================
    # STEP 7: PASPA REGIME ANALYSIS
    # ============================================================
    print("\n--- STEP 7: PASPA Regime Analysis ---")

    regime_processes = extract_regime_processes(df, outcome="home")
    regime_comparison = compare_processes(regime_processes)
    print("\nProcess by Regime:")
    print(regime_comparison.to_string(index=False))
    regime_comparison.to_csv(TABLES_DIR / "process_by_regime.csv", index=False)

    regime_pricing = regime_pricing_comparison(regime_processes)
    print("\nDerivative Pricing by Regime:")
    print(regime_pricing.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    regime_pricing.to_csv(TABLES_DIR / "regime_pricing.csv", index=False)

    if "pre_paspa" in regime_processes and "post_paspa_ex_covid" in regime_processes:
        pre = regime_processes["pre_paspa"]
        post = regime_processes["post_paspa_ex_covid"]

        print(f"\n  Pre-PASPA:  σ={pre.std:.5f}, kurt={pre.kurtosis:.2f}, skew={pre.skew:.3f}")
        print(f"  Post-PASPA: σ={post.std:.5f}, kurt={post.kurtosis:.2f}, skew={post.skew:.3f}")

        vol_change = (post.std - pre.std) / pre.std * 100
        kurt_change = post.kurtosis - pre.kurtosis
        print(f"\n  Volatility change: {vol_change:+.1f}%")
        print(f"  Kurtosis change:  {kurt_change:+.2f}")

        if abs(vol_change) > 5:
            direction = "increased" if vol_change > 0 else "decreased"
            print(f"  → Line movement volatility {direction} after PASPA repeal")
        else:
            print(f"  → Line movement volatility roughly stable across regimes")

        if kurt_change > 1:
            print(f"  → Tails got FATTER post-PASPA (more extreme line moves)")
        elif kurt_change < -1:
            print(f"  → Tails got THINNER post-PASPA (less extreme line moves)")
        else:
            print(f"  → Tail thickness roughly stable across regimes")

    # ============================================================
    # STEP 8: BOOKMAKER MICROSTRUCTURE
    # ============================================================
    print("\n--- STEP 8: Bookmaker Microstructure ---")

    if processes_by_book:
        sharp_books = ["Pinnacle", "Bet-in-asia"]
        rec_books = ["888sport", "bet365", "bet-at-home"]

        for category, book_list in [("Sharp", sharp_books), ("Recreational", rec_books)]:
            matching = {k: v for k, v in processes_by_book.items() if k in book_list}
            if matching:
                avg_std = np.mean([p.std for p in matching.values()])
                avg_kurt = np.mean([p.kurtosis for p in matching.values()])
                avg_mean = np.mean([p.mean for p in matching.values()])
                print(f"\n  {category} books ({', '.join(matching.keys())}):")
                print(f"    Avg σ={avg_std:.5f}, Avg kurt={avg_kurt:.2f}, "
                      f"Avg mean(Δp)={avg_mean:+.5f}")

        if "Pinnacle" in processes_by_book:
            pin = processes_by_book["Pinnacle"]
            print(f"\n  Pinnacle line drift: {pin.mean:+.5f}")
            if pin.mean < -0.001:
                print(f"    → Sharp money systematically moves Pinnacle lines "
                      f"away from the home team")

        if "888sport" in processes_by_book:
            eight = processes_by_book["888sport"]
            print(f"  888sport line drift: {eight.mean:+.5f}")
            if eight.mean > 0.0005:
                print(f"    → Recreational action pushes 888sport lines "
                      f"toward the home team (public bias)")

    # ============================================================
    # STEP 9: GENERATE ALL FIGURES
    # ============================================================
    print("\n--- STEP 9: Generating Figures ---")

    plot_line_movement_distribution(process)

    top_sports = sorted(processes_by_sport.keys(),
                        key=lambda s: processes_by_sport[s].n_obs,
                        reverse=True)[:6]
    plot_distribution_by_segment(
        {s: processes_by_sport[s] for s in top_sports},
        segment_label="Sport",
    )

    plot_payoff_diagrams()
    plot_pricing_comparison(pricing_df)
    plot_three_model_pricing(pricing_df)
    plot_implied_vol_smile(smile_df)
    plot_mispricing_improvement(pricing_df)
    plot_backtest_cumulative(all_results)
    plot_strategy_summary_table(summary_df)

    if len(acf_df) > 0:
        plot_autocorrelation(acf_df)
    plot_efficient_frontiers(
        frontiers["frontiers"],
        frontiers.get("spot_optimal"),
        frontiers.get("deriv_optimal"),
    )

    if regime_processes:
        plot_regime_distributions(regime_processes)
        plot_regime_pricing_table(regime_pricing)

    print("  Rolling volatility (this may take a moment)...")
    df_sorted = df.sort_values("match_date")
    subsample = df_sorted.iloc[::10].copy()
    plot_rolling_volatility(subsample, window=500)

    if processes_by_book:
        plot_bookmaker_comparison(processes_by_book)

    # ============================================================
    # STEP 10: SAVE RESULTS
    # ============================================================
    print("\n--- STEP 10: Saving Results ---")

    results_summary = {
        "data": summary,
        "process_mean": process.mean,
        "process_std": process.std,
        "process_skew": process.skew,
        "process_kurtosis": process.kurtosis,
        "jb_pvalue": process.normality_test["jarque_bera"]["pvalue"],
        "n_instruments": len(catalog),
        "best_derivative": best_deriv.strategy_name if best_deriv else "N/A",
        "best_derivative_sharpe": best_deriv.sharpe if best_deriv else 0,
        "benchmark_sharpe": bench_result.sharpe,
        "sharpe_improvement_portfolio": frontiers["sharpe_improvement"],
        "cvar_improvement_portfolio": frontiers["cvar_improvement"],
        "student_t_df": t_fit["df"],
        "student_t_scale": t_fit["scale"],
        "ljung_box_p": predictive_results["ljung_box"]["p_value"],
    }

    if "pre_paspa" in regime_processes and "post_paspa_ex_covid" in regime_processes:
        pre = regime_processes["pre_paspa"]
        post = regime_processes["post_paspa_ex_covid"]
        results_summary["pre_paspa_sigma"] = pre.std
        results_summary["post_paspa_sigma"] = post.std
        results_summary["sigma_change_pct"] = (post.std - pre.std) / pre.std * 100
        results_summary["pre_paspa_kurtosis"] = pre.kurtosis
        results_summary["post_paspa_kurtosis"] = post.kurtosis

    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)

    with open(RESULTS_DIR / "analysis_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {RESULTS_DIR}")

    from config import FIGURES_DIR
    figs = list(FIGURES_DIR.glob("*.png"))
    print(f"\nFigures generated ({len(figs)}):")
    for f in sorted(figs):
        print(f"  {f.name}")

    tables = list(TABLES_DIR.glob("*.csv"))
    print(f"\nTables generated ({len(tables)}):")
    for t in sorted(tables):
        print(f"  {t.name}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
