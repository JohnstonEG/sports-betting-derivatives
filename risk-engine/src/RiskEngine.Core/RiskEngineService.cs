namespace RiskEngine.Core;

/// <summary>
/// Top-level orchestration: turns a portfolio plus a Python-generated calibration
/// set into a complete <see cref="RiskReport"/>. Shared by the CLI and the dashboard
/// so both surfaces produce identical numbers from the same engine.
/// </summary>
public sealed class RiskEngineService
{
    private readonly MonteCarloSettings _mcSettings;

    public RiskEngineService(MonteCarloSettings? mcSettings = null)
        => _mcSettings = mcSettings ?? MonteCarloSettings.Default;

    public MonteCarloSettings Settings => _mcSettings;

    /// <summary>
    /// Runs the full risk pipeline: headline metrics, per-position diagnostics,
    /// per-regime analysis, the regime-switching mixture, the stress battery and
    /// the volatility-scenario sweep.
    /// </summary>
    public RiskReport Analyze(Portfolio portfolio, CalibrationSet calibration,
        PricingModel? modelOverride = null)
    {
        var model = modelOverride ?? calibration.Model;
        var engine = new MonteCarloEngine(_mcSettings);

        // Headline run on the base (full-sample) regime.
        var baseSampler = SamplerFactory.Create(calibration.Base, model);
        var baseSim = engine.Run(portfolio, baseSampler);
        var headline = RiskSummary.FromPnl($"Base regime ({calibration.Base.Name})", baseSim.PortfolioPnl);

        // Per-position diagnostics derived from the base run.
        var positions = BuildPositionRisk(portfolio, baseSim);

        // Regime and scenario analysis.
        var analyzer = new ScenarioAnalyzer(engine, model);
        var regimeResults = analyzer.ByRegime(portfolio, calibration.Regimes);
        var regimeSwitching = analyzer.RegimeSwitching(portfolio, calibration.Regimes);
        var volCurve = analyzer.VolatilityCurve(portfolio, calibration.Base);

        // Stress battery.
        var stressTester = new StressTester(engine, model);
        var stress = stressTester.Run(portfolio, calibration.Base, StressTester.DefaultBattery());

        return new RiskReport
        {
            PortfolioName = portfolio.Name,
            GeneratedUtc = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss 'UTC'"),
            CalibrationSource = calibration.Source,
            DistributionModel = model.ToString(),
            Paths = _mcSettings.Paths,
            PositionCount = portfolio.Count,
            NetPremium = portfolio.NetPremium(),
            Headline = headline,
            PnlHistogram = HistogramData.From(Histogram.BuildClipped(baseSim.PortfolioPnl)),
            Positions = positions,
            Regimes = regimeResults.Select(r => r.Summary).ToList(),
            RegimeSwitching = regimeSwitching,
            StressScenarios = stress.Select(s => s.Summary).ToList(),
            VolatilityCurve = volCurve
        };
    }

    private static List<PositionRisk> BuildPositionRisk(Portfolio portfolio, SimulationResult sim)
    {
        double totalMean = 0.0;
        foreach (var leg in sim.PositionPnl.Values)
            totalMean += RiskMetrics.Mean(leg);

        var list = new List<PositionRisk>(portfolio.Count);
        foreach (var pos in portfolio.Positions)
        {
            var legPnl = sim.PositionPnl[pos.Instrument.Id];
            double meanPnl = RiskMetrics.Mean(legPnl);

            // Model value = mean payoff per contract = meanPnl / quantity + entry.
            double modelValue = Math.Abs(pos.Quantity) > 1e-12
                ? meanPnl / pos.Quantity + pos.EntryPrice
                : double.NaN;

            list.Add(new PositionRisk
            {
                InstrumentId = pos.Instrument.Id,
                InstrumentType = pos.Instrument.Type.ToString(),
                Quantity = pos.Quantity,
                EntryPrice = pos.EntryPrice,
                ModelValue = modelValue,
                MeanPnl = meanPnl,
                PnlStdDev = RiskMetrics.StdDev(legPnl),
                Var95 = RiskMetrics.ValueAtRisk(legPnl, 0.95),
                PnlShare = Math.Abs(totalMean) > 1e-12 ? meanPnl / totalMean : 0.0
            });
        }
        return list;
    }
}
