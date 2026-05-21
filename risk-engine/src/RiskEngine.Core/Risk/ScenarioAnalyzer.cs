namespace RiskEngine.Core;

/// <summary>One point on the volatility-scenario curve.</summary>
public sealed class VolScenarioPoint
{
    public double VolMultiplier { get; set; }
    public double Sigma { get; set; }
    public double MeanPnl { get; set; }
    public double Var95 { get; set; }
    public double CVar95 { get; set; }
    public double ProbabilityOfLoss { get; set; }
}

/// <summary>Risk under one calibrated regime.</summary>
public sealed class RegimeResult
{
    public required RegimeParameters Regime { get; init; }
    public required RiskSummary Summary { get; init; }
}

/// <summary>Regime-switching and volatility-scenario analysis of a portfolio.</summary>
public sealed class ScenarioAnalyzer
{
    private readonly MonteCarloEngine _engine;
    private readonly PricingModel _model;

    public ScenarioAnalyzer(MonteCarloEngine engine, PricingModel model)
    {
        _engine = engine;
        _model = model;
    }

    /// <summary>Runs the portfolio under each calibrated regime separately.</summary>
    public List<RegimeResult> ByRegime(Portfolio portfolio, IEnumerable<RegimeParameters> regimes)
    {
        var results = new List<RegimeResult>();
        foreach (var r in regimes)
        {
            var sampler = SamplerFactory.Create(r, _model);
            var sim = _engine.Run(portfolio, sampler);
            results.Add(new RegimeResult
            {
                Regime = r,
                Summary = RiskSummary.FromPnl(r.Name, sim.PortfolioPnl)
            });
        }
        return results;
    }

    /// <summary>Runs the portfolio under the regime-switching mixture of all regimes.</summary>
    public RiskSummary RegimeSwitching(Portfolio portfolio, IEnumerable<RegimeParameters> regimes)
    {
        var sampler = SamplerFactory.CreateRegimeSwitching(regimes, _model);
        var sim = _engine.Run(portfolio, sampler);
        return RiskSummary.FromPnl("Regime-switching mixture", sim.PortfolioPnl);
    }

    /// <summary>Sweeps a range of volatility multipliers against the base regime.</summary>
    public List<VolScenarioPoint> VolatilityCurve(Portfolio portfolio, RegimeParameters baseline,
        double from = 0.5, double to = 2.0, int steps = 16)
    {
        if (steps < 2) steps = 2;
        var points = new List<VolScenarioPoint>(steps);
        for (int i = 0; i < steps; i++)
        {
            double mult = from + (to - from) * i / (steps - 1);
            var shocked = new RegimeParameters
            {
                Name = $"Vol x{mult:0.00}",
                Mean = baseline.Mean,
                Sigma = baseline.Sigma * mult,
                Skewness = baseline.Skewness,
                ExcessKurtosis = baseline.ExcessKurtosis,
                StudentTDof = baseline.StudentTDof,
                Weight = 1.0
            };
            var sampler = SamplerFactory.Create(shocked, _model);
            var sim = _engine.Run(portfolio, sampler);

            var sorted = sim.PortfolioPnl.ToArray();
            Array.Sort(sorted);
            points.Add(new VolScenarioPoint
            {
                VolMultiplier = mult,
                Sigma = shocked.Sigma,
                MeanPnl = RiskMetrics.Mean(sim.PortfolioPnl),
                Var95 = RiskMetrics.ValueAtRiskSorted(sorted, 0.95),
                CVar95 = RiskMetrics.ConditionalVaRSorted(sorted, 0.95),
                ProbabilityOfLoss = RiskMetrics.ProbabilityOfLoss(sim.PortfolioPnl)
            });
        }
        return points;
    }
}
