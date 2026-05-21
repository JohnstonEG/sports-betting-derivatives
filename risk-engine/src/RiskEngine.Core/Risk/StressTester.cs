namespace RiskEngine.Core;

/// <summary>A single stress scenario: a multiplicative/additive shock to a base regime.</summary>
public sealed class StressScenario
{
    public string Name { get; set; } = "";

    /// <summary>Multiplier applied to the base sigma.</summary>
    public double VolMultiplier { get; set; } = 1.0;

    /// <summary>Additive shift applied to the base mean of Δp.</summary>
    public double MeanShift { get; set; } = 0.0;

    /// <summary>Override for the Student-t degrees of freedom (≤ 0 keeps the calibrated value).</summary>
    public double DofOverride { get; set; } = 0.0;

    public string Description { get; set; } = "";

    /// <summary>Applies this scenario's shocks to a baseline regime.</summary>
    public RegimeParameters Apply(RegimeParameters baseline) => new()
    {
        Name = Name,
        Mean = baseline.Mean + MeanShift,
        Sigma = baseline.Sigma * VolMultiplier,
        Skewness = baseline.Skewness,
        ExcessKurtosis = baseline.ExcessKurtosis,
        StudentTDof = DofOverride > 0.0 ? DofOverride : baseline.StudentTDof,
        Weight = 1.0,
        Notes = Description
    };
}

/// <summary>The outcome of one stress scenario.</summary>
public sealed class StressResult
{
    public required StressScenario Scenario { get; init; }
    public required RegimeParameters ShockedParameters { get; init; }
    public required RiskSummary Summary { get; init; }
}

/// <summary>Re-prices a portfolio under a battery of stress scenarios.</summary>
public sealed class StressTester
{
    private readonly MonteCarloEngine _engine;
    private readonly PricingModel _model;

    public StressTester(MonteCarloEngine engine, PricingModel model)
    {
        _engine = engine;
        _model = model;
    }

    public StressResult RunOne(Portfolio portfolio, RegimeParameters baseline, StressScenario scenario)
    {
        var shocked = scenario.Apply(baseline);
        var sampler = SamplerFactory.Create(shocked, _model);
        var sim = _engine.Run(portfolio, sampler);
        return new StressResult
        {
            Scenario = scenario,
            ShockedParameters = shocked,
            Summary = RiskSummary.FromPnl(scenario.Name, sim.PortfolioPnl)
        };
    }

    public List<StressResult> Run(Portfolio portfolio, RegimeParameters baseline,
        IEnumerable<StressScenario> scenarios)
    {
        var results = new List<StressResult>();
        foreach (var s in scenarios)
            results.Add(RunOne(portfolio, baseline, s));
        return results;
    }

    /// <summary>The default stress battery, themed on the documented market history.</summary>
    public static IReadOnlyList<StressScenario> DefaultBattery() => new[]
    {
        new StressScenario
        {
            Name = "Baseline", VolMultiplier = 1.0,
            Description = "Calibrated base regime, no shock."
        },
        new StressScenario
        {
            Name = "Vol +50%", VolMultiplier = 1.5,
            Description = "Volatility spike comparable to the COVID regime."
        },
        new StressScenario
        {
            Name = "Vol -30%", VolMultiplier = 0.7,
            Description = "Calm, efficient market with compressed line movement."
        },
        new StressScenario
        {
            Name = "Fat-tail shock", VolMultiplier = 1.2, DofOverride = 2.3,
            Description = "Heavier tails (low dof) plus a moderate volatility increase."
        },
        new StressScenario
        {
            Name = "Adverse drift", VolMultiplier = 1.1, MeanShift = -0.010,
            Description = "Systematic line move against the book's net exposure."
        },
        new StressScenario
        {
            Name = "Crisis", VolMultiplier = 1.8, MeanShift = -0.015, DofOverride = 2.2,
            Description = "Combined volatility, adverse drift and tail stress."
        }
    };
}
