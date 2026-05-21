namespace RiskEngine.Core;

/// <summary>Configuration for a Monte Carlo run.</summary>
public sealed class MonteCarloSettings
{
    /// <summary>Number of simulated paths.</summary>
    public int Paths { get; set; } = 50_000;

    /// <summary>
    /// RNG seed. A fixed seed makes every run reproducible and means scenario
    /// comparisons use common random numbers — a standard variance-reduction
    /// technique that isolates the effect of a parameter change.
    /// </summary>
    public int Seed { get; set; } = 20240517;

    public static MonteCarloSettings Default => new();
}

/// <summary>The result of one Monte Carlo run over a portfolio.</summary>
public sealed class SimulationResult
{
    public required string SamplerName { get; init; }
    public required int Paths { get; init; }

    /// <summary>The sampled underlying draws, one per path.</summary>
    public required double[] DeltaP { get; init; }

    /// <summary>Total portfolio P&amp;L, one per path.</summary>
    public required double[] PortfolioPnl { get; init; }

    /// <summary>Per-instrument P&amp;L paths, keyed by instrument id.</summary>
    public required Dictionary<string, double[]> PositionPnl { get; init; }
}

/// <summary>
/// Monte Carlo simulation of portfolio P&amp;L under a chosen Δp distribution.
/// Deterministic for a fixed seed, so results are reproducible and unit-testable.
/// </summary>
public sealed class MonteCarloEngine
{
    private readonly MonteCarloSettings _settings;

    public MonteCarloEngine(MonteCarloSettings? settings = null)
        => _settings = settings ?? MonteCarloSettings.Default;

    public MonteCarloSettings Settings => _settings;

    /// <summary>Simulates the portfolio under the supplied Δp sampler.</summary>
    public SimulationResult Run(Portfolio portfolio, IDistributionSampler sampler)
    {
        int n = Math.Max(1, _settings.Paths);
        var rng = new Random(_settings.Seed);

        // Pre-build payoffs once so the hot loop only does arithmetic.
        int m = portfolio.Positions.Count;
        var payoffs = new IPayoff[m];
        var quantities = new double[m];
        var entries = new double[m];
        var ids = new string[m];
        var positionPnl = new Dictionary<string, double[]>(m);
        for (int j = 0; j < m; j++)
        {
            var pos = portfolio.Positions[j];
            payoffs[j] = PayoffFactory.Create(pos.Instrument);
            quantities[j] = pos.Quantity;
            entries[j] = pos.EntryPrice;
            ids[j] = pos.Instrument.Id;
            positionPnl[pos.Instrument.Id] = new double[n];
        }

        var deltaP = new double[n];
        var pnl = new double[n];

        for (int i = 0; i < n; i++)
        {
            double dp = sampler.Sample(rng);
            deltaP[i] = dp;

            double total = 0.0;
            for (int j = 0; j < m; j++)
            {
                double legPnl = quantities[j] * (payoffs[j].Evaluate(dp) - entries[j]);
                positionPnl[ids[j]][i] = legPnl;
                total += legPnl;
            }
            pnl[i] = total;
        }

        return new SimulationResult
        {
            SamplerName = sampler.Name,
            Paths = n,
            DeltaP = deltaP,
            PortfolioPnl = pnl,
            PositionPnl = positionPnl
        };
    }
}
