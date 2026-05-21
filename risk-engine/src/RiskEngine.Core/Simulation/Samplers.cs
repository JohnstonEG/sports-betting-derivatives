namespace RiskEngine.Core;

/// <summary>Draws realisations of the line-movement underlying Δp.</summary>
public interface IDistributionSampler
{
    string Name { get; }
    double Sample(Random rng);
}

/// <summary>Δp ~ Normal(mean, sigma) — the Bachelier model's data-generating process.</summary>
public sealed class NormalSampler : IDistributionSampler
{
    private readonly double _mean;
    private readonly double _sigma;

    public NormalSampler(double mean, double sigma)
    {
        _mean = mean;
        _sigma = Math.Max(sigma, 0.0);
    }

    public string Name => "Normal (Bachelier)";

    public double Sample(Random rng) => _mean + _sigma * rng.NextGaussian();
}

/// <summary>
/// Δp ~ scaled Student-t with the given degrees of freedom. The scale is chosen
/// so the sampled distribution has standard deviation equal to <c>sigma</c>,
/// matching the Student-t Bachelier model. Requires dof &gt; 2 for finite variance.
/// </summary>
public sealed class StudentTSampler : IDistributionSampler
{
    private readonly double _mean;
    private readonly double _dof;
    private readonly double _scale;

    public StudentTSampler(double mean, double sigma, double dof)
    {
        if (dof <= 2.0)
            throw new ArgumentOutOfRangeException(nameof(dof),
                "Student-t variance is finite only for dof > 2.");
        _mean = mean;
        _dof = dof;
        // Var(standard t) = dof/(dof-2); rescale so Var(Δp) = sigma².
        _scale = Math.Max(sigma, 0.0) * Math.Sqrt((dof - 2.0) / dof);
    }

    public string Name => $"Student-t (df={_dof:0.##})";

    public double Sample(Random rng)
    {
        double z = rng.NextGaussian();
        double w = rng.NextChiSquared(_dof);
        double standardT = z / Math.Sqrt(w / _dof);
        return _mean + _scale * standardT;
    }
}

/// <summary>
/// A regime-switching mixture: each draw first selects a regime in proportion
/// to its weight, then samples that regime's distribution. This reproduces the
/// fat-tailed, volatility-clustered behaviour seen across the PASPA regimes.
/// </summary>
public sealed class RegimeSwitchingSampler : IDistributionSampler
{
    private readonly IDistributionSampler[] _samplers;
    private readonly double[] _cumulativeWeights;

    public RegimeSwitchingSampler(IEnumerable<(IDistributionSampler sampler, double weight)> regimes)
    {
        var list = regimes.ToList();
        if (list.Count == 0)
            throw new ArgumentException("At least one regime is required.", nameof(regimes));

        double total = list.Sum(r => Math.Max(r.weight, 0.0));
        if (total <= 0.0)
            throw new ArgumentException("Regime weights must sum to a positive value.", nameof(regimes));

        _samplers = list.Select(r => r.sampler).ToArray();
        _cumulativeWeights = new double[list.Count];
        double running = 0.0;
        for (int i = 0; i < list.Count; i++)
        {
            running += Math.Max(list[i].weight, 0.0) / total;
            _cumulativeWeights[i] = running;
        }
    }

    public string Name => "Regime-switching mixture";

    public double Sample(Random rng)
    {
        double u = rng.NextDouble();
        for (int i = 0; i < _cumulativeWeights.Length; i++)
            if (u <= _cumulativeWeights[i])
                return _samplers[i].Sample(rng);
        return _samplers[^1].Sample(rng);
    }
}

/// <summary>Builds samplers from calibrated regime parameters.</summary>
public static class SamplerFactory
{
    /// <summary>
    /// Builds a sampler for one regime. Under the Student-t model, a fitted dof
    /// that cannot support finite variance falls back to the Normal model.
    /// </summary>
    public static IDistributionSampler Create(RegimeParameters p, PricingModel model)
    {
        if (model == PricingModel.Normal || p.StudentTDof <= 2.0)
            return new NormalSampler(p.Mean, p.Sigma);
        return new StudentTSampler(p.Mean, p.Sigma, p.StudentTDof);
    }

    /// <summary>Builds the regime-switching mixture over all supplied regimes.</summary>
    public static IDistributionSampler CreateRegimeSwitching(
        IEnumerable<RegimeParameters> regimes, PricingModel model)
    {
        var pairs = regimes.Select(r => (Create(r, model), r.Weight));
        return new RegimeSwitchingSampler(pairs);
    }
}
