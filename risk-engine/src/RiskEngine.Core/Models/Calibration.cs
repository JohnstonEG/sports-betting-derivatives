namespace RiskEngine.Core;

/// <summary>The distributional model used to simulate the line-movement underlying.</summary>
public enum PricingModel
{
    /// <summary>Δp ~ Normal — the Bachelier (arithmetic) model.</summary>
    Normal,

    /// <summary>Δp ~ scaled Student-t — the fat-tailed Student-t Bachelier model.</summary>
    StudentT
}

/// <summary>
/// Distributional parameters for the line-movement process Δp within a regime.
/// Produced by the Python calibration pipeline (see tools/export_calibration.py).
/// </summary>
public sealed class RegimeParameters
{
    public string Name { get; set; } = "";

    /// <summary>Mean of Δp.</summary>
    public double Mean { get; set; }

    /// <summary>Standard deviation of Δp.</summary>
    public double Sigma { get; set; }

    public double Skewness { get; set; }

    public double ExcessKurtosis { get; set; }

    /// <summary>Fitted Student-t degrees of freedom (requires &gt; 2 for finite variance).</summary>
    public double StudentTDof { get; set; } = 5.0;

    /// <summary>Historical fraction of observations in this regime (mixture weight).</summary>
    public double Weight { get; set; } = 1.0;

    public string Notes { get; set; } = "";
}

/// <summary>
/// The full set of calibrated parameters consumed by the C# risk engine —
/// the bridge between the Python research pipeline and this engine.
/// </summary>
public sealed class CalibrationSet
{
    public string Source { get; set; } = "";

    public string GeneratedUtc { get; set; } = "";

    public long SampleSize { get; set; }

    /// <summary>"Normal" or "StudentT" — the default model to simulate with.</summary>
    public string DistributionModel { get; set; } = "StudentT";

    /// <summary>Full-sample parameters.</summary>
    public RegimeParameters Base { get; set; } = new();

    /// <summary>Per-regime parameters (e.g. pre-/post-PASPA, COVID).</summary>
    public List<RegimeParameters> Regimes { get; set; } = new();

    /// <summary>The default <see cref="PricingModel"/> parsed from <see cref="DistributionModel"/>.</summary>
    public PricingModel Model =>
        string.Equals(DistributionModel, "Normal", StringComparison.OrdinalIgnoreCase)
            ? PricingModel.Normal
            : PricingModel.StudentT;
}
