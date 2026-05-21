namespace RiskEngine.Core;

/// <summary>Per-position diagnostics evaluated within the base regime.</summary>
public sealed class PositionRisk
{
    public string InstrumentId { get; set; } = "";
    public string InstrumentType { get; set; } = "";
    public double Quantity { get; set; }
    public double EntryPrice { get; set; }

    /// <summary>Monte Carlo model value (mean payoff per contract) under the base regime.</summary>
    public double ModelValue { get; set; }

    /// <summary>Mean P&amp;L contribution of this leg.</summary>
    public double MeanPnl { get; set; }

    public double PnlStdDev { get; set; }

    /// <summary>Standalone 95% VaR of this leg (positive = loss).</summary>
    public double Var95 { get; set; }

    /// <summary>Share of the portfolio's total mean P&amp;L attributable to this leg.</summary>
    public double PnlShare { get; set; }
}

/// <summary>Histogram payload in a serialisation-friendly shape.</summary>
public sealed class HistogramData
{
    public double[] BinCenters { get; set; } = Array.Empty<double>();
    public int[] Counts { get; set; } = Array.Empty<int>();
    public double BinWidth { get; set; }

    public static HistogramData From(Histogram h) => new()
    {
        BinCenters = h.BinCenters,
        Counts = h.Counts,
        BinWidth = h.BinWidth
    };
}

/// <summary>The complete risk report produced by <see cref="RiskEngineService"/>.</summary>
public sealed class RiskReport
{
    public string PortfolioName { get; set; } = "";
    public string GeneratedUtc { get; set; } = "";
    public string CalibrationSource { get; set; } = "";
    public string DistributionModel { get; set; } = "";
    public int Paths { get; set; }
    public int PositionCount { get; set; }

    /// <summary>Net premium outlay across the book (positive = premium paid).</summary>
    public double NetPremium { get; set; }

    /// <summary>Headline risk metrics under the base (full-sample) regime.</summary>
    public RiskSummary Headline { get; set; } = new();

    /// <summary>P&amp;L distribution histogram for the base regime.</summary>
    public HistogramData PnlHistogram { get; set; } = new();

    public List<PositionRisk> Positions { get; set; } = new();

    /// <summary>Risk metrics under each calibrated regime.</summary>
    public List<RiskSummary> Regimes { get; set; } = new();

    /// <summary>Risk metrics under the regime-switching mixture.</summary>
    public RiskSummary RegimeSwitching { get; set; } = new();

    /// <summary>Risk metrics under each stress scenario.</summary>
    public List<RiskSummary> StressScenarios { get; set; } = new();

    /// <summary>The volatility-scenario sweep.</summary>
    public List<VolScenarioPoint> VolatilityCurve { get; set; } = new();
}
