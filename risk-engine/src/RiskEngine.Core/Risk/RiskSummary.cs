namespace RiskEngine.Core;

/// <summary>A serialisable snapshot of risk metrics for a single P&amp;L distribution.</summary>
public sealed class RiskSummary
{
    public string Label { get; set; } = "";
    public int Paths { get; set; }
    public double MeanPnl { get; set; }
    public double StdDevPnl { get; set; }
    public double Skewness { get; set; }
    public double ExcessKurtosis { get; set; }
    public double MinPnl { get; set; }
    public double MaxPnl { get; set; }
    public double Median { get; set; }

    /// <summary>95% Value at Risk (positive = loss).</summary>
    public double Var95 { get; set; }

    /// <summary>99% Value at Risk (positive = loss).</summary>
    public double Var99 { get; set; }

    /// <summary>95% Conditional VaR / expected shortfall (positive = loss).</summary>
    public double CVar95 { get; set; }

    /// <summary>99% Conditional VaR / expected shortfall (positive = loss).</summary>
    public double CVar99 { get; set; }

    /// <summary>Fraction of paths that lose money.</summary>
    public double ProbabilityOfLoss { get; set; }

    /// <summary>Mean P&amp;L divided by its standard deviation (a Sharpe-like ratio).</summary>
    public double SharpeLike { get; set; }

    /// <summary>Builds a summary from a P&amp;L sample, sorting once for all quantiles.</summary>
    public static RiskSummary FromPnl(string label, IReadOnlyList<double> pnl)
    {
        var sorted = pnl.ToArray();
        Array.Sort(sorted);

        double mean = RiskMetrics.Mean(pnl);
        double sd = RiskMetrics.StdDev(pnl);

        return new RiskSummary
        {
            Label = label,
            Paths = pnl.Count,
            MeanPnl = mean,
            StdDevPnl = sd,
            Skewness = RiskMetrics.Skewness(pnl),
            ExcessKurtosis = RiskMetrics.ExcessKurtosis(pnl),
            MinPnl = sorted.Length > 0 ? sorted[0] : double.NaN,
            MaxPnl = sorted.Length > 0 ? sorted[^1] : double.NaN,
            Median = RiskMetrics.PercentileSorted(sorted, 50.0),
            Var95 = RiskMetrics.ValueAtRiskSorted(sorted, 0.95),
            Var99 = RiskMetrics.ValueAtRiskSorted(sorted, 0.99),
            CVar95 = RiskMetrics.ConditionalVaRSorted(sorted, 0.95),
            CVar99 = RiskMetrics.ConditionalVaRSorted(sorted, 0.99),
            ProbabilityOfLoss = RiskMetrics.ProbabilityOfLoss(pnl),
            SharpeLike = sd > 1e-12 ? mean / sd : 0.0
        };
    }
}
