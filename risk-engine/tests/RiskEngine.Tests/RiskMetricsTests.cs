using RiskEngine.Core;
using Xunit;

namespace RiskEngine.Tests;

public class RiskMetricsTests
{
    private static readonly double[] OneToFive = { 1, 2, 3, 4, 5 };

    [Fact]
    public void Mean_and_variance_match_hand_computed_values()
    {
        Assert.Equal(3.0, RiskMetrics.Mean(OneToFive), 12);
        Assert.Equal(2.5, RiskMetrics.Variance(OneToFive), 12);          // sample (N-1)
        Assert.Equal(2.0, RiskMetrics.Variance(OneToFive, sample: false), 12);
    }

    [Fact]
    public void Skewness_is_zero_for_symmetric_data()
    {
        double[] symmetric = { -2, -1, 0, 1, 2 };
        Assert.Equal(0.0, RiskMetrics.Skewness(symmetric), 10);
        Assert.Equal(-1.3, RiskMetrics.ExcessKurtosis(symmetric), 10);
    }

    [Fact]
    public void Percentile_uses_linear_interpolation()
    {
        var data = new double[101];
        for (int i = 0; i <= 100; i++) data[i] = i;
        Assert.Equal(50.0, RiskMetrics.PercentileSorted(data, 50.0), 8);
        Assert.Equal(95.0, RiskMetrics.PercentileSorted(data, 95.0), 8);
        Assert.Equal(0.0, RiskMetrics.PercentileSorted(data, 0.0), 8);
        Assert.Equal(100.0, RiskMetrics.PercentileSorted(data, 100.0), 8);
    }

    [Fact]
    public void Value_at_risk_and_cvar_match_hand_computed_values()
    {
        var pnl = new double[100];
        for (int i = 0; i < 100; i++) pnl[i] = i + 1; // 1..100

        // 5th percentile by interpolation = 5.95, so VaR (loss) = -5.95
        Assert.Equal(-5.95, RiskMetrics.ValueAtRisk(pnl, 0.95), 8);

        // mean of all values <= 5.95 is mean(1..5) = 3, so CVaR = -3
        Assert.Equal(-3.0, RiskMetrics.ConditionalVaR(pnl, 0.95), 8);
    }

    [Fact]
    public void Cvar_never_understates_var()
    {
        // A deliberately fat-tailed, asymmetric sample.
        var rng = new Random(123);
        var pnl = new double[20000];
        for (int i = 0; i < pnl.Length; i++)
        {
            double u = rng.NextDouble();
            pnl[i] = u < 0.05 ? -50.0 * rng.NextDouble() : rng.NextDouble();
        }
        double var95 = RiskMetrics.ValueAtRisk(pnl, 0.95);
        double cvar95 = RiskMetrics.ConditionalVaR(pnl, 0.95);
        double var99 = RiskMetrics.ValueAtRisk(pnl, 0.99);

        Assert.True(cvar95 >= var95, $"CVaR {cvar95} should be >= VaR {var95}");
        Assert.True(var99 >= var95, $"99% VaR {var99} should be >= 95% VaR {var95}");
    }

    [Fact]
    public void Probability_of_loss_counts_negative_paths()
    {
        double[] data = { -1, -2, 3, 4 };
        Assert.Equal(0.5, RiskMetrics.ProbabilityOfLoss(data), 12);
    }
}
