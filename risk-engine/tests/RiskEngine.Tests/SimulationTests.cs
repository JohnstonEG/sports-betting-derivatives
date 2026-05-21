using RiskEngine.Core;
using Xunit;

namespace RiskEngine.Tests;

public class SimulationTests
{
    private static Portfolio SingleCall(double strike, double quantity, double entry)
    {
        var pf = new Portfolio { Name = "Single" };
        pf.Positions.Add(new Position
        {
            Instrument = new Instrument { Id = "C", Type = InstrumentType.Call, Strike = strike },
            Quantity = quantity,
            EntryPrice = entry
        });
        return pf;
    }

    [Fact]
    public void Box_muller_gaussian_has_unit_mean_and_variance()
    {
        var rng = new Random(7);
        const int n = 300_000;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < n; i++)
        {
            double x = rng.NextGaussian();
            sum += x;
            sumSq += x * x;
        }
        double mean = sum / n;
        double std = Math.Sqrt(sumSq / n - mean * mean);
        Assert.True(Math.Abs(mean) < 0.02, $"mean = {mean}");
        Assert.True(Math.Abs(std - 1.0) < 0.02, $"std = {std}");
    }

    [Theory]
    [InlineData(3.0, 2.0)]   // shape >= 1 path
    [InlineData(0.5, 2.0)]   // shape < 1 boost path
    public void Gamma_sampler_mean_matches_shape_times_scale(double shape, double scale)
    {
        var rng = new Random(11);
        const int n = 300_000;
        double sum = 0;
        for (int i = 0; i < n; i++) sum += rng.NextGamma(shape, scale);
        double mean = sum / n;
        double expected = shape * scale;
        Assert.True(Math.Abs(mean - expected) < expected * 0.03, $"mean = {mean}, expected {expected}");
    }

    [Fact]
    public void Student_t_sampler_is_scaled_to_the_target_sigma()
    {
        // df = 8 keeps the fourth moment finite, so the sample std is well behaved.
        var sampler = new StudentTSampler(0.0, 0.05, 8.0);
        var rng = new Random(999);
        const int n = 300_000;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < n; i++)
        {
            double x = sampler.Sample(rng);
            sum += x;
            sumSq += x * x;
        }
        double mean = sum / n;
        double std = Math.Sqrt(sumSq / n - mean * mean);
        Assert.True(Math.Abs(std - 0.05) < 0.05 * 0.06, $"std = {std}");
    }

    [Fact]
    public void Student_t_sampler_rejects_non_finite_variance()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTSampler(0.0, 0.05, 2.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTSampler(0.0, 0.05, 1.5));
    }

    [Fact]
    public void Monte_carlo_is_deterministic_for_a_fixed_seed()
    {
        var pf = SingleCall(0.02, 5.0, 0.01);
        var engine = new MonteCarloEngine(new MonteCarloSettings { Paths = 8_000, Seed = 42 });
        var sampler = new NormalSampler(-0.0013, 0.0526);

        var first = engine.Run(pf, sampler);
        var second = engine.Run(pf, sampler);

        Assert.Equal(first.PortfolioPnl, second.PortfolioPnl);
        Assert.Equal(8_000, first.PortfolioPnl.Length);
    }

    [Fact]
    public void Monte_carlo_call_price_converges_to_the_bachelier_value()
    {
        const double mean = -0.0013, sigma = 0.0526, strike = 0.02;
        var pf = SingleCall(strike, 1.0, 0.0); // qty 1, no premium => pnl == payoff

        var engine = new MonteCarloEngine(new MonteCarloSettings { Paths = 400_000, Seed = 20240517 });
        var sim = engine.Run(pf, new NormalSampler(mean, sigma));

        double mc = RiskMetrics.Mean(sim.PortfolioPnl);
        double analytic = Bachelier.CallPrice(mean, sigma, strike);

        Assert.True(Math.Abs(mc - analytic) < 8e-4,
            $"Monte Carlo {mc} should converge to Bachelier {analytic}");
    }

    [Fact]
    public void Regime_switching_sampler_requires_positive_weights()
    {
        Assert.Throws<ArgumentException>(() =>
            new RegimeSwitchingSampler(Array.Empty<(IDistributionSampler, double)>()));
    }
}
