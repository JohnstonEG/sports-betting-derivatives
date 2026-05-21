using RiskEngine.Core;
using Xunit;

namespace RiskEngine.Tests;

public class EngineTests
{
    private static CalibrationSet TestCalibration()
    {
        var c = new CalibrationSet
        {
            Source = "unit-test",
            DistributionModel = "StudentT",
            SampleSize = 1000,
            Base = new RegimeParameters
            {
                Name = "Base", Mean = -0.0013, Sigma = 0.0526, StudentTDof = 2.70, Weight = 1.0
            }
        };
        c.Regimes.Add(new RegimeParameters { Name = "Calm", Mean = 0.0, Sigma = 0.040, StudentTDof = 3.0, Weight = 0.5 });
        c.Regimes.Add(new RegimeParameters { Name = "Stormy", Mean = 0.0, Sigma = 0.070, StudentTDof = 3.0, Weight = 0.5 });
        return c;
    }

    private static Portfolio TwoLegBook()
    {
        var pf = new Portfolio { Name = "Test Book" };
        pf.Positions.Add(new Position
        {
            Instrument = new Instrument { Id = "CALL", Type = InstrumentType.Call, Strike = 0.02 },
            Quantity = 10, EntryPrice = 0.011
        });
        pf.Positions.Add(new Position
        {
            Instrument = new Instrument { Id = "STRD", Type = InstrumentType.Straddle, Strike = 0.0 },
            Quantity = -5, EntryPrice = 0.036
        });
        return pf;
    }

    [Fact]
    public void Analyze_produces_a_complete_report()
    {
        var service = new RiskEngineService(new MonteCarloSettings { Paths = 20_000 });
        var report = service.Analyze(TwoLegBook(), TestCalibration());

        Assert.Equal(2, report.PositionCount);
        Assert.Equal(2, report.Positions.Count);
        Assert.Equal(2, report.Regimes.Count);
        Assert.Equal(6, report.StressScenarios.Count);   // the default stress battery
        Assert.NotEmpty(report.VolatilityCurve);
        Assert.True(report.PnlHistogram.Counts.Length > 0);

        // Tail-risk invariants must hold on the headline distribution.
        Assert.True(report.Headline.CVar95 >= report.Headline.Var95);
        Assert.True(report.Headline.CVar99 >= report.Headline.Var99);
        Assert.True(report.Headline.Var99 >= report.Headline.Var95);
    }

    [Fact]
    public void Student_t_model_has_tails_at_least_as_heavy_as_normal()
    {
        var service = new RiskEngineService(new MonteCarloSettings { Paths = 80_000 });
        var book = TwoLegBook();
        var calib = TestCalibration();

        var normal = service.Analyze(book, calib, PricingModel.Normal);
        var studentT = service.Analyze(book, calib, PricingModel.StudentT);

        Assert.Equal("Normal", normal.DistributionModel);
        Assert.Equal("StudentT", studentT.DistributionModel);

        // The fitted Student-t (df = 2.70) is markedly fatter-tailed than the
        // normal, so 99% tail risk should not be smaller.
        Assert.True(studentT.Headline.Var99 >= normal.Headline.Var99,
            $"Student-t Var99 {studentT.Headline.Var99} vs normal {normal.Headline.Var99}");
    }

    [Fact]
    public void Histogram_counts_sum_to_the_sample_size()
    {
        var data = new double[1000];
        for (int i = 0; i < data.Length; i++) data[i] = i;

        var h = Histogram.Build(data, 20);
        Assert.Equal(20, h.Counts.Length);
        Assert.Equal(1000, h.Total);
        Assert.Equal(1000, h.Counts.Sum());
    }

    [Fact]
    public void Higher_volatility_raises_var_for_a_short_volatility_book()
    {
        var pf = new Portfolio { Name = "Short Straddle" };
        pf.Positions.Add(new Position
        {
            Instrument = new Instrument { Id = "STRD", Type = InstrumentType.Straddle, Strike = 0.0 },
            Quantity = -10, EntryPrice = 0.036
        });

        var engine = new MonteCarloEngine(new MonteCarloSettings { Paths = 60_000 });
        var tester = new StressTester(engine, PricingModel.Normal);
        var baseRegime = new RegimeParameters { Name = "Base", Mean = 0.0, Sigma = 0.05, StudentTDof = 5.0 };

        var baseline = tester.RunOne(pf, baseRegime,
            new StressScenario { Name = "Baseline", VolMultiplier = 1.0 });
        var shocked = tester.RunOne(pf, baseRegime,
            new StressScenario { Name = "Vol +50%", VolMultiplier = 1.5 });

        Assert.True(shocked.Summary.Var95 > baseline.Summary.Var95,
            $"shocked VaR {shocked.Summary.Var95} should exceed baseline {baseline.Summary.Var95}");
    }

    [Fact]
    public void Clipped_histogram_excludes_the_extreme_tails()
    {
        var data = new double[1000];
        for (int i = 0; i < data.Length; i++) data[i] = i; // 0..999
        data[0] = -100_000.0;   // extreme low outlier
        data[999] = 100_000.0;  // extreme high outlier

        var h = Histogram.BuildClipped(data, 20, 1.0, 99.0);

        // Bin edges sit at the 1st/99th percentile, far inside the outliers.
        Assert.True(h.BinEdges[0] > -100_000.0);
        Assert.True(h.BinEdges[^1] < 100_000.0);

        // The outliers are excluded from the counts, but Total is the full sample.
        Assert.Equal(1000, h.Total);
        Assert.True(h.Counts.Sum() < data.Length);
        Assert.True(h.Counts.Sum() > 0);
    }
}
