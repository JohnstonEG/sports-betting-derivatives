using RiskEngine.Core;
using Xunit;

namespace RiskEngine.Tests;

public class IoTests
{
    [Fact]
    public void Portfolio_json_round_trips()
    {
        var pf = new Portfolio { Name = "RoundTrip", BaseCurrency = "USD" };
        pf.Positions.Add(new Position
        {
            Instrument = new Instrument
            {
                Id = "C1", Type = InstrumentType.Call, Strike = 0.02, Description = "call"
            },
            Quantity = 10, EntryPrice = 0.0102, Tag = "long"
        });
        pf.Positions.Add(new Position
        {
            Instrument = new Instrument
            {
                Id = "STR", Type = InstrumentType.Strangle, Strike = -0.04, SecondStrike = 0.04
            },
            Quantity = -3, EntryPrice = 0.015
        });

        string path = Path.GetTempFileName();
        try
        {
            PortfolioLoader.SaveJson(pf, path);
            var loaded = PortfolioLoader.LoadJson(path);

            Assert.Equal("RoundTrip", loaded.Name);
            Assert.Equal(2, loaded.Positions.Count);
            Assert.Equal("C1", loaded.Positions[0].Instrument.Id);
            Assert.Equal(InstrumentType.Strangle, loaded.Positions[1].Instrument.Type);
            Assert.Equal(0.04, loaded.Positions[1].Instrument.SecondStrike, 12);
            Assert.Equal(-3.0, loaded.Positions[1].Quantity, 12);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Portfolio_csv_loads_with_an_order_independent_header()
    {
        string csv =
            "Id,Type,Quantity,Strike,SecondStrike,EntryPrice\n" +
            "C1,Call,5,0.02,,0.01\n" +
            "B1,BullCallSpread,3,0.02,0.06,0.006\n";

        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, csv);
            var pf = PortfolioLoader.LoadCsv(path);

            Assert.Equal(2, pf.Positions.Count);
            Assert.Equal(5.0, pf.Positions[0].Quantity, 12);
            Assert.Equal(InstrumentType.BullCallSpread, pf.Positions[1].Instrument.Type);
            Assert.Equal(0.06, pf.Positions[1].Instrument.SecondStrike, 12);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Duplicate_instrument_ids_are_rejected()
    {
        var pf = new Portfolio();
        pf.Positions.Add(new Position { Instrument = new Instrument { Id = "X", Type = InstrumentType.Call } });
        pf.Positions.Add(new Position { Instrument = new Instrument { Id = "X", Type = InstrumentType.Put } });

        string path = Path.GetTempFileName();
        try
        {
            PortfolioLoader.SaveJson(pf, path);
            Assert.Throws<InvalidDataException>(() => PortfolioLoader.LoadJson(path));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Calibration_json_loads_and_parses_the_model()
    {
        const string json = """
        {
          "source": "unit-test",
          "distributionModel": "StudentT",
          "sampleSize": 1000,
          "base": { "name": "Base", "mean": -0.001, "sigma": 0.05, "studentTDof": 2.7, "weight": 1.0 },
          "regimes": [
            { "name": "R1", "mean": 0.0, "sigma": 0.04, "studentTDof": 3.0, "weight": 1.0 }
          ]
        }
        """;

        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, json);
            var calib = CalibrationLoader.Load(path);

            Assert.Equal(PricingModel.StudentT, calib.Model);
            Assert.Equal(0.05, calib.Base.Sigma, 12);
            Assert.Equal(2.7, calib.Base.StudentTDof, 12);
            Assert.Single(calib.Regimes);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Bundled_sample_data_loads_and_runs_end_to_end()
    {
        // Integration check against the data/ files shipped with the engine.
        string? portfolioPath = DataLocator.TryResolve("portfolio.json");
        string? calibrationPath = DataLocator.TryResolve("calibration.json");
        if (portfolioPath is null || calibrationPath is null)
            return; // data folder not reachable from this runner - skip quietly

        var portfolio = PortfolioLoader.LoadJson(portfolioPath);
        var calibration = CalibrationLoader.Load(calibrationPath);
        var report = new RiskEngineService(new MonteCarloSettings { Paths = 10_000 })
            .Analyze(portfolio, calibration);

        Assert.True(report.PositionCount >= 1);
        Assert.Equal(10_000, report.Headline.Paths);
    }
}
