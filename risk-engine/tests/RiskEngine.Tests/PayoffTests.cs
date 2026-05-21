using RiskEngine.Core;
using Xunit;

namespace RiskEngine.Tests;

public class PayoffTests
{
    [Fact]
    public void Call_pays_intrinsic_value_above_strike()
    {
        var c = new CallPayoff(0.02);
        Assert.Equal(0.03, c.Evaluate(0.05), 12);
        Assert.Equal(0.0, c.Evaluate(0.02), 12);
        Assert.Equal(0.0, c.Evaluate(-0.10), 12);
    }

    [Fact]
    public void Put_pays_intrinsic_value_below_strike()
    {
        var p = new PutPayoff(0.02);
        Assert.Equal(0.02, p.Evaluate(0.0), 12);
        Assert.Equal(0.0, p.Evaluate(0.05), 12);
        Assert.Equal(0.07, p.Evaluate(-0.05), 12);
    }

    [Fact]
    public void Digital_call_is_cash_or_nothing_and_strict()
    {
        var d = new DigitalCallPayoff(0.02);
        Assert.Equal(1.0, d.Evaluate(0.03), 12);
        Assert.Equal(0.0, d.Evaluate(0.01), 12);
        Assert.Equal(0.0, d.Evaluate(0.02), 12); // strict inequality at the strike
    }

    [Fact]
    public void Straddle_pays_absolute_distance_from_strike()
    {
        var s = new StraddlePayoff(0.0);
        Assert.Equal(0.04, s.Evaluate(0.04), 12);
        Assert.Equal(0.04, s.Evaluate(-0.04), 12);
        Assert.Equal(0.0, s.Evaluate(0.0), 12);
    }

    [Fact]
    public void Strangle_pays_only_outside_the_wings()
    {
        var s = new StranglePayoff(-0.04, 0.04);
        Assert.Equal(0.0, s.Evaluate(0.0), 12);
        Assert.Equal(0.06, s.Evaluate(0.10), 12);
        Assert.Equal(0.06, s.Evaluate(-0.10), 12);
    }

    [Fact]
    public void Bull_call_spread_is_capped()
    {
        var s = new BullCallSpreadPayoff(0.02, 0.06);
        Assert.Equal(0.0, s.Evaluate(0.0), 12);
        Assert.Equal(0.02, s.Evaluate(0.04), 12);
        Assert.Equal(0.04, s.Evaluate(0.20), 12); // capped at K2 - K1
    }

    [Fact]
    public void Bear_put_spread_is_capped()
    {
        var s = new BearPutSpreadPayoff(0.02, 0.06);
        Assert.Equal(0.04, s.Evaluate(0.0), 12);   // capped at K2 - K1
        Assert.Equal(0.0, s.Evaluate(0.10), 12);
    }

    [Fact]
    public void Butterfly_peaks_at_the_body_and_is_zero_at_the_wings()
    {
        var s = new ButterflyPayoff(0.0, 0.05, 0.10);
        Assert.Equal(0.0, s.Evaluate(0.0), 12);
        Assert.Equal(0.05, s.Evaluate(0.05), 12);  // peak at the body strike
        Assert.Equal(0.0, s.Evaluate(0.10), 12);
        Assert.Equal(0.0, s.Evaluate(0.20), 12);
    }

    [Fact]
    public void Variance_swap_pays_squared_move_less_strike()
    {
        var s = new VarianceSwapPayoff(0.0025);
        Assert.Equal(0.0, s.Evaluate(0.05), 12);
        Assert.Equal(0.0075, s.Evaluate(0.10), 12);
    }

    [Theory]
    [InlineData(-0.08)]
    [InlineData(0.0)]
    [InlineData(0.035)]
    [InlineData(0.12)]
    public void Put_call_parity_holds_at_the_payoff_level(double deltaP)
    {
        var call = new CallPayoff(0.02);
        var put = new PutPayoff(0.02);
        // call - put = underlying - strike, for any realised move
        Assert.Equal(deltaP - 0.02, call.Evaluate(deltaP) - put.Evaluate(deltaP), 12);
    }

    [Fact]
    public void Factory_builds_the_correct_payoff_for_each_type()
    {
        Assert.IsType<CallPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.Call }));
        Assert.IsType<PutPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.Put }));
        Assert.IsType<DigitalCallPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.DigitalCall }));
        Assert.IsType<DigitalPutPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.DigitalPut }));
        Assert.IsType<StraddlePayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.Straddle }));
        Assert.IsType<StranglePayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.Strangle }));
        Assert.IsType<BullCallSpreadPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.BullCallSpread }));
        Assert.IsType<BearPutSpreadPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.BearPutSpread }));
        Assert.IsType<ButterflyPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.Butterfly }));
        Assert.IsType<VarianceSwapPayoff>(PayoffFactory.Create(new Instrument { Type = InstrumentType.VarianceSwap }));
    }

    [Fact]
    public void Position_pnl_nets_off_the_entry_premium()
    {
        var pos = new Position
        {
            Instrument = new Instrument { Id = "C", Type = InstrumentType.Call, Strike = 0.02 },
            Quantity = 10,
            EntryPrice = 0.01
        };
        // payoff at dp = 0.05 is 0.03; pnl = 10 * (0.03 - 0.01) = 0.20
        Assert.Equal(0.20, pos.ProfitAndLoss(0.05), 12);
    }
}
