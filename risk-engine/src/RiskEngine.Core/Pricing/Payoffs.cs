namespace RiskEngine.Core;

/// <summary>Maps a realised line move Δp to a per-contract payoff.</summary>
public interface IPayoff
{
    double Evaluate(double deltaP);
}

/// <summary>Call on Δp: max(Δp − K, 0).</summary>
public sealed class CallPayoff(double strike) : IPayoff
{
    public double Strike => strike;
    public double Evaluate(double deltaP) => Math.Max(deltaP - strike, 0.0);
}

/// <summary>Put on Δp: max(K − Δp, 0).</summary>
public sealed class PutPayoff(double strike) : IPayoff
{
    public double Strike => strike;
    public double Evaluate(double deltaP) => Math.Max(strike - deltaP, 0.0);
}

/// <summary>Cash-or-nothing digital call: pays 1 if Δp &gt; K, else 0.</summary>
public sealed class DigitalCallPayoff(double strike) : IPayoff
{
    public double Strike => strike;
    public double Evaluate(double deltaP) => deltaP > strike ? 1.0 : 0.0;
}

/// <summary>Cash-or-nothing digital put: pays 1 if Δp &lt; K, else 0.</summary>
public sealed class DigitalPutPayoff(double strike) : IPayoff
{
    public double Strike => strike;
    public double Evaluate(double deltaP) => deltaP < strike ? 1.0 : 0.0;
}

/// <summary>Straddle: long call + long put at the same strike, payoff |Δp − K|.</summary>
public sealed class StraddlePayoff(double strike) : IPayoff
{
    public double Strike => strike;
    public double Evaluate(double deltaP) => Math.Abs(deltaP - strike);
}

/// <summary>Strangle: long OTM put at K1 + long OTM call at K2 (K1 &lt; K2).</summary>
public sealed class StranglePayoff(double putStrike, double callStrike) : IPayoff
{
    public double Evaluate(double deltaP)
        => Math.Max(putStrike - deltaP, 0.0) + Math.Max(deltaP - callStrike, 0.0);
}

/// <summary>Bull call spread: long call at K1, short call at K2 (K1 &lt; K2).</summary>
public sealed class BullCallSpreadPayoff(double lowStrike, double highStrike) : IPayoff
{
    public double Evaluate(double deltaP)
        => Math.Max(deltaP - lowStrike, 0.0) - Math.Max(deltaP - highStrike, 0.0);
}

/// <summary>Bear put spread: long put at K2, short put at K1 (K1 &lt; K2).</summary>
public sealed class BearPutSpreadPayoff(double lowStrike, double highStrike) : IPayoff
{
    public double Evaluate(double deltaP)
        => Math.Max(highStrike - deltaP, 0.0) - Math.Max(lowStrike - deltaP, 0.0);
}

/// <summary>Butterfly: long K1, short 2× K2, long K3 (K1 &lt; K2 &lt; K3).</summary>
public sealed class ButterflyPayoff(double lowStrike, double bodyStrike, double highStrike) : IPayoff
{
    public double Evaluate(double deltaP)
        => Math.Max(deltaP - lowStrike, 0.0)
           - 2.0 * Math.Max(deltaP - bodyStrike, 0.0)
           + Math.Max(deltaP - highStrike, 0.0);
}

/// <summary>Variance swap: Δp² − K, a single-period realised-variance proxy.</summary>
public sealed class VarianceSwapPayoff(double varianceStrike) : IPayoff
{
    public double Evaluate(double deltaP) => deltaP * deltaP - varianceStrike;
}

/// <summary>Builds the correct <see cref="IPayoff"/> for an instrument.</summary>
public static class PayoffFactory
{
    public static IPayoff Create(Instrument inst) => inst.Type switch
    {
        InstrumentType.Call           => new CallPayoff(inst.Strike),
        InstrumentType.Put            => new PutPayoff(inst.Strike),
        InstrumentType.DigitalCall    => new DigitalCallPayoff(inst.Strike),
        InstrumentType.DigitalPut     => new DigitalPutPayoff(inst.Strike),
        InstrumentType.Straddle       => new StraddlePayoff(inst.Strike),
        InstrumentType.Strangle       => new StranglePayoff(inst.Strike, inst.SecondStrike),
        InstrumentType.BullCallSpread => new BullCallSpreadPayoff(inst.Strike, inst.SecondStrike),
        InstrumentType.BearPutSpread  => new BearPutSpreadPayoff(inst.Strike, inst.SecondStrike),
        InstrumentType.Butterfly      => new ButterflyPayoff(inst.Strike, inst.SecondStrike, inst.ThirdStrike),
        InstrumentType.VarianceSwap   => new VarianceSwapPayoff(inst.Strike),
        _ => throw new ArgumentOutOfRangeException(nameof(inst), $"Unknown instrument type: {inst.Type}")
    };
}
