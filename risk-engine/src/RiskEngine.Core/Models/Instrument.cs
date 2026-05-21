namespace RiskEngine.Core;

/// <summary>
/// A synthetic derivative written on the line-movement underlying Δp.
/// Strike fields are interpreted per <see cref="Type"/>:
/// <list type="bullet">
///   <item>Call / Put / Digital* — <see cref="Strike"/></item>
///   <item>Straddle — <see cref="Strike"/> (centre)</item>
///   <item>Strangle — <see cref="Strike"/> (put leg), <see cref="SecondStrike"/> (call leg)</item>
///   <item>BullCallSpread — <see cref="Strike"/> (long call), <see cref="SecondStrike"/> (short call)</item>
///   <item>BearPutSpread — <see cref="Strike"/> (short put), <see cref="SecondStrike"/> (long put)</item>
///   <item>Butterfly — <see cref="Strike"/> (lower), <see cref="SecondStrike"/> (body), <see cref="ThirdStrike"/> (upper)</item>
///   <item>VarianceSwap — <see cref="Strike"/> (variance strike, in Δp² units)</item>
/// </list>
/// </summary>
public sealed class Instrument
{
    /// <summary>Unique identifier within a portfolio.</summary>
    public string Id { get; set; } = "";

    public InstrumentType Type { get; set; }

    public double Strike { get; set; }

    public double SecondStrike { get; set; }

    public double ThirdStrike { get; set; }

    /// <summary>Free-text label, e.g. "ATM straddle on Pinnacle line moves".</summary>
    public string Description { get; set; } = "";
}
