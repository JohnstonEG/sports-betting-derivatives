namespace RiskEngine.Core;

/// <summary>
/// A holding in a single instrument.
/// <see cref="Quantity"/> &gt; 0 is long (premium paid); &lt; 0 is short (premium received).
/// <see cref="EntryPrice"/> is the premium per contract at which the position was struck.
/// </summary>
public sealed class Position
{
    public Instrument Instrument { get; set; } = new();

    public double Quantity { get; set; }

    public double EntryPrice { get; set; }

    /// <summary>Optional grouping tag, e.g. "vol-short" or "directional".</summary>
    public string Tag { get; set; } = "";

    /// <summary>Profit/loss of this position for a realised line move Δp.</summary>
    public double ProfitAndLoss(IPayoff payoff, double deltaP)
        => Quantity * (payoff.Evaluate(deltaP) - EntryPrice);

    /// <summary>Profit/loss of this position, building the payoff on demand.</summary>
    public double ProfitAndLoss(double deltaP)
        => ProfitAndLoss(PayoffFactory.Create(Instrument), deltaP);
}
