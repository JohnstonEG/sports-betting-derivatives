namespace RiskEngine.Core;

/// <summary>A book of derivative positions on the line-movement underlying.</summary>
public sealed class Portfolio
{
    public string Name { get; set; } = "Portfolio";

    public string BaseCurrency { get; set; } = "USD";

    /// <summary>Valuation/as-of date for the book (free text).</summary>
    public string AsOf { get; set; } = "";

    public List<Position> Positions { get; set; } = new();

    public int Count => Positions.Count;

    /// <summary>
    /// Net premium outlay across the book: positive means net premium paid
    /// (cash out), negative means net premium received (cash in).
    /// </summary>
    public double NetPremium()
    {
        double s = 0.0;
        foreach (var p in Positions) s += p.Quantity * p.EntryPrice;
        return s;
    }

    /// <summary>Total portfolio P&amp;L for a single realised line move Δp.</summary>
    public double ProfitAndLoss(double deltaP)
    {
        double s = 0.0;
        foreach (var p in Positions) s += p.ProfitAndLoss(deltaP);
        return s;
    }
}
