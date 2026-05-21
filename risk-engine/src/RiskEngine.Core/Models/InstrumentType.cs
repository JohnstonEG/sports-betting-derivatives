namespace RiskEngine.Core;

/// <summary>
/// Synthetic derivative instrument types written on the line-movement
/// process Δp = p_close − p_open (the change in bookmaker-implied probability).
/// </summary>
public enum InstrumentType
{
    /// <summary>Call on Δp: payoff max(Δp − K, 0).</summary>
    Call,

    /// <summary>Put on Δp: payoff max(K − Δp, 0).</summary>
    Put,

    /// <summary>Cash-or-nothing digital call: pays 1 if Δp &gt; K.</summary>
    DigitalCall,

    /// <summary>Cash-or-nothing digital put: pays 1 if Δp &lt; K.</summary>
    DigitalPut,

    /// <summary>Long call + long put at the same strike: payoff |Δp − K|.</summary>
    Straddle,

    /// <summary>Long OTM put + long OTM call: payoff max(K1 − Δp, 0) + max(Δp − K2, 0).</summary>
    Strangle,

    /// <summary>Long call at K1, short call at K2 (K1 &lt; K2).</summary>
    BullCallSpread,

    /// <summary>Long put at K2, short put at K1 (K1 &lt; K2).</summary>
    BearPutSpread,

    /// <summary>Long wings, short body: long K1, short 2× K2, long K3.</summary>
    Butterfly,

    /// <summary>Variance swap: payoff Δp² − K (single-period realised-variance proxy).</summary>
    VarianceSwap
}
