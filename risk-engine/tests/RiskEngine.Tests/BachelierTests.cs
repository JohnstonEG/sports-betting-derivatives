using RiskEngine.Core;
using Xunit;

namespace RiskEngine.Tests;

public class BachelierTests
{
    [Fact]
    public void Normal_cdf_matches_known_values()
    {
        Assert.Equal(0.5, Bachelier.NormalCdf(0.0), 6);
        Assert.Equal(0.975, Bachelier.NormalCdf(1.959964), 3);
        Assert.Equal(0.025, Bachelier.NormalCdf(-1.959964), 3);
    }

    [Fact]
    public void Call_price_is_intrinsic_when_volatility_is_zero()
    {
        Assert.Equal(0.03, Bachelier.CallPrice(0.05, 0.0, 0.02), 12);
        Assert.Equal(0.0, Bachelier.CallPrice(0.01, 0.0, 0.02), 12);
    }

    [Fact]
    public void Bachelier_put_call_parity_holds()
    {
        double call = Bachelier.CallPrice(-0.0013, 0.0526, 0.02);
        double put = Bachelier.PutPrice(-0.0013, 0.0526, 0.02);
        // C - P = forward - strike = mean - strike
        Assert.Equal(-0.0013 - 0.02, call - put, 12);
    }

    [Fact]
    public void Call_price_increases_with_volatility()
    {
        double lo = Bachelier.CallPrice(0.0, 0.03, 0.02);
        double hi = Bachelier.CallPrice(0.0, 0.08, 0.02);
        Assert.True(hi > lo);
    }

    [Fact]
    public void Implied_volatility_round_trips()
    {
        double price = Bachelier.CallPrice(0.0, 0.05, 0.03);
        double iv = Bachelier.ImpliedVol(0.0, 0.03, price);
        Assert.Equal(0.05, iv, 4);
    }

    [Fact]
    public void Digital_call_price_is_a_probability()
    {
        double p = Bachelier.DigitalCallPrice(0.0, 0.05, 0.0);
        Assert.Equal(0.5, p, 6); // at-the-money digital under a symmetric distribution
    }
}
