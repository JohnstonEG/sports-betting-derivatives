namespace RiskEngine.Core;

/// <summary>
/// Closed-form Bachelier (arithmetic / normal) pricing for options on Δp.
/// Bachelier — not Black-Scholes — is the correct analogue here: Δp can be
/// negative, so the lognormal assumption fails. Used as an analytic benchmark
/// against the Monte Carlo engine and to invert the implied-volatility smile.
/// </summary>
public static class Bachelier
{
    private static readonly double Sqrt2 = Math.Sqrt(2.0);
    private static readonly double Sqrt2Pi = Math.Sqrt(2.0 * Math.PI);

    /// <summary>Standard normal probability density.</summary>
    public static double NormalPdf(double x) => Math.Exp(-0.5 * x * x) / Sqrt2Pi;

    /// <summary>Standard normal cumulative distribution.</summary>
    public static double NormalCdf(double x) => 0.5 * Erfc(-x / Sqrt2);

    /// <summary>
    /// Complementary error function via the Numerical Recipes rational
    /// approximation (fractional error &lt; 1.2e-7 everywhere).
    /// </summary>
    public static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double ans = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
            t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0.0 ? ans : 2.0 - ans;
    }

    /// <summary>Fair value of a call on Δp under N(mean, sigma).</summary>
    public static double CallPrice(double mean, double sigma, double strike)
    {
        if (sigma <= 0.0) return Math.Max(mean - strike, 0.0);
        double d = (mean - strike) / sigma;
        return (mean - strike) * NormalCdf(d) + sigma * NormalPdf(d);
    }

    /// <summary>Fair value of a put on Δp under N(mean, sigma).</summary>
    public static double PutPrice(double mean, double sigma, double strike)
        // Bachelier put-call parity: C − P = mean − strike.
        => CallPrice(mean, sigma, strike) - (mean - strike);

    /// <summary>Fair value of a cash-or-nothing digital call (pays 1 if Δp &gt; K).</summary>
    public static double DigitalCallPrice(double mean, double sigma, double strike)
    {
        if (sigma <= 0.0) return mean > strike ? 1.0 : 0.0;
        return NormalCdf((mean - strike) / sigma);
    }

    /// <summary>
    /// Implied volatility: the sigma reproducing a target call price, solved by
    /// bisection. Returns NaN if the target lies outside no-arbitrage bounds.
    /// </summary>
    public static double ImpliedVol(double mean, double strike, double targetCallPrice,
        double lo = 1e-6, double hi = 5.0, int maxIter = 128)
    {
        double intrinsic = Math.Max(mean - strike, 0.0);
        if (targetCallPrice < intrinsic - 1e-12) return double.NaN;

        double a = lo, b = hi;
        double fa = CallPrice(mean, a, strike) - targetCallPrice;
        double fb = CallPrice(mean, b, strike) - targetCallPrice;
        if (fa * fb > 0.0) return double.NaN;

        for (int i = 0; i < maxIter; i++)
        {
            double m = 0.5 * (a + b);
            double fm = CallPrice(mean, m, strike) - targetCallPrice;
            if (Math.Abs(fm) < 1e-12) return m;
            if (fa * fm <= 0.0) { b = m; }
            else { a = m; fa = fm; }
        }
        return 0.5 * (a + b);
    }
}
