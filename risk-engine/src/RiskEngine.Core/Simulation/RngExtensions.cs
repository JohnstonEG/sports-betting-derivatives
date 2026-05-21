namespace RiskEngine.Core;

/// <summary>Random-variate sampling helpers used by the Monte Carlo engine.</summary>
public static class RngExtensions
{
    /// <summary>Standard normal draw via the Box-Muller transform.</summary>
    public static double NextGaussian(this Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(); // in (0, 1] so Log is finite
        double u2 = rng.NextDouble();       // in [0, 1)
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Gamma(shape, scale) draw via the Marsaglia-Tsang method. Valid for any
    /// shape &gt; 0; shapes below 1 are handled by the standard boost transform.
    /// </summary>
    public static double NextGamma(this Random rng, double shape, double scale)
    {
        if (shape <= 0.0) throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be positive.");
        if (scale <= 0.0) throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be positive.");

        if (shape < 1.0)
        {
            double u = 1.0 - rng.NextDouble();
            return rng.NextGamma(shape + 1.0, scale) * Math.Pow(u, 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = rng.NextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0.0);

            v = v * v * v;
            double u = 1.0 - rng.NextDouble();
            double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2) return d * v * scale;
            if (Math.Log(u) < 0.5 * x2 + d * (1.0 - v + Math.Log(v))) return d * v * scale;
        }
    }

    /// <summary>Chi-squared(dof) draw — a Gamma(dof/2, 2) variate.</summary>
    public static double NextChiSquared(this Random rng, double dof)
        => rng.NextGamma(dof / 2.0, 2.0);
}
