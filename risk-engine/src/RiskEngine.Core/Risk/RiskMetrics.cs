namespace RiskEngine.Core;

/// <summary>Descriptive statistics and tail-risk measures over a P&amp;L sample.</summary>
public static class RiskMetrics
{
    public static double Mean(IReadOnlyList<double> x)
    {
        if (x.Count == 0) return double.NaN;
        double s = 0.0;
        for (int i = 0; i < x.Count; i++) s += x[i];
        return s / x.Count;
    }

    /// <summary>Variance — sample (N−1) by default, population if requested.</summary>
    public static double Variance(IReadOnlyList<double> x, bool sample = true)
    {
        int n = x.Count;
        if (n < 2) return 0.0;
        double mean = Mean(x);
        double s = 0.0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; s += d * d; }
        return s / (sample ? n - 1 : n);
    }

    public static double StdDev(IReadOnlyList<double> x, bool sample = true)
        => Math.Sqrt(Variance(x, sample));

    public static double Skewness(IReadOnlyList<double> x)
    {
        int n = x.Count;
        if (n < 3) return 0.0;
        double mean = Mean(x);
        double m2 = 0.0, m3 = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = x[i] - mean;
            double d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
        }
        m2 /= n; m3 /= n;
        return m2 <= 0.0 ? 0.0 : m3 / Math.Pow(m2, 1.5);
    }

    public static double ExcessKurtosis(IReadOnlyList<double> x)
    {
        int n = x.Count;
        if (n < 4) return 0.0;
        double mean = Mean(x);
        double m2 = 0.0, m4 = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = x[i] - mean;
            double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n; m4 /= n;
        return m2 <= 0.0 ? 0.0 : m4 / (m2 * m2) - 3.0;
    }

    /// <summary>Percentile (p in [0,100]) by linear interpolation. Sorts a copy.</summary>
    public static double Percentile(IReadOnlyList<double> x, double p)
    {
        if (x.Count == 0) return double.NaN;
        var s = x.ToArray();
        Array.Sort(s);
        return PercentileSorted(s, p);
    }

    /// <summary>Percentile (p in [0,100]) over an already ascending-sorted array.</summary>
    public static double PercentileSorted(double[] sorted, double p)
    {
        if (sorted.Length == 0) return double.NaN;
        if (sorted.Length == 1) return sorted[0];
        double rank = Math.Clamp(p, 0.0, 100.0) / 100.0 * (sorted.Length - 1);
        int lo = (int)Math.Floor(rank);
        int hi = (int)Math.Ceiling(rank);
        double frac = rank - lo;
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }

    /// <summary>
    /// Value at Risk at the given confidence (e.g. 0.95), reported as a positive
    /// loss number. VaR is the loss not exceeded with probability <c>confidence</c>.
    /// </summary>
    public static double ValueAtRisk(IReadOnlyList<double> pnl, double confidence)
    {
        var s = pnl.ToArray();
        Array.Sort(s);
        return ValueAtRiskSorted(s, confidence);
    }

    public static double ValueAtRiskSorted(double[] sortedPnl, double confidence)
        => -PercentileSorted(sortedPnl, (1.0 - confidence) * 100.0);

    /// <summary>
    /// Conditional VaR (expected shortfall): the mean loss in the worst
    /// (1 − confidence) tail, reported as a positive loss number. By construction
    /// CVaR ≥ VaR.
    /// </summary>
    public static double ConditionalVaR(IReadOnlyList<double> pnl, double confidence)
    {
        var s = pnl.ToArray();
        Array.Sort(s);
        return ConditionalVaRSorted(s, confidence);
    }

    public static double ConditionalVaRSorted(double[] sortedPnl, double confidence)
    {
        if (sortedPnl.Length == 0) return double.NaN;
        double q = PercentileSorted(sortedPnl, (1.0 - confidence) * 100.0);
        double sum = 0.0;
        int n = 0;
        for (int i = 0; i < sortedPnl.Length; i++)
        {
            if (sortedPnl[i] <= q) { sum += sortedPnl[i]; n++; }
            else break; // ascending order — past the tail
        }
        return n == 0 ? -q : -(sum / n);
    }

    /// <summary>Fraction of paths with negative P&amp;L.</summary>
    public static double ProbabilityOfLoss(IReadOnlyList<double> pnl)
    {
        if (pnl.Count == 0) return double.NaN;
        int losses = 0;
        for (int i = 0; i < pnl.Count; i++) if (pnl[i] < 0.0) losses++;
        return (double)losses / pnl.Count;
    }
}
