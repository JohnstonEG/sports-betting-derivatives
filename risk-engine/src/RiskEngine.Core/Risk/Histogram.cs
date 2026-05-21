namespace RiskEngine.Core;

/// <summary>An equal-width histogram of a P&amp;L sample, used for payoff visualisation.</summary>
public sealed class Histogram
{
    public required double[] BinEdges { get; init; }   // length = bins + 1
    public required double[] BinCenters { get; init; } // length = bins
    public required int[] Counts { get; init; }        // length = bins
    public required double BinWidth { get; init; }
    public required int Total { get; init; }

    /// <summary>Builds an equal-width histogram with the requested number of bins.</summary>
    public static Histogram Build(IReadOnlyList<double> data, int bins = 41)
    {
        if (bins < 1) bins = 1;

        if (data.Count == 0)
        {
            return new Histogram
            {
                BinEdges = new[] { 0.0, 1.0 },
                BinCenters = new[] { 0.5 },
                Counts = new[] { 0 },
                BinWidth = 1.0,
                Total = 0
            };
        }

        double min = data[0], max = data[0];
        for (int i = 1; i < data.Count; i++)
        {
            if (data[i] < min) min = data[i];
            if (data[i] > max) max = data[i];
        }
        if (max - min < 1e-12) { min -= 0.5; max += 0.5; }

        double width = (max - min) / bins;
        var edges = new double[bins + 1];
        var centers = new double[bins];
        for (int i = 0; i <= bins; i++) edges[i] = min + i * width;
        for (int i = 0; i < bins; i++) centers[i] = edges[i] + width / 2.0;

        var counts = new int[bins];
        for (int i = 0; i < data.Count; i++)
        {
            int b = (int)((data[i] - min) / width);
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            counts[b]++;
        }

        return new Histogram
        {
            BinEdges = edges,
            BinCenters = centers,
            Counts = counts,
            BinWidth = width,
            Total = data.Count
        };
    }

    /// <summary>
    /// Builds a histogram clipped to a central percentile window. Values outside
    /// [<paramref name="lowerPercentile"/>, <paramref name="upperPercentile"/>] are
    /// excluded from the bin counts, so a fat-tailed distribution shows its shape
    /// instead of collapsing into a single bar. The extreme tails are still
    /// reflected in the risk metrics (VaR, CVaR, min/max) computed elsewhere.
    /// </summary>
    public static Histogram BuildClipped(IReadOnlyList<double> data, int bins = 41,
        double lowerPercentile = 1.0, double upperPercentile = 99.0)
    {
        if (bins < 1) bins = 1;
        if (data.Count == 0) return Build(data, bins);

        var sorted = data.ToArray();
        Array.Sort(sorted);
        double lo = RiskMetrics.PercentileSorted(sorted, lowerPercentile);
        double hi = RiskMetrics.PercentileSorted(sorted, upperPercentile);

        // Degenerate window — fall back to the full-range histogram.
        if (hi - lo < 1e-12) return Build(data, bins);

        double width = (hi - lo) / bins;
        var edges = new double[bins + 1];
        var centers = new double[bins];
        for (int i = 0; i <= bins; i++) edges[i] = lo + i * width;
        for (int i = 0; i < bins; i++) centers[i] = edges[i] + width / 2.0;

        var counts = new int[bins];
        for (int i = 0; i < data.Count; i++)
        {
            double v = data[i];
            if (v < lo || v > hi) continue; // tail value — outside the shown window
            int b = (int)((v - lo) / width);
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            counts[b]++;
        }

        return new Histogram
        {
            BinEdges = edges,
            BinCenters = centers,
            Counts = counts,
            BinWidth = width,
            Total = data.Count
        };
    }
}
