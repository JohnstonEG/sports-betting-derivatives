using System.Globalization;
using System.Text;
using RiskEngine.Core;

namespace RiskEngine.Dashboard;

/// <summary>
/// Builds self-contained inline-SVG charts (no JavaScript, no CDN) for the
/// dashboard. Each method returns SVG markup rendered via a MarkupString.
/// </summary>
public static class SvgCharts
{
    private const string AxisColor = "#9aa4b2";
    private const string GridColor = "#e6e9ef";
    private const string TextColor = "#5b6472";
    private const string PosColor = "#2f9e6b";
    private const string NegColor = "#d1495b";
    private const string LineA = "#3a6ea5";
    private const string LineB = "#d1495b";

    private static string N(double v) =>
        (double.IsFinite(v) ? v : 0.0).ToString("0.###", CultureInfo.InvariantCulture);

    private static string Esc(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

    private static string Txt(double x, double y, string s, string anchor = "middle",
        int size = 11, string color = TextColor, bool bold = false)
        => $"<text x=\"{N(x)}\" y=\"{N(y)}\" text-anchor=\"{anchor}\" " +
           $"font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"{size}\" " +
           $"font-weight=\"{(bold ? "600" : "400")}\" fill=\"{color}\">{Esc(s)}</text>";

    private static string Line(double x1, double y1, double x2, double y2, string color, double w = 1)
        => $"<line x1=\"{N(x1)}\" y1=\"{N(y1)}\" x2=\"{N(x2)}\" y2=\"{N(y2)}\" " +
           $"stroke=\"{color}\" stroke-width=\"{N(w)}\"/>";

    // -------------------------------------------------------------------------
    /// <summary>Bar histogram of a portfolio P&amp;L distribution.</summary>
    public static string Histogram(HistogramData h, int width = 700, int height = 280)
    {
        var sb = new StringBuilder();
        sb.Append($"<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" " +
                  "role=\"img\" style=\"width:100%;height:auto\">");

        double ml = 62, mr = 20, mt = 18, mb = 42;
        double pw = width - ml - mr, ph = height - mt - mb;
        int bins = h.Counts.Length;

        if (bins == 0)
        {
            sb.Append(Txt(width / 2.0, height / 2.0, "No data"));
            sb.Append("</svg>");
            return sb.ToString();
        }

        int maxCount = 1;
        for (int i = 0; i < bins; i++) if (h.Counts[i] > maxCount) maxCount = h.Counts[i];

        double minEdge = h.BinCenters[0] - h.BinWidth / 2.0;
        double maxEdge = h.BinCenters[^1] + h.BinWidth / 2.0;
        double span = Math.Max(maxEdge - minEdge, 1e-12);

        // Gridlines + y labels.
        for (int g = 0; g <= 2; g++)
        {
            double yy = mt + ph * g / 2.0;
            sb.Append(Line(ml, yy, ml + pw, yy, GridColor));
            int val = (int)Math.Round(maxCount * (1 - g / 2.0));
            sb.Append(Txt(ml - 8, yy + 4, val.ToString("N0"), "end"));
        }

        // Bars.
        double slot = pw / bins;
        for (int i = 0; i < bins; i++)
        {
            double barH = ph * h.Counts[i] / maxCount;
            double x = ml + i * slot;
            double y = mt + ph - barH;
            string fill = h.BinCenters[i] >= 0 ? PosColor : NegColor;
            sb.Append($"<rect x=\"{N(x + slot * 0.08)}\" y=\"{N(y)}\" " +
                      $"width=\"{N(slot * 0.84)}\" height=\"{N(barH)}\" fill=\"{fill}\" opacity=\"0.85\"/>");
        }

        // Axes.
        sb.Append(Line(ml, mt + ph, ml + pw, mt + ph, AxisColor));
        sb.Append(Line(ml, mt, ml, mt + ph, AxisColor));

        // Zero reference line.
        if (minEdge < 0 && maxEdge > 0)
        {
            double zx = ml + pw * (0 - minEdge) / span;
            sb.Append(Line(zx, mt, zx, mt + ph, "#33404f", 1.4));
            sb.Append(Txt(zx, mt - 5, "P&L = 0", "middle", 10));
        }

        // X labels.
        for (int t = 0; t <= 4; t++)
        {
            double xx = ml + pw * t / 4.0;
            double val = minEdge + span * t / 4.0;
            sb.Append(Txt(xx, mt + ph + 16, val.ToString("0.###", CultureInfo.InvariantCulture)));
        }
        sb.Append(Txt(ml + pw / 2.0, height - 6, "Portfolio P&L (per path)", "middle", 11, TextColor, true));

        sb.Append("</svg>");
        return sb.ToString();
    }

    // -------------------------------------------------------------------------
    /// <summary>Line chart of 95% VaR and CVaR across the volatility sweep.</summary>
    public static string VolCurve(IReadOnlyList<VolScenarioPoint> pts, int width = 700, int height = 300)
    {
        var sb = new StringBuilder();
        sb.Append($"<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" " +
                  "role=\"img\" style=\"width:100%;height:auto\">");

        double ml = 64, mr = 20, mt = 26, mb = 46;
        double pw = width - ml - mr, ph = height - mt - mb;

        if (pts.Count < 2)
        {
            sb.Append(Txt(width / 2.0, height / 2.0, "No data"));
            sb.Append("</svg>");
            return sb.ToString();
        }

        double xMin = pts[0].VolMultiplier, xMax = pts[^1].VolMultiplier;
        double xSpan = Math.Max(xMax - xMin, 1e-12);
        double yMax = 1e-9;
        foreach (var p in pts) yMax = Math.Max(yMax, Math.Max(p.Var95, p.CVar95));
        yMax *= 1.1;

        double Px(double v) => ml + pw * (v - xMin) / xSpan;
        double Py(double v) => mt + ph * (1 - v / yMax);

        // Gridlines + y labels.
        for (int g = 0; g <= 4; g++)
        {
            double yy = mt + ph * g / 4.0;
            sb.Append(Line(ml, yy, ml + pw, yy, GridColor));
            double val = yMax * (1 - g / 4.0);
            sb.Append(Txt(ml - 8, yy + 4, val.ToString("0.###", CultureInfo.InvariantCulture), "end"));
        }

        // Axes.
        sb.Append(Line(ml, mt + ph, ml + pw, mt + ph, AxisColor));
        sb.Append(Line(ml, mt, ml, mt + ph, AxisColor));

        // X labels (vol multiplier).
        for (int t = 0; t <= 5; t++)
        {
            double mult = xMin + xSpan * t / 5.0;
            double xx = Px(mult);
            sb.Append(Txt(xx, mt + ph + 16, "x" + mult.ToString("0.0", CultureInfo.InvariantCulture)));
        }

        // Series.
        sb.Append(Polyline(pts, Px, Py, p => p.VolMultiplier, p => p.Var95, LineA));
        sb.Append(Polyline(pts, Px, Py, p => p.VolMultiplier, p => p.CVar95, LineB));

        // Legend.
        sb.Append($"<rect x=\"{N(ml + 8)}\" y=\"{N(mt + 2)}\" width=\"12\" height=\"3\" fill=\"{LineA}\"/>");
        sb.Append(Txt(ml + 26, mt + 8, "95% VaR", "start", 11));
        sb.Append($"<rect x=\"{N(ml + 96)}\" y=\"{N(mt + 2)}\" width=\"12\" height=\"3\" fill=\"{LineB}\"/>");
        sb.Append(Txt(ml + 114, mt + 8, "95% CVaR (ES)", "start", 11));

        sb.Append(Txt(ml + pw / 2.0, height - 6, "Volatility multiplier", "middle", 11, TextColor, true));
        sb.Append("</svg>");
        return sb.ToString();
    }

    private static string Polyline(IReadOnlyList<VolScenarioPoint> pts,
        Func<double, double> px, Func<double, double> py,
        Func<VolScenarioPoint, double> fx, Func<VolScenarioPoint, double> fy, string color)
    {
        var sb = new StringBuilder();
        var coords = new StringBuilder();
        foreach (var p in pts)
            coords.Append($"{N(px(fx(p)))},{N(py(fy(p)))} ");
        sb.Append($"<polyline points=\"{coords.ToString().Trim()}\" fill=\"none\" " +
                  $"stroke=\"{color}\" stroke-width=\"2.2\"/>");
        foreach (var p in pts)
            sb.Append($"<circle cx=\"{N(px(fx(p)))}\" cy=\"{N(py(fy(p)))}\" r=\"2.6\" fill=\"{color}\"/>");
        return sb.ToString();
    }

    // -------------------------------------------------------------------------
    /// <summary>Per-contract payoff and P&amp;L curve of one instrument vs Δp.</summary>
    public static string PayoffCurve(Instrument inst, double entryPrice,
        int width = 460, int height = 240)
    {
        var sb = new StringBuilder();
        sb.Append($"<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" " +
                  "role=\"img\" style=\"width:100%;height:auto\">");

        double ml = 52, mr = 14, mt = 16, mb = 36;
        double pw = width - ml - mr, ph = height - mt - mb;

        var payoff = PayoffFactory.Create(inst);
        const double xMin = -0.15, xMax = 0.15;
        const int steps = 121;

        var xs = new double[steps];
        var gross = new double[steps];
        var pnl = new double[steps];
        double yMin = 0, yMax = 0;
        for (int i = 0; i < steps; i++)
        {
            double dp = xMin + (xMax - xMin) * i / (steps - 1);
            xs[i] = dp;
            gross[i] = payoff.Evaluate(dp);
            pnl[i] = gross[i] - entryPrice;
            yMin = Math.Min(yMin, Math.Min(gross[i], pnl[i]));
            yMax = Math.Max(yMax, Math.Max(gross[i], pnl[i]));
        }
        if (yMax - yMin < 1e-9) { yMax += 0.01; yMin -= 0.01; }
        double pad = (yMax - yMin) * 0.08;
        yMin -= pad; yMax += pad;
        double ySpan = yMax - yMin;

        double Px(double v) => ml + pw * (v - xMin) / (xMax - xMin);
        double Py(double v) => mt + ph * (1 - (v - yMin) / ySpan);

        // Axes + zero lines.
        sb.Append(Line(ml, mt + ph, ml + pw, mt + ph, AxisColor));
        sb.Append(Line(ml, mt, ml, mt + ph, AxisColor));
        if (yMin < 0 && yMax > 0)
            sb.Append(Line(ml, Py(0), ml + pw, Py(0), GridColor, 1.2));
        double zx = Px(0);
        sb.Append(Line(zx, mt, zx, mt + ph, GridColor, 1.2));

        // Y labels.
        for (int g = 0; g <= 2; g++)
        {
            double val = yMax - ySpan * g / 2.0;
            sb.Append(Txt(ml - 7, mt + ph * g / 2.0 + 4,
                val.ToString("0.###", CultureInfo.InvariantCulture), "end", 10));
        }
        // X labels.
        for (int t = 0; t <= 2; t++)
        {
            double dp = xMin + (xMax - xMin) * t / 2.0;
            sb.Append(Txt(Px(dp), mt + ph + 15,
                dp.ToString("0.##", CultureInfo.InvariantCulture), "middle", 10));
        }

        sb.Append(Curve(xs, gross, Px, Py, LineA));
        sb.Append(Curve(xs, pnl, Px, Py, PosColor));

        // Legend.
        sb.Append($"<rect x=\"{N(ml + 6)}\" y=\"{N(mt + 2)}\" width=\"11\" height=\"3\" fill=\"{LineA}\"/>");
        sb.Append(Txt(ml + 22, mt + 8, "Payoff", "start", 10));
        sb.Append($"<rect x=\"{N(ml + 74)}\" y=\"{N(mt + 2)}\" width=\"11\" height=\"3\" fill=\"{PosColor}\"/>");
        sb.Append(Txt(ml + 90, mt + 8, "P&L (net of premium)", "start", 10));

        sb.Append(Txt(ml + pw / 2.0, height - 5, "Line move Δp", "middle", 10, TextColor, true));
        sb.Append("</svg>");
        return sb.ToString();
    }

    private static string Curve(double[] xs, double[] ys,
        Func<double, double> px, Func<double, double> py, string color)
    {
        var coords = new StringBuilder();
        for (int i = 0; i < xs.Length; i++)
            coords.Append($"{N(px(xs[i]))},{N(py(ys[i]))} ");
        return $"<polyline points=\"{coords.ToString().Trim()}\" fill=\"none\" " +
               $"stroke=\"{color}\" stroke-width=\"2\"/>";
    }

    // -------------------------------------------------------------------------
    /// <summary>Horizontal bar chart for comparing a metric across scenarios.</summary>
    public static string HBars(IReadOnlyList<(string Label, double Value)> rows,
        string unit = "", int width = 700)
    {
        int rowH = 30;
        int mt = 12, mb = 14;
        int height = mt + mb + rows.Count * rowH;
        var sb = new StringBuilder();
        sb.Append($"<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" " +
                  "role=\"img\" style=\"width:100%;height:auto\">");

        if (rows.Count == 0)
        {
            sb.Append(Txt(width / 2.0, height / 2.0, "No data"));
            sb.Append("</svg>");
            return sb.ToString();
        }

        double ml = 150, mr = 70;
        double pw = width - ml - mr;
        double maxV = 1e-9, minV = 0;
        foreach (var r in rows) { maxV = Math.Max(maxV, r.Value); minV = Math.Min(minV, r.Value); }
        double span = Math.Max(maxV - minV, 1e-9);
        double zeroX = ml + pw * (0 - minV) / span;

        for (int i = 0; i < rows.Count; i++)
        {
            double cy = mt + i * rowH + rowH / 2.0;
            var (label, value) = rows[i];
            double vx = ml + pw * (value - minV) / span;
            double barX = Math.Min(zeroX, vx);
            double barW = Math.Abs(vx - zeroX);
            string fill = value >= 0 ? NegColor : PosColor; // losses are "bad" -> red
            sb.Append(Txt(ml - 10, cy + 4, label, "end", 11));
            sb.Append($"<rect x=\"{N(barX)}\" y=\"{N(cy - 9)}\" width=\"{N(barW)}\" " +
                      $"height=\"18\" rx=\"2\" fill=\"{fill}\" opacity=\"0.82\"/>");
            sb.Append(Txt(vx + (value >= 0 ? 6 : -6), cy + 4,
                value.ToString("0.###", CultureInfo.InvariantCulture) + unit,
                value >= 0 ? "start" : "end", 10));
        }

        sb.Append(Line(zeroX, mt, zeroX, mt + rows.Count * rowH, AxisColor));
        sb.Append("</svg>");
        return sb.ToString();
    }
}
