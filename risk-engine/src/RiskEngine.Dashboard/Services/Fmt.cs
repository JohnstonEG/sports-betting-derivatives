using System.Globalization;

namespace RiskEngine.Dashboard;

/// <summary>Invariant-culture number formatting helpers for the dashboard UI.</summary>
public static class Fmt
{
    private static string Digits(int dp) => dp <= 0 ? "0" : "0." + new string('0', dp);

    public static string Num(double v, int dp = 4)
        => double.IsFinite(v)
            ? v.ToString(Digits(dp), CultureInfo.InvariantCulture)
            : "—";

    public static string Signed(double v, int dp = 4)
    {
        if (!double.IsFinite(v)) return "—";
        string d = Digits(dp);
        return v.ToString("+" + d + ";-" + d, CultureInfo.InvariantCulture);
    }

    public static string Pct(double v, int dp = 1)
        => double.IsFinite(v)
            ? (v * 100.0).ToString(Digits(dp), CultureInfo.InvariantCulture) + "%"
            : "—";

    public static string Int(double v)
        => v.ToString("#,##0", CultureInfo.InvariantCulture);
}
