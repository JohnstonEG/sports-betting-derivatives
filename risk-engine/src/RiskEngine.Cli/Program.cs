using System.Text;
using System.Text.Json;
using RiskEngine.Core;

// =============================================================================
//  Synthetic Derivatives Risk Engine - command-line runner
//
//  Loads a portfolio and a Python-generated calibration set, runs the full
//  Monte Carlo risk pipeline, prints a formatted report and writes a JSON copy.
// =============================================================================

var options = CliOptions.Parse(args);
if (options.ShowHelp)
{
    CliOptions.PrintHelp();
    return 0;
}

try
{
    return Run(options);
}
catch (Exception ex)
{
    Console.Error.WriteLine();
    Console.Error.WriteLine($"  ERROR: {ex.Message}");
    Console.Error.WriteLine();
    Console.Error.WriteLine("  Run with --help for usage. Ensure the data/ folder contains");
    Console.Error.WriteLine("  calibration.json and portfolio.json, or pass explicit paths.");
    return 1;
}

// -----------------------------------------------------------------------------
static int Run(CliOptions options)
{
    // ---- Load inputs --------------------------------------------------------
    var portfolio = options.PortfolioPath is { } pp
        ? PortfolioLoader.Load(pp)
        : PortfolioLoader.LoadDefault();

    var calibration = options.CalibrationPath is { } cp
        ? CalibrationLoader.Load(cp)
        : CalibrationLoader.LoadDefault();

    var settings = new MonteCarloSettings { Paths = options.Paths, Seed = options.Seed };
    var service = new RiskEngineService(settings);
    var report = service.Analyze(portfolio, calibration, options.ModelOverride);

    // ---- Print report -------------------------------------------------------
    PrintReport(report);

    // ---- Persist JSON -------------------------------------------------------
    string outPath = options.OutputPath ?? "risk-report.json";
    File.WriteAllText(outPath, JsonSerializer.Serialize(report, JsonConfig.Options));
    Console.WriteLine();
    Console.WriteLine($"  Full report written to: {Path.GetFullPath(outPath)}");
    Console.WriteLine();
    return 0;
}

// -----------------------------------------------------------------------------
static void PrintReport(RiskReport r)
{
    Rule('=');
    Console.WriteLine("  SYNTHETIC DERIVATIVES RISK ENGINE");
    Console.WriteLine("  Quantitative risk & pricing engine  -  C# / .NET");
    Rule('=');
    Console.WriteLine();
    Console.WriteLine($"  Portfolio    : {r.PortfolioName}  ({r.PositionCount} positions)");
    Console.WriteLine($"  Calibration  : {Truncate(r.CalibrationSource, 58)}");
    Console.WriteLine($"  Model        : {r.DistributionModel}      Paths: {r.Paths:N0}");
    Console.WriteLine($"  Generated    : {r.GeneratedUtc}");
    string premiumNote = r.NetPremium >= 0 ? "premium paid" : "premium received";
    Console.WriteLine($"  Net premium  : {r.NetPremium:+0.0000;-0.0000}  ({premiumNote})");

    // ---- Headline -----------------------------------------------------------
    Section($"HEADLINE RISK  -  {r.Headline.Label}");
    var h = r.Headline;
    Field("Mean P&L", Signed(h.MeanPnl));
    Field("Std deviation", h.StdDevPnl.ToString("0.0000"));
    Field("Skewness", Signed(h.Skewness, 2));
    Field("Excess kurtosis", Signed(h.ExcessKurtosis, 2));
    Field("Probability of loss", Pct(h.ProbabilityOfLoss));
    Field("95% VaR", h.Var95.ToString("0.0000"));
    Field("99% VaR", h.Var99.ToString("0.0000"));
    Field("95% CVaR (expected shortfall)", h.CVar95.ToString("0.0000"));
    Field("99% CVaR (expected shortfall)", h.CVar99.ToString("0.0000"));
    Field("Sharpe-like ratio (mean / sd)", Signed(h.SharpeLike, 3));

    // ---- Positions ----------------------------------------------------------
    Section("POSITIONS  (base regime)");
    Console.WriteLine($"  {"Id",-11}{"Type",-16}{"Qty",8}{"Entry",10}{"Model",10}{"Mean P&L",12}{"95% VaR",11}");
    Console.WriteLine($"  {new string('-', 76)}");
    foreach (var p in r.Positions)
    {
        Console.WriteLine(
            $"  {Truncate(p.InstrumentId, 10),-11}{Truncate(p.InstrumentType, 15),-16}" +
            $"{p.Quantity,8:0}{p.EntryPrice,10:0.0000}{p.ModelValue,10:0.0000}" +
            $"{p.MeanPnl,12:+0.0000;-0.0000}{p.Var95,11:0.0000}");
    }

    // ---- Histogram ----------------------------------------------------------
    Section("P&L DISTRIBUTION  (base regime)");
    PrintHistogram(r.PnlHistogram);

    // ---- Regimes ------------------------------------------------------------
    Section("REGIME ANALYSIS");
    PrintSummaryTable(r.Regimes);
    PrintSummaryRow(r.RegimeSwitching);
    Console.WriteLine($"  {new string('-', 72)}");
    Console.WriteLine("  Regime-switching is the weighted mixture of all regimes above.");

    // ---- Stress battery -----------------------------------------------------
    Section("STRESS TEST BATTERY");
    PrintSummaryTable(r.StressScenarios);

    // ---- Volatility sweep ---------------------------------------------------
    Section("VOLATILITY SCENARIO SWEEP");
    Console.WriteLine($"  {"VolMult",-10}{"Sigma",10}{"Mean P&L",13}{"95% VaR",11}{"95% CVaR",11}{"P(loss)",10}");
    Console.WriteLine($"  {new string('-', 63)}");
    foreach (var v in r.VolatilityCurve)
    {
        Console.WriteLine(
            $"  {"x" + v.VolMultiplier.ToString("0.00"),-10}{v.Sigma,10:0.0000}" +
            $"{v.MeanPnl,13:+0.0000;-0.0000}{v.Var95,11:0.0000}{v.CVar95,11:0.0000}" +
            $"{Pct(v.ProbabilityOfLoss),10}");
    }
    Console.WriteLine();
    Rule('=');
}

// -----------------------------------------------------------------------------
static void PrintHistogram(HistogramData hist)
{
    if (hist.Counts.Length == 0) { Console.WriteLine("  (no data)"); return; }

    int max = 0;
    foreach (int c in hist.Counts) if (c > max) max = c;
    if (max == 0) { Console.WriteLine("  (empty)"); return; }

    const int width = 46;
    for (int i = 0; i < hist.Counts.Length; i++)
    {
        int bar = (int)Math.Round(width * (double)hist.Counts[i] / max);
        Console.WriteLine($"  {hist.BinCenters[i],10:+0.000;-0.000}  {new string('#', bar)}");
    }
    Console.WriteLine("  P&L bin centre (left). Central 98% shown - the deep tails sit in the VaR/CVaR figures.");
}

// -----------------------------------------------------------------------------
static void PrintSummaryTable(List<RiskSummary> rows)
{
    Console.WriteLine($"  {"Scenario",-22}{"Mean P&L",13}{"95% VaR",11}{"95% CVaR",11}{"P(loss)",10}");
    Console.WriteLine($"  {new string('-', 72)}");
    foreach (var s in rows) PrintSummaryRow(s);
}

static void PrintSummaryRow(RiskSummary s)
{
    Console.WriteLine(
        $"  {Truncate(s.Label, 21),-22}{s.MeanPnl,13:+0.0000;-0.0000}" +
        $"{s.Var95,11:0.0000}{s.CVar95,11:0.0000}{Pct(s.ProbabilityOfLoss),10}");
}

// -----------------------------------------------------------------------------
static void Rule(char c) => Console.WriteLine(new string(c, 64));
static void Section(string title)
{
    Console.WriteLine();
    Console.WriteLine($"  {new string('-', 60)}");
    Console.WriteLine($"  {title}");
    Console.WriteLine($"  {new string('-', 60)}");
}
static void Field(string label, string value) => Console.WriteLine($"  {label,-34}: {value}");
static string Signed(double v, int dp = 4) => v.ToString("+0." + new string('0', dp) + ";-0." + new string('0', dp));
static string Pct(double v) => (v * 100.0).ToString("0.0") + "%";
static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n - 1) + "~";

// =============================================================================
//  Command-line options
// =============================================================================
sealed class CliOptions
{
    public string? PortfolioPath { get; private set; }
    public string? CalibrationPath { get; private set; }
    public string? OutputPath { get; private set; }
    public PricingModel? ModelOverride { get; private set; }
    public int Paths { get; private set; } = 50_000;
    public int Seed { get; private set; } = 20240517;
    public bool ShowHelp { get; private set; }

    public static CliOptions Parse(string[] args)
    {
        var o = new CliOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i].ToLowerInvariant();
            string Next() => i + 1 < args.Length ? args[++i]
                : throw new ArgumentException($"Missing value for '{args[i]}'.");
            switch (a)
            {
                case "-h": case "--help": o.ShowHelp = true; break;
                case "-p": case "--portfolio": o.PortfolioPath = Next(); break;
                case "-c": case "--calibration": o.CalibrationPath = Next(); break;
                case "-o": case "--out": o.OutputPath = Next(); break;
                case "-n": case "--paths": o.Paths = int.Parse(Next()); break;
                case "--seed": o.Seed = int.Parse(Next()); break;
                case "-m": case "--model":
                    o.ModelOverride = Enum.Parse<PricingModel>(Next(), ignoreCase: true);
                    break;
                default:
                    throw new ArgumentException($"Unknown argument: '{args[i]}'. Use --help.");
            }
        }
        if (o.Paths < 1000) o.Paths = 1000;
        return o;
    }

    public static void PrintHelp()
    {
        Console.WriteLine("""
        Synthetic Derivatives Risk Engine - CLI

        Usage:
          dotnet run --project src/RiskEngine.Cli -- [options]

        Options:
          -p, --portfolio <path>    Portfolio file (.json or .csv).  Default: bundled data/portfolio.json
          -c, --calibration <path>  Calibration file (.json).        Default: bundled data/calibration.json
          -m, --model <name>        Pricing model: Normal | StudentT. Default: from the calibration file
          -n, --paths <int>         Monte Carlo paths.                Default: 50000
              --seed <int>          RNG seed (reproducibility).       Default: 20240517
          -o, --out <path>          JSON report output.               Default: risk-report.json
          -h, --help                Show this help.

        Examples:
          dotnet run --project src/RiskEngine.Cli
          dotnet run --project src/RiskEngine.Cli -- --model Normal --paths 100000
          dotnet run --project src/RiskEngine.Cli -- --portfolio data/portfolio.csv
        """);
    }
}
