using RiskEngine.Core;

namespace RiskEngine.Dashboard;

/// <summary>
/// Singleton that loads the portfolio and the Python-generated calibration once,
/// then serves cached risk reports to the dashboard pages.
/// </summary>
public sealed class RiskDataService
{
    private readonly object _lock = new();
    private RiskReport? _cached;
    private (PricingModel Model, int Paths) _cachedKey;

    public Portfolio? Portfolio { get; }
    public CalibrationSet? Calibration { get; }
    public string? LoadError { get; }

    public RiskDataService()
    {
        try
        {
            Portfolio = PortfolioLoader.LoadDefault();
            Calibration = CalibrationLoader.LoadDefault();
        }
        catch (Exception ex)
        {
            LoadError = ex.Message;
        }
    }

    public bool IsReady => Portfolio is not null && Calibration is not null;

    public PricingModel DefaultModel => Calibration?.Model ?? PricingModel.StudentT;

    /// <summary>Returns a risk report, recomputing only when the inputs change.</summary>
    public RiskReport GetReport(PricingModel model, int paths)
    {
        if (!IsReady)
            throw new InvalidOperationException(LoadError ?? "Engine data is not loaded.");

        lock (_lock)
        {
            if (_cached is not null && _cachedKey == (model, paths))
                return _cached;

            var service = new RiskEngineService(new MonteCarloSettings { Paths = paths });
            _cached = service.Analyze(Portfolio!, Calibration!, model);
            _cachedKey = (model, paths);
            return _cached;
        }
    }
}
