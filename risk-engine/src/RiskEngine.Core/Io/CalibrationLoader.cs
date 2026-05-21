using System.Text.Json;

namespace RiskEngine.Core;

/// <summary>Loads a <see cref="CalibrationSet"/> from the Python-generated JSON bridge file.</summary>
public static class CalibrationLoader
{
    public static CalibrationSet Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Calibration file not found: {path}", path);

        string json = File.ReadAllText(path);
        var set = JsonSerializer.Deserialize<CalibrationSet>(json, JsonConfig.Options)
                  ?? throw new InvalidDataException($"Calibration file is empty or invalid: {path}");

        Validate(set);
        return set;
    }

    /// <summary>Loads <c>calibration.json</c> from the bundled data folder.</summary>
    public static CalibrationSet LoadDefault()
        => Load(DataLocator.Resolve("calibration.json"));

    public static void Save(CalibrationSet set, string path)
        => File.WriteAllText(path, JsonSerializer.Serialize(set, JsonConfig.Options));

    private static void Validate(CalibrationSet set)
    {
        if (set.Base.Sigma <= 0.0)
            throw new InvalidDataException("Calibration base sigma must be positive.");
        if (set.Regimes.Count == 0)
            set.Regimes.Add(set.Base);
    }
}
