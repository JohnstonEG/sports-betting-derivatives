using System.Globalization;
using System.Text;
using System.Text.Json;

namespace RiskEngine.Core;

/// <summary>Loads a <see cref="Portfolio"/> from JSON or CSV.</summary>
public static class PortfolioLoader
{
    /// <summary>Loads a portfolio, dispatching on file extension (.json or .csv).</summary>
    public static Portfolio Load(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        return ext switch
        {
            ".json" => LoadJson(path),
            ".csv" => LoadCsv(path),
            _ => throw new NotSupportedException($"Unsupported portfolio format: '{ext}'. Use .json or .csv.")
        };
    }

    /// <summary>Loads <c>portfolio.json</c> from the bundled data folder.</summary>
    public static Portfolio LoadDefault()
        => LoadJson(DataLocator.Resolve("portfolio.json"));

    public static Portfolio LoadJson(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Portfolio file not found: {path}", path);

        string json = File.ReadAllText(path);
        var portfolio = JsonSerializer.Deserialize<Portfolio>(json, JsonConfig.Options)
                        ?? throw new InvalidDataException($"Portfolio file is empty or invalid: {path}");

        Validate(portfolio);
        return portfolio;
    }

    /// <summary>
    /// Loads a portfolio from CSV. Expected header (order-independent):
    /// <c>Id,Type,Strike,SecondStrike,ThirdStrike,Quantity,EntryPrice,Tag,Description</c>.
    /// Lines starting with '#' are treated as comments.
    /// </summary>
    public static Portfolio LoadCsv(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Portfolio file not found: {path}", path);

        var portfolio = new Portfolio { Name = Path.GetFileNameWithoutExtension(path) };
        var header = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        bool headerSeen = false;

        foreach (var raw in File.ReadAllLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;

            var cells = SplitCsv(line);
            if (!headerSeen)
            {
                for (int i = 0; i < cells.Count; i++) header[cells[i].Trim()] = i;
                headerSeen = true;
                continue;
            }

            string Text(string col) =>
                header.TryGetValue(col, out int idx) && idx < cells.Count ? cells[idx].Trim() : "";

            double Num(string col)
            {
                string v = Text(col);
                return string.IsNullOrEmpty(v)
                    ? 0.0
                    : double.Parse(v, NumberStyles.Any, CultureInfo.InvariantCulture);
            }

            var instrument = new Instrument
            {
                Id = Text("Id"),
                Type = Enum.Parse<InstrumentType>(Text("Type"), ignoreCase: true),
                Strike = Num("Strike"),
                SecondStrike = Num("SecondStrike"),
                ThirdStrike = Num("ThirdStrike"),
                Description = Text("Description")
            };
            portfolio.Positions.Add(new Position
            {
                Instrument = instrument,
                Quantity = Num("Quantity"),
                EntryPrice = Num("EntryPrice"),
                Tag = Text("Tag")
            });
        }

        Validate(portfolio);
        return portfolio;
    }

    public static void SaveJson(Portfolio portfolio, string path)
        => File.WriteAllText(path, JsonSerializer.Serialize(portfolio, JsonConfig.Options));

    /// <summary>Minimal RFC-4180-style CSV field splitter (handles quoted fields).</summary>
    private static List<string> SplitCsv(string line)
    {
        var result = new List<string>();
        var sb = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else
            {
                if (c == '"') inQuotes = true;
                else if (c == ',') { result.Add(sb.ToString()); sb.Clear(); }
                else sb.Append(c);
            }
        }
        result.Add(sb.ToString());
        return result;
    }

    private static void Validate(Portfolio portfolio)
    {
        if (portfolio.Positions.Count == 0)
            throw new InvalidDataException("Portfolio contains no positions.");

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var p in portfolio.Positions)
        {
            if (string.IsNullOrWhiteSpace(p.Instrument.Id))
                throw new InvalidDataException("Every instrument must have a non-empty Id.");
            if (!seen.Add(p.Instrument.Id))
                throw new InvalidDataException($"Duplicate instrument Id '{p.Instrument.Id}'. Ids must be unique.");
        }
    }
}
