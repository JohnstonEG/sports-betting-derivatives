using System.Text.Json;
using System.Text.Json.Serialization;

namespace RiskEngine.Core;

/// <summary>Shared System.Text.Json options for the engine's JSON I/O.</summary>
public static class JsonConfig
{
    /// <summary>
    /// camelCase property names, string-valued enums, case-insensitive reads,
    /// trailing commas and comments tolerated (helpful for hand-edited files).
    /// </summary>
    public static readonly JsonSerializerOptions Options = Build();

    private static JsonSerializerOptions Build()
    {
        var o = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };
        o.Converters.Add(new JsonStringEnumConverter());
        return o;
    }
}
