namespace RiskEngine.Core;

/// <summary>Locates bundled data files at runtime.</summary>
public static class DataLocator
{
    /// <summary>
    /// Resolves a data file by walking up from both the executable directory and
    /// the current working directory, looking for the file directly or inside a
    /// sibling <c>data</c> folder.
    /// </summary>
    public static string Resolve(string fileName)
    {
        var candidates = new List<string>();
        foreach (var start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var dir = new DirectoryInfo(start);
            for (int depth = 0; depth < 8 && dir != null; depth++)
            {
                candidates.Add(Path.Combine(dir.FullName, "data", fileName));
                candidates.Add(Path.Combine(dir.FullName, fileName));
                dir = dir.Parent;
            }
        }

        foreach (var c in candidates)
            if (File.Exists(c)) return c;

        throw new FileNotFoundException(
            $"Could not locate '{fileName}'. Searched a 'data' folder near the executable " +
            "and the working directory. Pass an explicit path instead.", fileName);
    }

    /// <summary>Returns the resolved path, or null if the file cannot be found.</summary>
    public static string? TryResolve(string fileName)
    {
        try { return Resolve(fileName); }
        catch (FileNotFoundException) { return null; }
    }
}
