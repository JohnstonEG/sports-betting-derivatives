using RiskEngine.Dashboard;
using RiskEngine.Dashboard.Components;

var builder = WebApplication.CreateBuilder(args);

// Blazor Web App with interactive server components.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// The risk engine data service: loads the portfolio and the Python-generated
// calibration once, then serves cached risk reports to the pages.
builder.Services.AddSingleton<RiskDataService>();

var app = builder.Build();

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
