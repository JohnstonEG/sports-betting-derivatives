"""
Configuration for Sports Derivatives project.
All paths, constants, and parameter defaults live here.
"""
from pathlib import Path

# ============================================================
# DATA PATHS
# ============================================================
# Place your data files in the data/ directory at the project root,
# or set the DATA_DIR environment variable to point elsewhere.
#
# Expected files:
#   data/cleaned_data.csv (or .parquet)
#   data/analysis_data.csv (or .parquet)
#
# Example override:
#   export DATA_DIR="D:\Data\Odds\Output\processed"
# ============================================================
import os

_PROJECT_ROOT = Path(__file__).resolve().parent
_default_data = _PROJECT_ROOT / "data"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_default_data)))

CLEANED_DATA = DATA_DIR / "cleaned_data.csv"
ANALYSIS_DATA = DATA_DIR / "analysis_data.csv"
MATCHED_SAMPLE = DATA_DIR / "matched_sample.parquet"
FILTERED_SAMPLE = DATA_DIR / "filtered_sample.parquet"

# Output
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# COLUMN MAPPINGS
# ============================================================
# Odds columns (decimal format)
CLOSE_ODDS = {
    "home": "close_home_odds",
    "away": "close_away_odds",
    "draw": "close_draw_odds",
}
OPEN_ODDS = {
    "home": "open_home_odds",
    "away": "open_away_odds",
    "draw": "open_draw_odds",
}

# Margin columns
CLOSE_MARGIN = "close_margin"
OPEN_MARGIN = "open_margin"

# Identifiers
MATCH_ID = "match_id"
SPORT = "sport"
LEAGUE = "league"
COUNTRY = "country"
SPORTSBOOK = "sportsbook"
BOOKMAKER = "bookmaker_name"

# Timestamps
START_TIME = "start_time"
MATCH_DATE = "match_date"

# Outcomes
HOME_SCORE = "home_score"
AWAY_SCORE = "away_score"
HOME_WIN = "home_win"
IS_DRAW = "is_draw"

# ============================================================
# DERIVATIVE PARAMETERS
# ============================================================

# Implied probability conversion
def odds_to_prob(odds):
    """Convert decimal odds to raw implied probability."""
    return 1.0 / odds

def prob_to_odds(prob):
    """Convert implied probability back to decimal odds."""
    return 1.0 / prob

# Default strike offsets for vanilla options (in probability space)
DEFAULT_STRIKES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

# Kernel bandwidth for empirical density estimation
KDE_BANDWIDTH = "silverman"  # or float

# Monte Carlo parameters
MC_N_SIMS = 50_000
MC_SEED = 42

# Rolling window for backtesting (number of matches)
BACKTEST_TRAIN_WINDOW = 5000
BACKTEST_STEP = 500

# Portfolio optimization
MIN_WEIGHT = -0.5   # allow modest short positions
MAX_WEIGHT = 2.0
RISK_FREE_RATE = 0.0
CVAR_ALPHA = 0.05   # 5% CVaR

# ============================================================
# BOOKMAKER ID → NAME MAPPING
# ============================================================
BOOKMAKER_NAMES = {
    "3":   "Pinnacle",
    "18":  "bet365",
    "27":  "1xBet",
    "549": "Betway",
    "575": "Unibet",
    "635": "Betfair",
    "851": "Betsson",
    "909": "888sport",
}

# ============================================================
# SPORT CATEGORIES
# ============================================================
TWO_WAY_SPORTS = [
    "basketball", "baseball", "tennis", "volleyball",
    "american-football", "handball", "esports",
]
THREE_WAY_SPORTS = [
    "soccer", "hockey", "rugby",
]

# ============================================================
# PASPA / REGIME DATES
# ============================================================
PASPA_REPEAL = "2018-05-14"
COVID_START = "2020-03-11"
COVID_END = "2020-07-23"

# ============================================================
# VISUALIZATION
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLOR_PALETTE = {
    "primary": "#1a1a2e",
    "accent": "#e94560",
    "secondary": "#0f3460",
    "background": "#f8f9fa",
    "grid": "#dee2e6",
    "profit": "#2ecc71",
    "loss": "#e74c3c",
}
