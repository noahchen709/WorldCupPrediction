from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLE_TEAMS_PATH = DATA_DIR / "sample_teams.csv"
SAMPLE_MATCHES_PATH = DATA_DIR / "sample_matches.csv"
ELO_WORLD_URL = "https://www.eloratings.net/World.tsv"
ELO_TEAM_NAMES_URL = "https://www.eloratings.net/en.teams.tsv"
ELO_WORLD_PATH = RAW_DATA_DIR / "world_elo.tsv"
ELO_TEAM_NAMES_PATH = RAW_DATA_DIR / "elo_team_names.tsv"
DERIVED_TEAMS_PATH = DATA_DIR / "derived_team_strengths.csv"
DERIVED_TEAMS_JSON_PATH = DATA_DIR / "derived_team_strengths.json"
