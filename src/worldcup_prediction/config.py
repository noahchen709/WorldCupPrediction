from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_TEAMS_PATH = DATA_DIR / "sample_teams.csv"
SAMPLE_MATCHES_PATH = DATA_DIR / "sample_matches.csv"
