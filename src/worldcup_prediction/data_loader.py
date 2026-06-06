import csv
from dataclasses import dataclass

from .config import SAMPLE_TEAMS_PATH


@dataclass(frozen=True)
class TeamRecord:
    team: str
    confederation: str
    rating: float
    attack_rating: float
    defense_rating: float


def load_sample_teams(path=SAMPLE_TEAMS_PATH) -> list[TeamRecord]:
    """Load starter team data from CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [
            TeamRecord(
                team=row["team"],
                confederation=row["confederation"],
                rating=float(row["rating"]),
                attack_rating=float(row["attack_rating"]),
                defense_rating=float(row["defense_rating"]),
            )
            for row in rows
        ]
