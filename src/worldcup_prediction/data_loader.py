import csv
from dataclasses import dataclass

from .config import DERIVED_TEAMS_PATH


@dataclass(frozen=True)
class TeamRecord:
    team: str
    confederation: str
    rating: float
    attack_rating: float
    defense_rating: float
    elo: float = 0
    rank: int = 0
    matches: int = 0


def load_derived_teams(path=DERIVED_TEAMS_PATH) -> list[TeamRecord]:
    """Load team ratings exported from World Football Elo data."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [
            TeamRecord(
                team=row["team"],
                confederation=row["confederation"],
                rating=float(row["rating"]),
                attack_rating=float(row["attack_rating"]),
                defense_rating=float(row["defense_rating"]),
                elo=float(row.get("elo") or row["rating"]),
                rank=int(row.get("rank") or 0),
                matches=int(row.get("matches") or 0),
            )
            for row in rows
        ]
