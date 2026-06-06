from dataclasses import dataclass

from .team_strength import TeamStrength


@dataclass(frozen=True)
class ExpectedGoals:
    home_xg: float
    away_xg: float


def predict_expected_goals(home: TeamStrength, away: TeamStrength) -> ExpectedGoals:
    """Placeholder expected-goals estimate for a future scoreline model."""
    home_xg = 1.25 + (home.attack - away.defense) / 35
    away_xg = 1.05 + (away.attack - home.defense) / 35
    return ExpectedGoals(
        home_xg=round(max(0.2, home_xg), 2),
        away_xg=round(max(0.2, away_xg), 2),
    )
