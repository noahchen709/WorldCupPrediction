from dataclasses import dataclass
from .elo import elo_win_draw_loss
from .team_strength import TeamStrength


@dataclass(frozen=True)
class MatchOutcomeProbabilities:
    home_win: float
    draw: float
    away_win: float


def predict_win_draw_loss(home: TeamStrength, away: TeamStrength) -> MatchOutcomeProbabilities:
    """Simple placeholder W/D/L estimate from an Elo rating gap."""
    home_win, draw, away_win = elo_win_draw_loss(home.overall, away.overall)
    return MatchOutcomeProbabilities(
        home_win=round(home_win, 4),
        draw=round(draw, 4),
        away_win=round(away_win, 4),
    )
