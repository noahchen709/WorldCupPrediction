from dataclasses import dataclass
from math import exp

from .team_strength import TeamStrength


@dataclass(frozen=True)
class MatchOutcomeProbabilities:
    home_win: float
    draw: float
    away_win: float


def predict_win_draw_loss(home: TeamStrength, away: TeamStrength) -> MatchOutcomeProbabilities:
    """Simple placeholder W/D/L estimate from rating gap."""
    rating_gap = home.overall - away.overall
    draw = max(0.16, 0.27 - abs(rating_gap) / 120)
    home_without_draw = 1 / (1 + exp(-rating_gap / 9))
    home_win = home_without_draw * (1 - draw)
    away_win = 1 - draw - home_win
    return MatchOutcomeProbabilities(
        home_win=round(home_win, 4),
        draw=round(draw, 4),
        away_win=round(away_win, 4),
    )
