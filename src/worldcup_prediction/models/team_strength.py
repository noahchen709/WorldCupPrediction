from dataclasses import dataclass

from worldcup_prediction.data_loader import TeamRecord


@dataclass(frozen=True)
class TeamStrength:
    team: str
    overall: float
    attack: float
    defense: float


def estimate_team_strength(team: TeamRecord) -> TeamStrength:
    """Placeholder blend for a future calibrated strength model."""
    overall = team.rating * 0.65 + team.attack_rating * 0.2 + team.defense_rating * 0.15
    return TeamStrength(
        team=team.team,
        overall=round(overall, 2),
        attack=team.attack_rating,
        defense=team.defense_rating,
    )
