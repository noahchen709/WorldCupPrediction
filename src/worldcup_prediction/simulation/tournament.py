from dataclasses import dataclass

from worldcup_prediction.models.team_strength import TeamStrength


@dataclass(frozen=True)
class ChampionProbability:
    team: str
    probability: float


def estimate_demo_champion_probabilities(
    strengths: list[TeamStrength],
) -> list[ChampionProbability]:
    """Placeholder champion probabilities for dashboard-style smoke tests."""
    field_average = sum(team.overall for team in strengths) / len(strengths)
    weights = [max(0.01, (team.overall / field_average) ** 8) for team in strengths]
    total = sum(weights)
    probabilities = [
        ChampionProbability(team=team.team, probability=round(weight / total, 4))
        for team, weight in zip(strengths, weights)
    ]
    return sorted(probabilities, key=lambda item: item.probability, reverse=True)
