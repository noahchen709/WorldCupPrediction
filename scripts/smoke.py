from worldcup_prediction.config import DERIVED_TEAMS_PATH
from worldcup_prediction.data_loader import load_derived_teams, load_sample_teams
from worldcup_prediction.models.match_outcome import predict_win_draw_loss
from worldcup_prediction.models.scoreline import predict_expected_goals
from worldcup_prediction.models.team_strength import estimate_team_strength
from worldcup_prediction.simulation.tournament import estimate_demo_champion_probabilities


def main() -> None:
    if DERIVED_TEAMS_PATH.exists():
        teams = load_derived_teams()
        source = "World Football Elo ratings"
    else:
        teams = load_sample_teams()
        source = "sample fallback ratings"

    strengths = [estimate_team_strength(team) for team in teams]
    probabilities = estimate_demo_champion_probabilities(strengths)
    outcome = predict_win_draw_loss(strengths[0], strengths[1])
    xg = predict_expected_goals(strengths[0], strengths[1])

    leader = teams[0]
    print(f"Loaded teams: {len(teams)} ({source})")
    print(f"Top Elo team: {leader.team} ({leader.elo:.0f})")
    print(f"Demo champion leader: {probabilities[0].team} ({probabilities[0].probability:.1%})")
    print(
        "Sample match W/D/L: "
        f"{outcome.home_win:.1%} / {outcome.draw:.1%} / {outcome.away_win:.1%}"
    )
    print(f"Sample expected goals: {xg.home_xg:.2f} - {xg.away_xg:.2f}")


if __name__ == "__main__":
    main()
