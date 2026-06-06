from worldcup_prediction.data_loader import load_sample_teams
from worldcup_prediction.models.match_outcome import predict_win_draw_loss
from worldcup_prediction.models.scoreline import predict_expected_goals
from worldcup_prediction.models.team_strength import estimate_team_strength
from worldcup_prediction.simulation.tournament import estimate_demo_champion_probabilities


def main() -> None:
    teams = load_sample_teams()
    strengths = [estimate_team_strength(team) for team in teams]
    probabilities = estimate_demo_champion_probabilities(strengths)
    outcome = predict_win_draw_loss(strengths[0], strengths[1])
    xg = predict_expected_goals(strengths[0], strengths[1])

    print(f"Loaded teams: {len(teams)}")
    print(f"Demo champion leader: {probabilities[0].team} ({probabilities[0].probability:.1%})")
    print(
        "Sample match W/D/L: "
        f"{outcome.home_win:.1%} / {outcome.draw:.1%} / {outcome.away_win:.1%}"
    )
    print(f"Sample expected goals: {xg.home_xg:.2f} - {xg.away_xg:.2f}")


if __name__ == "__main__":
    main()
