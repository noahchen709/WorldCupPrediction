from worldcup_prediction.data_loader import load_derived_teams
from worldcup_prediction.models.match_outcome import predict_win_draw_loss
from worldcup_prediction.models.scoreline import predict_expected_goals
from worldcup_prediction.models.team_strength import estimate_team_strength
from worldcup_prediction.simulation.monte_carlo import simulate_tournament
from worldcup_prediction.simulation.tournament import estimate_baseline_champion_probabilities


def main() -> None:
    teams = load_derived_teams()
    strengths = [estimate_team_strength(team) for team in teams]
    probabilities = estimate_baseline_champion_probabilities(strengths)
    outcome = predict_win_draw_loss(strengths[0], strengths[1])
    xg = predict_expected_goals(strengths[0], strengths[1])
    simulation = simulate_tournament(teams, iterations=100, seed=2026)

    leader = teams[0]
    print(f"Loaded teams: {len(teams)} (World Football Elo ratings)")
    print(f"Top Elo team: {leader.team} ({leader.elo:.0f})")
    print(f"Baseline champion leader: {probabilities[0].team} ({probabilities[0].probability:.1%})")
    print(f"Monte Carlo smoke leader: {simulation[0].team} ({simulation[0].champion_probability:.1%})")
    print(
        "Match W/D/L: "
        f"{outcome.home_win:.1%} / {outcome.draw:.1%} / {outcome.away_win:.1%}"
    )
    print(f"Expected goals: {xg.home_xg:.2f} - {xg.away_xg:.2f}")


if __name__ == "__main__":
    main()
