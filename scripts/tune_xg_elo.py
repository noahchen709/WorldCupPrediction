import argparse
import csv

from worldcup_prediction.config import DATA_DIR, REPORTS_DIR
from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.simulation.backtest import (
    HISTORICAL_TOURNAMENTS,
    XGModelConfig,
    load_historical_team_ratings,
    run_backtest,
)


DEFAULT_RATINGS = {
    "world-cup-2022": DATA_DIR / "backtests" / "world_cup_2022_elo.csv",
}


def parse_int_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_backtest_teams(path) -> list[TeamRecord]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return [
            TeamRecord(
                team=row["team"],
                confederation="",
                rating=float(row["elo"]),
                attack_rating=float(row["elo"]),
                defense_rating=float(row["elo"]),
                elo=float(row["elo"]),
            )
            for row in rows
        ]


def load_teams(tournament_key: str, tournament) -> list[TeamRecord]:
    if tournament_key in DEFAULT_RATINGS:
        return load_backtest_teams(DEFAULT_RATINGS[tournament_key])
    return load_historical_team_ratings(tournament)


def objective_score(row: dict[str, float], objective: str) -> float:
    if objective == "stage_brier":
        return row["stage_brier_score"]
    if objective == "champion_log_loss":
        return row["champion_log_loss"]
    return row["stage_brier_score"] + 0.02 * row["champion_log_loss"]


def evaluate_config(
    config: XGModelConfig,
    tournament_teams,
    iterations: int,
    seed: int,
    objective: str,
) -> dict[str, float]:
    summaries = []
    for offset, (tournament_key, tournament, teams) in enumerate(tournament_teams):
        result = run_backtest(
            tournament,
            teams,
            iterations=iterations,
            seed=seed + offset,
            method="xg",
            xg_config=config,
        )
        summary = result.summary
        summaries.append(
            {
                "tournament": tournament_key,
                "champion_log_loss": summary.champion_log_loss,
                "round_of_16_brier_score": summary.round_of_16_brier_score,
                "stage_brier_score": summary.stage_brier_score,
                "stage_score_mae": summary.stage_score_mae,
                "calibration_error": summary.calibration_error,
            }
        )

    aggregate = {
        "history_years": config.history_years,
        "recency_half_life_days": config.recency_half_life_days,
        "elo_goal_adjustment_scale": config.elo_goal_adjustment_scale,
        "match_type_weights": config.match_type_weights,
    }
    for key in (
        "champion_log_loss",
        "round_of_16_brier_score",
        "stage_brier_score",
        "stage_score_mae",
        "calibration_error",
    ):
        aggregate[key] = sum(summary[key] for summary in summaries) / len(summaries)
    aggregate["objective_score"] = objective_score(aggregate, objective)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search xG/Elo-adjusted backtest parameters."
    )
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--history-years", default="3,4,5")
    parser.add_argument("--half-life-days", default="365,500,730")
    parser.add_argument("--elo-scales", default="450,650,850")
    parser.add_argument(
        "--objective",
        choices=("combined", "stage_brier", "champion_log_loss"),
        default="combined",
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    tournament_teams = [
        (key, tournament, load_teams(key, tournament))
        for key, tournament in sorted(HISTORICAL_TOURNAMENTS.items())
    ]
    rows = []
    for history_years in parse_int_values(args.history_years):
        for half_life_days in parse_float_values(args.half_life_days):
            for elo_scale in parse_float_values(args.elo_scales):
                config = XGModelConfig(
                    history_years=history_years,
                    recency_half_life_days=half_life_days,
                    elo_goal_adjustment_scale=elo_scale,
                )
                rows.append(
                    evaluate_config(
                        config,
                        tournament_teams,
                        iterations=args.iterations,
                        seed=args.seed,
                        objective=args.objective,
                    )
                )

    rows.sort(key=lambda row: row["objective_score"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "xg_elo_tuning.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print("Best xG/Elo config:")
    print(
        "history_years={history_years}, half_life_days={recency_half_life_days:.0f}, "
        "elo_scale={elo_goal_adjustment_scale:.0f}".format(**best)
    )
    print(
        "objective={objective_score:.5f}, champion_log_loss={champion_log_loss:.5f}, "
        "r16_brier={round_of_16_brier_score:.5f}, stage_brier={stage_brier_score:.5f}, "
        "stage_mae={stage_score_mae:.5f}, calibration_error={calibration_error:.5f}".format(
            **best
        )
    )
    print()
    print(f"Top {min(args.top, len(rows))}:")
    for row in rows[: args.top]:
        print(
            "history={history_years}, half_life={recency_half_life_days:.0f}, "
            "elo_scale={elo_goal_adjustment_scale:.0f}, objective={objective_score:.5f}, "
            "stage_brier={stage_brier_score:.5f}, champion_log_loss={champion_log_loss:.5f}".format(
                **row
            )
        )
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
