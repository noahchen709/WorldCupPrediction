import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from worldcup_prediction.config import DATA_DIR, REPORTS_DIR
from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.simulation.backtest import (
    HISTORICAL_TOURNAMENTS,
    compare_backtest_methods,
    run_backtest,
)


DEFAULT_RATINGS = {
    "world-cup-2022": DATA_DIR / "backtests" / "world_cup_2022_elo.csv",
}


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


def write_csv(result, path) -> None:
    fieldnames = [
        "team",
        "elo",
        "actual_stage",
        "champion_probability",
        "final_probability",
        "semifinal_probability",
        "quarterfinal_probability",
        "round_of_16_probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for team in result.teams:
            writer.writerow(asdict(team))


def write_json(result, tournament, path, method_comparisons=None, xg_result=None) -> None:
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "method": "Elo Monte Carlo backtest using pre-tournament historical ratings",
        "ratingSource": tournament.source,
        "summary": asdict(result.summary),
        "teams": [asdict(team) for team in result.teams],
    }
    if method_comparisons is not None:
        payload["methodComparison"] = [asdict(row) for row in method_comparisons]
    if xg_result is not None:
        payload["xgEloAdjusted"] = {
            "method": (
                "Monte Carlo using recency-weighted team scoring history as an expected-goals "
                "proxy, adjusted for opponent Elo difference while retaining the fitted draw rate"
            ),
            "summary": asdict(xg_result.summary),
            "teams": [asdict(team) for team in xg_result.teams],
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the Elo tournament model against completed tournaments."
    )
    parser.add_argument(
        "--tournament",
        choices=sorted(HISTORICAL_TOURNAMENTS),
        default="world-cup-2022",
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--ratings", default=None, help="Optional CSV with team,elo columns.")
    parser.add_argument(
        "--method",
        choices=("elo", "xg", "compare"),
        default="compare",
        help="Backtest method to run. Default compares Elo and xG/Elo-adjusted history.",
    )
    args = parser.parse_args()

    tournament = HISTORICAL_TOURNAMENTS[args.tournament]
    ratings_path = DEFAULT_RATINGS[args.tournament] if args.ratings is None else Path(args.ratings)
    teams = load_backtest_teams(ratings_path)
    xg_result = None
    comparisons = None
    if args.method == "compare":
        result, xg_result, comparisons = compare_backtest_methods(
            tournament,
            teams,
            iterations=args.iterations,
            seed=args.seed,
        )
    else:
        result = run_backtest(
            tournament,
            teams,
            iterations=args.iterations,
            seed=args.seed,
            method=args.method,
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"{args.tournament}_backtest.csv"
    json_path = REPORTS_DIR / f"{args.tournament}_backtest.json"
    write_csv(result, csv_path)
    if xg_result is not None:
        write_csv(xg_result, REPORTS_DIR / f"{args.tournament}_xg_backtest.csv")
    write_json(result, tournament, json_path, method_comparisons=comparisons, xg_result=xg_result)

    summary = result.summary
    print(f"Backtest: {summary.tournament} ({summary.iterations} iterations)")
    print(
        f"Top pick: {summary.top_pick} {percent(summary.top_pick_probability)} "
        f"(actual: {summary.top_pick_actual_stage})"
    )
    print(
        f"Actual champion: {summary.actual_champion} "
        f"{percent(summary.actual_champion_probability)} "
        f"(model rank #{summary.actual_champion_rank})"
    )
    print(f"Champion log loss: {summary.champion_log_loss:.3f}")
    print(f"Stage Brier score: {summary.stage_brier_score:.3f}")
    print(f"Calibration error: {summary.calibration_error:.3f}")
    print(f"Actual finalist probability mass: {percent(summary.finalist_probability_total)}")
    if comparisons is not None:
        print()
        print("Method comparison:")
        for row in comparisons:
            print(
                f"{row.model}: top_pick={row.top_pick}, "
                f"actual_champion={percent(row.actual_champion_probability)}, "
                f"champion_log_loss={row.champion_log_loss:.3f}, "
                f"r16_brier={row.round_of_16_brier_score:.3f}, "
                f"stage_brier={row.stage_brier_score:.3f}, "
                f"stage_mae={row.stage_score_mae:.3f}, "
                f"calibration_error={row.calibration_error:.3f}"
            )
    print(f"Wrote: {csv_path}")
    if xg_result is not None:
        print(f"Wrote: {REPORTS_DIR / f'{args.tournament}_xg_backtest.csv'}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
