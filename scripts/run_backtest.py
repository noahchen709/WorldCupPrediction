import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from worldcup_prediction.config import DATA_DIR, REPORTS_DIR
from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.simulation.backtest import HISTORICAL_TOURNAMENTS, run_backtest


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


def write_json(result, tournament, path) -> None:
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "method": "Elo Monte Carlo backtest using pre-tournament historical ratings",
        "ratingSource": tournament.source,
        "summary": asdict(result.summary),
        "teams": [asdict(team) for team in result.teams],
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
    args = parser.parse_args()

    tournament = HISTORICAL_TOURNAMENTS[args.tournament]
    ratings_path = DEFAULT_RATINGS[args.tournament] if args.ratings is None else Path(args.ratings)
    teams = load_backtest_teams(ratings_path)
    result = run_backtest(tournament, teams, iterations=args.iterations, seed=args.seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"{args.tournament}_backtest.csv"
    json_path = REPORTS_DIR / f"{args.tournament}_backtest.json"
    write_csv(result, csv_path)
    write_json(result, tournament, json_path)

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
    print(f"Actual finalist probability mass: {percent(summary.finalist_probability_total)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
