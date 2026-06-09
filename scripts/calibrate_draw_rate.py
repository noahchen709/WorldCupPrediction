import argparse
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from urllib.request import Request, urlopen

from worldcup_prediction.config import DATA_DIR, RAW_DATA_DIR
from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.models import elo
from worldcup_prediction.simulation.backtest import HISTORICAL_TOURNAMENTS, run_backtest


RESULTS_URL_TEMPLATE = "https://www.eloratings.net/{year}_results.tsv"
BACKTEST_RATINGS = DATA_DIR / "backtests" / "world_cup_2022_elo.csv"


@dataclass(frozen=True)
class MatchSample:
    year: int
    gap: int
    draw: bool


@dataclass(frozen=True)
class DrawParameters:
    floor: float
    ceiling: float
    scale: float


@dataclass(frozen=True)
class DrawMetrics:
    matches: int
    observed_draw_rate: float
    average_probability: float
    log_loss: float
    brier_score: float


def fetch_year_results(year: int, refresh: bool = False) -> str:
    path = RAW_DATA_DIR / f"elo_results_{year}.tsv"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    request = Request(
        RESULTS_URL_TEMPLATE.format(year=year),
        headers={"User-Agent": "WorldCupPredictionStarter/0.1"},
    )
    with urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return content


def parse_samples(content: str) -> list[MatchSample]:
    samples = []
    for line in content.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        samples.append(
            MatchSample(
                year=int(fields[0]),
                gap=abs(int(fields[10]) - int(fields[11])),
                draw=int(fields[5]) == int(fields[6]),
            )
        )
    return samples


def load_samples(
    start_year: int,
    end_year: int,
    refresh: bool = False,
) -> list[MatchSample]:
    samples = []
    for year in range(start_year, end_year + 1):
        samples.extend(parse_samples(fetch_year_results(year, refresh=refresh)))
    return samples


def old_linear_probability(gap: int) -> float:
    return max(0.16, min(0.30, 0.30 - gap / 1200))


def fitted_probability(gap: int, parameters: DrawParameters) -> float:
    return parameters.floor + (parameters.ceiling - parameters.floor) * exp(
        -gap / parameters.scale
    )


def clip_probability(probability: float) -> float:
    return min(1 - 1e-12, max(1e-12, probability))


def metrics(samples: list[MatchSample], probability_fn) -> DrawMetrics:
    if not samples:
        raise ValueError("No samples available for metrics.")

    log_loss_total = 0.0
    brier_total = 0.0
    probability_total = 0.0
    draws = 0
    for sample in samples:
        probability = clip_probability(probability_fn(sample.gap))
        actual = 1.0 if sample.draw else 0.0
        draws += sample.draw
        probability_total += probability
        log_loss_total += -(
            actual * log(probability) + (1 - actual) * log(1 - probability)
        )
        brier_total += (probability - actual) ** 2

    count = len(samples)
    return DrawMetrics(
        matches=count,
        observed_draw_rate=draws / count,
        average_probability=probability_total / count,
        log_loss=log_loss_total / count,
        brier_score=brier_total / count,
    )


def grouped_counts(samples: list[MatchSample]) -> list[tuple[int, int, int]]:
    counts: dict[int, list[int]] = {}
    for sample in samples:
        row = counts.setdefault(sample.gap, [0, 0])
        row[0] += int(sample.draw)
        row[1] += 1
    return [(gap, draws, total) for gap, (draws, total) in counts.items()]


def negative_log_loss(
    counts: list[tuple[int, int, int]],
    parameters: DrawParameters,
) -> float:
    total = 0.0
    for gap, draws, matches in counts:
        probability = clip_probability(fitted_probability(gap, parameters))
        total -= draws * log(probability) + (matches - draws) * log(1 - probability)
    return total


def candidate_values(center: float, radius: float, step: float, lower: float, upper: float):
    start = max(lower, center - radius)
    end = min(upper, center + radius)
    value = start
    while value <= end + step / 2:
        yield round(value, 6)
        value += step


def fit_draw_parameters(samples: list[MatchSample]) -> DrawParameters:
    counts = grouped_counts(samples)
    best = DrawParameters(floor=0.14, ceiling=0.30, scale=250.0)
    best_score = negative_log_loss(counts, best)

    search_rounds = [
        (0.12, 0.18, 700.0, 0.01, 0.01, 50.0),
        (0.04, 0.06, 150.0, 0.0025, 0.0025, 10.0),
        (0.01, 0.02, 40.0, 0.0005, 0.0005, 2.0),
    ]
    for (
        floor_radius,
        ceiling_radius,
        scale_radius,
        floor_step,
        ceiling_step,
        scale_step,
    ) in search_rounds:
        for floor in candidate_values(best.floor, floor_radius, floor_step, 0.02, 0.35):
            ceilings = candidate_values(
                best.ceiling,
                ceiling_radius,
                ceiling_step,
                floor,
                0.50,
            )
            for ceiling in ceilings:
                scales = candidate_values(
                    best.scale,
                    scale_radius,
                    scale_step,
                    20.0,
                    1500.0,
                )
                for scale in scales:
                    candidate = DrawParameters(floor=floor, ceiling=ceiling, scale=scale)
                    score = negative_log_loss(counts, candidate)
                    if score < best_score:
                        best = candidate
                        best_score = score
    return best


def load_backtest_teams(path: Path) -> list[TeamRecord]:
    import csv

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


def backtest_summary(
    label: str,
    draw_probability_fn,
    iterations: int,
    seed: int,
) -> dict[str, float | str]:
    original = elo.estimate_draw_probability
    elo.estimate_draw_probability = lambda rating_a, rating_b: draw_probability_fn(
        abs(rating_a - rating_b)
    )
    try:
        result = run_backtest(
            HISTORICAL_TOURNAMENTS["world-cup-2022"],
            load_backtest_teams(BACKTEST_RATINGS),
            iterations=iterations,
            seed=seed,
        )
    finally:
        elo.estimate_draw_probability = original

    summary = result.summary
    return {
        "model": label,
        "top_pick": summary.top_pick,
        "actual_champion_probability": summary.actual_champion_probability,
        "champion_log_loss": summary.champion_log_loss,
        "round_of_16_brier_score": summary.round_of_16_brier_score,
        "stage_brier_score": summary.stage_brier_score,
        "stage_score_mae": summary.stage_score_mae,
        "calibration_error": summary.calibration_error,
    }


def print_draw_metrics(label: str, value: DrawMetrics) -> None:
    print(
        f"{label}: matches={value.matches}, observed={value.observed_draw_rate:.4f}, "
        f"avg_pred={value.average_probability:.4f}, log_loss={value.log_loss:.5f}, "
        f"brier={value.brier_score:.5f}"
    )


def print_backtest_metrics(row: dict[str, float | str]) -> None:
    print(
        f"{row['model']}: top_pick={row['top_pick']}, "
        f"actual_champion_probability={row['actual_champion_probability']:.4f}, "
        f"champion_log_loss={row['champion_log_loss']:.4f}, "
        f"r16_brier={row['round_of_16_brier_score']:.5f}, "
        f"stage_brier={row['stage_brier_score']:.5f}, "
        f"stage_mae={row['stage_score_mae']:.4f}, "
        f"calibration_error={row['calibration_error']:.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit draw-rate parameters from World Football Elo result history."
    )
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--backtest-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    samples = load_samples(args.start_year, args.end_year, refresh=args.refresh)
    train = [sample for sample in samples if sample.year < args.test_start_year]
    test = [sample for sample in samples if sample.year >= args.test_start_year]
    parameters = fit_draw_parameters(train)

    print("Fitted draw model:")
    print(
        f"floor={parameters.floor:.4f}, ceiling={parameters.ceiling:.4f}, "
        f"scale={parameters.scale:.1f}"
    )
    print()

    print("Draw prediction metrics:")
    print_draw_metrics("baseline train", metrics(train, old_linear_probability))
    print_draw_metrics(
        "fitted train",
        metrics(train, lambda gap: fitted_probability(gap, parameters)),
    )
    print_draw_metrics("baseline test", metrics(test, old_linear_probability))
    print_draw_metrics(
        "fitted test",
        metrics(test, lambda gap: fitted_probability(gap, parameters)),
    )
    print()

    print("2022 World Cup backtest metrics:")
    print_backtest_metrics(
        backtest_summary(
            "baseline",
            old_linear_probability,
            iterations=args.backtest_iterations,
            seed=args.seed,
        )
    )
    print_backtest_metrics(
        backtest_summary(
            "fitted",
            lambda gap: fitted_probability(gap, parameters),
            iterations=args.backtest_iterations,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
