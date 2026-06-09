import math
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from worldcup_prediction.config import RAW_DATA_DIR
from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.models.elo import elo_win_draw_loss, estimate_draw_probability


ELO_GOAL_ADJUSTMENT_SCALE = 650.0
GOAL_RATE_FLOOR = 0.15
GOAL_RATE_CEILING = 4.5
DEFAULT_HISTORY_YEARS = 4
RECENCY_HALF_LIFE_DAYS = 500.0
DEFAULT_MATCH_TYPE_WEIGHTS = {
    "F": 0.35,
    "FT": 0.45,
    "FQ": 0.65,
    "WQ": 0.85,
    "WC": 1.35,
    "AR": 1.25,
    "AM": 1.25,
    "EU": 1.25,
    "GC": 1.25,
    "NL": 1.10,
    "UNL": 1.10,
    "CNL": 1.10,
}


@dataclass(frozen=True)
class XGModelConfig:
    history_years: int = DEFAULT_HISTORY_YEARS
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    elo_goal_adjustment_scale: float = ELO_GOAL_ADJUSTMENT_SCALE
    match_type_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MATCH_TYPE_WEIGHTS)
    )


@dataclass(frozen=True)
class HistoricalTournament:
    name: str
    as_of: str
    source: str
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    actual_finish: dict[str, str]


@dataclass(frozen=True)
class BacktestTeamResult:
    team: str
    elo: float
    actual_stage: str
    champion_probability: float
    final_probability: float
    semifinal_probability: float
    quarterfinal_probability: float
    round_of_16_probability: float


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    average_probability: float
    observed_frequency: float
    brier_score: float


@dataclass(frozen=True)
class BacktestSummary:
    tournament: str
    as_of: str
    iterations: int
    seed: int
    actual_champion: str
    actual_champion_probability: float
    actual_champion_rank: int
    top_pick: str
    top_pick_probability: float
    top_pick_actual_stage: str
    finalist_probability_total: float
    semifinalist_probability_total: float
    quarterfinalist_probability_total: float
    champion_log_loss: float
    champion_brier_score: float
    round_of_16_brier_score: float
    finalist_brier_score: float
    semifinalist_brier_score: float
    quarterfinalist_brier_score: float
    stage_brier_score: float
    stage_score_mae: float
    top_pick_accuracy: float
    calibration_error: float
    calibration_bins: list[CalibrationBin]


@dataclass(frozen=True)
class BacktestResult:
    summary: BacktestSummary
    teams: list[BacktestTeamResult]


@dataclass(frozen=True)
class MethodComparison:
    model: str
    top_pick: str
    actual_champion_probability: float
    champion_log_loss: float
    round_of_16_brier_score: float
    stage_brier_score: float
    stage_score_mae: float
    calibration_error: float


@dataclass(frozen=True)
class TeamExpectedGoalsProfile:
    team: str
    matches: int
    adjusted_goals_for: float
    adjusted_goals_against: float


@dataclass(frozen=True)
class ExpectedGoalsModel:
    profiles: dict[str, TeamExpectedGoalsProfile]
    average_goals: float
    average_rating: float
    config: XGModelConfig


@dataclass(frozen=True)
class HistoricalMatch:
    played_on: date
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    match_type: str
    rating_a: float
    rating_b: float


ROUND_OF_16_MATCHES: tuple[tuple[int, tuple[str, int], tuple[str, int]], ...] = (
    (49, ("A", 1), ("B", 2)),
    (50, ("C", 1), ("D", 2)),
    (51, ("B", 1), ("A", 2)),
    (52, ("D", 1), ("C", 2)),
    (53, ("E", 1), ("F", 2)),
    (54, ("G", 1), ("H", 2)),
    (55, ("F", 1), ("E", 2)),
    (56, ("H", 1), ("G", 2)),
)

KNOCKOUT_ROUNDS_32_TEAM: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((57, 49, 50), (58, 53, 54), (59, 51, 52), (60, 55, 56)),
    ((61, 57, 58), (62, 59, 60)),
    ((64, 61, 62),),
)

STAGE_ORDER = {
    "group": 0,
    "round_of_16": 1,
    "quarterfinal": 2,
    "semifinal": 3,
    "runner_up": 4,
    "champion": 5,
}

STAGE_PROBABILITY_FIELDS = (
    ("round_of_16", "round_of_16_probability"),
    ("quarterfinal", "quarterfinal_probability"),
    ("semifinal", "semifinal_probability"),
    ("runner_up", "final_probability"),
    ("champion", "champion_probability"),
)


WORLD_CUP_2022 = HistoricalTournament(
    name="2022 FIFA World Cup",
    as_of="2022-11-19",
    source=(
        "World Football Elo ratings as of 2022-11-19, mirrored from "
        "international-football.net/elo-ratings-table"
    ),
    groups=(
        ("A", ("Qatar", "Ecuador", "Senegal", "Netherlands")),
        ("B", ("England", "Iran", "United States", "Wales")),
        ("C", ("Argentina", "Saudi Arabia", "Mexico", "Poland")),
        ("D", ("France", "Australia", "Denmark", "Tunisia")),
        ("E", ("Spain", "Costa Rica", "Germany", "Japan")),
        ("F", ("Belgium", "Canada", "Morocco", "Croatia")),
        ("G", ("Brazil", "Serbia", "Switzerland", "Cameroon")),
        ("H", ("Portugal", "Ghana", "Uruguay", "South Korea")),
    ),
    actual_finish={
        "Argentina": "champion",
        "France": "runner_up",
        "Croatia": "semifinal",
        "Morocco": "semifinal",
        "Netherlands": "quarterfinal",
        "Brazil": "quarterfinal",
        "England": "quarterfinal",
        "Portugal": "quarterfinal",
        "United States": "round_of_16",
        "Australia": "round_of_16",
        "Poland": "round_of_16",
        "Senegal": "round_of_16",
        "Japan": "round_of_16",
        "Spain": "round_of_16",
        "Switzerland": "round_of_16",
        "South Korea": "round_of_16",
    },
)

WORLD_CUP_2018 = HistoricalTournament(
    name="2018 FIFA World Cup",
    as_of="2018-06-13",
    source=(
        "World Football Elo ratings reconstructed from local Elo result history "
        "before the 2018 FIFA World Cup"
    ),
    groups=(
        ("A", ("Russia", "Saudi Arabia", "Egypt", "Uruguay")),
        ("B", ("Portugal", "Spain", "Morocco", "Iran")),
        ("C", ("France", "Australia", "Peru", "Denmark")),
        ("D", ("Argentina", "Iceland", "Croatia", "Nigeria")),
        ("E", ("Brazil", "Switzerland", "Costa Rica", "Serbia")),
        ("F", ("Germany", "Mexico", "Sweden", "South Korea")),
        ("G", ("Belgium", "Panama", "Tunisia", "England")),
        ("H", ("Poland", "Senegal", "Colombia", "Japan")),
    ),
    actual_finish={
        "France": "champion",
        "Croatia": "runner_up",
        "Belgium": "semifinal",
        "England": "semifinal",
        "Uruguay": "quarterfinal",
        "Brazil": "quarterfinal",
        "Sweden": "quarterfinal",
        "Russia": "quarterfinal",
        "Portugal": "round_of_16",
        "Argentina": "round_of_16",
        "Mexico": "round_of_16",
        "Japan": "round_of_16",
        "Spain": "round_of_16",
        "Denmark": "round_of_16",
        "Switzerland": "round_of_16",
        "Colombia": "round_of_16",
    },
)

WORLD_CUP_2014 = HistoricalTournament(
    name="2014 FIFA World Cup",
    as_of="2014-06-11",
    source=(
        "World Football Elo ratings reconstructed from local Elo result history "
        "before the 2014 FIFA World Cup"
    ),
    groups=(
        ("A", ("Brazil", "Croatia", "Mexico", "Cameroon")),
        ("B", ("Spain", "Netherlands", "Chile", "Australia")),
        ("C", ("Colombia", "Greece", "Ivory Coast", "Japan")),
        ("D", ("Uruguay", "Costa Rica", "England", "Italy")),
        ("E", ("Switzerland", "Ecuador", "France", "Honduras")),
        ("F", ("Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria")),
        ("G", ("Germany", "Portugal", "Ghana", "United States")),
        ("H", ("Belgium", "Algeria", "Russia", "South Korea")),
    ),
    actual_finish={
        "Germany": "champion",
        "Argentina": "runner_up",
        "Brazil": "semifinal",
        "Netherlands": "semifinal",
        "Colombia": "quarterfinal",
        "France": "quarterfinal",
        "Costa Rica": "quarterfinal",
        "Belgium": "quarterfinal",
        "Chile": "round_of_16",
        "Uruguay": "round_of_16",
        "Nigeria": "round_of_16",
        "Algeria": "round_of_16",
        "Switzerland": "round_of_16",
        "United States": "round_of_16",
        "Mexico": "round_of_16",
        "Greece": "round_of_16",
    },
)

HISTORICAL_TOURNAMENTS = {
    "world-cup-2014": WORLD_CUP_2014,
    "world-cup-2018": WORLD_CUP_2018,
    "world-cup-2022": WORLD_CUP_2022,
}


def parse_team_code_names(path: Path = RAW_DATA_DIR / "elo_team_names.tsv") -> dict[str, str]:
    code_names = {}
    if not path.exists():
        return code_names

    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            code_names[fields[0]] = fields[1]
    return code_names


def parse_historical_matches(
    start_year: int,
    end_year: int,
    as_of: date,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> list[HistoricalMatch]:
    code_names = parse_team_code_names(raw_data_dir / "elo_team_names.tsv")
    matches = []
    for year in range(start_year, end_year + 1):
        path = raw_data_dir / f"elo_results_{year}.tsv"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 12:
                continue
            try:
                played_on = date(int(fields[0]), int(fields[1]), int(fields[2]))
            except ValueError:
                continue
            if played_on >= as_of:
                continue
            team_a = code_names.get(fields[3])
            team_b = code_names.get(fields[4])
            if not team_a or not team_b:
                continue
            matches.append(
                HistoricalMatch(
                    played_on=played_on,
                    team_a=team_a,
                    team_b=team_b,
                    goals_a=int(fields[5]),
                    goals_b=int(fields[6]),
                    match_type=fields[7],
                    rating_a=float(fields[10]),
                    rating_b=float(fields[11]),
                )
            )
    return matches


def match_type_weight(match_type: str, config: XGModelConfig) -> float:
    return config.match_type_weights.get(match_type, 1.0)


def parse_rating_delta(value: str) -> float:
    return float(value.replace("−", "-").replace("+", "") or 0)


def load_historical_team_ratings(
    tournament: HistoricalTournament,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> list[TeamRecord]:
    tournament_date = date.fromisoformat(tournament.as_of)
    field = {
        team_name
        for _, group in tournament.groups
        for team_name in group
    }
    code_names = parse_team_code_names(raw_data_dir / "elo_team_names.tsv")
    latest: dict[str, tuple[date, float]] = {}

    for year in range(1994, tournament_date.year + 1):
        path = raw_data_dir / f"elo_results_{year}.tsv"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 14:
                continue
            try:
                played_on = date(int(fields[0]), int(fields[1]), int(fields[2]))
            except ValueError:
                continue
            if played_on >= tournament_date:
                continue
            team_a = code_names.get(fields[3])
            team_b = code_names.get(fields[4])
            if team_a in field:
                latest[team_a] = (
                    played_on,
                    float(fields[10]) + parse_rating_delta(fields[12]),
                )
            if team_b in field:
                latest[team_b] = (
                    played_on,
                    float(fields[11]) + parse_rating_delta(fields[13]),
                )

    missing = sorted(field - set(latest))
    if missing:
        raise ValueError(
            "Could not reconstruct historical Elo ratings for: "
            + ", ".join(missing)
        )

    return [
        TeamRecord(
            team=team_name,
            confederation="",
            rating=latest[team_name][1],
            attack_rating=latest[team_name][1],
            defense_rating=latest[team_name][1],
            elo=latest[team_name][1],
        )
        for team_name in sorted(field)
    ]


def build_expected_goals_model(
    teams: list[TeamRecord],
    tournament: HistoricalTournament,
    history_years: int = DEFAULT_HISTORY_YEARS,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    elo_goal_adjustment_scale: float = ELO_GOAL_ADJUSTMENT_SCALE,
    config: XGModelConfig | None = None,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> ExpectedGoalsModel:
    if config is None:
        config = XGModelConfig(
            history_years=history_years,
            recency_half_life_days=recency_half_life_days,
            elo_goal_adjustment_scale=elo_goal_adjustment_scale,
        )
    tournament_date = date.fromisoformat(tournament.as_of)
    matches = parse_historical_matches(
        tournament_date.year - config.history_years,
        tournament_date.year,
        tournament_date,
        raw_data_dir=raw_data_dir,
    )
    field = {team.team for team in teams}
    totals = {
        team_name: {"matches": 0, "weight": 0.0, "for": 0.0, "against": 0.0}
        for team_name in field
    }

    all_goals = 0
    all_team_matches = 0
    weighted_rating_total = 0.0
    weighted_rating_count = 0.0
    if config.recency_half_life_days <= 0:
        raise ValueError("recency_half_life_days must be greater than zero")
    if config.elo_goal_adjustment_scale <= 0:
        raise ValueError("elo_goal_adjustment_scale must be greater than zero")
    if any(weight <= 0 for weight in config.match_type_weights.values()):
        raise ValueError("match_type_weights must be greater than zero")

    weighted_matches = []
    for match in matches:
        age_days = max(0, (tournament_date - match.played_on).days)
        recency_weight = 0.5 ** (age_days / config.recency_half_life_days)
        effective_weight = recency_weight * match_type_weight(match.match_type, config)
        weighted_matches.append((match, effective_weight))
        all_goals += effective_weight * (match.goals_a + match.goals_b)
        all_team_matches += 2 * effective_weight
        weighted_rating_total += effective_weight * (match.rating_a + match.rating_b)
        weighted_rating_count += 2 * effective_weight

    average_rating = (
        weighted_rating_total / weighted_rating_count
        if weighted_rating_count
        else sum(team.rating for team in teams) / len(teams)
    )

    for match, effective_weight in weighted_matches:
        for_name = match.team_a
        against_name = match.team_b
        if for_name in totals:
            totals[for_name]["matches"] += 1
            totals[for_name]["weight"] += effective_weight
            totals[for_name]["for"] += (
                effective_weight
                * match.goals_a
                * math.exp(
                    (match.rating_b - average_rating)
                    / config.elo_goal_adjustment_scale
                )
            )
            totals[for_name]["against"] += (
                effective_weight
                * match.goals_b
                * math.exp(
                    (average_rating - match.rating_b)
                    / config.elo_goal_adjustment_scale
                )
            )
        if against_name in totals:
            totals[against_name]["matches"] += 1
            totals[against_name]["weight"] += effective_weight
            totals[against_name]["for"] += (
                effective_weight
                * match.goals_b
                * math.exp(
                    (match.rating_a - average_rating)
                    / config.elo_goal_adjustment_scale
                )
            )
            totals[against_name]["against"] += (
                effective_weight
                * match.goals_a
                * math.exp(
                    (average_rating - match.rating_a)
                    / config.elo_goal_adjustment_scale
                )
            )

    average_goals = all_goals / all_team_matches if all_team_matches else 1.25
    profiles = {}
    for team in teams:
        row = totals[team.team]
        if row["weight"]:
            adjusted_for = row["for"] / row["weight"]
            adjusted_against = row["against"] / row["weight"]
        else:
            adjusted_for = average_goals
            adjusted_against = average_goals
        profiles[team.team] = TeamExpectedGoalsProfile(
            team=team.team,
            matches=row["matches"],
            adjusted_goals_for=max(GOAL_RATE_FLOOR, adjusted_for),
            adjusted_goals_against=max(GOAL_RATE_FLOOR, adjusted_against),
        )
    return ExpectedGoalsModel(
        profiles=profiles,
        average_goals=average_goals,
        average_rating=average_rating,
        config=config,
    )


def clamp_goal_rate(value: float) -> float:
    return min(GOAL_RATE_CEILING, max(GOAL_RATE_FLOOR, value))


def expected_goal_rates(
    team_a: TeamRecord,
    team_b: TeamRecord,
    model: ExpectedGoalsModel,
) -> tuple[float, float]:
    profile_a = model.profiles[team_a.team]
    profile_b = model.profiles[team_b.team]
    attack_a = profile_a.adjusted_goals_for * math.exp(
        (model.average_rating - team_b.rating) / model.config.elo_goal_adjustment_scale
    )
    defense_b = profile_b.adjusted_goals_against * math.exp(
        (team_a.rating - model.average_rating) / model.config.elo_goal_adjustment_scale
    )
    attack_b = profile_b.adjusted_goals_for * math.exp(
        (model.average_rating - team_a.rating) / model.config.elo_goal_adjustment_scale
    )
    defense_a = profile_a.adjusted_goals_against * math.exp(
        (team_b.rating - model.average_rating) / model.config.elo_goal_adjustment_scale
    )
    goals_a = math.sqrt(attack_a * defense_b)
    goals_b = math.sqrt(attack_b * defense_a)
    return clamp_goal_rate(goals_a), clamp_goal_rate(goals_b)


def poisson_probabilities(lam: float, max_goals: int = 10) -> list[float]:
    probabilities = [math.exp(-lam)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * lam / goals)
    return probabilities


def poisson_win_draw_loss(goals_a: float, goals_b: float) -> tuple[float, float, float]:
    probabilities_a = poisson_probabilities(goals_a)
    probabilities_b = poisson_probabilities(goals_b)
    win = 0.0
    draw = 0.0
    loss = 0.0
    for score_a, probability_a in enumerate(probabilities_a):
        for score_b, probability_b in enumerate(probabilities_b):
            probability = probability_a * probability_b
            if score_a > score_b:
                win += probability
            elif score_a == score_b:
                draw += probability
            else:
                loss += probability
    total = win + draw + loss
    return win / total, draw / total, loss / total


def xg_win_draw_loss(
    team_a: TeamRecord,
    team_b: TeamRecord,
    model: ExpectedGoalsModel,
    allow_draw: bool = True,
) -> tuple[float, float, float]:
    goals_a, goals_b = expected_goal_rates(team_a, team_b, model)
    poisson_win, poisson_draw, poisson_loss = poisson_win_draw_loss(goals_a, goals_b)
    decisive = poisson_win + poisson_loss
    if decisive <= 0:
        non_draw_win = 0.5
    else:
        non_draw_win = poisson_win / decisive

    if not allow_draw:
        return non_draw_win, 0.0, 1 - non_draw_win

    draw = estimate_draw_probability(team_a.rating, team_b.rating)
    win = (1 - draw) * non_draw_win
    loss = 1 - win - draw
    return win, draw, loss


def sample_poisson_score(lam: float, rng: random.Random) -> int:
    threshold = math.exp(-lam)
    probability = threshold
    cumulative = probability
    roll = rng.random()
    goals = 0
    while roll > cumulative and goals < 12:
        goals += 1
        probability *= lam / goals
        cumulative += probability
    return goals


def sample_xg_regulation_result(
    team_a: TeamRecord,
    team_b: TeamRecord,
    rng: random.Random,
    model: ExpectedGoalsModel,
) -> tuple[int, int]:
    win, draw, _ = xg_win_draw_loss(team_a, team_b, model, allow_draw=True)
    roll = rng.random()
    goals_a, goals_b = expected_goal_rates(team_a, team_b, model)
    if roll < draw:
        drawn_goals = max(0, round((sample_poisson_score(goals_a, rng) + sample_poisson_score(goals_b, rng)) / 2))
        return drawn_goals, drawn_goals
    if roll < draw + win:
        score_a = sample_poisson_score(goals_a, rng)
        score_b = sample_poisson_score(goals_b, rng)
        if score_a <= score_b:
            score_a = score_b + 1
        return score_a, score_b

    score_a = sample_poisson_score(goals_a, rng)
    score_b = sample_poisson_score(goals_b, rng)
    if score_b <= score_a:
        score_b = score_a + 1
    return score_a, score_b


def sample_regulation_result(
    team_a: TeamRecord,
    team_b: TeamRecord,
    rng: random.Random,
    xg_model: ExpectedGoalsModel | None = None,
) -> tuple[int, int]:
    if xg_model is not None:
        return sample_xg_regulation_result(team_a, team_b, rng, xg_model)

    win, draw, _ = elo_win_draw_loss(team_a.rating, team_b.rating, allow_draw=True)
    roll = rng.random()
    if roll < win:
        return 1, 0
    if roll < win + draw:
        return 0, 0
    return 0, 1


def sample_knockout_winner(
    team_a: TeamRecord,
    team_b: TeamRecord,
    rng: random.Random,
    xg_model: ExpectedGoalsModel | None = None,
) -> TeamRecord:
    if xg_model is not None:
        win, _, _ = xg_win_draw_loss(team_a, team_b, xg_model, allow_draw=False)
        return team_a if rng.random() < win else team_b

    win, _, _ = elo_win_draw_loss(team_a.rating, team_b.rating, allow_draw=False)
    return team_a if rng.random() < win else team_b


def simulate_group(
    group: list[TeamRecord],
    rng: random.Random,
    xg_model: ExpectedGoalsModel | None = None,
) -> list[TeamRecord]:
    table = {
        team.team: {
            "record": team,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "wins": 0,
        }
        for team in group
    }
    for index, team_a in enumerate(group):
        for team_b in group[index + 1 :]:
            goals_a, goals_b = sample_regulation_result(team_a, team_b, rng, xg_model)
            row_a = table[team_a.team]
            row_b = table[team_b.team]
            row_a["goals_for"] += goals_a
            row_a["goals_against"] += goals_b
            row_b["goals_for"] += goals_b
            row_b["goals_against"] += goals_a

            if goals_a > goals_b:
                row_a["points"] += 3
                row_a["wins"] += 1
            elif goals_b > goals_a:
                row_b["points"] += 3
                row_b["wins"] += 1
            else:
                row_a["points"] += 1
                row_b["points"] += 1

    return [
        row["record"]
        for row in sorted(
            table.values(),
            key=lambda row: (
                row["points"],
                row["goals_for"] - row["goals_against"],
                row["goals_for"],
                row["wins"],
                row["record"].rating,
                rng.random(),
            ),
            reverse=True,
        )
    ]


def make_groups(
    tournament: HistoricalTournament,
    teams: list[TeamRecord],
) -> dict[str, list[TeamRecord]]:
    records = {team.team: team for team in teams}
    missing = [
        team_name
        for _, group in tournament.groups
        for team_name in group
        if team_name not in records
    ]
    if missing:
        raise ValueError(f"Missing historical rating rows: {', '.join(missing)}")
    return {
        group_name: [records[team_name] for team_name in group]
        for group_name, group in tournament.groups
    }


def qualify_round_of_16(
    groups: dict[str, list[TeamRecord]],
    rng: random.Random,
    xg_model: ExpectedGoalsModel | None = None,
) -> dict[int, tuple[TeamRecord, TeamRecord]]:
    group_tables = {
        group_name: simulate_group(group, rng, xg_model)
        for group_name, group in groups.items()
    }
    return {
        match_number: (
            group_tables[left_group][left_position - 1],
            group_tables[right_group][right_position - 1],
        )
        for match_number, (left_group, left_position), (right_group, right_position)
        in ROUND_OF_16_MATCHES
    }


def simulate_knockout(
    round_of_16_matches: dict[int, tuple[TeamRecord, TeamRecord]],
    rng: random.Random,
    xg_model: ExpectedGoalsModel | None = None,
) -> tuple[dict[str, list[TeamRecord]], TeamRecord]:
    winners = {}
    for match_number, (team_a, team_b) in round_of_16_matches.items():
        winners[match_number] = sample_knockout_winner(team_a, team_b, rng, xg_model)

    round_winners = {"qf": list(winners.values())}
    for round_name, round_matches in zip(("sf", "final", "champion"), KNOCKOUT_ROUNDS_32_TEAM):
        for match_number, left_match, right_match in round_matches:
            winners[match_number] = sample_knockout_winner(
                winners[left_match],
                winners[right_match],
                rng,
                xg_model,
            )
        round_winners[round_name] = [winners[match_number] for match_number, _, _ in round_matches]

    return round_winners, winners[64]


def actual_stage(tournament: HistoricalTournament, team: str) -> str:
    return tournament.actual_finish.get(team, "group")


def probability_total(
    results: list[BacktestTeamResult],
    tournament: HistoricalTournament,
    minimum_stage: str,
    probability_field: str,
) -> float:
    minimum = STAGE_ORDER[minimum_stage]
    return sum(
        getattr(result, probability_field)
        for result in results
        if STAGE_ORDER[actual_stage(tournament, result.team)] >= minimum
    )


def actual_reached(tournament: HistoricalTournament, team: str, minimum_stage: str) -> int:
    return int(STAGE_ORDER[actual_stage(tournament, team)] >= STAGE_ORDER[minimum_stage])


def brier_score(
    results: list[BacktestTeamResult],
    tournament: HistoricalTournament,
    minimum_stage: str,
    probability_field: str,
) -> float:
    return sum(
        (getattr(result, probability_field) - actual_reached(tournament, result.team, minimum_stage)) ** 2
        for result in results
    ) / len(results)


def stage_brier_score(results: list[BacktestTeamResult], tournament: HistoricalTournament) -> float:
    event_count = len(results) * len(STAGE_PROBABILITY_FIELDS)
    return sum(
        (getattr(result, probability_field) - actual_reached(tournament, result.team, minimum_stage)) ** 2
        for result in results
        for minimum_stage, probability_field in STAGE_PROBABILITY_FIELDS
    ) / event_count


def expected_stage_score(result: BacktestTeamResult) -> float:
    return sum(getattr(result, probability_field) for _, probability_field in STAGE_PROBABILITY_FIELDS)


def stage_score_mae(results: list[BacktestTeamResult], tournament: HistoricalTournament) -> float:
    return sum(
        abs(expected_stage_score(result) - STAGE_ORDER[actual_stage(tournament, result.team)])
        for result in results
    ) / len(results)


def calibration_bins(
    results: list[BacktestTeamResult],
    tournament: HistoricalTournament,
    bin_count: int = 5,
) -> list[CalibrationBin]:
    bins = [[] for _ in range(bin_count)]
    for result in results:
        for minimum_stage, probability_field in STAGE_PROBABILITY_FIELDS:
            probability = getattr(result, probability_field)
            outcome = actual_reached(tournament, result.team, minimum_stage)
            bin_index = min(int(probability * bin_count), bin_count - 1)
            bins[bin_index].append((probability, outcome))

    summaries = []
    for index, values in enumerate(bins):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if values:
            average_probability = sum(probability for probability, _ in values) / len(values)
            observed_frequency = sum(outcome for _, outcome in values) / len(values)
            bin_brier_score = sum(
                (probability - outcome) ** 2
                for probability, outcome in values
            ) / len(values)
        else:
            average_probability = 0.0
            observed_frequency = 0.0
            bin_brier_score = 0.0
        summaries.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(values),
                average_probability=average_probability,
                observed_frequency=observed_frequency,
                brier_score=bin_brier_score,
            )
        )
    return summaries


def calibration_error(bins: list[CalibrationBin]) -> float:
    total = sum(bin.count for bin in bins)
    if total == 0:
        return 0.0
    return sum(
        abs(bin.average_probability - bin.observed_frequency) * bin.count
        for bin in bins
    ) / total


def method_comparison(model: str, result: BacktestResult) -> MethodComparison:
    summary = result.summary
    return MethodComparison(
        model=model,
        top_pick=summary.top_pick,
        actual_champion_probability=summary.actual_champion_probability,
        champion_log_loss=summary.champion_log_loss,
        round_of_16_brier_score=summary.round_of_16_brier_score,
        stage_brier_score=summary.stage_brier_score,
        stage_score_mae=summary.stage_score_mae,
        calibration_error=summary.calibration_error,
    )


def compare_backtest_methods(
    tournament: HistoricalTournament,
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2022,
    xg_config: XGModelConfig | None = None,
) -> tuple[BacktestResult, BacktestResult, list[MethodComparison]]:
    groups = make_groups(tournament, teams)
    field = [team for group in groups.values() for team in group]
    xg_model = build_expected_goals_model(field, tournament, config=xg_config)
    elo_result = run_backtest(
        tournament,
        teams,
        iterations=iterations,
        seed=seed,
        method="elo",
    )
    xg_result = run_backtest(
        tournament,
        teams,
        iterations=iterations,
        seed=seed,
        method="xg",
        xg_model=xg_model,
    )
    return (
        elo_result,
        xg_result,
        [
            method_comparison("elo", elo_result),
            method_comparison("xg_elo_adjusted", xg_result),
        ],
    )


def run_backtest(
    tournament: HistoricalTournament,
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2022,
    method: str = "elo",
    xg_model: ExpectedGoalsModel | None = None,
    xg_config: XGModelConfig | None = None,
) -> BacktestResult:
    if method not in {"elo", "xg"}:
        raise ValueError("method must be 'elo' or 'xg'")

    rng = random.Random(seed)
    groups = make_groups(tournament, teams)
    field = [team for group in groups.values() for team in group]
    if method == "xg" and xg_model is None:
        xg_model = build_expected_goals_model(field, tournament, config=xg_config)

    counts = {
        team.team: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for team in field
    }

    for _ in range(iterations):
        round_of_16_matches = qualify_round_of_16(groups, rng, xg_model)
        round_of_16 = {
            team.team
            for match in round_of_16_matches.values()
            for team in match
        }
        for team_name in round_of_16:
            counts[team_name]["r16"] += 1

        round_winners, _ = simulate_knockout(round_of_16_matches, rng, xg_model)
        for round_name, winners in round_winners.items():
            for team in winners:
                counts[team.team][round_name] += 1

    records = {team.team: team for team in field}
    team_results = [
        BacktestTeamResult(
            team=team_name,
            elo=records[team_name].elo,
            actual_stage=actual_stage(tournament, team_name),
            champion_probability=team_counts["champion"] / iterations,
            final_probability=team_counts["final"] / iterations,
            semifinal_probability=team_counts["sf"] / iterations,
            quarterfinal_probability=team_counts["qf"] / iterations,
            round_of_16_probability=team_counts["r16"] / iterations,
        )
        for team_name, team_counts in counts.items()
    ]
    team_results = sorted(team_results, key=lambda item: item.champion_probability, reverse=True)
    champion = next(team for team, stage in tournament.actual_finish.items() if stage == "champion")
    champion_result = next(result for result in team_results if result.team == champion)
    top_pick = team_results[0]
    bins = calibration_bins(team_results, tournament)

    summary = BacktestSummary(
        tournament=tournament.name,
        as_of=tournament.as_of,
        iterations=iterations,
        seed=seed,
        actual_champion=champion,
        actual_champion_probability=champion_result.champion_probability,
        actual_champion_rank=team_results.index(champion_result) + 1,
        top_pick=top_pick.team,
        top_pick_probability=top_pick.champion_probability,
        top_pick_actual_stage=top_pick.actual_stage,
        finalist_probability_total=probability_total(
            team_results,
            tournament,
            "runner_up",
            "final_probability",
        ),
        semifinalist_probability_total=probability_total(
            team_results,
            tournament,
            "semifinal",
            "semifinal_probability",
        ),
        quarterfinalist_probability_total=probability_total(
            team_results,
            tournament,
            "quarterfinal",
            "quarterfinal_probability",
        ),
        champion_log_loss=-math.log(max(champion_result.champion_probability, 1e-15)),
        champion_brier_score=brier_score(
            team_results,
            tournament,
            "champion",
            "champion_probability",
        ),
        round_of_16_brier_score=brier_score(
            team_results,
            tournament,
            "round_of_16",
            "round_of_16_probability",
        ),
        finalist_brier_score=brier_score(
            team_results,
            tournament,
            "runner_up",
            "final_probability",
        ),
        semifinalist_brier_score=brier_score(
            team_results,
            tournament,
            "semifinal",
            "semifinal_probability",
        ),
        quarterfinalist_brier_score=brier_score(
            team_results,
            tournament,
            "quarterfinal",
            "quarterfinal_probability",
        ),
        stage_brier_score=stage_brier_score(team_results, tournament),
        stage_score_mae=stage_score_mae(team_results, tournament),
        top_pick_accuracy=float(top_pick.actual_stage == "champion"),
        calibration_error=calibration_error(bins),
        calibration_bins=bins,
    )
    return BacktestResult(summary=summary, teams=team_results)
