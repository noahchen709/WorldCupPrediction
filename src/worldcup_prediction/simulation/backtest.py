import random
from dataclasses import dataclass

from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.models.elo import elo_win_draw_loss


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


@dataclass(frozen=True)
class BacktestResult:
    summary: BacktestSummary
    teams: list[BacktestTeamResult]


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

HISTORICAL_TOURNAMENTS = {
    "world-cup-2022": WORLD_CUP_2022,
}


def sample_regulation_result(
    team_a: TeamRecord,
    team_b: TeamRecord,
    rng: random.Random,
) -> tuple[int, int]:
    win, draw, _ = elo_win_draw_loss(team_a.rating, team_b.rating, allow_draw=True)
    roll = rng.random()
    if roll < win:
        return 1, 0
    if roll < win + draw:
        return 0, 0
    return 0, 1


def sample_knockout_winner(team_a: TeamRecord, team_b: TeamRecord, rng: random.Random) -> TeamRecord:
    win, _, _ = elo_win_draw_loss(team_a.rating, team_b.rating, allow_draw=False)
    return team_a if rng.random() < win else team_b


def simulate_group(group: list[TeamRecord], rng: random.Random) -> list[TeamRecord]:
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
            goals_a, goals_b = sample_regulation_result(team_a, team_b, rng)
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
) -> dict[int, tuple[TeamRecord, TeamRecord]]:
    group_tables = {
        group_name: simulate_group(group, rng)
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
) -> tuple[dict[str, list[TeamRecord]], TeamRecord]:
    winners = {}
    for match_number, (team_a, team_b) in round_of_16_matches.items():
        winners[match_number] = sample_knockout_winner(team_a, team_b, rng)

    round_winners = {"qf": list(winners.values())}
    for round_name, round_matches in zip(("sf", "final", "champion"), KNOCKOUT_ROUNDS_32_TEAM):
        for match_number, left_match, right_match in round_matches:
            winners[match_number] = sample_knockout_winner(
                winners[left_match],
                winners[right_match],
                rng,
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


def run_backtest(
    tournament: HistoricalTournament,
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2022,
) -> BacktestResult:
    rng = random.Random(seed)
    groups = make_groups(tournament, teams)
    field = [team for group in groups.values() for team in group]
    counts = {
        team.team: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for team in field
    }

    for _ in range(iterations):
        round_of_16_matches = qualify_round_of_16(groups, rng)
        round_of_16 = {
            team.team
            for match in round_of_16_matches.values()
            for team in match
        }
        for team_name in round_of_16:
            counts[team_name]["r16"] += 1

        round_winners, _ = simulate_knockout(round_of_16_matches, rng)
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
    )
    return BacktestResult(summary=summary, teams=team_results)
