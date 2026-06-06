import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.models.elo import elo_win_draw_loss

OFFICIAL_2026_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A", ("Mexico", "South Africa", "South Korea", "Czechia")),
    ("B", ("Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina")),
    ("C", ("Brazil", "Morocco", "Haiti", "Scotland")),
    ("D", ("United States", "Paraguay", "Australia", "Turkey")),
    ("E", ("Germany", "Ecuador", "Curaçao", "Ivory Coast")),
    ("F", ("Netherlands", "Japan", "Tunisia", "Sweden")),
    ("G", ("Belgium", "Iran", "Egypt", "New Zealand")),
    ("H", ("Spain", "Uruguay", "Saudi Arabia", "Cape Verde")),
    ("I", ("France", "Senegal", "Iraq", "Norway")),
    ("J", ("Argentina", "Austria", "Algeria", "Jordan")),
    ("K", ("Portugal", "Colombia", "Uzbekistan", "DR Congo")),
    ("L", ("England", "Croatia", "Ghana", "Panama")),
)

ROUND_OF_32_MATCHES: tuple[tuple[int, tuple[str, str], tuple[str, str]], ...] = (
    (73, ("2", "A"), ("2", "B")),
    (74, ("1", "E"), ("3", "ABCDF")),
    (75, ("1", "F"), ("2", "C")),
    (76, ("1", "C"), ("2", "F")),
    (77, ("1", "I"), ("3", "CDFGH")),
    (78, ("2", "E"), ("2", "I")),
    (79, ("1", "A"), ("3", "CEFHI")),
    (80, ("1", "L"), ("3", "EHIJK")),
    (81, ("1", "D"), ("3", "BEFIJ")),
    (82, ("1", "G"), ("3", "AEHIJ")),
    (83, ("2", "K"), ("2", "L")),
    (84, ("1", "H"), ("2", "J")),
    (85, ("1", "B"), ("3", "EFGIJ")),
    (86, ("1", "J"), ("2", "H")),
    (87, ("1", "K"), ("3", "DEIJL")),
    (88, ("2", "D"), ("2", "G")),
)

KNOCKOUT_ROUNDS: tuple[tuple[tuple[int, int, int], ...], ...] = (
    (
        (89, 74, 77),
        (90, 73, 75),
        (91, 76, 78),
        (92, 79, 80),
        (93, 83, 84),
        (94, 81, 82),
        (95, 86, 88),
        (96, 85, 87),
    ),
    (
        (97, 89, 90),
        (98, 93, 94),
        (99, 91, 92),
        (100, 95, 96),
    ),
    (
        (101, 97, 98),
        (102, 99, 100),
    ),
    ((104, 101, 102),),
)

MATCH_ROUNDS = {
    **{match_number: "Round of 32" for match_number, _, _ in ROUND_OF_32_MATCHES},
    **{match_number: "Round of 16" for match_number, _, _ in KNOCKOUT_ROUNDS[0]},
    **{match_number: "Quarter-finals" for match_number, _, _ in KNOCKOUT_ROUNDS[1]},
    **{match_number: "Semi-finals" for match_number, _, _ in KNOCKOUT_ROUNDS[2]},
    **{match_number: "Final" for match_number, _, _ in KNOCKOUT_ROUNDS[3]},
}


@dataclass
class SimTeam:
    record: TeamRecord
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class TournamentSimulationResult:
    team: str
    rank: int
    elo: float
    champion_probability: float
    final_probability: float
    semifinal_probability: float
    quarterfinal_probability: float
    round_of_16_probability: float


@dataclass(frozen=True)
class BracketMatchSimulationResult:
    match: int
    round: str
    entrant_probabilities: list[dict]
    matchup_probabilities: list[dict]


def make_official_2026_groups(teams: list[TeamRecord]) -> dict[str, list[TeamRecord]]:
    records = {team.team: team for team in teams}
    missing = [
        team_name
        for _, group in OFFICIAL_2026_GROUPS
        for team_name in group
        if team_name not in records
    ]
    if missing:
        raise ValueError(f"Missing World Cup 2026 teams in rating data: {', '.join(missing)}")
    return {
        group_name: [records[team_name] for team_name in group]
        for group_name, group in OFFICIAL_2026_GROUPS
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


def simulate_group(group: list[TeamRecord], rng: random.Random) -> list[SimTeam]:
    table = {team.team: SimTeam(team) for team in group}
    for team_a, team_b in combinations(group, 2):
        goals_a, goals_b = sample_regulation_result(team_a, team_b, rng)
        row_a = table[team_a.team]
        row_b = table[team_b.team]
        row_a.goals_for += goals_a
        row_a.goals_against += goals_b
        row_b.goals_for += goals_b
        row_b.goals_against += goals_a

        if goals_a > goals_b:
            row_a.points += 3
            row_a.wins += 1
        elif goals_b > goals_a:
            row_b.points += 3
            row_b.wins += 1
        else:
            row_a.points += 1
            row_b.points += 1

    return sorted(
        table.values(),
        key=lambda row: (
            row.points,
            row.goal_difference,
            row.goals_for,
            row.wins,
            row.record.rating,
            rng.random(),
        ),
        reverse=True,
    )


def assign_third_place_slots(
    third_place_groups: list[str],
    slot_options: list[tuple[int, str]],
) -> dict[int, str]:
    remaining_slots = list(slot_options)
    remaining_groups = set(third_place_groups)

    def search(slots: list[tuple[int, str]], groups: set[str]) -> dict[int, str] | None:
        if not slots:
            return {}

        match_number, allowed_groups = min(slots, key=lambda slot: len(set(slot[1]) & groups))
        candidates = [group for group in third_place_groups if group in groups and group in allowed_groups]
        next_slots = [slot for slot in slots if slot[0] != match_number]

        for group in candidates:
            assignment = search(next_slots, groups - {group})
            if assignment is not None:
                assignment[match_number] = group
                return assignment
        return None

    assignment = search(remaining_slots, remaining_groups)
    if assignment is None:
        raise ValueError(f"Could not assign third-place groups to bracket slots: {third_place_groups}")
    return assignment


def qualify_official_round_of_32(
    groups: dict[str, list[TeamRecord]],
    rng: random.Random,
) -> tuple[dict[int, tuple[TeamRecord, TeamRecord]], list[TeamRecord]]:
    group_tables = {
        group_name: simulate_group(group, rng)
        for group_name, group in groups.items()
    }
    best_thirds = sorted(
        ((group_name, table[2]) for group_name, table in group_tables.items()),
        key=lambda item: (
            item[1].points,
            item[1].goal_difference,
            item[1].goals_for,
            item[1].wins,
            item[1].record.rating,
            rng.random(),
        ),
        reverse=True,
    )[:8]
    third_records = {group_name: row.record for group_name, row in best_thirds}
    third_slots = [
        (match_number, selector[1])
        for match_number, _, selector in ROUND_OF_32_MATCHES
        if selector[0] == "3"
    ]
    third_assignment = assign_third_place_slots(list(third_records), third_slots)

    def resolve(selector: tuple[str, str], match_number: int) -> TeamRecord:
        position, group_selector = selector
        if position == "1":
            return group_tables[group_selector][0].record
        if position == "2":
            return group_tables[group_selector][1].record
        if position == "3":
            return third_records[third_assignment[match_number]]
        raise ValueError(f"Unsupported bracket selector: {selector}")

    matches = {
        match_number: (
            resolve(left_selector, match_number),
            resolve(right_selector, match_number),
        )
        for match_number, left_selector, right_selector in ROUND_OF_32_MATCHES
    }
    qualifiers = [
        row.record
        for table in group_tables.values()
        for row in table[:2]
    ] + list(third_records.values())
    return matches, qualifiers


def simulate_official_knockout(
    round_of_32_matches: dict[int, tuple[TeamRecord, TeamRecord]],
    rng: random.Random,
) -> tuple[
    dict[str, list[TeamRecord]],
    TeamRecord,
    dict[int, tuple[TeamRecord, TeamRecord, TeamRecord]],
]:
    match_results = {}
    winners = {}
    for match_number, (team_a, team_b) in round_of_32_matches.items():
        winner = sample_knockout_winner(team_a, team_b, rng)
        winners[match_number] = winner
        match_results[match_number] = (team_a, team_b, winner)

    round_winners = {"r16": list(winners.values())}

    round_names = ("qf", "sf", "final", "champion")
    for round_name, round_matches in zip(round_names, KNOCKOUT_ROUNDS):
        for match_number, left_match, right_match in round_matches:
            team_a = winners[left_match]
            team_b = winners[right_match]
            winner = sample_knockout_winner(team_a, team_b, rng)
            winners[match_number] = winner
            match_results[match_number] = (team_a, team_b, winner)
        round_winners[round_name] = [winners[match_number] for match_number, _, _ in round_matches]

    return round_winners, winners[104], match_results


def summarize_bracket_matches(
    entrant_counts: dict[int, Counter],
    win_counts: dict[int, Counter],
    matchup_counts: dict[int, Counter],
    matchup_win_counts: dict[int, Counter],
    iterations: int,
    top_matchups: int = 5,
) -> list[BracketMatchSimulationResult]:
    results = []
    for match_number in sorted(MATCH_ROUNDS):
        entrants = [
            {
                "team": team,
                "appearance_probability": count / iterations,
                "win_probability": win_counts[match_number][team] / iterations,
                "conditional_win_probability": (
                    win_counts[match_number][team] / count
                    if count
                    else 0
                ),
            }
            for team, count in entrant_counts[match_number].most_common()
        ]
        matchups = []
        for matchup, count in matchup_counts[match_number].most_common(top_matchups):
            team_a, team_b = matchup
            wins = matchup_win_counts[match_number][matchup]
            team_a_wins = wins.get(team_a, 0)
            team_b_wins = wins.get(team_b, 0)
            matchups.append(
                {
                    "teams": [team_a, team_b],
                    "probability": count / iterations,
                    "team_win_probabilities": {
                        team_a: team_a_wins / count if count else 0,
                        team_b: team_b_wins / count if count else 0,
                    },
                }
            )
        results.append(
            BracketMatchSimulationResult(
                match=match_number,
                round=MATCH_ROUNDS[match_number],
                entrant_probabilities=entrants,
                matchup_probabilities=matchups,
            )
        )
    return results


def simulate_tournament_with_bracket(
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2026,
) -> tuple[list[TournamentSimulationResult], list[BracketMatchSimulationResult]]:
    rng = random.Random(seed)
    groups = make_official_2026_groups(teams)
    field = [team for group in groups.values() for team in group]
    counts = {
        team.team: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for team in field
    }
    entrant_counts = {match_number: Counter() for match_number in MATCH_ROUNDS}
    win_counts = {match_number: Counter() for match_number in MATCH_ROUNDS}
    matchup_counts = {match_number: Counter() for match_number in MATCH_ROUNDS}
    matchup_win_counts = defaultdict(lambda: defaultdict(Counter))

    for _ in range(iterations):
        round_of_32_matches, round_of_32 = qualify_official_round_of_32(groups, rng)
        for team in round_of_32:
            counts[team.team]["r32"] += 1

        round_winners, champion, match_results = simulate_official_knockout(round_of_32_matches, rng)
        for round_name, winners in round_winners.items():
            for team in winners:
                counts[team.team][round_name] += 1
        for match_number, (team_a, team_b, winner) in match_results.items():
            matchup = tuple(sorted((team_a.team, team_b.team)))
            entrant_counts[match_number][team_a.team] += 1
            entrant_counts[match_number][team_b.team] += 1
            win_counts[match_number][winner.team] += 1
            matchup_counts[match_number][matchup] += 1
            matchup_win_counts[match_number][matchup][winner.team] += 1

    team_results = []
    records = {team.team: team for team in field}
    for team_name, team_counts in counts.items():
        team = records[team_name]
        team_results.append(
            TournamentSimulationResult(
                team=team.team,
                rank=team.rank,
                elo=team.elo,
                champion_probability=team_counts["champion"] / iterations,
                final_probability=team_counts["final"] / iterations,
                semifinal_probability=team_counts["sf"] / iterations,
                quarterfinal_probability=team_counts["qf"] / iterations,
                round_of_16_probability=team_counts["r16"] / iterations,
            )
        )

    bracket_results = summarize_bracket_matches(
        entrant_counts,
        win_counts,
        matchup_counts,
        matchup_win_counts,
        iterations,
    )
    return (
        sorted(team_results, key=lambda item: item.champion_probability, reverse=True),
        bracket_results,
    )


def simulate_tournament(
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2026,
) -> list[TournamentSimulationResult]:
    results, _ = simulate_tournament_with_bracket(teams, iterations=iterations, seed=seed)
    return results
