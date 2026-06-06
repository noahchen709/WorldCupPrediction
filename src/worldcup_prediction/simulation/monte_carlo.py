import random
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


def make_seeded_groups(teams: list[TeamRecord], group_count: int = 12) -> list[list[TeamRecord]]:
    field = sorted(teams, key=lambda team: team.rating, reverse=True)
    groups = [[] for _ in range(group_count)]
    for index, team in enumerate(field):
        pot_index = index // group_count
        group_index = index % group_count
        if pot_index % 2:
            group_index = group_count - 1 - group_index
        groups[group_index].append(team)
    return groups


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


def qualify_round_of_32(groups: list[list[TeamRecord]], rng: random.Random) -> list[TeamRecord]:
    group_tables = [simulate_group(group, rng) for group in groups]
    automatic = [row.record for table in group_tables for row in table[:2]]
    third_place = [table[2] for table in group_tables if len(table) > 2]
    best_thirds = sorted(
        third_place,
        key=lambda row: (
            row.points,
            row.goal_difference,
            row.goals_for,
            row.wins,
            row.record.rating,
            rng.random(),
        ),
        reverse=True,
    )[:8]
    return sorted(automatic + [row.record for row in best_thirds], key=lambda team: team.rating, reverse=True)


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


def simulate_knockout_round(
    teams: list[TeamRecord],
    rng: random.Random,
) -> list[TeamRecord]:
    winners = []
    for index in range(len(teams) // 2):
        team_a = teams[index]
        team_b = teams[-index - 1]
        winners.append(sample_knockout_winner(team_a, team_b, rng))
    return sorted(winners, key=lambda team: team.rating, reverse=True)


def simulate_official_knockout(
    round_of_32_matches: dict[int, tuple[TeamRecord, TeamRecord]],
    rng: random.Random,
) -> tuple[
    dict[str, list[TeamRecord]],
    TeamRecord,
]:
    winners = {
        match_number: sample_knockout_winner(team_a, team_b, rng)
        for match_number, (team_a, team_b) in round_of_32_matches.items()
    }
    round_winners = {"r16": list(winners.values())}

    round_names = ("qf", "sf", "final", "champion")
    for round_name, round_matches in zip(round_names, KNOCKOUT_ROUNDS):
        for match_number, left_match, right_match in round_matches:
            winners[match_number] = sample_knockout_winner(
                winners[left_match],
                winners[right_match],
                rng,
            )
        round_winners[round_name] = [winners[match_number] for match_number, _, _ in round_matches]

    return round_winners, winners[104]


def simulate_tournament(
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2026,
    field_size: int | None = None,
) -> list[TournamentSimulationResult]:
    rng = random.Random(seed)
    groups = make_official_2026_groups(teams)
    field = [team for group in groups.values() for team in group]
    counts = {
        team.team: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for team in field
    }

    for _ in range(iterations):
        round_of_32_matches, round_of_32 = qualify_official_round_of_32(groups, rng)
        for team in round_of_32:
            counts[team.team]["r32"] += 1

        round_winners, champion = simulate_official_knockout(round_of_32_matches, rng)
        for round_name, winners in round_winners.items():
            for team in winners:
                counts[team.team][round_name] += 1

    results = []
    records = {team.team: team for team in field}
    for team_name, team_counts in counts.items():
        team = records[team_name]
        results.append(
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

    return sorted(results, key=lambda item: item.champion_probability, reverse=True)
