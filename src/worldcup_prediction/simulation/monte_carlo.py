import random
from dataclasses import dataclass
from itertools import combinations

from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.models.elo import elo_win_draw_loss


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


def simulate_tournament(
    teams: list[TeamRecord],
    iterations: int = 10_000,
    seed: int = 2026,
    field_size: int = 48,
) -> list[TournamentSimulationResult]:
    rng = random.Random(seed)
    field = sorted(teams, key=lambda team: team.rating, reverse=True)[:field_size]
    counts = {
        team.team: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for team in field
    }

    for _ in range(iterations):
        groups = make_seeded_groups(field)
        round_of_32 = qualify_round_of_32(groups, rng)
        for team in round_of_32:
            counts[team.team]["r32"] += 1

        round_of_16 = simulate_knockout_round(round_of_32, rng)
        for team in round_of_16:
            counts[team.team]["r16"] += 1

        quarterfinalists = simulate_knockout_round(round_of_16, rng)
        for team in quarterfinalists:
            counts[team.team]["qf"] += 1

        semifinalists = simulate_knockout_round(quarterfinalists, rng)
        for team in semifinalists:
            counts[team.team]["sf"] += 1

        finalists = simulate_knockout_round(semifinalists, rng)
        for team in finalists:
            counts[team.team]["final"] += 1

        champion = simulate_knockout_round(finalists, rng)[0]
        counts[champion.team]["champion"] += 1

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
