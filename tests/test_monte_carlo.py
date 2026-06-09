import pytest

from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.models.elo import elo_expected_score, elo_win_draw_loss
from worldcup_prediction.simulation.monte_carlo import (
    HOST_ELO_ADVANTAGE,
    host_advantage_for_match,
)


def team_record(team: str, rating: float = 1500) -> TeamRecord:
    return TeamRecord(
        team=team,
        confederation="",
        rating=rating,
        attack_rating=rating,
        defense_rating=rating,
        elo=rating,
    )


def test_host_advantage_is_neutral_without_venue_country() -> None:
    mexico = team_record("Mexico")
    south_africa = team_record("South Africa")

    assert host_advantage_for_match(mexico, south_africa) == 0


def test_host_advantage_applies_only_in_teams_host_country() -> None:
    mexico = team_record("Mexico")
    south_africa = team_record("South Africa")

    assert (
        host_advantage_for_match(mexico, south_africa, venue_country="Mexico")
        == HOST_ELO_ADVANTAGE
    )
    assert host_advantage_for_match(mexico, south_africa, venue_country="Canada") == 0
    assert (
        host_advantage_for_match(south_africa, mexico, venue_country="Mexico")
        == -HOST_ELO_ADVANTAGE
    )


def test_host_advantage_follows_the_venue_for_two_host_teams() -> None:
    assert host_advantage_for_match(
        team_record("Canada"),
        team_record("United States"),
        venue_country="Canada",
    ) == HOST_ELO_ADVANTAGE
    assert host_advantage_for_match(
        team_record("Canada"),
        team_record("United States"),
        venue_country="United States",
    ) == -HOST_ELO_ADVANTAGE
    assert host_advantage_for_match(
        team_record("Canada"),
        team_record("United States"),
        venue_country="Mexico",
    ) == 0


def test_host_advantage_flows_into_match_probability() -> None:
    mexico = team_record("Mexico")
    south_africa = team_record("South Africa")
    win, draw, _ = elo_win_draw_loss(
        mexico.rating,
        south_africa.rating,
        home_advantage=host_advantage_for_match(mexico, south_africa, venue_country="Mexico"),
    )
    expected = elo_expected_score(1500, 1500, home_advantage=HOST_ELO_ADVANTAGE)

    assert win + 0.5 * draw == pytest.approx(expected)
    assert win > 0.5
