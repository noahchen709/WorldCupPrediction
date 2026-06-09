import pytest

from worldcup_prediction.models.elo import elo_expected_score, elo_win_draw_loss


@pytest.mark.parametrize(
    ("rating_gap", "expected"),
    [
        (0, 0.5),
        (120, 0.666),
        (800, 0.99),
    ],
)
def test_elo_expected_score_matches_world_football_elo_landmarks(
    rating_gap: int,
    expected: float,
) -> None:
    assert elo_expected_score(1500 + rating_gap, 1500) == pytest.approx(expected, abs=0.005)


def test_elo_expected_score_applies_home_advantage_to_team_a() -> None:
    assert elo_expected_score(1500, 1500, home_advantage=100) == pytest.approx(
        elo_expected_score(1600, 1500)
    )


@pytest.mark.parametrize(
    ("rating_a", "rating_b", "home_advantage"),
    [
        (1500, 1500, 0),
        (1620, 1500, 0),
        (1500, 1620, 0),
        (2100, 1300, 0),
        (1300, 2100, 0),
        (1500, 1500, 100),
    ],
)
def test_win_draw_loss_preserves_elo_expected_score(
    rating_a: int,
    rating_b: int,
    home_advantage: int,
) -> None:
    win, draw, loss = elo_win_draw_loss(rating_a, rating_b, home_advantage=home_advantage)
    expected = elo_expected_score(rating_a, rating_b, home_advantage=home_advantage)

    assert win + draw + loss == pytest.approx(1)
    assert win + 0.5 * draw == pytest.approx(expected)
    assert all(0 <= probability <= 1 for probability in (win, draw, loss))


def test_no_draw_mode_uses_elo_expected_score_as_win_probability() -> None:
    expected = elo_expected_score(1620, 1500)

    assert elo_win_draw_loss(1620, 1500, allow_draw=False) == pytest.approx(
        (expected, 0, 1 - expected)
    )
