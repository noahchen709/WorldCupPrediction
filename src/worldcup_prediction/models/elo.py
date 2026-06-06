from math import pow


def elo_expected_score(rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
    """Return Elo expected result for team A on the 1/0.5/0 scale."""
    rating_gap = rating_a + home_advantage - rating_b
    return 1 / (1 + pow(10, -rating_gap / 400))


def estimate_draw_probability(rating_a: float, rating_b: float) -> float:
    """Starter draw model based on absolute Elo gap.

    Elo itself gives expected result, not a draw probability. This curve is a
    transparent assumption that can later be replaced by calibration on match
    history with pre-match Elo ratings.
    """
    gap = abs(rating_a - rating_b)
    return max(0.16, min(0.30, 0.30 - gap / 1200))


def elo_win_draw_loss(
    rating_a: float,
    rating_b: float,
    home_advantage: float = 0,
    allow_draw: bool = True,
) -> tuple[float, float, float]:
    """Convert Elo expected result into W/D/L probabilities for team A."""
    expected = elo_expected_score(rating_a, rating_b, home_advantage)

    if not allow_draw:
        return expected, 0, 1 - expected

    draw = estimate_draw_probability(rating_a + home_advantage, rating_b)
    win = expected - 0.5 * draw
    loss = 1 - win - draw

    if win < 0 or loss < 0:
        draw = min(draw, expected * 2, (1 - expected) * 2)
        win = expected - 0.5 * draw
        loss = 1 - win - draw

    return win, draw, loss
