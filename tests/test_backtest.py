from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.simulation.backtest import (
    HISTORICAL_TOURNAMENTS,
    WORLD_CUP_2022,
    HistoricalTournament,
    XGModelConfig,
    build_expected_goals_model,
    compare_backtest_methods,
    load_historical_team_ratings,
    run_backtest,
)


def test_world_cup_2022_backtest_ranks_actual_champion() -> None:
    ratings = {
        "Qatar": 1680,
        "Ecuador": 1833,
        "Senegal": 1687,
        "Netherlands": 2041,
        "England": 1920,
        "Iran": 1797,
        "United States": 1798,
        "Wales": 1790,
        "Argentina": 2143,
        "Saudi Arabia": 1635,
        "Mexico": 1809,
        "Poland": 1814,
        "France": 2005,
        "Australia": 1719,
        "Denmark": 1971,
        "Tunisia": 1707,
        "Spain": 2048,
        "Costa Rica": 1744,
        "Germany": 1963,
        "Japan": 1787,
        "Belgium": 2007,
        "Canada": 1776,
        "Morocco": 1766,
        "Croatia": 1927,
        "Brazil": 2169,
        "Serbia": 1898,
        "Switzerland": 1902,
        "Cameroon": 1610,
        "Portugal": 2005,
        "Ghana": 1567,
        "Uruguay": 1936,
        "South Korea": 1786,
    }
    teams = [
        TeamRecord(
            team=team,
            confederation="",
            rating=elo,
            attack_rating=elo,
            defense_rating=elo,
            elo=elo,
        )
        for team, elo in ratings.items()
    ]

    result = run_backtest(WORLD_CUP_2022, teams, iterations=500, seed=2022)

    assert result.summary.actual_champion == "Argentina"
    assert result.summary.actual_champion_probability > 0
    assert result.summary.actual_champion_rank <= 4
    assert result.summary.top_pick in ratings
    assert result.summary.champion_log_loss > 0
    assert 0 <= result.summary.champion_brier_score <= 1
    assert 0 <= result.summary.round_of_16_brier_score <= 1
    assert 0 <= result.summary.stage_brier_score <= 1
    assert result.summary.stage_score_mae >= 0
    assert result.summary.top_pick_accuracy in {0.0, 1.0}
    assert 0 <= result.summary.calibration_error <= 1
    assert len(result.summary.calibration_bins) == 5
    assert sum(bin.count for bin in result.summary.calibration_bins) == len(result.teams) * 5
    assert len(result.teams) == 32


def test_world_cup_2022_backtest_compares_xg_history_method() -> None:
    ratings = {
        "Qatar": 1680,
        "Ecuador": 1833,
        "Senegal": 1687,
        "Netherlands": 2041,
        "England": 1920,
        "Iran": 1797,
        "United States": 1798,
        "Wales": 1790,
        "Argentina": 2143,
        "Saudi Arabia": 1635,
        "Mexico": 1809,
        "Poland": 1814,
        "France": 2005,
        "Australia": 1719,
        "Denmark": 1971,
        "Tunisia": 1707,
        "Spain": 2048,
        "Costa Rica": 1744,
        "Germany": 1963,
        "Japan": 1787,
        "Belgium": 2007,
        "Canada": 1776,
        "Morocco": 1766,
        "Croatia": 1927,
        "Brazil": 2169,
        "Serbia": 1898,
        "Switzerland": 1902,
        "Cameroon": 1610,
        "Portugal": 2005,
        "Ghana": 1567,
        "Uruguay": 1936,
        "South Korea": 1786,
    }
    teams = [
        TeamRecord(
            team=team,
            confederation="",
            rating=elo,
            attack_rating=elo,
            defense_rating=elo,
            elo=elo,
        )
        for team, elo in ratings.items()
    ]

    elo_result, xg_result, comparisons = compare_backtest_methods(
        WORLD_CUP_2022,
        teams,
        iterations=100,
        seed=2022,
    )

    assert len(elo_result.teams) == 32
    assert len(xg_result.teams) == 32
    assert [row.model for row in comparisons] == ["elo", "xg_elo_adjusted"]
    assert all(row.actual_champion_probability >= 0 for row in comparisons)
    assert xg_result.summary.actual_champion == "Argentina"


def test_historical_world_cups_have_complete_rating_fields() -> None:
    for key, tournament in HISTORICAL_TOURNAMENTS.items():
        field = {
            team
            for _, group in tournament.groups
            for team in group
        }
        teams = (
            [
                TeamRecord(team, "", 1500, 1500, 1500, 1500)
                for team in field
            ]
            if key == "world-cup-2022"
            else load_historical_team_ratings(tournament)
        )

        assert len(field) == 32
        assert len(teams) == 32
        assert {team.team for team in teams} == field


def test_xg_history_model_weights_recent_games_more_heavily(tmp_path) -> None:
    raw_dir = tmp_path
    (raw_dir / "elo_team_names.tsv").write_text("AA	Alpha\nBB	Beta\n", encoding="utf-8")
    (raw_dir / "elo_results_2021.tsv").write_text(
        "\n".join(
            [
                "2021\t01\t01\tAA\tBB\t0\t4\tF\t\t0\t1500\t1500\t0\t0\t0\t0",
                "2021\t12\t15\tAA\tBB\t4\t0\tF\t\t0\t1500\t1500\t0\t0\t0\t0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tournament = HistoricalTournament(
        name="Fixture",
        as_of="2022-01-01",
        source="test",
        groups=(("A", ("Alpha", "Beta")),),
        actual_finish={},
    )
    teams = [
        TeamRecord("Alpha", "", 1500, 1500, 1500, 1500),
        TeamRecord("Beta", "", 1500, 1500, 1500, 1500),
    ]

    recency_weighted = build_expected_goals_model(
        teams,
        tournament,
        history_years=1,
        recency_half_life_days=90,
        raw_data_dir=raw_dir,
    )
    unweighted_like = build_expected_goals_model(
        teams,
        tournament,
        history_years=1,
        recency_half_life_days=100_000,
        raw_data_dir=raw_dir,
    )

    assert recency_weighted.profiles["Alpha"].adjusted_goals_for > 3
    assert unweighted_like.profiles["Alpha"].adjusted_goals_for < 2.1


def test_xg_history_model_downweights_friendlies(tmp_path) -> None:
    raw_dir = tmp_path
    (raw_dir / "elo_team_names.tsv").write_text("AA\tAlpha\nBB\tBeta\n", encoding="utf-8")
    (raw_dir / "elo_results_2021.tsv").write_text(
        "\n".join(
            [
                "2021\t06\t01\tAA\tBB\t6\t0\tF\t\t0\t1500\t1500\t0\t0\t0\t0",
                "2021\t06\t02\tAA\tBB\t0\t2\tWQ\t\t0\t1500\t1500\t0\t0\t0\t0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tournament = HistoricalTournament(
        name="Fixture",
        as_of="2022-01-01",
        source="test",
        groups=(("A", ("Alpha", "Beta")),),
        actual_finish={},
    )
    teams = [
        TeamRecord("Alpha", "", 1500, 1500, 1500, 1500),
        TeamRecord("Beta", "", 1500, 1500, 1500, 1500),
    ]

    neutral_match_types = build_expected_goals_model(
        teams,
        tournament,
        config=XGModelConfig(
            history_years=1,
            recency_half_life_days=100_000,
            elo_goal_adjustment_scale=650,
            match_type_weights={"F": 1, "WQ": 1},
        ),
        raw_data_dir=raw_dir,
    )
    weighted_match_types = build_expected_goals_model(
        teams,
        tournament,
        config=XGModelConfig(
            history_years=1,
            recency_half_life_days=100_000,
            elo_goal_adjustment_scale=650,
        ),
        raw_data_dir=raw_dir,
    )

    assert neutral_match_types.profiles["Alpha"].adjusted_goals_for > 2.9
    assert weighted_match_types.profiles["Alpha"].adjusted_goals_for < 2
