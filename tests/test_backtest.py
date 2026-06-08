from worldcup_prediction.data_loader import TeamRecord
from worldcup_prediction.simulation.backtest import WORLD_CUP_2022, run_backtest


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
    assert len(result.teams) == 32
