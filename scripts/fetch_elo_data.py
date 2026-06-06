import csv
import json
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen

from worldcup_prediction.config import (
    DERIVED_TEAMS_JSON_PATH,
    DERIVED_TEAMS_PATH,
    ELO_TEAM_NAMES_PATH,
    ELO_TEAM_NAMES_URL,
    ELO_WORLD_PATH,
    ELO_WORLD_URL,
    RAW_DATA_DIR,
)

CONFEDERATION_URLS = {
    "UEFA": "https://www.eloratings.net/UEFA.tsv",
    "CONMEBOL": "https://www.eloratings.net/CONMEBOL.tsv",
    "CONCACAF": "https://www.eloratings.net/CONCACAF.tsv",
    "CAF": "https://www.eloratings.net/CAF.tsv",
    "AFC": "https://www.eloratings.net/AFC.tsv",
    "OFC": "https://www.eloratings.net/OFC.tsv",
}

MAX_DASHBOARD_TEAMS = 48


def normalize_number(value: str) -> int:
    normalized = value.replace("−", "-").replace("+", "")
    if normalized in {"", "-"}:
        return 0
    return int(normalized)


def fetch_text(url: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "WorldCupPredictionStarter/0.1"})
    with urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        last_modified = response.headers.get("Last-Modified", "")
        return body, last_modified


def write_raw(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_team_names(content: str) -> dict[str, str]:
    names = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        names[fields[0]] = fields[1]
    return names


def parse_codes(content: str) -> set[str]:
    return {
        fields[2]
        for line in content.splitlines()
        if line.strip() and len(fields := line.split("\t")) >= 3
    }


def parse_world_elo(
    content: str,
    team_names: dict[str, str],
    confederations: dict[str, str],
) -> list[dict[str, str | int | float]]:
    teams = []
    for line in content.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        team_code = fields[2]
        rating = int(fields[3])
        goals_for = int(fields[29])
        goals_against = int(fields[30])
        matches = int(fields[22])
        goals_for_per_match = goals_for / matches if matches else 0
        goals_against_per_match = goals_against / matches if matches else 0

        teams.append(
            {
                "rank": int(fields[1]),
                "team_code": team_code,
                "team": team_names.get(team_code, team_code),
                "confederation": confederations.get(team_code, "Other"),
                "elo": rating,
                "rating": rating,
                "attack_rating": round(rating + goals_for_per_match * 35, 2),
                "defense_rating": round(rating - goals_against_per_match * 35, 2),
                "rank_max": int(fields[4]),
                "rating_max": int(fields[5]),
                "rank_avg": int(fields[6]),
                "rating_avg": int(fields[7]),
                "rank_min": int(fields[8]),
                "rating_min": int(fields[9]),
                "rank_one_year_change": normalize_number(fields[14]),
                "rating_one_year_change": normalize_number(fields[15]),
                "rank_chg": 0,
                "rating_chg": 0,
                "matches": matches,
                "wins": int(fields[26]),
                "draws": int(fields[28]),
                "losses": int(fields[27]),
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        )
    return sorted(teams, key=lambda item: int(item["rank"]))


def write_outputs(
    teams: list[dict[str, str | int | float]],
    ratings_last_modified: str,
) -> None:
    DERIVED_TEAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "team_code",
        "team",
        "confederation",
        "elo",
        "rating",
        "attack_rating",
        "defense_rating",
        "rank_max",
        "rating_max",
        "rank_avg",
        "rating_avg",
        "rank_min",
        "rating_min",
        "rank_one_year_change",
        "rating_one_year_change",
        "rank_chg",
        "rating_chg",
        "matches",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
    ]

    with DERIVED_TEAMS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(teams)

    as_of = ""
    if ratings_last_modified:
        as_of = parsedate_to_datetime(ratings_last_modified).date().isoformat()

    dashboard_teams = [
        {
            "name": team["team"],
            "code": team["team_code"],
            "confederation": team["confederation"],
            "rank": team["rank"],
            "elo": team["elo"],
            "rating": team["rating"],
            "attack": team["attack_rating"],
            "defense": team["defense_rating"],
            "matches": team["matches"],
            "ratingChange": team["rating_chg"],
            "oneYearRatingChange": team["rating_one_year_change"],
        }
        for team in teams[:MAX_DASHBOARD_TEAMS]
    ]

    DERIVED_TEAMS_JSON_PATH.write_text(
        json.dumps(
            {
                "source": ELO_WORLD_URL,
                "sourceName": "World Football Elo Ratings",
                "asOf": as_of,
                "teams": dashboard_teams,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    world_content, world_last_modified = fetch_text(ELO_WORLD_URL)
    names_content, _ = fetch_text(ELO_TEAM_NAMES_URL)
    write_raw(ELO_WORLD_PATH, world_content)
    write_raw(ELO_TEAM_NAMES_PATH, names_content)

    confederations = {}
    for confederation, url in CONFEDERATION_URLS.items():
        content, _ = fetch_text(url)
        write_raw(RAW_DATA_DIR / f"elo_{confederation.lower()}.tsv", content)
        for code in parse_codes(content):
            confederations[code] = confederation

    team_names = parse_team_names(names_content)
    teams = parse_world_elo(world_content, team_names, confederations)
    write_outputs(teams, world_last_modified)

    print(f"Downloaded: {ELO_WORLD_PATH}")
    print(f"Ratings as of: {world_last_modified or 'unknown'}")
    print(f"Derived Elo teams: {len(teams)}")
    print(f"Dashboard data: {DERIVED_TEAMS_JSON_PATH}")


if __name__ == "__main__":
    main()
