import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json

from worldcup_prediction.config import DERIVED_TEAMS_JSON_PATH, REPORTS_DIR
from worldcup_prediction.data_loader import load_derived_teams
from worldcup_prediction.models.elo import elo_win_draw_loss
from worldcup_prediction.simulation.monte_carlo import (
    KNOCKOUT_ROUNDS,
    OFFICIAL_2026_GROUPS,
    ROUND_OF_32_MATCHES,
    build_2026_expected_goals_model,
    host_advantage_for_match,
    match_win_draw_loss,
    simulate_tournament_with_bracket,
)

ROUND_NAMES = ("Round of 16", "Quarter-finals", "Semi-finals", "Final")
GROUP_STAGE_FIXTURES: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "A",
        (
            ("June 11, 2026", "Mexico", "South Africa"),
            ("June 11, 2026", "South Korea", "Czechia"),
            ("June 18, 2026", "Czechia", "South Africa"),
            ("June 18, 2026", "Mexico", "South Korea"),
            ("June 24, 2026", "Czechia", "Mexico"),
            ("June 24, 2026", "South Africa", "South Korea"),
        ),
    ),
    (
        "B",
        (
            ("June 12, 2026", "Canada", "Bosnia and Herzegovina"),
            ("June 13, 2026", "Qatar", "Switzerland"),
            ("June 18, 2026", "Switzerland", "Bosnia and Herzegovina"),
            ("June 18, 2026", "Canada", "Qatar"),
            ("June 24, 2026", "Switzerland", "Canada"),
            ("June 24, 2026", "Bosnia and Herzegovina", "Qatar"),
        ),
    ),
    (
        "C",
        (
            ("June 13, 2026", "Brazil", "Morocco"),
            ("June 13, 2026", "Haiti", "Scotland"),
            ("June 19, 2026", "Scotland", "Morocco"),
            ("June 19, 2026", "Brazil", "Haiti"),
            ("June 24, 2026", "Scotland", "Brazil"),
            ("June 24, 2026", "Morocco", "Haiti"),
        ),
    ),
    (
        "D",
        (
            ("June 12, 2026", "United States", "Paraguay"),
            ("June 13, 2026", "Australia", "Turkey"),
            ("June 19, 2026", "Turkey", "Paraguay"),
            ("June 19, 2026", "United States", "Australia"),
            ("June 25, 2026", "Turkey", "United States"),
            ("June 25, 2026", "Paraguay", "Australia"),
        ),
    ),
    (
        "E",
        (
            ("June 14, 2026", "Ivory Coast", "Ecuador"),
            ("June 14, 2026", "Germany", "Curaçao"),
            ("June 20, 2026", "Germany", "Ivory Coast"),
            ("June 20, 2026", "Ecuador", "Curaçao"),
            ("June 25, 2026", "Curaçao", "Ivory Coast"),
            ("June 25, 2026", "Ecuador", "Germany"),
        ),
    ),
    (
        "F",
        (
            ("June 14, 2026", "Netherlands", "Japan"),
            ("June 14, 2026", "Sweden", "Tunisia"),
            ("June 20, 2026", "Netherlands", "Sweden"),
            ("June 20, 2026", "Tunisia", "Japan"),
            ("June 25, 2026", "Japan", "Sweden"),
            ("June 25, 2026", "Tunisia", "Netherlands"),
        ),
    ),
    (
        "G",
        (
            ("June 15, 2026", "Iran", "New Zealand"),
            ("June 15, 2026", "Belgium", "Egypt"),
            ("June 21, 2026", "Belgium", "Iran"),
            ("June 21, 2026", "New Zealand", "Egypt"),
            ("June 26, 2026", "Egypt", "Iran"),
            ("June 26, 2026", "New Zealand", "Belgium"),
        ),
    ),
    (
        "H",
        (
            ("June 15, 2026", "Saudi Arabia", "Uruguay"),
            ("June 15, 2026", "Spain", "Cape Verde"),
            ("June 21, 2026", "Uruguay", "Cape Verde"),
            ("June 21, 2026", "Spain", "Saudi Arabia"),
            ("June 26, 2026", "Cape Verde", "Saudi Arabia"),
            ("June 26, 2026", "Uruguay", "Spain"),
        ),
    ),
    (
        "I",
        (
            ("June 16, 2026", "France", "Senegal"),
            ("June 16, 2026", "Iraq", "Norway"),
            ("June 22, 2026", "Norway", "Senegal"),
            ("June 22, 2026", "France", "Iraq"),
            ("June 26, 2026", "Norway", "France"),
            ("June 26, 2026", "Senegal", "Iraq"),
        ),
    ),
    (
        "J",
        (
            ("June 16, 2026", "Argentina", "Algeria"),
            ("June 16, 2026", "Austria", "Jordan"),
            ("June 22, 2026", "Argentina", "Austria"),
            ("June 22, 2026", "Jordan", "Algeria"),
            ("June 27, 2026", "Algeria", "Austria"),
            ("June 27, 2026", "Jordan", "Argentina"),
        ),
    ),
    (
        "K",
        (
            ("June 17, 2026", "Portugal", "DR Congo"),
            ("June 17, 2026", "Uzbekistan", "Colombia"),
            ("June 23, 2026", "Portugal", "Uzbekistan"),
            ("June 23, 2026", "Colombia", "DR Congo"),
            ("June 27, 2026", "Colombia", "Portugal"),
            ("June 27, 2026", "DR Congo", "Uzbekistan"),
        ),
    ),
    (
        "L",
        (
            ("June 17, 2026", "Ghana", "Panama"),
            ("June 17, 2026", "England", "Croatia"),
            ("June 23, 2026", "England", "Ghana"),
            ("June 23, 2026", "Panama", "Croatia"),
            ("June 27, 2026", "Panama", "England"),
            ("June 27, 2026", "Croatia", "Ghana"),
        ),
    ),
)


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_ratings_as_of() -> str:
    if DERIVED_TEAMS_JSON_PATH.exists():
        payload = json.loads(DERIVED_TEAMS_JSON_PATH.read_text(encoding="utf-8"))
        return payload.get("asOf") or datetime.now(timezone.utc).date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


def selector_label(selector: tuple[str, str], compact: bool = False) -> str:
    position, group_selector = selector
    if position == "1":
        return f"1{group_selector}" if compact else f"Winner Group {group_selector}"
    if position == "2":
        return f"2{group_selector}" if compact else f"Runner-up Group {group_selector}"
    if position == "3":
        groups = "/".join(group_selector)
        return f"3{groups}" if compact else f"Third place Group {groups}"
    raise ValueError(f"Unsupported bracket selector: {selector}")


def fixture_probability_payload(home, away, xg_model=None) -> dict:
    home_win, draw, away_win = match_win_draw_loss(
        home,
        away,
        xg_model=xg_model,
        allow_draw=True,
    )
    baseline_home, baseline_draw, baseline_away = elo_win_draw_loss(
        home.rating,
        away.rating,
        home_advantage=host_advantage_for_match(home, away),
        allow_draw=True,
    )
    return {
        "homeWin": home_win,
        "draw": draw,
        "awayWin": away_win,
        "eloBaseline": {
            "homeWin": baseline_home,
            "draw": baseline_draw,
            "awayWin": baseline_away,
        },
    }


def tournament_structure(records_by_name: dict | None = None, xg_model=None) -> dict:
    fixtures_by_group = dict(GROUP_STAGE_FIXTURES)
    knockout_rounds = []
    for round_name, round_matches in zip(ROUND_NAMES, KNOCKOUT_ROUNDS):
        knockout_rounds.append(
            {
                "round": round_name,
                "matches": [
                    {
                        "match": match_number,
                        "left": f"W{left_match}",
                        "right": f"W{right_match}",
                    }
                    for match_number, left_match, right_match in round_matches
                ],
            }
        )

    return {
        "groups": [
            {
                "group": group_name,
                "teams": list(teams),
                "fixtures": [
                    {
                        "date": date,
                        "home": home,
                        "away": away,
                        **(
                            {
                                "probabilities": fixture_probability_payload(
                                    records_by_name[home],
                                    records_by_name[away],
                                    xg_model=xg_model,
                                )
                            }
                            if records_by_name is not None
                            else {}
                        ),
                    }
                    for date, home, away in fixtures_by_group[group_name]
                ],
            }
            for group_name, teams in OFFICIAL_2026_GROUPS
        ],
        "roundOf32": [
            {
                "match": match_number,
                "left": selector_label(left_selector),
                "right": selector_label(right_selector),
            }
            for match_number, left_selector, right_selector in ROUND_OF_32_MATCHES
        ],
        "knockoutRounds": knockout_rounds,
    }


def bracket_sources() -> dict[int, tuple[str, str]]:
    sources = {
        match_number: (selector_label(left_selector), selector_label(right_selector))
        for match_number, left_selector, right_selector in ROUND_OF_32_MATCHES
    }
    for round_matches in KNOCKOUT_ROUNDS:
        for match_number, left_match, right_match in round_matches:
            sources[match_number] = (f"W{left_match}", f"W{right_match}")
    return sources


def bracket_result_payload(bracket_results) -> list[dict]:
    return [
        {
            "match": result.match,
            "round": result.round,
            "entrantProbabilities": result.entrant_probabilities,
            "matchupProbabilities": result.matchup_probabilities,
        }
        for result in bracket_results
    ]


def write_csv(results, path) -> None:
    fieldnames = [
        "team",
        "rank",
        "elo",
        "champion_probability",
        "final_probability",
        "semifinal_probability",
        "quarterfinal_probability",
        "round_of_16_probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_json(
    results,
    bracket_results,
    teams,
    path,
    iterations: int,
    team_count: int,
    seed: int,
    ratings_as_of: str,
    xg_model,
) -> None:
    records_by_name = {team.team: team for team in teams}
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "teamCount": team_count,
        "seed": seed,
        "ratingsAsOf": ratings_as_of,
        "method": (
            "Tuned xG/Elo-adjusted Monte Carlo using recency-weighted scoring history, "
            "fitted Elo draw curve, official 2026 groups and knockout bracket"
        ),
        "xgConfig": xg_model.config.__dict__,
        "tournamentStructure": tournament_structure(records_by_name, xg_model),
        "bracketProbabilities": bracket_result_payload(bracket_results),
        "results": [result.__dict__ for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def top_entrants_html(match_result, limit: int = 4) -> str:
    entrants = match_result.entrant_probabilities[:limit]
    return "<br />".join(
        f"{escape(entrant['team'])} {percent(entrant['appearance_probability'])}"
        for entrant in entrants
    )


def top_matchups_html(match_result, limit: int = 3) -> str:
    if not match_result.matchup_probabilities:
        return ""
    lines = []
    for matchup in match_result.matchup_probabilities[:limit]:
        team_a, team_b = matchup["teams"]
        team_a_win = matchup["team_win_probabilities"][team_a]
        team_b_win = matchup["team_win_probabilities"][team_b]
        lines.append(
            f"{escape(team_a)} v {escape(team_b)} "
            f"{percent(matchup['probability'])} "
            f"<span class=\"subtle\">({escape(team_a)} {percent(team_a_win)} / "
            f"{escape(team_b)} {percent(team_b_win)})</span>"
        )
    return "<br />".join(lines)


def likely_winner_html(match_result) -> str:
    if not match_result.entrant_probabilities:
        return ""
    winner = max(
        match_result.entrant_probabilities,
        key=lambda entrant: entrant["win_probability"],
    )
    return (
        f"{escape(winner['team'])}<br />"
        f"<span class=\"subtle\">wins match {percent(winner['win_probability'])}; "
        f"if appears {percent(winner['conditional_win_probability'])}</span>"
    )


def probability_bracket_html(bracket_results) -> str:
    sources = bracket_sources()
    sections = []
    for round_name in ("Round of 32", *ROUND_NAMES):
        rows = []
        for result in [item for item in bracket_results if item.round == round_name]:
            left_source, right_source = sources[result.match]
            rows.append(
                f"""
                <tr>
                  <td>M{result.match}</td>
                  <td>{escape(left_source)}<br /><span class="subtle">{escape(right_source)}</span></td>
                  <td>{top_entrants_html(result)}</td>
                  <td>{top_matchups_html(result)}</td>
                  <td>{likely_winner_html(result)}</td>
                </tr>
                """
            )
        sections.append(
            f"""
            <section class="bracket-round">
              <h3>{escape(round_name)}</h3>
              <table>
                <thead>
                  <tr>
                    <th>Match</th>
                    <th>Sources</th>
                    <th>Likely Entrants</th>
                    <th>Top Matchup Odds</th>
                    <th>Likely Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {"".join(rows)}
                </tbody>
              </table>
            </section>
            """
        )
    return "\n".join(sections)


def group_match_rows_html(group_name: str, records_by_name: dict, xg_model) -> str:
    rows = []
    fixtures = dict(GROUP_STAGE_FIXTURES)[group_name]
    for date, home_name, away_name in fixtures:
        home = records_by_name[home_name]
        away = records_by_name[away_name]
        probabilities = fixture_probability_payload(home, away, xg_model=xg_model)
        home_win = probabilities["homeWin"]
        draw = probabilities["draw"]
        away_win = probabilities["awayWin"]
        baseline = probabilities["eloBaseline"]
        baseline_label = (
            f"Elo {percent(baseline['homeWin'])} / "
            f"{percent(baseline['draw'])} / {percent(baseline['awayWin'])}"
        )
        rows.append(
            f"""
            <tr>
              <td>{escape(date)}</td>
              <td>
                <span class="fixture-team">{escape(home_name)}</span>
                <span class="fixture-v">v</span>
                <span class="fixture-team">{escape(away_name)}</span>
              </td>
              <td>
                <span class="odds-chip">{escape(home_name)} {percent(home_win)}</span>
                <span class="odds-chip draw-pick">Draw {percent(draw)}</span>
                <span class="odds-chip">{escape(away_name)} {percent(away_win)}</span>
                <span class="subtle">{escape(baseline_label)}</span>
              </td>
            </tr>
            """
        )
    return "".join(rows)


def write_html_report(
    results,
    bracket_results,
    teams,
    path,
    iterations: int,
    team_count: int,
    seed: int,
    ratings_as_of: str,
    xg_model,
) -> None:
    records_by_name = {team.team: team for team in teams}
    group_cards = "\n".join(
        f"""
        <section class="group-card">
          <h3>Group {escape(group_name)}</h3>
          <ol>
            {"".join(f"<li>{escape(team)}</li>" for team in teams)}
          </ol>
          <h4>Group Game Probabilities</h4>
          <table class="group-matches">
            <thead>
              <tr>
                <th>Date</th>
                <th>Fixture</th>
                <th>Simulated Result Odds</th>
              </tr>
            </thead>
            <tbody>
              {group_match_rows_html(group_name, records_by_name, xg_model)}
            </tbody>
          </table>
        </section>
        """
        for group_name, teams in OFFICIAL_2026_GROUPS
    )
    probability_bracket = probability_bracket_html(bracket_results)
    rows = "\n".join(
        f"""
        <tr>
          <td>{index}</td>
          <td>{escape(result.team)}</td>
          <td>{result.rank}</td>
          <td>{result.elo:.0f}</td>
          <td>{percent(result.champion_probability)}</td>
          <td>{percent(result.final_probability)}</td>
          <td>{percent(result.semifinal_probability)}</td>
          <td>{percent(result.quarterfinal_probability)}</td>
          <td>{percent(result.round_of_16_probability)}</td>
        </tr>
        """
        for index, result in enumerate(results[:32], start=1)
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>World Cup 2026 xG/Elo Monte Carlo Report</title>
    <style>
      body {{
        margin: 40px;
        color: #17202a;
        font-family: Inter, Arial, sans-serif;
      }}
      h1 {{
        margin-bottom: 4px;
        font-size: 30px;
      }}
      .meta {{
        margin: 0 0 22px;
        color: #667085;
      }}
      h2 {{
        margin: 30px 0 12px;
        font-size: 20px;
      }}
      h3 {{
        margin: 0 0 8px;
        font-size: 14px;
      }}
      h4 {{
        margin: 12px 0 6px;
        color: #344054;
        font-size: 12px;
      }}
      .tabs {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 26px 0 18px;
        border-bottom: 1px solid #d9dee7;
      }}
      .tab-button {{
        appearance: none;
        border: 1px solid #d9dee7;
        border-bottom: 0;
        border-radius: 8px 8px 0 0;
        background: #f8fafc;
        color: #344054;
        cursor: pointer;
        font: inherit;
        font-size: 13px;
        font-weight: 700;
        padding: 10px 14px;
      }}
      .tab-button[aria-selected="true"] {{
        background: #fff;
        color: #17202a;
        box-shadow: 0 1px 0 #fff;
      }}
      .tab-panel {{
        display: none;
      }}
      .tab-panel.active {{
        display: block;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      th, td {{
        padding: 8px 7px;
        border-bottom: 1px solid #d9dee7;
        text-align: right;
      }}
      th:nth-child(2), td:nth-child(2) {{
        text-align: left;
      }}
      th {{
        background: #f3f5f8;
      }}
      .groups {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 10px;
      }}
      .group-card {{
        border: 1px solid #d9dee7;
        border-radius: 8px;
        padding: 12px;
      }}
      .group-card ol {{
        margin: 0;
        padding-left: 20px;
        font-size: 13px;
        line-height: 1.6;
      }}
      .group-matches {{
        margin-top: 12px;
        font-size: 12px;
      }}
      .group-matches th,
      .group-matches td {{
        padding: 6px 5px;
      }}
      .group-matches th,
      .group-matches td {{
        text-align: left;
      }}
      .group-matches th:nth-child(1),
      .group-matches td:nth-child(1) {{
        white-space: nowrap;
        width: 108px;
      }}
      .fixture-team {{
        font-weight: 700;
      }}
      .fixture-v {{
        color: #667085;
        display: inline-block;
        margin: 0 6px;
      }}
      .odds-chip {{
        background: #f3f5f8;
        border: 1px solid #d9dee7;
        border-radius: 999px;
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        margin: 2px 4px 2px 0;
        padding: 3px 7px;
        white-space: nowrap;
      }}
      .draw-pick {{
        background: #fff7e6;
        border-color: #f4d29a;
        color: #8a5a00;
      }}
      .bracket-grid {{
        display: grid;
        gap: 22px;
      }}
      .bracket-round th,
      .bracket-round td,
      .probability-bracket th,
      .probability-bracket td {{
        text-align: left;
        vertical-align: top;
        line-height: 1.45;
      }}
      .bracket-round th:nth-child(1),
      .bracket-round td:nth-child(1) {{
        width: 54px;
      }}
      .subtle {{
        color: #667085;
        font-size: 12px;
      }}
      .note {{
        margin-top: 22px;
        color: #667085;
        font-size: 12px;
        line-height: 1.45;
      }}
      @media print {{
        body {{ margin: 20mm; }}
        .tabs {{ display: none; }}
        .tab-panel {{ display: block; page-break-inside: avoid; }}
        .groups {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
    </style>
  </head>
  <body>
    <h1>World Cup 2026 xG/Elo Monte Carlo Report</h1>
    <p class="meta">Generated {generated} · ratings as of {escape(ratings_as_of)} · {iterations:,} simulations · official 2026 field ({team_count} teams) · seed {seed}</p>
    <nav class="tabs" role="tablist" aria-label="Report sections">
      <button class="tab-button" id="tab-probabilities" type="button" role="tab" aria-selected="true" aria-controls="panel-probabilities" data-tab="probabilities">Team Probabilities</button>
      <button class="tab-button" id="tab-bracket" type="button" role="tab" aria-selected="false" aria-controls="panel-bracket" data-tab="bracket">Probabilistic Bracket</button>
      <button class="tab-button" id="tab-groups" type="button" role="tab" aria-selected="false" aria-controls="panel-groups" data-tab="groups">Groups</button>
    </nav>
    <section class="tab-panel active" id="panel-probabilities" role="tabpanel" aria-labelledby="tab-probabilities">
      <h2>Team Probabilities</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Team</th>
            <th>Elo Rank</th>
            <th>Elo</th>
            <th>Win Cup</th>
            <th>Final</th>
            <th>Semi</th>
            <th>Quarter</th>
            <th>Round of 16</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </section>
    <section class="tab-panel" id="panel-bracket" role="tabpanel" aria-labelledby="tab-bracket">
      <h2>Probabilistic Bracket</h2>
      <div class="bracket-grid probability-bracket">
        {probability_bracket}
      </div>
    </section>
    <section class="tab-panel" id="panel-groups" role="tabpanel" aria-labelledby="tab-groups">
      <h2>Groups</h2>
      <div class="groups">
        {group_cards}
      </div>
    </section>
    <p class="note">
      Method: tuned xG/Elo-adjusted Monte Carlo using recency-weighted scoring history, opponent Elo adjustment, and the fitted Elo draw curve. Official 2026 groups plus the match-numbered knockout bracket are simulated. Host advantage is not applied until fixture venue countries are available. The model does not yet encode exact venues, injuries, live squad news, or FIFA disciplinary tie-breakers in full detail.
    </p>
    <script>
      const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
      const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

      function activateTab(tabName) {{
        tabButtons.forEach((button) => {{
          const isActive = button.dataset.tab === tabName;
          button.setAttribute("aria-selected", String(isActive));
        }});
        tabPanels.forEach((panel) => {{
          panel.classList.toggle("active", panel.id === `panel-${{tabName}}`);
        }});
      }}

      tabButtons.forEach((button) => {{
        button.addEventListener("click", () => activateTab(button.dataset.tab));
      }});
    </script>
  </body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a starter World Cup Elo Monte Carlo simulation.")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    teams = load_derived_teams()
    ratings_as_of = load_ratings_as_of()
    xg_model = build_2026_expected_goals_model(teams, ratings_as_of)
    results, bracket_results = simulate_tournament_with_bracket(
        teams,
        iterations=args.iterations,
        seed=args.seed,
        xg_model=xg_model,
    )
    team_count = len(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(results, REPORTS_DIR / "monte_carlo_results.csv")
    write_json(
        results,
        bracket_results,
        teams,
        REPORTS_DIR / "monte_carlo_results.json",
        args.iterations,
        team_count,
        args.seed,
        ratings_as_of,
        xg_model,
    )
    write_html_report(
        results,
        bracket_results,
        teams,
        REPORTS_DIR / "monte_carlo_report.html",
        args.iterations,
        team_count,
        args.seed,
        ratings_as_of,
        xg_model,
    )

    print(f"Champion favorite: {results[0].team} ({percent(results[0].champion_probability)})")
    print(f"Wrote: {REPORTS_DIR / 'monte_carlo_results.csv'}")
    print(f"Wrote: {REPORTS_DIR / 'monte_carlo_results.json'}")
    print(f"Wrote: {REPORTS_DIR / 'monte_carlo_report.html'}")


if __name__ == "__main__":
    main()
