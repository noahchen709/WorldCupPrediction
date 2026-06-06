import argparse
import csv
import json
from datetime import datetime, timezone
from html import escape

from worldcup_prediction.config import REPORTS_DIR
from worldcup_prediction.data_loader import load_derived_teams
from worldcup_prediction.simulation.monte_carlo import simulate_tournament


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def write_json(results, path, iterations: int, field_size: int, seed: int) -> None:
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "fieldSize": field_size,
        "seed": seed,
        "method": "Elo expected result + draw curve + official 2026 groups and knockout bracket",
        "results": [result.__dict__ for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_html_report(results, path, iterations: int, field_size: int, seed: int) -> None:
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
    <title>World Cup 2026 Elo Monte Carlo Report</title>
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
      .note {{
        margin-top: 22px;
        color: #667085;
        font-size: 12px;
        line-height: 1.45;
      }}
      @media print {{
        body {{ margin: 20mm; }}
      }}
    </style>
  </head>
  <body>
    <h1>World Cup 2026 Elo Monte Carlo Report</h1>
    <p class="meta">Generated {generated} · {iterations:,} simulations · official 2026 field ({field_size} teams) · seed {seed}</p>
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
    <p class="note">
      Method: World Football Elo ratings are converted to match expected result, a draw probability is estimated from Elo gap, and the official 2026 groups plus match-numbered knockout bracket are simulated. The model does not yet encode venues, injuries, live squad news, or FIFA disciplinary tie-breakers in full detail.
    </p>
  </body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a starter World Cup Elo Monte Carlo simulation.")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--field-size", type=int, default=48, help="Deprecated; the official 2026 field is always used.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    teams = load_derived_teams()
    results = simulate_tournament(
        teams,
        iterations=args.iterations,
        seed=args.seed,
        field_size=args.field_size,
    )
    field_size = len(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(results, REPORTS_DIR / "monte_carlo_results.csv")
    write_json(results, REPORTS_DIR / "monte_carlo_results.json", args.iterations, field_size, args.seed)
    write_html_report(results, REPORTS_DIR / "monte_carlo_report.html", args.iterations, field_size, args.seed)

    print(f"Champion favorite: {results[0].team} ({percent(results[0].champion_probability)})")
    print(f"Wrote: {REPORTS_DIR / 'monte_carlo_results.csv'}")
    print(f"Wrote: {REPORTS_DIR / 'monte_carlo_results.json'}")
    print(f"Wrote: {REPORTS_DIR / 'monte_carlo_report.html'}")


if __name__ == "__main__":
    main()
