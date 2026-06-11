# 2026 FIFA World Cup Quant Prediction
## Project Layout

```text
.
├── app/                         # Static dashboard prototype
│   ├── index.html
│   ├── main.js
│   └── styles.css
├── data/
│   ├── raw/world_elo.tsv        # Downloaded World Football Elo table
│   ├── raw/elo_team_names.tsv   # Downloaded Elo team-code names
│   ├── derived_team_strengths.csv
│   └── derived_team_strengths.json
├── notebooks/                   # Exploratory analysis notebooks
├── reports/                     # Generated simulation outputs
├── scripts/
│   ├── fetch_elo_data.py        # Download and export Elo starter ratings
│   ├── run_simulation.py        # Run Monte Carlo and write report files
│   ├── make_infographic.py      # Render the PDF/PNG forecast infographic
│   └── smoke.py                 # Import and data smoke check
├── requirements.txt             # matplotlib, for the infographic generator
├── src/worldcup_prediction/
│   ├── config.py
│   ├── data_loader.py
│   ├── models/
│   │   ├── match_outcome.py     # Win/draw/loss prediction stub
│   │   ├── scoreline.py         # Scoreline prediction stub
│   │   └── team_strength.py     # Team strength model stub
│   └── simulation/
│       └── tournament.py        # Tournament simulation stub
└── tests/                       # Future tests
```

## Data Source

The real data source is World Football Elo Ratings:

```text
https://www.eloratings.net/
```

This project downloads the current world ratings table from:

```text
https://www.eloratings.net/World.tsv
```

It also downloads Elo team names and confederation pages from `eloratings.net` so the dashboard can show readable team names and regional filters. The exported starter ratings preserve the official Elo value directly. The `attack_rating` and `defense_rating` fields are lightweight dashboard placeholders derived from historical goals for/against per match; they are not official Elo fields.

## Refresh Elo Data

```bash
PYTHONPATH=src python3 scripts/fetch_elo_data.py
```

This writes:

- `data/raw/world_elo.tsv`
- `data/raw/elo_team_names.tsv`
- `data/raw/elo_*.tsv`
- `data/derived_team_strengths.csv`
- `data/derived_team_strengths.json`

## Run The Dashboard

From the project root:

```bash
python3 -m http.server 5173
```

Then open:

```text
http://localhost:5173/app/
```

The dashboard reads `data/derived_team_strengths.json` and `reports/monte_carlo_results.json` when served locally.

## Run The Python Smoke Check

```bash
PYTHONPATH=src python3 scripts/smoke.py
```

## Run Monte Carlo Simulation

```bash
PYTHONPATH=src python3 scripts/run_simulation.py --iterations 10000
```

This writes:

- `reports/monte_carlo_results.csv`
- `reports/monte_carlo_results.json`
- `reports/monte_carlo_report.html`

The HTML report is designed as a print-friendly snapshot that can be exported to PDF from a browser.
It includes the official groups, a probabilistic match-numbered knockout bracket, and simulation probability table. The JSON output also includes `tournamentStructure` and `bracketProbabilities` blocks for programmatic bracket rendering.

The Monte Carlo uses the official 48-team 2026 field, 12 groups of four, the top two teams from each group plus the eight best third-place teams, and the official match-numbered knockout path from the Round of 32 to the final. Match strength still comes from World Football Elo ratings.

## Run Historical Backtest

```bash
PYTHONPATH=src python3 scripts/run_backtest.py --tournament world-cup-2022 --iterations 10000
```

This replays the 2022 FIFA World Cup format using pre-tournament Elo ratings from November 19, 2022, then compares the simulated probabilities with the actual finish. It writes:

- `reports/world-cup-2022_backtest.csv`
- `reports/world-cup-2022_backtest.json`

The summary reports the model's top pick, the actual champion's predicted probability and rank, and how much probability mass the model assigned to the actual finalists, semifinalists, and quarterfinalists. It also includes evaluation metrics: champion log loss, Brier scores for champion and stage-progression events, Round of 16 qualification Brier score, average stage error, top-pick accuracy, and calibration buckets.

## Generate The PDF Infographic

Render a polished, print-ready PDF infographic (plus optional shareable PNGs) from the
simulation and backtest outputs:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/make_infographic.py --png
```

This writes:

- `reports/world_cup_2026_infographic.pdf`
- `reports/world_cup_2026_infographic_page1.png` … `_page3.png` (with `--png`)

The infographic is a three-page editorial poster:

1. **The forecast** — title odds for the top contenders, champion probability by
   confederation, and the favourite's stage-by-stage path.
2. **The bracket** — a stage-progression matrix (round of 16 through the title) and the two
   most likely sides to advance from each group.
3. **Model validation** — how the same model scored on past World Cups using only
   pre-tournament ratings, with headline accuracy and calibration stats.

It reads `reports/monte_carlo_results.json`, `data/derived_team_strengths.json`, and
`reports/world-cup-backtests.json`, so run the simulation (and, optionally, the backtests)
first.

## Calibrate Draw Rate

```bash
PYTHONPATH=src python3 scripts/calibrate_draw_rate.py --start-year 1994 --end-year 2025 --test-start-year 2022
```

This downloads yearly World Football Elo result tables into `data/raw/elo_results_*.tsv`, fits `P(draw | Elo gap)` on matches before the test start year, and compares the old linear draw heuristic with the fitted exponential curve on held-out match results. It also runs the 2022 World Cup backtest with both draw models.

The current fitted curve is:

```text
P(draw) = 0.0200 + (0.3850 - 0.0200) * exp(-abs(Elo gap) / 344.0)
```

## Modeling Roadmap

1. **Team strength model**
   - Blend Elo, recent form, squad value, confederation strength, travel/rest effects, and injury availability.
   - Output attack, defense, and overall strength ratings.

2. **Match win/draw/loss prediction**
   - Convert team strengths into calibrated probabilities.
   - Benchmark multinomial logistic regression, Dixon-Coles style models, and gradient boosting.

3. **Scoreline prediction**
   - Estimate expected goals for each team.
   - Use Poisson, bivariate Poisson, or Dixon-Coles corrections for low-score dependence.

4. **Tournament simulation**
   - Simulate groups, knockouts, extra time, and penalties.
   - Run thousands of draws to estimate progression and champion probabilities.

5. **Dashboard**
   - Display champion probabilities, expected goals, match cards, uncertainty, and scenario controls.
