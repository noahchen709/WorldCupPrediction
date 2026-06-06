# 2026 FIFA World Cup Quant Prediction Starter

A clean starter project for building a quantitative 2026 FIFA World Cup prediction system.

This repo intentionally does not include a full prediction model yet. It provides:

- A project structure for modeling, simulation, data, and dashboard work
- A real-data ingestion script for World Football Elo ratings
- A working dashboard prototype showing estimated champion likelihoods
- Lightweight Python stubs for the future model pipeline

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
│   ├── derived_team_strengths.json
│   ├── sample_teams.csv         # Offline fallback inputs
│   └── sample_matches.csv       # Starter fixture/result-style data
├── notebooks/                   # Exploratory analysis notebooks
├── reports/                     # Generated simulation outputs
├── scripts/
│   ├── fetch_elo_data.py        # Download and export Elo starter ratings
│   ├── run_simulation.py        # Run Monte Carlo and write report files
│   └── smoke.py                 # Import and data smoke check
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

This starter downloads the current world ratings table from:

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

The dashboard reads `data/derived_team_strengths.json` when served locally. If that file is missing or the page is opened directly from disk, it falls back to embedded demo values.

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

## Current Status

This is scaffolding plus a dashboard prototype. The next useful milestone is to implement a small deterministic baseline model and connect it to the dashboard-generated probabilities.
