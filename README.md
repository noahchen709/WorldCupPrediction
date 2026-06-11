# 2026 FIFA World Cup Quant Prediction Starter

A clean starter project for building a quantitative 2026 FIFA World Cup prediction system.

This repo intentionally does not include a full prediction model yet. It provides:

- A project structure for modeling, simulation, data, and dashboard work
- A real-data ingestion script for World Football Elo ratings
- A working dashboard prototype showing estimated champion likelihoods
- An Elo Monte Carlo using the actual 2026 World Cup groups and knockout bracket
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

This starter downloads the current world ratings table from:

```text
https://www.eloratings.net/World.tsv
```

It also downloads Elo team names and confederation pages from `eloratings.net` so the dashboard can show readable team names and regional filters. The exported starter ratings preserve the official Elo value directly. The `attack_rating` and `defense_rating` fields are lightweight dashboard placeholders derived from historical goals for/against per match; they are not official Elo fields.

## Refresh Elo Data

```bash
PYTHONPATH=src python3 scripts/run_simulation.py --iterations 50000
```

Serve the dashboard locally:

```bash
python3 -m http.server 5173
```

Then open:

```text
http://localhost:5173/app/
```

## Project Structure

```text
.
├── app/                         # Static dashboard prototype
├── data/
│   ├── raw/                     # Downloaded Elo ratings and result files
│   ├── backtests/               # Historical tournament rating snapshots
│   ├── derived_team_strengths.csv
│   └── derived_team_strengths.json
├── reports/                     # Generated simulation, backtest, and infographic outputs
├── scripts/
│   ├── fetch_elo_data.py        # Refresh current Elo data
│   ├── run_simulation.py        # Run the 2026 Monte Carlo forecast
│   ├── run_backtest.py          # Replay historical World Cups
│   ├── calibrate_draw_rate.py   # Fit draw probability by Elo gap
│   ├── tune_xg_elo.py           # Tune xG/Elo model settings
│   ├── make_infographic.py      # Render PDF/PNG forecast artifact
│   └── smoke.py                 # End-to-end import/data check
├── src/worldcup_prediction/
│   ├── data_loader.py
│   ├── models/                  # Elo, W/D/L, scoreline, and team-strength logic
│   └── simulation/              # Monte Carlo and backtest engines
└── tests/                       # Model, simulation, and backtest tests
```

## Core Workflows

Refresh current Elo inputs:

```bash
PYTHONPATH=src python3 scripts/fetch_elo_data.py
```

Run all historical backtests:

```bash
PYTHONPATH=src python3 scripts/run_backtest.py --tournament all --iterations 10000
```

Calibrate the draw-rate curve:

```bash
PYTHONPATH=src python3 scripts/calibrate_draw_rate.py --start-year 1994 --end-year 2025 --test-start-year 2022
```

Generate the PDF infographic and PNG pages:

```bash
python3 scripts/make_infographic.py --png
```

Generated infographic pages:

- `reports/world_cup_2026_infographic.pdf`
- `reports/world_cup_2026_infographic_page1.png` … `_page3.png` (with `--png`)

The infographic is a three-page editorial poster:

1. **The forecast** — title odds for the top contenders, champion probability by
   confederation, and the favourite's stage-by-stage path.
2. **The bracket** — a stage-progression matrix (round of 16 through the title) and the two
   most likely sides to advance from each group.
3. **Model validation** — how the same model scored on past World Cups using only
   pre-tournament ratings, with headline accuracy and calibration stats.

Useful flags:

- `--png` also export each page as a high-resolution PNG.
- `--dpi 300` raise the export resolution (default `200`).
- `--out path/to/file.pdf` choose a different output location.

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

## Current Status

This is scaffolding plus a dashboard prototype. The next useful milestone is to implement a small deterministic baseline model and connect it to the dashboard-generated probabilities.
