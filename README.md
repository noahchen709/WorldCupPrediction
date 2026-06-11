# World Cup Prediction

A Python-based forecasting project for the 2026 FIFA World Cup. It combines World Football Elo ratings, a calibrated draw model, expected-goals style scoring profiles, Monte Carlo tournament simulation, historical backtesting, and a small static dashboard.

The goal is not to pretend soccer is fully predictable. The goal is to build a transparent forecasting pipeline that can ingest real ratings, simulate the tournament structure thousands of times, validate the approach on past World Cups, and publish the results in recruiter-friendly artifacts.

## Highlights

- **Real data pipeline**: fetches World Football Elo ratings, confederation files, team names, and historical result tables from `eloratings.net`.
- **Tournament simulation**: models the 48-team 2026 format with 12 groups, best third-place qualifiers, Round of 32, and match-numbered knockout paths.
- **Calibrated match probabilities**: uses Elo expected score, a fitted draw-rate curve, host advantage, and an xG/Elo-adjusted scoring-history model.
- **Backtesting**: replays completed World Cups using pre-tournament ratings and reports log loss, Brier scores, calibration buckets, stage error, and top-pick accuracy.
- **Recruiter-visible outputs**: includes generated CSV/JSON reports, an HTML report, a three-page PDF infographic, and a static dashboard prototype.

## Current Forecast Snapshot

Latest checked-in simulation: `50,000` iterations, generated from World Football Elo data and the configured 2026 tournament field.

| Team | Champion | Final | Semi-final |
| --- | ---: | ---: | ---: |
| Spain | 21.3% | 33.2% | 45.4% |
| Argentina | 20.5% | 31.0% | 44.4% |
| England | 12.1% | 20.8% | 34.8% |
| France | 8.0% | 15.6% | 29.3% |
| Colombia | 4.8% | 10.4% | 19.7% |

Full outputs:

- [Monte Carlo CSV](reports/monte_carlo_results.csv)
- [Monte Carlo JSON](reports/monte_carlo_results.json)
- [HTML report](reports/monte_carlo_report.html)
- [PDF infographic](reports/world_cup_2026_infographic.pdf)
- [Dashboard](app/index.html)

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the smoke check:

```bash
PYTHONPATH=src python3 scripts/smoke.py
```

Run the test suite:

```bash
PYTHONPATH=src pytest
```

Regenerate the main forecast:

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
- `reports/world_cup_2026_infographic_page1.png`
- `reports/world_cup_2026_infographic_page2.png`
- `reports/world_cup_2026_infographic_page3.png`

## Model Notes

The baseline starts with World Football Elo expected score and converts it into win/draw/loss probabilities using a fitted draw curve:

```text
P(draw) = 0.0200 + (0.3850 - 0.0200) * exp(-abs(Elo gap) / 344.0)
```

The newer simulation path adds a lightweight expected-goals profile from recent scoring history. Matches are weighted by recency and match importance, adjusted for opponent Elo, and then combined with the fitted draw model. The pipeline remains intentionally inspectable: every major output is written to CSV or JSON so the assumptions can be audited.

## Data Source

This project uses World Football Elo Ratings:

```text
https://www.eloratings.net/
```

Primary downloaded files include:

- `https://www.eloratings.net/World.tsv`
- `https://www.eloratings.net/en.teams.tsv`
- Confederation files such as `UEFA.tsv`, `CONMEBOL.tsv`, `CONCACAF.tsv`, `CAF.tsv`, `AFC.tsv`, and `OFC.tsv`
- Historical result files such as `https://www.eloratings.net/2022_results.tsv`

The derived `rating` and `elo` fields preserve official Elo values. Dashboard compatibility fields such as `attack_rating` and `defense_rating` are derived features, not official Elo fields.

## What This Demonstrates

- Designing a reproducible data-to-report pipeline
- Turning sports ratings into calibrated probabilistic predictions
- Building Monte Carlo simulations with deterministic seeds and test coverage
- Evaluating forecasts with proper scoring rules instead of only narrative accuracy
- Publishing results in formats useful to both engineers and non-technical readers

