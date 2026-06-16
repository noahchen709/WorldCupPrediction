# 2026 FIFA World Cup Prediction

A Python project for forecasting the 2026 FIFA World Cup with Elo ratings, Monte Carlo simulation, historical backtests, and a small static dashboard.

## What It Includes

- World Football Elo data ingestion
- 2026 tournament simulation with group and knockout rounds
- Historical World Cup backtests
- Draw-rate calibration and xG/Elo tuning scripts
- Generated CSV, JSON, HTML, PDF, and PNG reports
- A static dashboard in `app/`

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

Refresh Elo data:

```bash
PYTHONPATH=src python3 scripts/fetch_elo_data.py
```

Run the 2026 Monte Carlo forecast:

```bash
PYTHONPATH=src python3 scripts/run_simulation.py --iterations 50000
```

Run historical backtests:

```bash
PYTHONPATH=src python3 scripts/run_backtest.py --tournament all --iterations 10000
```

Calibrate the draw-rate model:

```bash
PYTHONPATH=src python3 scripts/calibrate_draw_rate.py --start-year 1994 --end-year 2025 --test-start-year 2022
```

Generate the infographic:

```bash
python3 scripts/make_infographic.py --png
```

Run tests:

```bash
pytest
```

## Dashboard

Serve the repo locally:

```bash
python3 -m http.server 5173
```

Open:

```text
http://localhost:5173/app/
```

## Project Layout

```text
app/                         Static dashboard
data/raw/                    Downloaded Elo ratings and match results
data/backtests/              Historical tournament rating snapshots
reports/                     Generated forecasts, backtests, and infographics
scripts/                     Data, simulation, calibration, and reporting commands
src/worldcup_prediction/     Core Python package
tests/                       Pytest suite
```

## Data Source

The project uses World Football Elo Ratings:

```text
https://www.eloratings.net/
```

Current ratings are downloaded from:

```text
https://www.eloratings.net/World.tsv
```

Team names and confederation data also come from `eloratings.net`.

## Outputs

Key generated files include:

- `reports/monte_carlo_results.json`
- `reports/monte_carlo_results.csv`
- `reports/monte_carlo_report.html`
- `reports/world-cup-backtests.json`
- `reports/world_cup_2026_infographic.pdf`
- `reports/world_cup_2026_infographic_page1.png`
- `reports/world_cup_2026_infographic_page2.png`
- `reports/world_cup_2026_infographic_page3.png`
