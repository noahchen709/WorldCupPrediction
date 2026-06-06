# 2026 FIFA World Cup Quant Prediction Starter

A clean starter project for building a quantitative 2026 FIFA World Cup prediction system.

This repo intentionally does not include a full prediction model yet. It provides:

- A project structure for modeling, simulation, data, and dashboard work
- Sample team data with placeholder ratings and qualification assumptions
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
│   ├── sample_teams.csv         # Starter team-strength inputs
│   └── sample_matches.csv       # Starter fixture/result-style data
├── notebooks/                   # Exploratory analysis notebooks
├── scripts/
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

## Run The Dashboard

From the project root:

```bash
python3 -m http.server 5173
```

Then open:

```text
http://localhost:5173/app/
```

The dashboard currently uses transparent placeholder formulas and sample data. Treat its probabilities as demo values only.

## Run The Python Smoke Check

```bash
PYTHONPATH=src python3 scripts/smoke.py
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
