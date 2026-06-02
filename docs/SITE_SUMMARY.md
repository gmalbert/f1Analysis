> **AI Onboarding Guide** — See also `.github/copilot-instructions.md` for full coding conventions.

# F1 Analysis — Site Summary

## What This App Does

Two-step Formula 1 race analytics platform. A heavy data generator (`f1-generate-analysis.py`) precomputes 2200+ feature columns and writes tab-separated CSVs; the Streamlit UI (`raceAnalysis.py`) then reads those CSVs and renders an interactive sidebar-filtered analysis dashboard. Predicts race finishing positions using XGBoost, LightGBM, CatBoost, and an ensemble stacking model. Current best MAE: ~1.94 (target: ≤1.5).

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Set environment variables
$env:LOCAL_RUN = "1"               # Enables FastF1 local caching

# 3. Run the data generator (~10-30 minutes)
python f1-generate-analysis.py

# 4. Run the app
streamlit run raceAnalysis.py
```

Step 3 must complete before Step 4. GitHub Actions re-runs the generator after qualifying data arrives each weekend.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (sidebar-filtered single page) |
| ML | XGBoost (default), LightGBM, CatBoost, Ensemble stacking |
| Hyperparameter tuning | Optuna (Bayesian optimization) |
| Feature selection | SHAP, Boruta, RFE |
| Data sources | F1DB JSON, FastF1 (2018+), Open-Meteo (weather) |
| Data storage | Tab-separated CSV (NOT comma-separated) |

## Key Files

| File | Purpose |
|---|---|
| `f1-generate-analysis.py` | Data generator (2543 lines) — runs offline, writes all CSV outputs |
| `raceAnalysis.py` | Streamlit UI — reads CSVs, renders filtered dashboard |
| `pit_constants.py` | Track-specific pit lane times (entry → exit) — used for pit stop calculations |
| `scripts/` | 20+ utility/diagnostic scripts — leakage audit, smoke tests, email sender |
| `data_files/f1ForAnalysis.csv` | Primary output — 2200+ columns, tab-separated |
| `data_files/f1db-*.json` | F1DB source data — all seasons, drivers, circuits |

## Data Flow

1. **F1DB JSON**: `data_files/f1db-*.json` → historical races, drivers, circuits, qualifying
2. **FastF1**: race control messages (2018+), practice lap data → `data_files/all_race_control_messages.csv`
3. **Open-Meteo**: weather per race day (free, no key) → `data_files/f1WeatherData_Grouped.csv`
4. **Generator**: all sources merged → feature engineering (lines 1699+) → `f1ForAnalysis.csv`
5. **UI**: reads `f1ForAnalysis.csv` → sidebar filters → model predictions → interactive dashboard

## CSVs Use TAB Separator

All CSVs in this project use `sep='\t'` — never comma:
```python
df = pd.read_csv("data_files/f1ForAnalysis.csv", sep='\t')   # Correct
df.to_csv("output.csv", sep='\t', index=False)                # Correct
```

## Cache Invalidation

All `@st.cache_data` functions include `CACHE_VERSION="v2.3"` as a parameter. Update this version string whenever making breaking changes to cached data structures — prevents stale model errors on Streamlit Cloud.

## ML Model Details

| Model | Use case |
|---|---|
| XGBoost (default) | Best general MAE; uses `xgb.DMatrix` for prediction |
| LightGBM | Fast training; uses standard `model.predict(X)` |
| CatBoost | Best with categorical features; uses `get_feature_importance()` |
| Ensemble | Stacks all three via sklearn `StackingRegressor`; slowest |

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `LOCAL_RUN=1` | Enables FastF1 local caching for the generator | Required for local dev |

## Critical Conventions

- **Generator must run before UI** — never expect the UI to generate data
- **All CSVs are tab-separated** — always specify `sep='\t'`
- Feature engineering is at **lines 1699+** in `f1-generate-analysis.py` — all features use `shift(1)` before rolling to prevent leakage
- Adding new features requires updating **both** the generator AND corresponding `read_csv` calls in `raceAnalysis.py`
- Run `python -c "import py_compile; py_compile.compile('f1-generate-analysis.py')"` after every edit to check for syntax errors

## Common Gotchas

- `scripts/` contains 20+ utilities — read `scripts/README.md` before adding a new one
- `data_files/sidebar_exclusion_debug.json` was a temporary debug file that has been removed; do not recreate it unless debugging
- If sidebar filters show unexpected columns, check the `exclusionList` and `suffixes_to_exclude` in `raceAnalysis.py`
- Run `scripts/run_all_smoke_checks.py` after generator runs to validate data quality
