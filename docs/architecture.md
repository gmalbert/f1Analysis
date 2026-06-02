# F1 Analysis — Architecture

## Overview
Prediction-focused Formula 1 analysis app using a two-phase precompute-then-display pattern. Predicts race winners, DNFs, and pit stop times using XGBoost, LightGBM, CatBoost, and an Ensemble stack.

## Two-Phase Architecture
```
Phase 1 — Data Generation (~10-30 min):
    FastF1 + F1DB JSON + Open-Meteo + qualifying CSV
            ↓
    f1-generate-analysis.py
            ↓
    data_files/f1ForAnalysis.csv (tab-separated, 2200+ columns)
    data_files/f1WeatherData_*.csv
    data_files/f1PitStopsData_*.csv
    data_files/all_race_control_messages.csv

Phase 2 — UI (sub-second):
    raceAnalysis.py (Streamlit, @st.cache_data)
            ↓
    Reads CSVs → trains model on-demand → displays predictions
```

## ML Models
All models target **MAE minimisation** for final race position (current MAE ~1.94, target ≤1.5):
| Model | Notes |
|-------|-------|
| XGBoost | Default, best MAE, `xgb.DMatrix` for predict |
| LightGBM | Fast, `feature_importances_` attribute |
| CatBoost | Best with categoricals, `get_feature_importance()` |
| Ensemble | StackingRegressor (XGB+LGB+CAT), highest accuracy, slowest |

Feature engineering: 70+ leakage-free features (all use `shift(1)` before rolling windows). Key: `practice_improvement`, `constructor_recent_form_3_races`, `podium_potential`, `track_experience`.

## Data Sources
| Source | Purpose | Notes |
|--------|---------|-------|
| F1DB JSON files | Drivers, constructors, circuits, results | `data_files/f1db-*.json` |
| FastF1 | Race control messages, practice data | Requires `LOCAL_RUN=1` env var for cache |
| Open-Meteo | Historical weather per race | Archive + forecast endpoints |
| football-data.co.uk (via UK flag) | N/A | Not used here |

## Key Files
- `f1-generate-analysis.py` — 2543-line generator (run first)
- `raceAnalysis.py` — Streamlit UI (run second)
- `pit_constants.py` — track-specific pit lane times
- `scripts/` — 20+ utility/diagnostic scripts (see `scripts/README.md`)
- `data_files/` — all CSV/JSON outputs

## Feature Engineering Rules
- Always `sort_values` by group + date before rolling calculations
- Use `shift(1)` before every cumulative/rolling window
- Never use future-race data for pre-race features
- Run `scripts/audit_temporal_leakage.py` after adding features

## Cache Invalidation
`CACHE_VERSION = "v2.3"` passed to all `@st.cache_data` functions. Increment when making breaking changes to cached data structures.

## Email Notifications
`scripts/send_rich_email_now.py` → checks upcoming races in `f1db-races.json` → calls `scripts/export_email_context.py` → sends SMTP email with embedded HTML + TSV attachment. Use `--force` to bypass race-timing checks.
