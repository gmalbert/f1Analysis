## Purpose
Short, actionable guidance for AI coding agents working on the Formula 1 Analysis project.

## Top-level summary
- **Prediction-focused F1 analysis app**: The primary goal is predicting race winners, DNFs, and pit stop times. All data processing, feature engineering, and UI filters serve these prediction objectives.
- **Success metric: MAE minimization**: Mean Absolute Error (MAE) is the key performance indicator. Current target is MAE ≤1.5 for final position predictions. All feature engineering and model improvements should focus on reducing MAE.
- This is a Streamlit-based data analysis app. The canonical runtime flow is:
  1. Run `f1-generate-analysis.py` to precompute data and write CSVs into `data_files/`.
  2. Run the Streamlit app `raceAnalysis.py` which reads those CSVs and displays the UI.
- Heavy computations live in the generator step so the UI remains responsive. Don't modify UI behavior without checking how CSVs are produced.

## Key files & locations
- `f1-generate-analysis.py` — generator that creates grouped CSVs and processed JSONs (2531 lines).
- `raceAnalysis.py` — Streamlit UI that consumes the outputs in `data_files/` (4305 lines).
- `data_files/` — contains generated CSVs, F1DB JSONs, and FastF1 cache. Key outputs:
  - `f1ForAnalysis.csv` — main dataset (tab-separated)
  - `f1WeatherData_Grouped.csv` — weather by race
  - `f1PitStopsData_Grouped.csv` — pit stop analysis
  - `f1SafetyCarFeatures.csv` — safety car data

## Important project-specific patterns
- **MAE-driven feature engineering**: All features are designed to minimize Mean Absolute Error when predicting final race positions. When adding features, evaluate their impact on MAE reduction. Current best MAE is ~1.5.
- **Prediction-centric approach**: Features target race winners, DNFs, and pit stop times. Monte Carlo simulation (1000 iterations) and feature selection (RFE, Boruta) are used to optimize MAE.
- **Tab-separated CSVs**: All CSVs use `sep='\t'` not commas. When reading/writing, always specify `sep='\t'`.
- **Precompute-first**: The generator intentionally separates heavy computation from the UI. Any change to column names in the generator must be propagated to the UI and vice-versa.
- **CSV contract**: The UI expects specific columns. Key MAE-critical ones include:
  - Pit stops: `['raceId', 'driverId', 'constructorId', 'numberOfStops', 'averageStopTime', 'totalStopTime']`
  - Weather: `['grandPrixId', 'short_date', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed']`
  - Race results: `['resultsFinalPositionNumber', 'driverDNFCount', 'SafetyCarStatus']`
- **ML features**: XGBoost model uses 70+ derived features (see README table) specifically selected to minimize MAE for final position predictions. Feature engineering happens in the generator around lines 1699+ ("NEW LEAKAGE-FREE FEATURES").

## External integrations
- **F1DB**: Source JSON files from github.com/f1db/f1db (all `f1db-*.json` files in `data_files/`)
- **FastF1**: Used for race control messages, practice data, and caching (`f1_cache/` directory)
- **Open-Meteo**: Weather API for historical data by race hour

## How to run
```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Generate the data (required first step)
python f1-generate-analysis.py

# Run the Streamlit app
streamlit run raceAnalysis.py
```

## What to watch for when editing
- **MAE impact first**: Always measure MAE before/after feature changes. The goal is MAE ≤1.5 for final position predictions. Use cross-validation to validate improvements.
- **Feature selection validation**: New features should improve MAE in Monte Carlo simulations (1000+ iterations). Consider RFE and Boruta for feature selection.
- **Generator output changes**: If you modify the final `to_csv()` calls in `f1-generate-analysis.py` (around lines 2217-2218), update corresponding `read_csv()` calls in `raceAnalysis.py`.
- **Column name changes**: The UI has ~30 filter controls. Changing column names breaks filters silently. Search for the column name in `raceAnalysis.py` before renaming.
- **Data types**: The UI uses pandas type checking (`is_numeric_dtype`, `is_object_dtype`, etc.) for dynamic filtering. Ensure consistent dtypes.

## Useful examples from this codebase
- **Adding an MAE-improving feature**: If you add `driver_recent_form`:
  1. Add it to the feature engineering section (around line 1699+)
  2. Include it in the final `to_csv()` call
  3. **Test MAE impact**: Compare before/after MAE using cross-validation
  4. Consider its correlation with `resultsFinalPositionNumber`
  5. Add it to UI filters in `raceAnalysis.py` if user-facing
- **High-impact MAE features** (from README): `practice_improvement`, `constructor_recent_form_3_races`, `best_s1_sec`, `podium_potential`, `track_experience`
- **DNF prediction features**: `driverDNFCount`, `recent_dnf_rate_3_races`, `SafetyCarStatus` (affects position predictions)
- **Pit stop timing features**: `numberOfStops`, `averageStopTime`, `totalStopTime`, `pit_stop_delta` (critical for race position)
- **Feature interaction examples**: `qualPos_x_last_practicePos`, `grid_x_avg_pit_time`, `recent_form_x_qual` (often reduce MAE more than individual features)
- **Tab-separated format**: Always use `pd.read_csv(path, sep='\t')` and `df.to_csv(path, sep='\t')`

## Dependencies (inferred from imports)
Core: `streamlit`, `pandas`, `numpy`, `fastf1`, `openmeteo_requests`
ML: `scikit-learn`, `xgboost`, `boruta`, `shap`  
Viz: `altair`, `matplotlib`, `seaborn`
Utils: `requests_cache`, `retry_requests`

## File locations confirmed
All Python files are in the `.venv/` directory, not the repository root. The `data_files/` subdirectory contains 100+ JSON/CSV files including the F1DB dataset and generated outputs.
