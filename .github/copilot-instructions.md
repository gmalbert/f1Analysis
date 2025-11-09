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
- `f1-generate-analysis.py` — generator that creates grouped CSVs and processed JSONs (2543 lines).
  - Uses `LOCAL_RUN` environment variable to enable FastF1 caching for local development.
  - Feature engineering section starts at line 1699 with comment "NEW LEAKAGE-FREE FEATURES".
  - Final CSV writes at lines 2217+ define the data contract with the UI.
  - Weather fetching logic updated to pull data for all missing races, not just the most recent one.
- `raceAnalysis.py` — Streamlit UI that consumes the outputs in `data_files/` (4692 lines).
  - All data loading functions use `@st.cache_data` decorator for performance.
  - Three prediction models: position (final placement), DNF (did not finish), and safety car likelihood.
  - Rookie simulation functions at lines 231+ and 309+ for handling drivers without historical data.
  - Includes warning suppression for numpy RuntimeWarnings and pandas deprecation warnings.
- `data_files/` — contains generated CSVs, F1DB JSONs, and FastF1 cache. Key outputs:
  - `f1ForAnalysis.csv` — main dataset (tab-separated), ~2200+ columns including all engineered features
  - `f1WeatherData_Grouped.csv` — weather by race (grouped from hourly data)
  - `f1WeatherData_AllData.csv` — hourly weather records (raw Open-Meteo API responses)
  - `f1PitStopsData_Grouped.csv` — pit stop analysis
  - `f1SafetyCarFeatures.csv` — safety car prediction features (leakage-free subset)
  - `f1PositionCorrelation.csv` — Pearson correlation matrix for key position features
  - `all_race_control_messages.csv` — safety car/flag data from FastF1 (2018-present)
- `pit_constants.py` — track-specific pit lane times (entry to exit) as a dictionary. Used for pit stop calculations.
- Helper scripts (mostly utility/exploration, not part of main workflow):
  - `f1-raceMessages.py` — pulls race control messages from FastF1 API (skips existing sessions)
  - `f1-pit-stop-loss.py` — calculates pit stop time loss using constants from `pit_constants.py`
  - `f1-analysis-weather.py`, `f1-constructorStandings.py`, etc. — older data pull scripts

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
- **Leakage prevention**: All features use `.shift()` or filtering to avoid using future data. Safety car features (line 2224+) are specifically designed to be available before race starts.
- **Incremental data pulls**: Scripts like `f1-raceMessages.py` track previously processed sessions and only pull new data to avoid API rate limits and reduce runtime.
- **Streamlit caching**: All heavy data operations use `@st.cache_data` to avoid recomputation on every widget interaction.

## External integrations
- **F1DB**: Source JSON files from github.com/f1db/f1db (all `f1db-*.json` files in `data_files/`)
  - Covers drivers, constructors, circuits, races, qualifying, practice, and race results
  - Historical data from Formula 1 inception to recent seasons
- **FastF1**: Used for race control messages, practice data, and caching (`f1_cache/` directory)
  - Requires `LOCAL_RUN=1` environment variable to enable caching in generator
  - Primary source for 2018+ race control messages (safety cars, flags, pit stops)
- **Open-Meteo**: Weather API for historical data by race hour
  - Two endpoints: archive API (past) and forecast API (next 16 days)
  - Hourly data pulled for race day, then grouped to daily averages
  - Results cached in `f1WeatherData_AllData.csv` to minimize API calls
  - Weather fetching logic updated to pull data for all missing races, not just the most recent one
  - Uses proper YYYY-MM-DD date format for API compatibility

## How to run
```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Generate the data (required first step, ~10-30 min depending on incremental updates)
python f1-generate-analysis.py

# Run the Streamlit app (opens in browser at localhost:8501)
streamlit run raceAnalysis.py
```

## What to watch for when editing
- **MAE impact first**: Always measure MAE before/after feature changes. The goal is MAE ≤1.5 for final position predictions. Use cross-validation to validate improvements.
- **Feature selection validation**: New features should improve MAE in Monte Carlo simulations (1000+ iterations). Consider RFE and Boruta for feature selection.
- **Generator output changes**: If you modify the final `to_csv()` calls in `f1-generate-analysis.py` (around lines 2217-2218), update corresponding `read_csv()` calls in `raceAnalysis.py`.
- **Column name changes**: The UI has ~30 filter controls. Changing column names breaks filters silently. Search for the column name in `raceAnalysis.py` before renaming.
- **Data types**: The UI uses pandas type checking (`is_numeric_dtype`, `is_object_dtype`, etc.) for dynamic filtering. Ensure consistent dtypes.
- **Time string conversions**: Use `time_to_seconds()` helper function (line 250 in generator) for lap time conversions. Handles various F1DB time formats.
- **Rookie handling**: Rookies lack historical features. Use simulation functions in `raceAnalysis.py` (lines 231+, 309+) to generate synthetic features based on truncated normal distributions.
- **FastF1 caching**: Enable with `LOCAL_RUN=1` env var. Cache directory is `data_files/f1_cache/`. Without it, FastF1 re-downloads data every run.

## Useful examples from this codebase
- **Adding an MAE-improving feature**: If you add `driver_recent_form`:
  1. Add it to the feature engineering section (around line 1699+)
  2. Include it in the `static_columns` or `bin_fields` list before the final `to_csv()` call (line ~2210)
  3. **Test MAE impact**: Compare before/after MAE using cross-validation
  4. Consider its correlation with `resultsFinalPositionNumber`
  5. Add it to UI filters in `raceAnalysis.py` if user-facing
- **High-impact MAE features** (from README): `practice_improvement`, `constructor_recent_form_3_races`, `best_s1_sec`, `podium_potential`, `track_experience`
- **DNF prediction features**: `driverDNFCount`, `recent_dnf_rate_3_races`, `SafetyCarStatus` (affects position predictions)
- **Pit stop timing features**: `numberOfStops`, `averageStopTime`, `totalStopTime`, `pit_stop_delta` (critical for race position)
- **Feature interaction examples**: `qualPos_x_last_practicePos`, `grid_x_avg_pit_time`, `recent_form_x_qual` (often reduce MAE more than individual features)
- **Tab-separated format**: Always use `pd.read_csv(path, sep='\t')` and `df.to_csv(path, sep='\t')`
- **Streamlit chart width parameters**:
  ```python
  # Native Streamlit charts - use width parameter
  st.bar_chart(data, width='stretch')     # Full width
  st.line_chart(data, width='content')    # Auto-size
  st.scatter_chart(data, width='stretch')
  
  # Altair charts - use use_container_width parameter
  st.altair_chart(chart, use_container_width=True)   # Full width
  st.altair_chart(chart, use_container_width=False)  # Default size
  ```
- **Streamlit DataFrame height optimization**:
  ```python
  # Use get_dataframe_height() to prevent scrolling
  height = get_dataframe_height(df)  # Auto-calculates based on rows
  st.dataframe(df, height=height)
  
  # With custom max height
  height = get_dataframe_height(df, max_height=400)
  st.dataframe(df, height=height)
  
  # No height limit (use carefully with large datasets)
  height = get_dataframe_height(df, max_height=None)
  st.dataframe(df, height=height)
  ```
- **Tab5 refactoring pattern** (lines 3782-4375): ONE expander + 6 tabs for organizing advanced options:
  ```python
  with st.expander("🔧 Advanced Options", expanded=True):
      tab_perf, tab_feat, tab_select, tab_hyper, tab_hist, tab_debug = st.tabs([...])
      with tab_perf:
          # Model performance metrics
      with tab_feat:
          # Feature analysis tools
      # ... etc
  ```
- **Cross-validation with XGBoost** (line 4264-4273):
  ```python
  # Create fresh estimator for sklearn compatibility
  xgb_for_cv = XGBRegressor(n_estimators=500, learning_rate=0.05, ...)
  # Always preprocess before passing to sklearn tools
  X_eval_preprocessed = preprocessor.fit_transform(X_eval)
  cv_scores = cross_val_score(xgb_for_cv, X_eval_preprocessed, y_eval, ...)
  ```
- **Proper categorical encoding** (line 4358-4377):
  ```python
  # Use preprocessor instead of naive .cat.codes
  preprocessor_q = get_preprocessor_position(X_temp_q)
  X_temp_q_preprocessed = preprocessor_q.fit_transform(X_temp_q)
  model_q.fit(X_temp_q_preprocessed, y_temp_q)
  ```
- **Handling Styler objects** (line 4113-4143):
  ```python
  # Extract DataFrame before calling DataFrame methods
  correlation_df = correlation_matrix.data if hasattr(correlation_matrix, 'data') else correlation_matrix
  correlation_matrix_display = correlation_df.rename(index={...})
  correlation_matrix_display = correlation_matrix_display.style.map(...)
  ```

## Dependencies (inferred from imports)
Core: `streamlit`, `pandas`, `numpy`, `fastf1`, `openmeteo_requests`
ML: `scikit-learn`, `xgboost`, `boruta`, `shap`  
Viz: `altair`, `matplotlib`, `seaborn`
Utils: `requests_cache`, `retry_requests`

## Notes for AI agents
- **Root directory structure**: All Python scripts are in the project root. The `.venv/` directory contains the virtual environment (ignored by git).
- **No formal tests**: Testing is manual via Streamlit UI and MAE evaluation. Use `xgb_test.py` as a reference for XGBoost early stopping.
- **Version control**: `.gitignore` excludes `.venv/`, `data_files/f1_cache/`, and many backup/intermediate files. Only track core scripts and final CSVs.
- **Naming conventions**: Files prefixed with `f1-` are data generation scripts. Files suffixed with `_bkup`, `_with_commented_out_code`, or dates are backups (not actively maintained).
- **Warning fixes**: Recent updates resolved imputation warnings, numpy RuntimeWarnings, pandas deprecation warnings (including combine_first FutureWarning), and scikit-learn warnings. Always ensure virtual environment is activated to avoid module import issues.
- **Data quality**: Active driver NaN values are filled with baseline values before CSV export. Weather data is fetched for all missing races, not just recent ones.
