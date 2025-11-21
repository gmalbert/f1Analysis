## Purpose
Short, actionable guidance for AI coding agents working on the Formula 1 Analysis project.

## Top-level summary
- **Prediction-focused F1 analysis app**: The primary goal is predicting race winners, DNFs, and pit stop times. All data processing, feature engineering, and UI filters serve these prediction objectives.
- **Success metric: MAE minimization**: Mean Absolute Error (MAE) is the key performance indicator. Current target is MAE ≤1.5 for final position predictions. All feature engineering and model improvements should focus on reducing MAE.
- **Multiple Model Support**: Now supports XGBoost, LightGBM, CatBoost, and Ensemble stacking for maximum prediction accuracy and flexibility.
- **Multiple Model Support**: Now supports XGBoost, LightGBM, CatBoost, and Ensemble stacking for maximum prediction accuracy and flexibility.
- This is a Streamlit-based data analysis app. The canonical runtime flow is:
  1. Run `f1-generate-analysis.py` to precompute data and write CSVs into `data_files/`.
  2. Run the Streamlit app `raceAnalysis.py` which reads those CSVs and displays the UI.
- Heavy computations live in the generator step so the UI remains responsive. Don't modify UI behavior without checking how CSVs are produced.

## Model Architecture & Selection
- **XGBoost** (Default): Excellent general-purpose performance, handles missing data, built-in feature importance. Best MAE performance.
- **LightGBM**: Very fast training, good for large datasets, handles categorical features well. Use when speed is critical.
- **CatBoost**: Excellent with categorical data, robust to overfitting, handles missing values automatically. Best for stability.
- **Ensemble (XGBoost + LightGBM + CatBoost)**: Stacks all three models using sklearn StackingRegressor. Highest accuracy but slowest training.
- **Hyperparameter Optimization**: Bayesian optimization (Optuna) and grid search with season-stratified GroupKFold CV to prevent data leakage.
- **Model-Specific Handling**: Conditional code for different APIs (feature importance, prediction formats, boosting rounds display).

## Key files & locations
 - `f1-generate-analysis.py` — generator that creates grouped CSVs and processed JSONs (2543 lines).
  - Uses `LOCAL_RUN` environment variable to enable FastF1 caching for local development.
  - Feature engineering section starts at line 1699 with comment "NEW LEAKAGE-FREE FEATURES".
  - Final CSV writes at lines 2217+ define the data contract with the UI.
  - Weather fetching logic updated to pull data for all missing races, not just the most recent one.
  - Optional post-run smoke checks: pass `--check-smoke` to run `scripts/check_generation_smoke.py` (see helper scripts). Forward `--smoke-strict`, `--smoke-qual-threshold`, and `--smoke-tolerance-days` to control behavior.
  - Recent fixes batch high-cardinality bin creation to reduce DataFrame fragmentation and silence PerformanceWarnings.
- `raceAnalysis.py` — Streamlit UI that consumes the outputs in `data_files/` (4973 lines).
  - All data loading functions use `@st.cache_data` decorator for performance.
  - Uses CACHE_VERSION="v2.3" for version-based cache invalidation to prevent stale cached models on Streamlit Cloud.
  - Four prediction models: XGBoost, LightGBM, CatBoost, and Ensemble stacking (XGBoost + LightGBM + CatBoost).
  - Hyperparameter optimization with Bayesian optimization (Optuna) and grid search.
  - Season-stratified cross-validation to prevent data leakage.
  - Model-specific feature importance extraction and prediction handling.
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
  - `scripts/audit_temporal_leakage.py` — conservative audit for temporal leakage; integrated into `scripts/run_all_smoke_checks.py`.
  - `scripts/check_points_leader_gap.py` — quick diagnostic to validate `points_leader_gap` and `round` population per race snapshot.
  
Additional preferred practice-best filenames:
  - `data_files/practice_best_by_session.imputed.round_filled.csv` — preferred by the generator when present (produced by `scripts/impute_missing_practice.py` + `scripts/aggregate_practice_laps.py` and a small `round` fill).
  - Fallbacks: `data_files/practice_best_by_session.imputed.csv`, `data_files/practice_best_by_session.csv`, then legacy `practice_best_fp1_fp2.csv` via `get_preferred_file()`.
- `pit_constants.py` — track-specific pit lane times (entry to exit) as a dictionary. Used for pit stop calculations.
 - Helper scripts (mostly utility/exploration, not part of main workflow):
  - `scripts/export_feature_selection.py` — consolidates SHAP, Boruta, and correlation artifacts into `scripts/output/feature_selection_summary.csv` and `scripts/output/feature_selection_report.html` (CSV/HTML-first, intended for sharing and quick review).
  - `f1-raceMessages.py` — pulls race control messages from FastF1 API (skips existing sessions)
  - `f1-pit-stop-loss.py` — calculates pit stop time loss using constants from `pit_constants.py`
  - `scripts/check_generation_smoke.py` — smoke test that validates `f1ForAnalysis.csv` coverage vs `f1db-races.json` and checks qualifying completeness (use `--strict` to fail CI).
    - `scripts/run_all_smoke_checks.py` — new convenience runner that discovers and executes smoke/check scripts in `scripts/` and prints a summarized pass/fail report. Use it from the repo root to run the full battery of checks (see script header for usage examples).
  - `scripts/repair_qualifying.py` — repair helpers for `all_qualifying_races.csv` used during the recent data-repair workflow.
  - `fastF1-qualifying.py` — fetches qualifying sessions (FastF1), merges with active drivers, and writes `data_files/all_qualifying_races.csv`.
    - New behavior: the qualifying script now computes `teammate_qual_delta` vectorially and will infer missing `constructorId`/`constructorName` from `f1db-races-race-results.json` when possible (this prevents missing constructor metadata from blocking teammate-delta computation).
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
 - **Leakage prevention**: All features use `.shift()` or filtering to avoid using future data. Safety car features (line 2224+) are specifically designed to be available before race starts. A temporal leakage audit script (`scripts/audit_temporal_leakage.py`) was added and runs as part of smoke checks to flag suspicious columns; `points_leader_gap` was updated to compute the per-race snapshot leader gap (grouped by `grandPrixYear` + `round`/`raceId`/`short_date`) to avoid season-wide leakage.
- **Incremental data pulls**: Scripts like `f1-raceMessages.py` track previously processed sessions and only pull new data to avoid API rate limits and reduce runtime.
- **Streamlit caching**: All heavy data operations use `@st.cache_data` to avoid recomputation on every widget interaction.
- **Cache invalidation**: Uses CACHE_VERSION system for all cached functions to prevent feature shape mismatches on Streamlit Cloud deployments.

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
- **Compile and test after fixes**: Every time you claim to have fixed a bug or made a change, immediately use `py_compile.compile()` on the affected script (e.g., `python -c "import py_compile; py_compile.compile('f1-generate-analysis.py')"` or `python -c "import py_compile; py_compile.compile('raceAnalysis.py')"` ) to verify it compiles without syntax errors. Don't assume the fix works—validate it.
- **MAE impact first**: Always measure MAE before/after feature changes. The goal is MAE ≤1.5 for final position predictions. Use cross-validation to validate improvements.
- **Feature selection validation**: New features should improve MAE in Monte Carlo simulations (1000+ iterations). Consider RFE and Boruta for feature selection.
- **Generator output changes**: If you modify the final `to_csv()` calls in `f1-generate-analysis.py` (around lines 2217-2218), update corresponding `read_csv()` calls in `raceAnalysis.py`.
- **Column name changes**: The UI has ~30 filter controls. Changing column names breaks filters silently. Search for the column name in `raceAnalysis.py` before renaming.
- **Data types**: The UI uses pandas type checking (`is_numeric_dtype`, `is_object_dtype`, etc.) for dynamic filtering. Ensure consistent dtypes.

- **Creating checks and repair helpers**: When you add diagnostic, smoke-test, or repair logic, prefer creating a new script under the `scripts/` directory (e.g., `scripts/my_check.py`) instead of editing existing production scripts. This keeps the generator and UI code stable and makes CI smoke tests easier to review and run.
  The new `scripts/run_all_smoke_checks.py` script is intended to be the one-stop command for running these scripts locally or in CI. It will list discovered checks with `--list-only`, and supports `--continue-on-fail` for non-fatal runs.
 - **Feature-selection artifacts & exporter**: The feature-selection pipeline emits lightweight artifacts into `scripts/output/` (SHAP ranking, Boruta selected, correlated pairs). Use `scripts/export_feature_selection.py` to create a CSV summary and an HTML report. The Streamlit UI includes download buttons and an on-demand "Regenerate CSV/HTML exporters" action in the Feature Selection tab.
- **Time string conversions**: Use `time_to_seconds()` helper function (line 250 in generator) for lap time conversions. Handles various F1DB time formats.
- **Model compatibility**: When adding features that interact with models, ensure compatibility across all four model types (XGBoost, LightGBM, CatBoost, Ensemble). Use isinstance() checks and hasattr() for API differences.
- **Feature importance**: Different models have different APIs - XGBoost uses get_score(), LightGBM uses feature_importances_, CatBoost uses get_feature_importance().
- **Prediction formats**: XGBoost uses xgb.DMatrix for prediction, others use regular numpy arrays.
- **Early stopping**: Different models have different early stopping APIs and attribute names.
- **Cache invalidation**: Always include CACHE_VERSION in cached function parameters to prevent stale models on Streamlit Cloud. Update CACHE_VERSION when making breaking changes to cached data structures.

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
- **Model-specific prediction handling**:
  ```python
  # Different models need different prediction formats
  if isinstance(model, xgb.Booster):  # XGBoost
      y_pred = model.predict(xgb.DMatrix(X_test_prep))
  else:  # LightGBM, CatBoost, sklearn models
      y_pred = model.predict(X_test_prep)
  ```
- **Model-specific feature importance**:
  ```python
  # Different APIs for feature importance
  if hasattr(model, 'get_score'):  # XGBoost
      importances_dict = model.get_score(importance_type='weight')
  elif hasattr(model, 'feature_importances_'):  # LightGBM, CatBoost
      importances = model.feature_importances_
  elif hasattr(model, 'get_feature_importance'):  # CatBoost
      importances = model.get_feature_importance()
  ```
- **Model-specific boosting rounds display**:
  ```python
  # Different attribute names for boosting rounds
  if hasattr(model, 'best_iteration_'):  # LightGBM
      st.write(f"Boosting rounds used: {model.best_iteration_}")
  elif hasattr(model, 'best_iteration'):  # XGBoost
      st.write(f"Boosting rounds used: {model.best_iteration + 1}")
  elif hasattr(model, 'get_best_iteration'):  # CatBoost
      st.write(f"Boosting rounds used: {model.get_best_iteration()}")
  ```
- **Cache invalidation pattern**:
  ```python
  # Always include CACHE_VERSION in cached functions to prevent stale models
  @st.cache_data
  def load_and_preprocess_data(CACHE_VERSION):
      # Data loading and preprocessing logic
      return processed_data
  
  @st.cache_data  
  def train_model(X, y, model_type, CACHE_VERSION):
      # Model training logic
      return trained_model
  ```
- **Model-specific prediction handling**:
  ```python
  # Different models need different prediction formats
  if isinstance(model, xgb.Booster):  # XGBoost
      y_pred = model.predict(xgb.DMatrix(X_test_prep))
  else:  # LightGBM, CatBoost, sklearn models
      y_pred = model.predict(X_test_prep)
  ```
- **Model-specific parameter handling**:
  ```python
  # CatBoost handles categorical features automatically
  elif model_type == 'CatBoost':
      model = CatBoostRegressor(cat_features=categorical_cols, ...)
  ```
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
ML: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `boruta`, `shap`  
Viz: `altair`, `matplotlib`, `seaborn`, `plotly`
Utils: `requests_cache`, `retry_requests`

## Notes for AI agents
- **Root directory structure**: All Python scripts are in the project root. The `.venv/` directory contains the virtual environment (ignored by git).
- **No formal tests**: Testing is manual via Streamlit UI and MAE evaluation. Use `xgb_test.py` as a reference for XGBoost early stopping.
- **Version control**: `.gitignore` excludes `.venv/`, `data_files/f1_cache/`, and many backup/intermediate files. Only track core scripts and final CSVs.
- **Naming conventions**: Files prefixed with `f1-` are data generation scripts. Files suffixed with `_bkup`, `_with_commented_out_code`, or dates are backups (not actively maintained).
- **Warning fixes**: Recent updates resolved imputation warnings, numpy RuntimeWarnings, pandas deprecation warnings (including combine_first FutureWarning), and scikit-learn warnings. Always ensure virtual environment is activated to avoid module import issues.
- **Data quality**: Active driver NaN values are filled with baseline values before CSV export. Weather data is fetched for all missing races, not just recent ones.
- **Model compatibility**: All new code must handle XGBoost, LightGBM, CatBoost, and Ensemble models. Use isinstance() and hasattr() checks for API differences.
- **Cache invalidation**: Streamlit Cloud requires explicit cache invalidation for all cached objects. Use CACHE_VERSION system to prevent feature mismatch errors on deployment.
