<img src="data_files/gridlocked-logo-with-text.png" alt="Gridlocked Logo" width="450">

# Formula 1 Data Analysis
Analysis of Formula 1 ```.json``` files based on the very generous data files from [F1DB](https://github.com/f1db/f1db) for the vast majority of the analysis. F1DB did not have race control messages which include Safety Cars and flags. For that data, I used [FastF1](https://docs.fastf1.dev/). Full data analysis is available through the [Formula 1 Analysis - Streamlit app](https://f1analysis-app.streamlit.app/).

## Table of Contents
- [How to run](#how-to-run)
- [Deployment and caching](#deployment-and-caching)
- [File organization](#file-organization)
- [Filtering](#filtering)
- [Linear regression](#linear-regression)
- [Predictive Data Modeling](#predictive-data-modeling)
- [Features used in data model](#features-used-in-data-model)
- [Other options](#other-options)
- [Weather](#weather)
- [To do](#to-do)

## How to run
```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Generate the data (required first step, ~10-30 min depending on incremental updates)
python f1-generate-analysis.py

# Run the Streamlit app (opens in browser at localhost:8501)
streamlit run raceAnalysis.py
```

## Email notifications

This repository includes a small email-notification pipeline that can embed a compact prediction table inline in the message and attach the full predictions CSV/TSV for the upcoming race.

Key scripts:

- `scripts/export_email_context.py` — builds an inline HTML snippet and a TSV attachment from a predictions CSV. If `--input` is not provided it will attempt to select the next race prediction (calendar-aware), then newest `predictions_*.csv`, and finally fall back to `scripts/headless_predict_and_write.py`.
- `scripts/send_rich_email_now.py` — sends the email via SMTP. It always calls the exporter before sending so the inline snippet/attachment are regenerated on each run. Credentials and SMTP config are read from a `.env` file in the repo root.
- `scripts/headless_predict_and_write.py` — lightweight fallback generator that writes `data_files/predictions_headless_next-race_<year>.csv` when no per-GP predictions are available.
- `scripts/generate_driver_constructor_map.py` — helper to generate `data_files/driver_to_constructor.csv` to help populate missing constructor names in headless outputs.

Environment variables (put in `.env`):

```
EMAIL_FROM=you@domain.tld
EMAIL_TO=recipient@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
EMAIL_PASSWORD=yourpassword
USE_TLS=true
```

Quick commands:

```pwsh
# Preview what would be embedded/attached (no email sent)
python .\scripts\email_preview_for_sender.py

# Send now (exporter will be called first to regenerate snippet/TSV)
python .\scripts\send_rich_email_now.py

# Force export from a specific predictions file
python .\scripts\export_email_context.py --input data_files\predictions_abu-dhabi_2025.csv
```

Notes & recommendations:

- **Automatic race checking**: The sender checks `f1db-races.json` for upcoming races and will exit (no send) if no upcoming races are found (season is over) or if the next race is more than 3 days away. Use `--force` flag to bypass: `python .\scripts\send_rich_email_now.py --force`
- The sender regenerates the snippet and TSV on every run so scheduled tasks will always include the latest predictions. For scheduling on Windows use Task Scheduler to call `python .\scripts\send_rich_email_now.py` at the desired cadence.
- Add persistent send logging for auditability (suggested: append timestamp, recipient, exported file path, exporter exit code and SMTP response to `logs/email_sends.log`).
- The exporter will include MAE in inline/attachment when present in the predictions file.
- To change the email layout edit `notifications/templates/race_results.html.j2`.

### Optional post-generation smoke check
After generating data you can run a small smoke test that verifies the analysis CSV covers the races present in `f1db-races.json` and that qualifying data is reasonably complete. Run the generator with the optional `--check-smoke` flag to execute the smoke test automatically:

```powershell
python f1-generate-analysis.py --check-smoke
# to fail the script on problems (useful for CI):
python f1-generate-analysis.py --check-smoke --smoke-strict
```

The smoke test is implemented in `scripts/check_generation_smoke.py` and supports `--qual-threshold` and `--tolerance-days` to tune warning/failure thresholds.

Note about future races: the generator treats future (scheduled) races as "not-yet-processed". The smoke test compares the latest processed data date (`short_date.max` in `f1ForAnalysis.csv`) with the race schedule from `f1db-races.json` and will emit a warning if the schedule contains races beyond the last processed data. This is expected behavior — the pipeline does not fabricate practice/qualifying data for races that haven't occurred. See `scripts/check_generation_smoke.py` if you want to change this behavior (for example, to ignore future dates or adjust tolerance).

### Unified smoke checks

There is a convenience runner that executes all smoke/check scripts under `scripts/` and prints a summarized pass/fail report. Run it from the repository root:

```pwsh
# list discovered checks
python .\scripts\run_all_smoke_checks.py --list-only

# run all checks (stop on first failure)
python .\scripts\run_all_smoke_checks.py

# run all checks and continue past failures (still exits non-zero if any failed)
python .\scripts\run_all_smoke_checks.py --continue-on-fail
```

Use this runner locally before committing or wire it into CI to fail on smoke-test regressions.

### Qualifying CSV and teammate deltas
The qualifying exporter `fastF1-qualifying.py` now computes `teammate_qual_delta` during CSV generation. It will also attempt to infer missing `constructorId`/`constructorName` from `data_files/f1db-races-race-results.json` when those fields are absent so teammate deltas can be computed reliably. To regenerate qualifying data independently:

```powershell
python .\fastF1-qualifying.py
python f1-generate-analysis.py --check-smoke
```

### Adding checks and repair scripts
When adding smoke tests, diagnostics, or repair utilities, prefer creating a new script under the `scripts/` directory (for example `scripts/check_my_fix.py`) rather than modifying production generator files. This keeps the main generator stable and makes CI reviews simpler.

### Race Control Messages and DNF Counting

**Data Pull**: `f1-raceMessages.py` incrementally pulls race control messages from FastF1 API (2018-present) and tracks previously processed sessions by (Year, Round) to avoid redundant API calls.

**DNF Counting Fix (Dec 2025)**: The DNF summary logic was corrected to:
- Exclude "0" as a retirement reason ("0" means "Finished", not DNF)
- Deduplicate by race and driver to avoid counting practice session duplicates

### Practice File Delimiter Standardization

**Delimiter Fix (Dec 2025)**: Fixed "practice_best is missing 'raceId' and/or 'driverId'" warning by standardizing practice file delimiters:
- Converted all `practice_best_by_session*.csv` files from comma to tab delimiters
- All CSVs in workspace now consistently use tab (`\t`) as separator
- Matches workspace standard used throughout the project

### Driver Team Assignment Updates

**Automatic Team Updates (Dec 2025)**: The generator now automatically updates driver constructor assignments for the current year after the main CSV is written. This happens in `f1-generate-analysis.py` (lines 2773-2832) and:
- Reads the latest race results from `f1db-races-race-results.json` for the current year
- Identifies each driver's most recent constructor assignment (sorted by raceId descending)
- Updates `f1ForAnalysis.csv` rows where `grandPrixYear == current_year` with latest team information
- Logs affected drivers and row counts to the console
- Runs automatically before smoke checks (no manual intervention needed)

This prevents drivers from showing outdated team information due to mid-season transfers or data staleness. Historical data (previous years) is not modified to preserve integrity.
- Count only actual retirements (Accident, mechanical failures, etc.)

This fix resolves inflated DNF counts (e.g., Australian GP 2025 showed 57 instead of the correct 6).

## Deployment and caching
The app is deployed on [Streamlit Cloud](https://f1analysis-app.streamlit.app/) and uses aggressive caching with `@st.cache_data` decorators to improve performance. 

### Cache Management
- **CACHE_VERSION**: The app uses a version-based cache invalidation system (`CACHE_VERSION="v2.3"`) to ensure cached data and models remain consistent across deployments
- **Cache Dependencies**: All cached functions include `CACHE_VERSION` as a dependency to force cache invalidation when the version changes
- **Model Caching**: ML models are cached with preprocessing pipelines to prevent feature shape mismatches between training and prediction

### Deployment Notes
- **Streamlit Cloud Compatibility**: The app is optimized for Streamlit Cloud deployment with proper cache management to prevent stale cached models
- **Environment Variables**: Set `LOCAL_RUN=1` for local development to enable FastF1 caching
- **Model Compatibility**: Supports XGBoost, LightGBM, CatBoost, and Ensemble models with model-specific parameter handling

## File organization
There are two python files involved in this app: ```raceAnalysis.py``` and ```f1-generate-analysis.py```, though there are other python files which generate content. The Race Analysis file is what runs the Streamlit code and displays the data, filters, charts, etc. Before that file is run, you need to run the Generate Analysis page. This creates a bunch of dataframes, and it creates several .csv files for easier retrievel during the Streamlit display. This is done so fewer calculations are required in the Streamlit app which should improve performance. However, it does require that you run the ```f1-generate-analysis.py``` before you run the Steamlit.

The CSV files and any associated .json files are included in the ```data_files``` directory. The ```.json``` files come from [F1DB](https://github.com/f1db/f1db). The following files are generated and then copied into the ```data_files``` directory:

1. ```f1WeatherData_Grouped.csv```
2. ```f1PitStopsData_Grouped.csv```
3. ```f1ForAnalysis.csv```
4. ```f1db-races.json```
5. ```f1db-drivers.json```
6. ```f1db-grands-prix.json```
7. ```f1db-races-race-results.json```

[↑ Back to top](#table-of-contents)

## Download UI & Icons

- **Download icons**: The Streamlit UI now displays a small clickable icon next to many download actions. The app prefers `data_files/csv_icon.png` and `data_files/pdf_icon.png` and falls back to `data_files/favicon.png` when those are missing.
- **Position Analysis**: The Position Group Analysis tab uses HTML-rendered, clickable icon buttons (icon + label). For small summary files the app embeds the file using a base64 data-URI so the download link is compact and self-contained; for larger artifacts the UI falls back to Streamlit's native `st.download_button` to stream files without inlining them into the page.
- **If you add assets**: Place your icons in `data_files/` with the names above; no code change is required.

## Filtering
There are currently more than 30 ways to filter the F1 data which spans from 2015 to present. You can filter by one or all of the data fields on the left side of the page. The data dynamically updates and gives you a new total record count.

[↑ Back to top](#table-of-contents)

## Developer notes (local)

- The Streamlit sidebar now displays friendly labels for fields by default (from `column_rename_for_filter`). The raw field names used for debugging were previously shown in parentheses; that debug label has been commented out in `raceAnalysis.py` to keep the UI cleaner.
- A temporary debug artifact `data_files/sidebar_exclusion_debug.json` (used to diagnose why some fields appeared in the sidebar) was removed from the repository after the issue was fixed.
- To re-enable raw-field debug labels, edit `raceAnalysis.py` and restore the commented `display_label = f"{column_friendly_name} ({column})"` line and optionally populate the `debug_fields` list near the sidebar code.


## Linear regression
In addition to correlation coefficients, I have added several linear regressions to help predict the results of the next race.

[↑ Back to top](#table-of-contents)

## Predictive Data Modeling
I used [scikit-learn](https://scikit-learn.org/stable/) to perform machine learning by using data points to predict the race winner. ~~The model is in its infancy, and I am still trying to figure out the right data points to feed it.~~ I'm also currently trying to predict a driver's final place rather than their final time. That means that the [Mean Absolute Error](https://www.sciencedirect.com/topics/engineering/mean-absolute-error) relates to finisher placement which feels less exact than what I need. 

After significant refinement with [Monte Carlo](https://www.ibm.com/think/topics/monte-carlo-simulation), [Recursive Feature Elimination (RFE)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html), and [Boruta](https://www.jstatsoft.org/v36/i11/) feature selection, I have achieved a MAE of **1.5 or below**.

### Multiple Model Support
The app now supports four different machine learning algorithms for maximum flexibility and accuracy:

- **🏆 XGBoost** (Recommended Default): Excellent performance, handles missing data, built-in feature importance
- **🚀 LightGBM**: Very fast training, good for large datasets, handles categorical features well  
- **🐱 CatBoost**: Excellent with categorical data, robust to overfitting, handles missing values automatically
- **🎯 Ensemble (XGBoost + LightGBM + CatBoost)**: Combines all three models for maximum prediction accuracy

### Advanced Hyperparameter Optimization
- **Bayesian Optimization**: Intelligent parameter tuning using Optuna
- **Grid Search**: Systematic parameter exploration
- **Season-Stratified Cross-Validation**: Prevents data leakage across racing seasons
- **Early Stopping**: Automatic training termination to prevent overfitting

### Model Selection Guide
Choose the right model based on your needs:

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| XGBoost | Fast | High | General-purpose predictions |
| LightGBM | Very Fast | High | Large datasets, speed-critical |
| CatBoost | Moderate | High | Categorical data, robustness |
| Ensemble | Slow | Highest | Maximum accuracy, final predictions |

### Advanced Options Structure (Tab 5)

The predictive modeling interface has been refactored into a streamlined, organized structure:

**🔧 Advanced Options** (Single Expander with 6 Tabs)

1. **📊 Model Performance**
   - Core metrics (MSE, R², MAE, Mean Error, boosting rounds)
   - Driver error statistics
   - MAE by position groups (Winner, Top3, Top10, Mid, Back, Bottom10)
   - Individual position MAE analysis (positions 1-20)
   - Error distribution visualization

2. **🔍 Feature Analysis**
   - Permutation importance (most/least helpful features)
   - High-cardinality features analysis (overfitting risk assessment)
   - Safety Car feature importance
   - Correlation matrix for all races

3. **🎯 Feature Selection**
   - Monte Carlo feature subset search (1000+ iterations)
   - Recursive Feature Elimination (RFE)
   - Boruta feature selection
   - RFE to minimize MAE
   - All tools are button-controlled to prevent accidental slow operations

4. **⚙️ Hyperparameters**
   - Early stopping details with visualization
   - MAE per round line chart
   - Top 50 feature importances after training
   - Hyperparameter tuning tool

5. **📈 Historical Validation**
   - Cross-validation metrics (Final Position, DNF, Safety Car models)
   - Model accuracy across all races
   - Scatter plot: Actual vs Predicted positions
   - Race-by-race performance breakdown

6. **🛠️ Debug & Experiments**
   - Bin count comparison (q-value experimentation)
   - Experimental features and debugging tools

This new structure reduces visual clutter by ~90% (1 expander vs 17 checkboxes) while organizing related features logically and improving navigation.

### Streamlit Chart Parameters

The application uses different chart width parameters depending on the chart type:

- **Native Streamlit Charts** (`st.bar_chart`, `st.line_chart`, `st.scatter_chart`):
  - Use `width='stretch'` for full-width display
  - Use `width='content'` for auto-sizing based on content
  
- **Altair/Vega Charts** (`st.altair_chart`):
  - Use `width='stretch'` for full-width display
  - Use `width='content'` for default sizing
  - Note: Prefer `width` over `use_container_width`; do NOT use `use_container_width`

### Streamlit DataFrame Height Optimization

To prevent scrolling in dataframes, use the `get_dataframe_height()` helper function:

```python
# Automatically calculate height based on number of rows
height = get_dataframe_height(df)
st.dataframe(df, height=height)

# With custom max height
height = get_dataframe_height(df, max_height=400)
st.dataframe(df, height=height)

# No height limit
height = get_dataframe_height(df, max_height=None)
st.dataframe(df, height=height)
```

**Function parameters:**
- `df`: The DataFrame to display (required)
- `row_height`: Pixels per row (default: 35)
- `header_height`: Header row height (default: 38)
- `padding`: Extra padding (default: 2)
- `max_height`: Maximum height cap (default: 600, use None for unlimited)
  
- **Altair/Vega Charts** (`st.altair_chart`):
  - Use `width='stretch'` for full-width display
  - Use `width='content'` for default sizing
  - Note: Prefer `width` over `use_container_width`; do NOT use `use_container_width`

[↑ Back to top](#table-of-contents)

## Features used in data model

After extensive feature engineering implementing all roadmap enhancements, Monte Carlo simulation (1000+ iterations), and feature selection (RFE, Boruta), I have achieved a MAE of **1.94** (target: ≤1.5). The model now includes 86+ engineered features, with automatic loading of Monte Carlo-selected best features when available. Feature lists are externalized to text files for easy maintenance and updates.

| Field Name                           | Description                                                                                   |
|-------------------------------------- |----------------------------------------------------------------------------------------------|
| practice_improvement                  | Change in practice position or time, indicating setup progress or adaptation.                 |
| practice_position_improvement_1P_3P   | Improvement in practice position from FP1 to FP3.                                             |
| driverDNFCount                       | Total number of times the driver did not finish (DNF) in their career.                        |
| CleanAirAvg_FP1                      | Driver's average lap time in clean air during FP1.                                            |
| constructor_recent_form_3_races      | Constructor's average finishing position over the last 3 races.                               |
| best_s1_sec                          | Driver's best sector 1 time in qualifying or practice.                                        |
| turns                                | Number of turns on the race circuit.                                                          |
| Delta_FP2                            | Change in lap time or position between FP1 and FP2.                                           |
| practice_time_improvement_2T_3T      | Improvement in practice lap time from FP2 to FP3.                                             |
| practice_position_improvement_2P_3P  | Improvement in practice position from FP2 to FP3.                                             |
| podium_potential                     | Estimated likelihood of finishing on the podium, based on historical and recent data.         |
| totalStopTime                        | Total time spent in pit stops during the race.                                                |
| driverFastestPracticeLap_sec         | Driver's fastest lap time in any practice session.                                            |
| practice_time_improvement_time_time  | General improvement in practice lap times across sessions.                                    |
| track_experience                     | Number of times the driver has raced on this circuit.                                         |
| driverTotalChampionshipWins          | Total number of championship titles won by the driver.                                        |
| qualifying_consistency_std           | Standard deviation of qualifying positions, measuring consistency.                            |
| pit_stop_delta                       | Time difference between the driver's pit stops and the average pit stop time.                 |
| totalChampionshipPoints              | Total championship points scored by the driver.                                               |
| delta_from_race_avg                  | Difference between driver's lap time and the race average lap time.                           |
| average_humidity                     | Average humidity during the race.                                                             |
| driverDNFAvg                         | Average DNF rate for the driver (DNFs per race).                                              |
| SpeedI2_mph                          | Speed at a specific intermediate point (I2) in mph.                                           |
| qualPos_x_last_practicePos           | Interaction between qualifying position and last practice position.                           |
| averageStopTime                      | Average time spent per pit stop.                                                              |
| recent_vs_season                     | Comparison of recent performance to season average.                                           |
| race_pace_vs_median                  | Driver's race pace compared to the median pace of the field.                                  |
| recent_positions_gained_3_races      | Total positions gained over the last 3 races.                                                 |
| driver_dnf_rate_5_races              | Driver's DNF rate over the last 5 races.                                                      |
| driver_positionsGained_3_races       | Total positions gained by the driver over the last 3 races.                                   |
| constructorTotalRaceWins             | Total number of race wins by the constructor.                                                 |
| grid_x_avg_pit_time                  | Interaction between grid position and average pit stop time.                                  |
| driver_starting_position_5_races     | Driver's average starting position over the last 5 races.                                     |
| recent_form_ratio                    | Ratio of recent performance to historical performance.                                        |
| fp1_lap_delta_vs_best                | Difference between FP1 lap time and the best lap time.                                        |
| recent_form_x_qual                   | Interaction between recent form and qualifying position.                                      |
| driver_starting_position_3_races     | Driver's average starting position over the last 3 races.                                     |
| practice_improvement_x_qual          | Interaction between practice improvement and qualifying position.                             |
| grid_penalty_x_constructor           | Interaction between grid penalty and constructor.                                             |
| constructor_recent_form_5_races      | Constructor's average finishing position over the last 5 races.                               |
| driver_rank_x_constructor_rank       | Interaction between driver rank and constructor rank.                                         |
| qualPos_x_avg_practicePos            | Interaction between qualifying position and average practice position.                        |
| constructorTotalRaceStarts           | Total number of race starts by the constructor.                                               |
| historical_avgLapPace                | Driver's historical average lap pace.                                                         |
| grid_penalty                         | Grid penalty applied to the driver for the race.                                              |
| avg_final_position_per_track         | Driver's average final position at this track.                                                |
| best_qual_time                       | Driver's best qualifying lap time.                                                            |
| lastFPPositionNumber                 | Driver's position in the last free practice session.                                          |
| SpeedI1_mph                          | Speed at a specific intermediate point (I1) in mph.                                           |
| driver_positionsGained_5_races       | Total positions gained by the driver over the last 5 races.                                   |
| SafetyCarStatus                      | Indicator if a safety car was deployed in the race.                                           |
| recent_dnf_rate_3_races              | Driver's DNF rate over the last 3 races.                                                      |
| top_speed_rank                       | Driver's rank based on top speed in the race.                                                 |
| LapTime_sec                          | Driver's lap time in seconds.                                                                 |
| teammate_qual_delta                  | Difference in qualifying time between the driver and their teammate.                          |
| recent_form_median_3_races           | Median of driver's finishing positions over the last 3 races.                                 |
| average_wind_speed                   | Average wind speed during the race.                                                           |
| average_temp                         | Average temperature during the race.                                                          |
| Delta_FP3                            | Change in lap time or position between FP2 and FP3.                                           |
| best_s2_sec                          | Driver's best sector 2 time in qualifying or practice.                                        |
| CleanAirAvg_FP2                      | Driver's average lap time in clean air during FP2.                                            |
| constructor_form_ratio               | Ratio of constructor's recent performance to historical performance.                          |
| totalPolePositions                   | Total number of pole positions by the driver or constructor.                                  |
| qual_x_constructor_wins              | Interaction between qualifying position and constructor's total wins.                         |
| numberOfStops                        | Number of pit stops made by the driver in the race.                                           |
| teammate_practice_delta              | Difference in practice lap time between the driver and their teammate.                        |
| best_s3_sec                          | Driver's best sector 3 time in qualifying or practice.                                        |
| last_final_position_per_track_constructor | Constructor's last final position at this track.                                         |
| street_experience                    | Number of street races the driver has participated in.                                        |
| total_experience                     | Total number of races the driver has participated in.                                         |
| best_theory_lap_sec                  | Driver's best theoretical lap time (sum of best sector times).                                |
| Points                               | Points scored by the driver in the current season.                                            |
| recent_form_median_5_races           | Median of driver's finishing positions over the last 5 races.                                 |
| trackRace                            | Indicator if the race is on a permanent circuit.                                              |
| yearsActive                          | Number of years the driver has been active in F1.                                             |
| Delta_FP1                            | Change in lap time or position between FP1 and previous session.                              |
| SpeedST_mph                          | Speed at the speed trap in mph.                                                               |
| avg_final_position_per_track_constructor | Constructor's average final position at this track.                                      |
| recent_form_5_races                  | Driver's average finishing position over the last 5 races.                                    |
| grid_x_constructor_rank              | Interaction between grid position and constructor rank.                                       |
| driverAge                            | Age of the driver.                                                                            |
| streetRace                           | Indicator if the race is on a street circuit.                                                 |
| averagePracticePosition              | Driver's average position across all practice sessions.                                       |
| SpeedFL_mph                          | Speed on the fastest lap in mph.                                                              |
| recent_form_best_3_races             | Best finishing position by the driver in the last 3 races.                                    |
| BestConstructorPracticeLap_sec       | Constructor's best practice lap time.                                                         |
| qualifying_gap_to_pole               | Time gap between the driver and pole position in qualifying.                                  |
| driverTotalRaceStarts                | Total number of race starts by the driver.                                                    |
| practice_time_improvement_1T_3T      | Improvement in practice lap time from FP1 to FP3.                                             |
| practice_time_improvement_1T_2T      | Improvement in practice lap time from FP1 to FP2.                                             |
| constructor_avg_practice_position    | Constructor's average practice position.                                                      |
| qual_vs_track_avg                    | Driver's qualifying position compared to track average.                                       |
| practice_std_x_qual                  | Interaction between practice position standard deviation and qualifying position.              |
| pit_stop_rate                        | Rate of pit stops per race for the driver.                                                    |
| positions_gained_first_lap_pct       | Percentage of positions gained on the first lap.                                              |
| driver_podium_rate_3y                | Driver's podium rate over the last 3 years.                                                   |
| constructor_podium_ratio             | Constructor's podium rate (podiums per entry).                                                |
| driver_age_squared                   | Driver's age squared, capturing nonlinear age effects.                                        |
| constructor_recent_win_streak        | Number of wins by the constructor in the last 3 races.                                        |
| rolling_3_race_win_percentage        | Rolling win percentage over the last 3 races, shifted to avoid data leakage.                   |
| head_to_head_teammate_performance_delta | Historical performance difference between driver and teammate.                               |
| recent_qualifying_improvement_trend  | Trend in qualifying position improvement over recent races.                                   |
| championship_position_pressure_factor | Pressure factor based on driver's current championship position.                              |
| recent_mechanical_dnf_rate_by_constructor | Constructor's mechanical DNF rate in recent races.                                          |
| engine_penalty_impact_on_grid        | Impact of engine penalties on starting grid position.                                         |
| component_age_vs_failure_probability | Probability of component failure based on age.                                               |
| constructor_development_rate         | Rate of mid-season upgrades and development by constructor.                                   |
| driver_historical_performance_at_circuit_types | Driver's performance across different circuit types (street, high-speed, technical).         |
| weather_pattern_analysis_by_circuit  | Weather pattern analysis specific to circuit location.                                        |
| tire_compound_strategy_effectiveness_by_track | Effectiveness of tire compound strategy at specific tracks.                               |
| overtaking_difficulty_index_per_circuit | Index measuring overtaking difficulty at each circuit.                                       |
| q1_q2_q3_sector_time_consistency     | Consistency of sector times across Q1, Q2, Q3 sessions.                                      |
| qualifying_position_vs_race_pace_delta_by_track | Delta between qualifying position and race pace by track.                                 |
| tire_compound_used_in_qualifying_vs_race_start | Comparison of tire compounds used in qualifying vs. race start.                              |
| traffic_impact_on_qualifying_laps    | Impact of traffic on qualifying lap times.                                                    |

[↑ Back to top](#table-of-contents)

## Other options
Besides filtering, you can also look at the upcoming race which shows historical and upcoming weather, the past winners, and data about the constructors. You can view the entire current season with details about each file. You can look at the raw, unfiltered data. Finally, you can view a correlation for the entire dataset.

[↑ Back to top](#table-of-contents)

## Feature Selection Exporter
The repository includes a small exporter that consolidates feature-selection artifacts into easy-to-share outputs.

- Location: `scripts/export_feature_selection.py`
- Inputs: `scripts/output/shap_ranking.txt`, `scripts/output/boruta_selected.txt`, `scripts/output/correlated_pairs.csv`
- Outputs (generated into `scripts/output/`):
   - `feature_selection_summary.csv` — tabular CSV summary with SHAP, Boruta selection, and correlation info
   - `feature_selection_report.html` — a simple HTML report with top SHAP features and correlation summaries

The Streamlit UI (`raceAnalysis.py`) has been wired to surface these artifacts under the Feature Selection tab: users can download the CSV and the HTML report directly from the app and run a one-click "Regenerate CSV/HTML exporters" action to recreate them from the latest artifacts. The SHAP reader in the app is robust to minor format changes and will fall back to a safe plain-text parse if a structured read fails.

Note: PDF export is intentionally not included to avoid introducing native binary dependencies into the repo; the HTML and CSV files are intended as portable, reviewable artifacts.

## Position Group Analysis (new)

There is a position-group analysis utility that produces diagnostic CSVs, plots, and a small HTML report in `scripts/output/`.

- Run it manually:

```pwsh
python .\scripts\position_group_analysis.py
```

- Outputs written to `scripts/output/` include (current names):
   - `mae_by_season.csv` and `mae_trends.png`
   - `confid_int_by_driver_track.csv`, `confid_int_by_driver.csv`, `confid_int_by_constructor.csv` (95% empirical residual intervals)
   - `heatmap_driver_by_circuit.png`, `heatmap_constructor_by_circuit.png`
   - `position_group_analysis_report.html`

- Notes:
   - The Streamlit UI now renders the Position Analysis natively (no iframe), showing plots and an informational shaded box explaining how to read the heatmaps.
   - The confidence-interval CSV files use an empirical percentile method (2.5th/97.5th percentiles) and are computed only for groups with at least 5 residuals; groups with fewer samples will contain NaNs.


## Weather
The weather is pulled from [Open-Meteo's free API](https://open-meteo.com/) which allows you to search historical weather data by hour going back to the 1940s. The hourly reports are pulled per race and then averaged to show a daily weather report on race day.

[↑ Back to top](#table-of-contents)

### To do
- ~~Figure out a way to reset the filters.~~
- ~~Incorporate the linear regression equations for predictive race results.~~

[↑ Back to top](#table-of-contents)

## Preferred precomputed filenames (practice-best)

- **Preferred:** `data_files/practice_best_by_session.imputed.round_filled.csv` — generated by the imputer+round-fill helpers and preferred by the generator when present.
- **Fallbacks:** `data_files/practice_best_by_session.imputed.csv`, `data_files/practice_best_by_session.csv`, then legacy `practice_best_fp1_fp2.csv` via the generator's `get_preferred_file()` logic.

Usage note: If you run the missing-data helpers (`scripts/impute_missing_practice.py` and `scripts/aggregate_practice_laps.py`) they will produce the `practice_best_by_session.*` variants; `f1-generate-analysis.py` will prefer the `round_filled` variant when available and will also attempt to backfill missing `round` values from `data_files/f1db-races.json`.

## Diagnostics & Temporal-Leakage Audit

- **Audit script:** `scripts/audit_temporal_leakage.py` — conservative automated checks to surface potential future-looking features, practice/qualifying timing on scheduled races, and suspicious SafetyCar/leader columns. This audit is integrated into the unified smoke-runner but is intentionally conservative and intended to flag items for manual review.
- **Quick diagnostic:** `scripts/check_points_leader_gap.py` — a lightweight diagnostic that verifies `round` population and sanity-checks `points_leader_gap` per race snapshot (non-negative and at least one zero-gap leader). Keep this in the repo as a manual troubleshooting helper.

Usage:
```pwsh
# Run the full audit (also executed by the unified smoke-runner)
python .\scripts\audit_temporal_leakage.py

# Run the quick points-leader diagnostic
python .\scripts\check_points_leader_gap.py
```

Note: The generator was updated to compute `points_leader_gap` using a per-race snapshot (grouped by `grandPrixYear` + `round`/`raceId`/`short_date`) to avoid leaking future race information into earlier rows.
 
## Dec 2025: Repair utilities & smoke test

Recent work added a small set of diagnostics and repair helpers to stabilize the race-control message pipeline and provide a lightweight smoke test for local validation:

- `scripts/check_unmapped_racemessages.py`: Detect rows in `data_files/all_race_control_messages.csv` that are missing canonical identifiers (for example when a prior ad-hoc merge produced `raceId_x` / `raceId_y`). Writes a small sample for inspection.
- `scripts/repair_unmapped_racemessages.py`: Backups the original combined CSV (timestamped backup) and coalesces suffixed identifier columns into canonical single columns suitable for grouping.
- `scripts/sanity_check_race_control_grouped.py`: Quick sanity check for the grouped `data_files/race_control_messages_grouped_with_dnf.csv` (row counts, missing ids, DNF stats).
- `scripts/smoke_run_core.py`: Lightweight import-safe smoke test that runs `headless_predict_and_write` helpers, validates repaired message files, writes a headless predictions TSV, and reads it back.

How to run the new smoke test (from repo root):

```pwsh
python .\scripts\smoke_run_core.py
```

If your `data_files/all_race_control_messages.csv` was previously merged and contains suffixed columns, run the repair helper first:

```pwsh
# backup will be written automatically by the script (timestamped filename)
python .\scripts\repair_unmapped_racemessages.py
# quick diagnostic
python .\scripts\check_unmapped_racemessages.py
# sanity-check grouped output
python .\scripts\sanity_check_race_control_grouped.py
```

After repair, `f1-raceMessages.py` will produce `data_files/race_control_messages_grouped_with_dnf.csv` for downstream analytics.
