# Missing-data helper scripts

This folder contains small, conservative scripts to help handle missing or
incomplete artifacts before running the main generator `f1-generate-analysis.py`.

Scripts
- `impute_missing_practice.py` — Impute numeric practice/session fields.
  - Methods: `season_mean`, `median_by_driver`, `last_race`
  - Usage: `python scripts/impute_missing_practice.py --input data_files/all_practice_results.csv`

- `handle_sprint_weekends.py` — Tag sprint weekends and normalize qualifying type.
  - Usage: `python scripts/handle_sprint_weekends.py --input data_files/all_practice_results.csv`

- `fill_weather_gaps.py` — Interpolate hourly weather and fill remaining gaps.
  - Usage: `python scripts/fill_weather_gaps.py --hourly data_files/f1WeatherData_AllData.csv --grouped data_files/f1WeatherData_Grouped.csv`

Integration
- The main generator calls these scripts early in the pipeline (best-effort).
- Each script is conservative and does not alter schema or invent new driver ids.

Notes
- The scripts are intentionally simple and local-only. If you want automatic
  Open-Meteo backfills or more aggressive imputation, we can extend them.
