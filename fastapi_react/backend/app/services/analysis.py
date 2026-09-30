from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress

from app.config import DATA_DIR
from app.services.data import (
    apply_filters,
    load_main_data,
    load_race_schedule,
    model_manifest,
    precomputed,
    records,
)


def _regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any] | None:
    if x_col not in df or y_col not in df:
        return None
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    if not x.notna().any() or not y.notna().any():
        return None
    x = x.fillna(x.mean())
    y = y.fillna(y.mean())
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    slope, intercept, r, p, stderr = linregress(x[mask], y[mask])
    return {
        "x": x_col, "y": y_col, "slope": float(slope), "intercept": float(intercept),
        "r_squared": float(r ** 2), "p_value": float(p), "std_err": float(stderr),
    }


def _regression_series(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any] | None:
    if x_col not in df or y_col not in df:
        return None
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    if not x.notna().any() or not y.notna().any():
        return None
    points = pd.DataFrame({x_col: x.fillna(x.mean()), y_col: y.fillna(y.mean())})
    if len(points) < 2:
        return None
    slope, intercept, _, _, _ = linregress(points[x_col], points[y_col])
    endpoints = np.linspace(float(points[x_col].min()), float(points[x_col].max()), num=60)
    return {
        "x": x_col,
        "y": y_col,
        "points": records(points),
        "fit": records(pd.DataFrame({x_col: endpoints, y_col: slope * endpoints + intercept})),
    }


def analytics(filters: Any, max_rows: int) -> dict[str, Any]:
    """Return every lightweight analysis block rendered by the Streamlit Analytics tab."""
    df = apply_filters(load_main_data(), filters).head(max_rows).copy()
    payload: dict[str, Any] = {"rows_considered": len(df), "charts": {}, "regressions": []}
    if df.empty:
        return payload

    pairs = {
        "active_years_vs_final": ("resultsFinalPositionNumber", "yearsActive"),
        "positions_gained_over_time": ("short_date", "positionsGained"),
        "practice_vs_final": ("lastFPPositionNumber", "resultsFinalPositionNumber"),
        "grid_vs_final": ("resultsStartingGridPositionNumber", "resultsFinalPositionNumber"),
        "avg_practice_vs_final": ("averagePracticePosition", "resultsFinalPositionNumber"),
        "pit_stop_vs_final": ("averageStopTime", "resultsFinalPositionNumber"),
        "track_turns_vs_final": ("turns", "resultsFinalPositionNumber"),
    }
    for name, (x, y) in pairs.items():
        if x in df and y in df:
            payload["charts"][name] = records(df[[x, y]].dropna().head(5000))

    for x in ("averagePracticePosition", "resultsStartingGridPositionNumber"):
        result = _regression(df, x, "resultsFinalPositionNumber")
        if result:
            payload["regressions"].append(result)
    regression_titles = {
        "averagePracticePosition": (
            "Linear Regression: Average Practice Position vs Final Position",
            "Average Practice Position",
        ),
        "resultsStartingGridPositionNumber": (
            "Linear Regression: Starting Position vs. Final Position",
            "Starting Position",
        ),
    }
    payload["regression_series"] = []
    for x, (title, x_label) in regression_titles.items():
        series = _regression_series(df, x, "resultsFinalPositionNumber")
        if series:
            series.update(title=title, x_label=x_label, y_label="Final Position")
            payload["regression_series"].append(series)

    corr_cols = [c for c in (
        "lastFPPositionNumber", "resultsFinalPositionNumber", "resultsStartingGridPositionNumber",
        "grandPrixLaps", "averagePracticePosition", "DNF", "resultsTop10", "resultsTop5",
        "resultsPodium", "streetRace", "trackRace", "constructorTotalRaceStarts",
        "constructorTotalRaceWins", "constructorTotalPolePositions", "turns", "positionsGained",
        "q1End", "q2End", "q3Top10", "driverBestStartingGridPosition", "yearsActive",
        "driverBestRaceResult", "driverTotalChampionshipWins", "driverTotalPolePositions",
        "driverTotalRaceEntries", "driverTotalRaceStarts", "driverTotalRaceWins",
        "driverTotalRaceLaps", "driverTotalPodiums", "avgLapPace", "finishingTime",
        "resultsQualificationPositionNumber", "numberOfStops",
    ) if c in df]
    if corr_cols:
        corr = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        payload["correlation"] = {
            "columns": list(corr.columns),
            "rows": [
                {"Feature": idx, **{col: (None if pd.isna(v) else float(v)) for col, v in row.items()}}
                for idx, row in corr.iterrows()
            ],
        }

    if {"grandPrixYear", "resultsDriverName", "resultsFinalPositionNumber"}.issubset(df.columns):
        agg: dict[str, tuple[str, Any]] = {
            "average_final_position": ("resultsFinalPositionNumber", "mean"),
        }
        if "resultsPodium" in df:
            agg["total_podiums"] = ("resultsPodium", "sum")
        driver = df.groupby(["grandPrixYear", "resultsDriverName"]).agg(**agg).reset_index()
        payload["driver_performance"] = records(driver)

    if {"grandPrixYear", "constructorName", "resultsFinalPositionNumber"}.issubset(df.columns):
        constructor = (
            df.groupby(["grandPrixYear", "constructorName"])
            .agg(
                total_wins=("resultsFinalPositionNumber", lambda values: int((values == 1).sum())),
                average_final_position=("resultsFinalPositionNumber", "mean"),
            ).reset_index()
        )
        if "resultsPodium" in df:
            podium = (
                df.groupby(["grandPrixYear", "constructorName"])["resultsPodium"]
                .sum().reset_index(name="total_podiums")
            )
            constructor = constructor.merge(podium, on=["grandPrixYear", "constructorName"], how="left")
        payload["constructor_performance"] = records(constructor)

    if {"constructorName", "resultsDriverName", "positionsGained", "resultsFinalPositionNumber"}.issubset(df.columns):
        driver_vs_constructor = (
            df.groupby(["constructorName", "resultsDriverName"])
            .agg(
                positionsGained=("positionsGained", "sum"),
                average_final_position=("resultsFinalPositionNumber", "mean"),
            )
            .reset_index()
            .sort_values("average_final_position")
        )
        driver_vs_constructor["average_final_position"] = driver_vs_constructor["average_final_position"].round(2)
        payload["driver_vs_constructor"] = records(driver_vs_constructor)

    dnf_rows = pd.DataFrame()
    if "DNF" in df:
        dnf_rows = df[pd.to_numeric(df["DNF"], errors="coerce").fillna(0).eq(1)].copy()
    if not dnf_rows.empty and "resultsReasonRetired" in dnf_rows:
        dnf = (
            dnf_rows.groupby("resultsReasonRetired").size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        payload["dnf_reasons"] = records(dnf)
    if not dnf_rows.empty and {"resultsDriverName", "driverTotalRaceEntries"}.issubset(dnf_rows.columns):
        grouped = (
            dnf_rows.groupby(["resultsDriverName", "driverTotalRaceEntries"]).size()
            .reset_index(name="dnf_count")
        )
        entries = pd.to_numeric(grouped["driverTotalRaceEntries"], errors="coerce")
        grouped["dnf_pct"] = (grouped["dnf_count"] / entries * 100).round(1)
        payload["dnf_by_driver"] = records(grouped.sort_values("dnf_pct", ascending=False))
    if "grandPrixName" in df:
        entries = df.groupby("grandPrixName").size().reset_index(name="race_entry_count")
        if not dnf_rows.empty:
            dnfs = dnf_rows.groupby("grandPrixName").size().reset_index(name="dnf_count")
            entries = entries.merge(dnfs, on="grandPrixName", how="left")
        else:
            entries["dnf_count"] = 0
        entries["dnf_count"] = entries["dnf_count"].fillna(0).astype(int)
        entries["dnf_pct"] = (entries["dnf_count"] / entries["race_entry_count"] * 100).round(1)
        payload["dnf_by_race"] = records(entries.sort_values("dnf_pct", ascending=False))
    if "constructorName" in df:
        entries = df.groupby("constructorName").size().reset_index(name="constructor_entry_count")
        if not dnf_rows.empty:
            dnfs = dnf_rows.groupby("constructorName").size().reset_index(name="dnf_count")
            entries = entries.merge(dnfs, on="constructorName", how="left")
        else:
            entries["dnf_count"] = 0
        entries["dnf_count"] = entries["dnf_count"].fillna(0).astype(int)
        entries["dnf_pct"] = (
            entries["dnf_count"] / entries["constructor_entry_count"] * 100
        ).round(1)
        payload["dnf_by_constructor"] = records(entries.sort_values("dnf_pct", ascending=False))

    if {"grandPrixYear", "resultsDriverName", "positionsGained", "resultsPodium"}.issubset(df.columns):
        year = int(pd.to_numeric(df["grandPrixYear"], errors="coerce").max())
        season = (
            df[pd.to_numeric(df["grandPrixYear"], errors="coerce") == year]
            .groupby("resultsDriverName")
            .agg(positions_gained=("positionsGained", "sum"), total_podiums=("resultsPodium", "sum"))
            .reset_index()
        )
        payload["season_year"] = year
        payload["season_summary"] = records(season)

    if {"resultsDriverName", "resultsFinalPositionNumber"}.issubset(df.columns):
        consistency = (
            df.groupby("resultsDriverName")
            .agg(finishing_position_std=("resultsFinalPositionNumber", "std"))
            .reset_index()
            .sort_values("finishing_position_std")
        )
        payload["driver_consistency"] = records(consistency)

    try:
        manifest = model_manifest("XGBoost")
    except (KeyError, OSError, ValueError):
        manifest = None
    payload["model_summary"] = manifest

    try:
        importance = precomputed("permutation")
    except (KeyError, OSError, ValueError):
        importance = None
    payload["feature_importance"] = importance
    return payload

def current_season() -> dict[str, Any]:
    schedule = load_race_schedule().copy()
    if "year" not in schedule:
        return {"year": None, "rows": [], "columns": []}
    year = int(pd.to_numeric(schedule["year"], errors="coerce").max())
    current = schedule[pd.to_numeric(schedule["year"], errors="coerce") == year].copy()
    sort = [c for c in ("round", "date") if c in current]
    if sort:
        current = current.sort_values(sort)

    # Preserve the Streamlit current-season next-race cue in API data.
    date_col = next((c for c in ("date", "raceDate", "short_date") if c in current.columns), None)
    current["seasonStatus"] = "Upcoming"
    if date_col:
        dates = pd.to_datetime(current[date_col], errors="coerce")
        today = pd.Timestamp.now().normalize()
        current.loc[dates < today, "seasonStatus"] = "Completed"
        future = dates[dates >= today]
        if not future.empty:
            next_idx = future.idxmin()
            current.loc[next_idx, "seasonStatus"] = "Next Race"
    return {"year": year, "rows": records(current), "columns": list(current.columns)}


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        if len(frame.columns) == 1:
            frame = pd.read_csv(path, low_memory=False)
        return frame
    except Exception:
        try:
            return pd.read_json(path)
        except Exception:
            return pd.DataFrame()


@lru_cache(maxsize=2)
def _load_tire_strategy(path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path_text, sep="\t", low_memory=False)


def tire_strategy(year: int | None = None, event_name: str | None = None) -> dict[str, Any]:
    """Return Streamlit-equivalent compound, degradation, and yearly tire summaries."""
    tire_path = DATA_DIR / "tire_strategy_data.csv"
    if not tire_path.is_file():
        return {"years": [], "events": [], "race_rows": [], "historical_rows": []}
    tire = _load_tire_strategy(str(tire_path), tire_path.stat().st_mtime_ns)
    if tire.empty or not {"year", "event_name", "driver"}.issubset(tire.columns):
        return {"years": [], "events": [], "race_rows": [], "historical_rows": []}

    years = sorted(
        (int(value) for value in pd.to_numeric(tire["year"], errors="coerce").dropna().unique()),
        reverse=True,
    )
    selected_year = int(year) if year in years else (years[0] if years else None)
    year_frame = tire[pd.to_numeric(tire["year"], errors="coerce") == selected_year].copy()
    events = sorted(year_frame["event_name"].dropna().astype(str).unique())
    selected_event = event_name if event_name in events else (events[0] if events else None)

    name_map: dict[str, str] = {}
    main_data = load_main_data()
    if {"abbreviation", "resultsDriverName"}.issubset(main_data.columns):
        names = main_data[["abbreviation", "resultsDriverName"]].dropna().drop_duplicates()
        name_map = names.set_index("abbreviation")["resultsDriverName"].to_dict()

    race_rows: list[dict[str, Any]] = []
    historical_rows: list[dict[str, Any]] = []
    if selected_event is not None:
        selected = year_frame[year_frame["event_name"].astype(str) == selected_event].copy()
        selected["driver"] = selected["driver"].map(name_map).fillna(selected["driver"])
        display_names = {
            "driver": "Driver", "starting_compound": "Start Compound", "num_stints": "Stints",
            "avg_stint_length": "Avg Stint (laps)", "max_stint_length": "Max Stint (laps)",
            "soft_ratio": "Soft Lap %", "used_soft": "Used Soft", "used_medium": "Used Medium",
            "used_hard": "Used Hard", "avg_tire_degradation_sec": "Avg Deg (s/lap)",
            "total_laps": "Laps",
        }
        available = [column for column in display_names if column in selected]
        display = selected[available].rename(columns=display_names)
        if "Soft Lap %" in display:
            display["Soft Lap %"] = (pd.to_numeric(display["Soft Lap %"], errors="coerce") * 100).round(1)
        if "Avg Stint (laps)" in display:
            display["Avg Stint (laps)"] = pd.to_numeric(display["Avg Stint (laps)"], errors="coerce").round(1)
        if "Avg Deg (s/lap)" in display:
            display["Avg Deg (s/lap)"] = pd.to_numeric(display["Avg Deg (s/lap)"], errors="coerce").round(3)
            display = display.sort_values("Avg Deg (s/lap)", na_position="last")
        race_rows = records(display)

    if selected_year is not None and "avg_tire_degradation_sec" in year_frame:
        yearly = year_frame.copy()
        yearly["driver"] = yearly["driver"].map(name_map).fillna(yearly["driver"])
        aggregations: dict[str, tuple[str, str]] = {
            "avg_deg": ("avg_tire_degradation_sec", "mean"),
            "races": ("event_name", "count"),
        }
        if "num_stints" in yearly:
            aggregations["avg_stints"] = ("num_stints", "mean")
        if "soft_ratio" in yearly:
            aggregations["soft_pct"] = ("soft_ratio", "mean")
        summary = yearly.groupby("driver").agg(**aggregations).reset_index()
        summary = summary.rename(columns={
            "driver": "Driver", "avg_deg": "Avg Deg (s/lap)", "avg_stints": "Avg Stints",
            "soft_pct": "Soft Lap %", "races": "Races",
        })
        summary["Avg Deg (s/lap)"] = summary["Avg Deg (s/lap)"].round(3)
        if "Avg Stints" in summary:
            summary["Avg Stints"] = summary["Avg Stints"].round(2)
        if "Soft Lap %" in summary:
            summary["Soft Lap %"] = (summary["Soft Lap %"] * 100).round(1)
        historical_rows = records(summary.sort_values("Avg Deg (s/lap)", na_position="last"))

    degradation_rows = [
        {"driver": row["Driver"], "degradation": row.get("Avg Deg (s/lap)")}
        for row in race_rows if row.get("Avg Deg (s/lap)") is not None
    ]
    return {
        "years": years,
        "events": events,
        "selected_year": selected_year,
        "selected_event": selected_event,
        "race_rows": race_rows,
        "degradation_rows": degradation_rows,
        "historical_rows": historical_rows,
    }


def fastest_pit_stops(race_id: str) -> dict[str, Any]:
    """Return the quickest prior constructor stops at a Grand Prix and stationary time."""
    pit_path = DATA_DIR / "f1db-races-pit-stops.json"
    if not pit_path.is_file():
        return {"rows": [], "total": 0, "pit_lane_time_constant": None}
    try:
        stops = pd.read_json(pit_path)
    except (ValueError, OSError):
        stops = pd.DataFrame()
    if stops.empty or not {"raceId", "constructorId", "timeMillis"}.issubset(stops.columns):
        return {"rows": [], "total": 0, "pit_lane_time_constant": None}
    stops = stops[pd.to_numeric(stops["year"], errors="coerce") >= 2018].copy()
    schedule = load_race_schedule()
    if not {"id", "grandPrixId"}.issubset(schedule.columns):
        return {"rows": [], "total": 0, "pit_lane_time_constant": None}
    race_map = schedule[["id", "grandPrixId"]].drop_duplicates()
    stops = stops.merge(race_map, left_on="raceId", right_on="id", how="left")
    prior = stops[stops["grandPrixId"].astype(str) == str(race_id)].copy()
    prior["timeMillis"] = pd.to_numeric(prior["timeMillis"], errors="coerce")
    prior = prior.dropna(subset=["timeMillis"])
    if prior.empty:
        return {"rows": [], "total": 0, "pit_lane_time_constant": None}

    fastest = prior.loc[prior.groupby(["raceId", "constructorId"])["timeMillis"].idxmin()].copy()
    fastest["pitStopSeconds"] = (fastest["timeMillis"] / 1000).round(3)
    names = load_main_data()
    if {"constructorId_results", "constructorName"}.issubset(names.columns):
        constructor_names = names[["constructorId_results", "constructorName"]].dropna().drop_duplicates()
        fastest = fastest.merge(
            constructor_names, left_on="constructorId", right_on="constructorId_results", how="left"
        )
    if "constructorName" not in fastest:
        fastest["constructorName"] = fastest["constructorId"]

    race_rows = names[names["grandPrixRaceId"].astype(str) == str(race_id)]
    constant = None
    if "pit_lane_time_constant" in race_rows:
        values = pd.to_numeric(race_rows["pit_lane_time_constant"], errors="coerce").dropna()
        if not values.empty:
            constant = float(values.iloc[0])
    fastest["pit_time_stationary"] = (fastest["pitStopSeconds"] - constant).round(3) if constant is not None else None
    columns = [column for column in (
        "year", "round", "constructorName", "lap", "pitStopSeconds", "pit_time_stationary"
    ) if column in fastest]
    fastest = fastest.sort_values([column for column in ("year", "pitStopSeconds") if column in fastest], ascending=[False, True])
    return {
        "rows": records(fastest[columns]),
        "total": len(fastest),
        "pit_lane_time_constant": constant,
    }


def find_prediction_artifact(
    race_id: str,
    year: str,
    race_name: str,
    expected_date: Any | None = None,
) -> dict[str, Any] | None:
    """Select the best committed next-race prediction artifact.

    The current precompute workflow writes JSON with predictions_by_model, while
    older/headless paths may write CSV. Both are supported.
    """
    import json

    candidates = []
    expected_day = pd.to_datetime(expected_date, errors="coerce")
    for directory in (DATA_DIR / "precomputed" / "predictions", DATA_DIR):
        if not directory.exists():
            continue
        for path in list(directory.glob("*.json")) + list(directory.glob("*.csv")):
            low = path.name.lower()
            if "prediction" not in low:
                continue
            terms = [
                race_id.lower(), year.lower(), race_name.lower().replace(" ", "_"),
                race_name.lower().replace(" ", "-"),
            ]
            score = sum(20 for term in terms if term and term in low)
            payload = None
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                artifact_date = pd.to_datetime(
                    (payload.get("metadata") or {}).get("next_race", {}).get("date"),
                    errors="coerce",
                )
                if pd.notna(expected_day) and pd.notna(artifact_date) and expected_day.date() == artifact_date.date():
                    score += 100
                score += 10 * len(payload.get("predictions_by_model") or {})
            candidates.append((score, path.stat().st_mtime_ns, path.name, path, payload))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, _, chosen, payload = candidates[0]
    relative = chosen.relative_to(DATA_DIR).as_posix()

    if chosen.suffix.lower() == ".json":
        by_model = payload.get("predictions_by_model") if isinstance(payload, dict) else None
        return {
            "file": relative,
            "format": "json",
            "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
            "predictions_by_model": by_model or {},
        }

    frame = _read_optional(chosen)
    return {
        "file": relative,
        "format": "csv",
        "columns": list(frame.columns),
        "rows": records(frame.head(1000)),
    }


def _legacy_prediction_rows(race_id: str, year: int | str, race_name: str) -> list[dict[str, Any]]:
    """Return the committed Streamlit-style CSV prediction rows when an exact race artifact exists."""
    slugs = {
        str(race_id).strip().lower().replace("_", "-").replace(" ", "-"),
        str(race_name).lower().replace(" grand prix", "").replace(" ", "-"),
    }
    candidates: list[Path] = []
    for slug in sorted(slugs):
        if slug:
            candidates.extend([
                DATA_DIR / f"predictions_{slug}_{year}.csv",
                DATA_DIR / f"predictions_{slug.replace('-', '_')}_{year}.csv",
            ])
    for candidate in candidates:
        if candidate.is_file():
            frame = _read_optional(candidate)
            if not frame.empty:
                sort_col = "Rank" if "Rank" in frame else (
                    "PredictedFinalPosition" if "PredictedFinalPosition" in frame else None
                )
                if sort_col:
                    frame = frame.sort_values(sort_col)
                return records(frame)
    return []


@lru_cache(maxsize=1)
def _load_safety_car_inputs() -> pd.DataFrame:
    """Load the same historical safety-car feature frame used by Streamlit."""
    path = DATA_DIR / "f1SafetyCarFeatures.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", low_memory=False)


@lru_cache(maxsize=1)
def _load_safety_car_model() -> Any:
    """Load the trusted, repository-generated safety-car inference artifact."""
    path = DATA_DIR / "models" / "safetycar_model.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted model artifact committed by this repository


def build_safety_car_predictions(
    next_race: pd.Series,
    race_name: str,
    year: int,
    weather: pd.DataFrame,
) -> dict[str, Any]:
    """Mirror Streamlit's historical + synthetic next-race safety-car inference."""
    frame = _load_safety_car_inputs().copy()
    model = _load_safety_car_model()
    if frame.empty or model is None or "SafetyCarStatus" not in frame:
        return {"rows": [], "mean": None, "min": None, "max": None}

    manifest_path = DATA_DIR / "models" / "safetycar_manifest.json"
    if not manifest_path.is_file():
        return {"rows": [], "mean": None, "min": None, "max": None}
    import json

    try:
        feature_names = json.loads(manifest_path.read_text(encoding="utf-8")).get("feature_names", [])
    except (OSError, json.JSONDecodeError):
        feature_names = []
    if not feature_names:
        return {"rows": [], "mean": None, "min": None, "max": None}

    for column in feature_names:
        if column not in frame:
            frame[column] = np.nan
    features = frame[feature_names].copy()
    try:
        probabilities = model.predict_proba(features)[:, 1]
    except Exception:
        return {"rows": [], "mean": None, "min": None, "max": None}

    history = pd.DataFrame({
        "grandPrixName": frame.get("grandPrixName"),
        "grandPrixYear": frame.get("grandPrixYear"),
        "PredictedSafetyCarProbabilityPercentage": (probabilities * 100).round(3),
    })
    history["Type"] = "Historical"

    synthetic: dict[str, Any] = {column: np.nan for column in feature_names}
    synthetic["grandPrixYear"] = year
    synthetic["grandPrixName"] = race_name
    schedule_map = {
        "circuitId": "circuitId",
        "grandPrixLaps": "laps",
        "turns": "turns",
        "streetRace": "streetRace",
        "trackRace": "trackRace",
    }
    for target, source in schedule_map.items():
        if target in synthetic and source in next_race.index and pd.notna(next_race[source]):
            synthetic[target] = next_race[source]

    if not weather.empty:
        weather_row = weather.iloc[0]
        for column in ("average_temp", "average_humidity", "average_wind_speed", "total_precipitation"):
            if column in synthetic and column in weather_row.index and pd.notna(weather_row[column]):
                synthetic[column] = weather_row[column]

    same_gp = frame[frame.get("grandPrixName", pd.Series(index=frame.index, dtype=object)) == race_name]
    for column in feature_names:
        if not pd.isna(synthetic[column]) or column not in frame:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.Series(dtype=float)
            if not same_gp.empty and "grandPrixYear" in same_gp:
                per_race = same_gp.groupby("grandPrixYear")[column].mean(numeric_only=True).dropna()
                if not per_race.empty:
                    values = per_race.sort_index().tail(2)
            synthetic[column] = (
                values.median()
                if not values.empty
                else pd.to_numeric(frame[column], errors="coerce").dropna().median()
            )

    synthetic_frame = pd.DataFrame([synthetic], columns=feature_names)
    try:
        next_probability = float(model.predict_proba(synthetic_frame)[:, 1][0])
    except Exception:
        next_probability = float("nan")

    current = history[
        (history["grandPrixName"].astype(str) == race_name)
        & (pd.to_numeric(history["grandPrixYear"], errors="coerce") != year)
    ].drop_duplicates(subset=["grandPrixYear"])

    if np.isfinite(next_probability):
        current = pd.concat([
            current,
            pd.DataFrame([{
                "grandPrixName": race_name,
                "grandPrixYear": year,
                "PredictedSafetyCarProbabilityPercentage": round(next_probability * 100, 3),
                "Type": "Next Race",
            }]),
        ], ignore_index=True)

    current = current.sort_values("grandPrixYear", ascending=False)
    percentage = history["PredictedSafetyCarProbabilityPercentage"]
    return {
        "rows": records(current),
        "mean": float(percentage.mean()),
        "min": float(percentage.min()),
        "max": float(percentage.max()),
    }

def next_race_bundle() -> dict[str, Any]:
    schedule = load_race_schedule().copy()
    date_col = "date" if "date" in schedule else ("short_date" if "short_date" in schedule else None)
    if not date_col:
        return {"next_race": None}
    dates = pd.to_datetime(schedule[date_col], errors="coerce")
    upcoming = schedule[dates >= pd.Timestamp.now().normalize()].copy()
    if upcoming.empty:
        return {"next_race": None}
    upcoming["_sort_date"] = pd.to_datetime(upcoming[date_col], errors="coerce")
    row = upcoming.sort_values("_sort_date").iloc[0]
    next_frame = pd.DataFrame([row.drop(labels=["_sort_date"], errors="ignore")])
    race_id = row.get("grandPrixId", row.get("grandPrixRaceId", row.get("id")))
    race_name = row.get("fullName", row.get("grandPrixName", "Upcoming Grand Prix"))
    year = row.get("year", pd.Timestamp(row[date_col]).year)

    data = load_main_data()
    if "grandPrixRaceId" in data and race_id is not None:
        past = data[data["grandPrixRaceId"].astype(str) == str(race_id)].copy()
    elif "grandPrixName" in data:
        past = data[data["grandPrixName"].astype(str) == str(race_name)].copy()
    else:
        past = pd.DataFrame()
    sort_cols = [c for c in ("grandPrixYear", "resultsFinalPositionNumber") if c in past]
    if sort_cols:
        past = past.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))

    driver_perf = pd.DataFrame()
    if {"resultsDriverName", "resultsStartingGridPositionNumber", "resultsFinalPositionNumber"}.issubset(past.columns):
        agg = {
            "average_starting_position": ("resultsStartingGridPositionNumber", "mean"),
            "average_ending_position": ("resultsFinalPositionNumber", "mean"),
            "driver_races": ("resultsFinalPositionNumber", "count"),
        }
        if "positionsGained" in past:
            agg["average_positions_gained"] = ("positionsGained", "mean")
        driver_perf = past.groupby("resultsDriverName").agg(**agg).reset_index().sort_values("average_ending_position")

    constructor_perf = pd.DataFrame()
    if {"constructorName", "resultsStartingGridPositionNumber", "resultsFinalPositionNumber"}.issubset(past.columns):
        agg = {
            "average_starting_position": ("resultsStartingGridPositionNumber", "mean"),
            "average_ending_position": ("resultsFinalPositionNumber", "mean"),
            "driver_races": ("resultsFinalPositionNumber", "count"),
        }
        if "positionsGained" in past:
            agg["average_positions_gained"] = ("positionsGained", "mean")
        constructor_perf = past.groupby("constructorName").agg(**agg).reset_index().sort_values("average_ending_position")

    weather = _read_optional(DATA_DIR / "f1WeatherData_Grouped.csv")
    if not weather.empty:
        if "grandPrixId" in weather and race_id is not None:
            weather = weather[weather["grandPrixId"].astype(str) == str(race_id)]
        elif "fullName" in weather:
            weather = weather[weather["fullName"].astype(str) == str(race_name)]

    messages = _read_optional(DATA_DIR / "race_control_messages_grouped_with_dnf.csv")
    if messages.empty:
        messages = _read_optional(DATA_DIR / "all_race_control_messages.csv")
    if not messages.empty and "grandPrixId" in messages and race_id is not None:
        messages = messages[messages["grandPrixId"].astype(str) == str(race_id)]

    predictions = find_prediction_artifact(str(race_id), str(year), str(race_name), row[date_col])
    legacy_predictions = _legacy_prediction_rows(str(race_id), int(year), str(race_name))
    safety_car_predictions = build_safety_car_predictions(row, str(race_name), int(year), weather)
    pit_stops = fastest_pit_stops(str(race_id))
    return {
        "next_race": records(next_frame)[0],
        "race_id": None if race_id is None else str(race_id),
        "race_name": str(race_name),
        "year": int(year) if pd.notna(year) else None,
        "past_results": records(
            past.drop_duplicates(
                subset=[column for column in ("resultsDriverName", "grandPrixYear") if column in past.columns]
            ).head(1000)
        ),
        "driver_performance": records(driver_perf),
        "constructor_performance": records(constructor_perf),
        "weather": records(weather.head(500)),
        "race_messages": records(messages.head(500)),
        "fastest_pit_stops": pit_stops,
        "predictions": predictions,
        "legacy_predictions": legacy_predictions,
        "safety_car_predictions": safety_car_predictions,
    }
