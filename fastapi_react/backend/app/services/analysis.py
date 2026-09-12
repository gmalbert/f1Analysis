from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress

from app.config import DATA_DIR
from app.services.data import apply_filters, load_main_data, load_race_schedule, records


def _regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any] | None:
    if x_col not in df or y_col not in df:
        return None
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    slope, intercept, r, p, stderr = linregress(x[mask], y[mask])
    return {
        "x": x_col, "y": y_col, "slope": float(slope), "intercept": float(intercept),
        "r_squared": float(r ** 2), "p_value": float(p), "std_err": float(stderr),
    }


def analytics(filters: Any, max_rows: int) -> dict[str, Any]:
    df = apply_filters(load_main_data(), filters).head(max_rows).copy()
    payload: dict[str, Any] = {"rows_considered": len(df), "charts": {}, "regressions": []}
    pairs = {
        "active_years_vs_final": ("resultsFinalPositionNumber", "yearsActive"),
        "positions_gained_over_time": ("short_date", "positionsGained"),
        "practice_vs_final": ("lastFPPositionNumber", "resultsFinalPositionNumber"),
        "grid_vs_final": ("resultsStartingGridPositionNumber", "resultsFinalPositionNumber"),
        "avg_practice_vs_final": ("averagePracticePosition", "resultsFinalPositionNumber"),
        "pit_stop_vs_final": ("averageStopTime", "resultsFinalPositionNumber"),
    }
    for name, (x, y) in pairs.items():
        if x in df and y in df:
            payload["charts"][name] = records(df[[x, y]].dropna().head(5000))

    for x in ("averagePracticePosition", "resultsStartingGridPositionNumber"):
        result = _regression(df, x, "resultsFinalPositionNumber")
        if result:
            payload["regressions"].append(result)

    corr_cols = [c for c in (
        "lastFPPositionNumber", "resultsFinalPositionNumber", "resultsStartingGridPositionNumber",
        "grandPrixLaps", "averagePracticePosition", "DNF", "resultsTop10", "resultsTop5",
        "resultsPodium", "streetRace", "trackRace", "constructorTotalRaceStarts",
        "constructorTotalRaceWins", "constructorTotalPolePositions", "turns", "positionsGained",
        "q1End", "q2End", "q3Top10", "driverBestStartingGridPosition", "yearsActive",
        "driverBestRaceResult", "driverTotalChampionshipWins", "driverTotalPolePositions",
        "driverTotalRaceEntries", "driverTotalRaceStarts", "driverTotalRaceWins",
        "driverTotalRaceLaps", "driverTotalPodiums", "avgLapPace", "finishingTime",
    ) if c in df]
    if corr_cols:
        corr = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        payload["correlation"] = {
            "columns": list(corr.columns),
            "rows": [
                {"feature": idx, **{col: (None if pd.isna(v) else float(v)) for col, v in row.items()}}
                for idx, row in corr.iterrows()
            ],
        }

    if {"grandPrixYear", "resultsDriverName", "resultsFinalPositionNumber"}.issubset(df.columns):
        agg = {"average_final_position": ("resultsFinalPositionNumber", "mean")}
        if "resultsPodium" in df:
            agg["total_podiums"] = ("resultsPodium", "sum")
        driver = df.groupby(["grandPrixYear", "resultsDriverName"]).agg(**agg).reset_index()
        payload["driver_performance"] = records(driver)

    if {"grandPrixYear", "constructorName", "resultsFinalPositionNumber"}.issubset(df.columns):
        constructor = (
            df.groupby(["grandPrixYear", "constructorName"])
            .agg(
                total_wins=("resultsFinalPositionNumber", lambda s: int((s == 1).sum())),
                average_final_position=("resultsFinalPositionNumber", "mean"),
            ).reset_index()
        )
        if "resultsPodium" in df:
            podium = df.groupby(["grandPrixYear", "constructorName"])["resultsPodium"].sum().reset_index(name="total_podiums")
            constructor = constructor.merge(podium, on=["grandPrixYear", "constructorName"], how="left")
        payload["constructor_performance"] = records(constructor)

    if {"DNF", "resultsReasonRetired"}.issubset(df.columns):
        dnf = (
            df[pd.to_numeric(df["DNF"], errors="coerce").fillna(0).eq(1)]
            .groupby("resultsReasonRetired").size().reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        payload["dnf_reasons"] = records(dnf)
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


def find_prediction_artifact(race_id: str, year: str, race_name: str) -> dict[str, Any] | None:
    """Select the best committed next-race prediction artifact.

    The current precompute workflow writes JSON with predictions_by_model, while
    older/headless paths may write CSV. Both are supported.
    """
    import json

    candidates = []
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
            score = 1 + sum(1 for term in terms if term and term in low)
            candidates.append((score, path.name, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = candidates[0][2]
    relative = chosen.relative_to(DATA_DIR).as_posix()

    if chosen.suffix.lower() == ".json":
        payload = json.loads(chosen.read_text(encoding="utf-8"))
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

    predictions = find_prediction_artifact(str(race_id), str(year), str(race_name))
    return {
        "next_race": records(next_frame)[0],
        "race_id": None if race_id is None else str(race_id),
        "race_name": str(race_name),
        "year": int(year) if pd.notna(year) else None,
        "past_results": records(past.drop_duplicates().head(1000)),
        "driver_performance": records(driver_perf),
        "constructor_performance": records(constructor_perf),
        "weather": records(weather.head(500)),
        "race_messages": records(messages.head(500)),
        "predictions": predictions,
    }
