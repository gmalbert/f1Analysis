from __future__ import annotations

import ast
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from app.config import DATA_DIR, MAX_TABLE_ROWS, PRECOMPUTED_DIR, REPO_ROOT

MAIN_DATA = DATA_DIR / "f1ForAnalysis.csv"
PARQUET_MAIN_DATA = DATA_DIR / "f1ForAnalysis.parquet"


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return value


def records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        frame = frame.head(limit)
    return [
        {str(k): _clean_scalar(v) for k, v in item.items()}
        for item in frame.to_dict(orient="records")
    ]


@lru_cache(maxsize=1)
def load_main_data() -> pd.DataFrame:
    use_parquet = os.environ.get("F1_USE_PARQUET", "1").strip().lower() in {"1", "true", "yes"}
    source = PARQUET_MAIN_DATA if use_parquet and PARQUET_MAIN_DATA.exists() else MAIN_DATA
    if not source.exists():
        raise FileNotFoundError(f"Missing required dataset: {source}")
    if source.suffix == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source, sep="\t", low_memory=False)
    for candidate in ("short_date", "date", "grandPrixDate"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_race_schedule() -> pd.DataFrame:
    candidates = [DATA_DIR / "f1db-races.json", DATA_DIR / "f1db-races-races.json"]
    for candidate in candidates:
        if candidate.exists():
            frame = pd.read_json(candidate)
            if "date" in frame:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            return frame
    df = load_main_data()
    cols = [c for c in (
        "grandPrixYear", "round", "grandPrixName", "grandPrixRaceId",
        "short_date", "grandPrixLaps", "turns", "courseLength", "circuitType"
    ) if c in df]
    schedule = df[cols].drop_duplicates()
    return schedule.rename(columns={
        "grandPrixYear": "year", "grandPrixName": "fullName",
        "grandPrixRaceId": "grandPrixId", "short_date": "date",
        "grandPrixLaps": "laps",
    })


def apply_filters(df: pd.DataFrame, filters: list[Any]) -> pd.DataFrame:
    result = df
    for spec in filters:
        column = spec.column
        if column not in result:
            continue
        value = spec.value
        if spec.kind == "range" and isinstance(value, (list, tuple)) and len(value) == 2:
            lo, hi = value
            numeric = pd.to_numeric(result[column], errors="coerce")
            result = result[numeric.between(lo, hi) | numeric.isna()]
        elif spec.kind == "date_range" and isinstance(value, (list, tuple)) and len(value) == 2:
            dates = pd.to_datetime(result[column], errors="coerce")
            lo, hi = pd.to_datetime(value[0]), pd.to_datetime(value[1])
            result = result[dates.between(lo, hi) | dates.isna()]
        elif spec.kind == "boolean":
            expected = bool(value)
            series = result[column]
            if not pd.api.types.is_bool_dtype(series):
                series = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
            result = result[(series == expected) | result[column].isna()]
        elif spec.kind == "exact" and value not in (None, "", " All", "All"):
            result = result[(result[column] == value) | result[column].isna()]
    return result


@lru_cache(maxsize=1)
def streamlit_filter_rules() -> tuple[dict[str, str], frozenset[str]]:
    """Read the Streamlit filter labels and exclusions without importing its app."""
    source_path = REPO_ROOT / "raceAnalysis.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    literal_names = {"column_rename_for_filter", "exclusionList", "suffixes_to_exclude"}
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in literal_names:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue

    labels = values.get("column_rename_for_filter", {})
    excluded = set(values.get("exclusionList", ()))
    suffixes = values.get("suffixes_to_exclude", ())
    excluded.update(column for column in load_main_data().columns if column.endswith(tuple(suffixes)))
    return labels, frozenset(excluded)


@lru_cache(maxsize=1)
def _streamlit_table_definitions() -> tuple[list[str], dict[str, str | None]]:
    source_path = REPO_ROOT / "raceAnalysis.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    display_config: dict[str, str | None] = {}
    selected_columns: list[str] = []

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(isinstance(target, ast.Name) and target.id == "columns_to_display" for target in node.targets)
        ):
            for key_node, value_node in zip(node.value.keys, node.value.values, strict=True):
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                if isinstance(value_node, ast.Constant) and value_node.value is None:
                    display_config[key_node.value] = None
                elif (
                    isinstance(value_node, ast.Call)
                    and value_node.args
                    and isinstance(value_node.args[0], ast.Constant)
                    and isinstance(value_node.args[0].value, str)
                ):
                    display_config[key_node.value] = value_node.args[0].value
        elif isinstance(node, ast.FunctionDef) and node.name == "load_data":
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(isinstance(target, ast.Name) and target.id == "selected_columns" for target in statement.targets):
                    try:
                        selected_columns = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        selected_columns = []
                    break
    return selected_columns, display_config


@lru_cache(maxsize=1)
def load_streamlit_raw_data() -> pd.DataFrame:
    """Build the same joined raw table that the Streamlit Data & Debug tab displays."""
    selected_columns, _ = _streamlit_table_definitions()
    source = load_main_data()
    bin_columns = [column for column in source.columns if column.endswith("_bin")]
    usecols = list(dict.fromkeys(column for column in selected_columns + bin_columns if column in source.columns))
    full_results = source[usecols].copy()

    pit_stops = pd.read_csv(
        DATA_DIR / "f1PitStopsData_Grouped.csv",
        sep="\t",
        nrows=10000,
        usecols=["raceId", "driverId", "constructorId", "numberOfStops", "averageStopTime", "totalStopTime"],
    )
    constructor_standings = pd.read_csv(DATA_DIR / "constructor_standings.csv", sep="\t")
    driver_standings = pd.read_csv(DATA_DIR / "driver_standings.csv", sep="\t")
    weather = pd.read_csv(
        DATA_DIR / "f1WeatherData_Grouped.csv",
        sep="\t",
        nrows=10000,
        usecols=["grandPrixId", "id_races", "average_temp", "average_humidity", "average_wind_speed", "total_precipitation"],
    )
    grand_prix = pd.read_json(DATA_DIR / "f1db-grands-prix.json")
    weather = weather.merge(
        grand_prix,
        left_on="grandPrixId",
        right_on="id",
        how="inner",
        suffixes=("_weather", "_grandPrix"),
    )[["id_races", "average_temp", "average_humidity", "average_wind_speed", "total_precipitation"]]
    qualifying = pd.read_csv(DATA_DIR / "all_qualifying_races.csv", sep="\t")

    full_results = full_results.merge(
        pit_stops,
        left_on=["raceId_results", "resultsDriverId"],
        right_on=["raceId", "driverId"],
        how="left",
        suffixes=("_results", "_pitStops"),
    )
    full_results = full_results.merge(
        constructor_standings,
        left_on="constructorId_results",
        right_on="id",
        how="left",
        suffixes=("_results", "_constructor_standings"),
    )
    full_results = full_results.merge(
        driver_standings,
        left_on="resultsDriverId",
        right_on="driverId",
        how="left",
        suffixes=("_results", "_driver_standings"),
    )
    full_results = full_results.merge(
        weather,
        left_on="raceId_results",
        right_on="id_races",
        how="left",
        suffixes=("_results", "_weather"),
    )
    full_results = full_results.merge(
        qualifying,
        left_on=["raceId_results", "resultsDriverId"],
        right_on=["raceId", "driverId"],
        how="left",
        suffixes=("_results_with_qualifying", "_qualifying"),
    )
    full_results = full_results.drop_duplicates(
        subset=["grandPrixYear", "grandPrixName", "resultsDriverName"]
    )
    full_results = full_results.loc[:, ~full_results.columns.duplicated()]

    renamed_columns = {
        "constructorName_results_with_qualifying": "constructorName",
        "best_qual_time_results_with_qualifying": "best_qual_time",
        "teammate_qual_delta_results_with_qualifying": "teammate_qual_delta",
    }
    full_results = full_results.rename(columns={old: new for old, new in renamed_columns.items() if old in full_results})
    return full_results


def streamlit_table_schema() -> dict[str, Any]:
    """Return the raw-data columns and labels configured by the Streamlit app."""
    _, display_config = _streamlit_table_definitions()
    columns: list[str] = []
    labels: dict[str, str] = {}
    for column in load_streamlit_raw_data().columns:
        configured_label = display_config.get(column, "visible")
        if configured_label is None:
            continue
        columns.append(column)
        if isinstance(configured_label, str) and configured_label != "visible":
            labels[column] = configured_label

    return {"columns": columns, "labels": labels}


def query_streamlit_raw_data(offset: int, limit: int) -> dict[str, Any]:
    frame = load_streamlit_raw_data()
    page = frame.iloc[offset: offset + limit]
    return {"total": len(frame), "columns": list(page.columns), "rows": records(page)}


def filter_schema() -> list[dict[str, Any]]:
    df = load_main_data()
    labels, excluded = streamlit_filter_rules()
    schema: list[dict[str, Any]] = []
    for column in sorted(df.columns):
        if column in excluded:
            continue
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        item: dict[str, Any] = {"column": column, "label": labels.get(column, column)}
        unique = non_null.nunique(dropna=True)
        numeric_non_null = pd.to_numeric(non_null, errors="coerce").dropna()
        bool_like = pd.api.types.is_bool_dtype(series) or (
            unique <= 2 and not numeric_non_null.empty and set(numeric_non_null.unique()).issubset({0, 1})
        )
        if bool_like:
            item["kind"] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            item.update(kind="date_range", min=_clean_scalar(non_null.min()), max=_clean_scalar(non_null.max()))
        elif pd.api.types.is_numeric_dtype(series):
            if not numeric_non_null.empty:
                item.update(kind="range", min=_clean_scalar(numeric_non_null.min()), max=_clean_scalar(numeric_non_null.max()))
        else:
            item["kind"] = "exact"
            if unique <= 250:
                item["options"] = sorted(str(v) for v in non_null.unique())
            else:
                item["high_cardinality"] = True
                item["unique_values"] = int(unique)
        schema.append(item)
    return schema


def query_main(request: Any) -> dict[str, Any]:
    df = apply_filters(load_main_data(), request.filters)
    total = len(df)
    if request.sort:
        valid = [c for c in request.sort if c in df.columns]
        if valid:
            df = df.sort_values(valid, ascending=not request.descending)
    if request.columns:
        valid = [c for c in request.columns if c in df.columns]
        if valid:
            df = df[valid]
    page = df.iloc[request.offset: request.offset + request.limit]
    return {"total": int(total), "columns": list(page.columns), "rows": records(page)}


def read_table(path: Path, limit: int = MAX_TABLE_ROWS) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        try:
            frame = pd.read_csv(path, sep="\t", low_memory=False)
            if len(frame.columns) == 1:
                frame = pd.read_csv(path, low_memory=False)
        except Exception:
            frame = pd.read_csv(path, low_memory=False)
        return {"kind": "table", "columns": list(frame.columns), "rows": records(frame, limit), "total": len(frame)}
    if suffix == ".json":
        return {"kind": "json", "data": json.loads(path.read_text(encoding="utf-8"))}
    if suffix in {".txt", ".md", ".log"}:
        return {"kind": "text", "data": path.read_text(encoding="utf-8", errors="replace")[:250_000]}
    return {"kind": "binary", "size": path.stat().st_size}


_ALLOWED_DATA_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".txt", ".md", ".log", ".png", ".html"})


def _data_file_index() -> dict[str, Path]:
    if not DATA_DIR.exists():
        return {}
    root = DATA_DIR.resolve()
    result: dict[str, Path] = {}
    for path in DATA_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _ALLOWED_DATA_SUFFIXES:
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            continue
        result[path.relative_to(DATA_DIR).as_posix()] = resolved
    return result


def list_data_files() -> list[dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    result = []
    for relative, path in sorted(_data_file_index().items()):
        result.append({
            "path": relative,
            "size": path.stat().st_size,
            "suffix": path.suffix.lower(),
        })
    return result


def resolve_data_file(relative: str) -> Path:
    candidate = _data_file_index().get(relative.replace("\\", "/"))
    if candidate is None:
        raise FileNotFoundError("Requested data file was not found")
    return candidate


def precomputed(name: str) -> Any:
    mapping = {
        "monte_carlo": "monte_carlo_results.json",
        "monte_carlo_log": "monte_carlo_run_log.json",
        "shap": "shap_results.json",
        "rfe": "rfe_results.json",
        "boruta": "boruta_results.json",
        "permutation": "permutation_results.json",
        "hyperparam_bayesian": "hyperparam_bayesian.json",
        "hyperparam_grid": "hyperparam_grid.json",
        "historical_validation": "historical_validation.json",
        "position_mae": "position_mae_detailed.json",
    }
    filename = mapping.get(name)
    if not filename:
        raise KeyError(name)
    target = PRECOMPUTED_DIR / filename
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def model_manifest(model_type: str) -> dict[str, Any] | None:
    directory_map = {
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "CatBoost": "catboost",
        "Ensemble (XGBoost + LightGBM + CatBoost)": "ensemble",
        "Position Group": "position_group",
        "Track-Weighted Ensemble": "track_weighted",
    }
    directory = directory_map.get(model_type)
    if not directory:
        raise KeyError(model_type)
    target = DATA_DIR / "models" / directory / "manifest.json"
    if not target.exists():
        return None
    return cast(dict[str, Any], json.loads(target.read_text(encoding="utf-8")))
