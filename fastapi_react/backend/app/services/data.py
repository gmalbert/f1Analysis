from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import DATA_DIR, MAX_TABLE_ROWS, PRECOMPUTED_DIR

MAIN_DATA = DATA_DIR / "f1ForAnalysis.csv"


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
    except Exception:
        pass
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
    if not MAIN_DATA.exists():
        raise FileNotFoundError(f"Missing required dataset: {MAIN_DATA}")
    df = pd.read_csv(MAIN_DATA, sep="\t", low_memory=False)
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


def filter_schema() -> list[dict[str, Any]]:
    df = load_main_data()
    schema: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        item: dict[str, Any] = {"column": column, "label": column}
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


def query_main(request) -> dict[str, Any]:
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


def list_data_files() -> list[dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    allowed = {".csv", ".tsv", ".json", ".txt", ".md", ".log", ".png", ".html"}
    result = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            result.append({
                "path": path.relative_to(DATA_DIR).as_posix(),
                "size": path.stat().st_size,
                "suffix": path.suffix.lower(),
            })
    return result


def resolve_data_file(relative: str) -> Path:
    candidate = (DATA_DIR / relative).resolve()
    root = DATA_DIR.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Invalid data path")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(relative)
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


def model_manifest(model_type: str) -> dict | None:
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
    return json.loads(target.read_text(encoding="utf-8"))
