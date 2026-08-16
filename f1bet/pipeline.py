"""Small orchestration helpers that keep validation outside the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .contracts import (
    FEATURE_SNAPSHOT_CONTRACT,
    RACE_MODEL_CONTRACT,
    SCHEMA_VERSION,
    add_event_identity,
    mask_fields_unavailable_at_stage,
    stamp_feature_snapshot,
)
from .domain import SessionStage, stable_id
from .features import add_pre_race_form_features, default_registry


@dataclass(frozen=True, slots=True)
class BuildResult:
    frame: pd.DataFrame
    contract_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def collapse_driver_event_grain(
    frame: pd.DataFrame,
    *,
    event_col: str = "event_id",
    driver_col: str = "resultsDriverId",
) -> pd.DataFrame:
    """Collapse duplicate session rows only when their non-null facts agree.

    Conflicting values fail closed because choosing one silently would invent a
    point-in-time state. Equal values and complementary nulls are safe to merge.
    """

    keys = [event_col, driver_col]
    missing = set(keys) - set(frame.columns)
    if missing:
        raise KeyError(f"missing grain columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, dropna=False, sort=False, observed=True):
        record: dict[str, object] = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for column in frame.columns:
            if column in keys:
                continue
            values = group[column].dropna()
            unique = values.astype("string").drop_duplicates()
            if len(unique) > 1:
                raise ValueError(f"conflicting values at driver-event grain for {key}: {column}")
            record[column] = values.iloc[-1] if len(values) else pd.NA
        rows.append(record)
    return pd.DataFrame(rows, columns=frame.columns)


def build_v2_race_snapshot(
    legacy: pd.DataFrame,
    *,
    as_of: datetime,
    stage: SessionStage = SessionStage.PRE_RACE,
    include_form_features: bool = True,
) -> BuildResult:
    frame = add_event_identity(legacy)
    if include_form_features:
        frame = add_pre_race_form_features(frame)
    frame = stamp_feature_snapshot(frame, as_of=as_of, stage=stage)
    report = RACE_MODEL_CONTRACT.validate(frame)
    errors = tuple(issue.message for issue in report.issues if issue.severity == "error")
    return BuildResult(frame, report.valid, errors)


def build_v2_event_snapshot(
    legacy: pd.DataFrame,
    *,
    event_id: str,
    as_of: datetime,
    stage: SessionStage,
    source_manifest_id: str,
    collapse_duplicates: bool = False,
    strict_feature_registry: bool = True,
) -> BuildResult:
    """Build one valid point-in-time event snapshot from the legacy history."""

    frame = add_event_identity(legacy) if "event_id" not in legacy else legacy.copy()
    frame = add_pre_race_form_features(frame)
    frame = frame.loc[frame["event_id"].astype(str) == str(event_id)].copy()
    if frame.empty:
        raise KeyError(f"event_id not found: {event_id}")
    if collapse_duplicates:
        frame = collapse_driver_event_grain(frame)
    elif frame.duplicated(["event_id", "resultsDriverId"], keep=False).any():
        raise ValueError("duplicate driver-event rows require an explicit grain-collapse decision")
    dropped_columns: tuple[str, ...] = ()
    if strict_feature_registry:
        registry = default_registry()
        available = set(registry.available(stage))
        core = {
            rule.name for rule in RACE_MODEL_CONTRACT.rules
        } | {
            "constructorId",
            "resultsDriverId",
            "event_id",
            "grandPrixYear",
            "round",
        }
        selected = [column for column in frame.columns if column in core or column in available]
        dropped_columns = tuple(sorted(set(frame.columns) - set(selected)))
        frame = frame[selected].copy()
    frame = stamp_feature_snapshot(frame, as_of=as_of, stage=stage)
    frame = mask_fields_unavailable_at_stage(frame, stage=stage)
    frame["source_manifest_id"] = source_manifest_id
    frame["driver_id"] = frame["resultsDriverId"].astype(str)
    constructor_source = "constructorId" if "constructorId" in frame else "constructorName"
    frame["constructor_id"] = (
        frame[constructor_source].astype("string") if constructor_source in frame else pd.Series(pd.NA, index=frame.index)
    )
    frame["snapshot_version"] = SCHEMA_VERSION
    frame["snapshot_id"] = [
        stable_id(event_id, driver, stage.name, frame.iloc[index]["feature_as_of"], source_manifest_id)
        for index, driver in enumerate(frame["resultsDriverId"].astype(str))
    ]
    race_report = RACE_MODEL_CONTRACT.validate(frame)
    snapshot_report = FEATURE_SNAPSHOT_CONTRACT.validate(frame)
    errors = tuple(
        issue.message
        for report in (race_report, snapshot_report)
        for issue in report.issues
        if issue.severity == "error"
    )
    warnings = (
        (f"dropped {len(dropped_columns)} unregistered or unavailable legacy columns",)
        if dropped_columns
        else ()
    )
    return BuildResult(frame, race_report.valid and snapshot_report.valid, errors, warnings)
