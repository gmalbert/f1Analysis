"""Versioned tabular contracts for point-in-time model and betting data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from .domain import MarketType, SessionStage


SCHEMA_VERSION = "2.0.0"
Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ContractIssue:
    severity: Severity
    code: str
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()


@dataclass(slots=True)
class ValidationReport:
    contract: str
    schema_version: str
    row_count: int
    issues: list[ContractIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue.message for issue in self.issues if issue.severity == "error"]
        if errors:
            raise ValueError("; ".join(errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "valid": self.valid,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "column": issue.column,
                    "rows": list(issue.rows),
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True, slots=True)
class ColumnRule:
    name: str
    kind: Literal["number", "integer", "string", "boolean", "datetime"]
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: frozenset[Any] | None = None
    stage: SessionStage = SessionStage.PRE_WEEKEND
    description: str = ""


@dataclass(frozen=True, slots=True)
class DatasetContract:
    name: str
    rules: tuple[ColumnRule, ...]
    unique_by: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    allow_extra_columns: bool = True

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        report = ValidationReport(self.name, self.schema_version, len(frame))
        rule_names = {rule.name for rule in self.rules}
        missing = sorted(rule_names - set(frame.columns))
        for column in missing:
            report.issues.append(
                ContractIssue("error", "missing_column", f"required column {column!r} is missing", column)
            )
        if not self.allow_extra_columns:
            for column in sorted(set(frame.columns) - rule_names):
                report.issues.append(
                    ContractIssue("error", "extra_column", f"unexpected column {column!r}", column)
                )

        for rule in self.rules:
            if rule.name not in frame:
                continue
            series = frame[rule.name]
            if not rule.nullable and series.isna().any():
                rows = tuple(int(value) for value in series.index[series.isna()][:10])
                report.issues.append(
                    ContractIssue(
                        "error", "null_not_allowed", f"{rule.name!r} contains null values", rule.name, rows
                    )
                )
            non_null = series.dropna()
            if non_null.empty:
                continue
            invalid_type = self._invalid_type_mask(non_null, rule.kind)
            if invalid_type.any():
                rows = tuple(int(value) for value in non_null.index[invalid_type][:10])
                report.issues.append(
                    ContractIssue(
                        "error",
                        "invalid_type",
                        f"{rule.name!r} contains values incompatible with {rule.kind}",
                        rule.name,
                        rows,
                    )
                )
                continue
            comparable = pd.to_numeric(non_null, errors="coerce") if rule.kind in {"number", "integer"} else None
            if comparable is not None and rule.minimum is not None:
                bad = comparable < rule.minimum
                if bad.any():
                    report.issues.append(
                        ContractIssue(
                            "error",
                            "below_minimum",
                            f"{rule.name!r} has values below {rule.minimum}",
                            rule.name,
                            tuple(int(value) for value in comparable.index[bad][:10]),
                        )
                    )
            if comparable is not None and rule.maximum is not None:
                bad = comparable > rule.maximum
                if bad.any():
                    report.issues.append(
                        ContractIssue(
                            "error",
                            "above_maximum",
                            f"{rule.name!r} has values above {rule.maximum}",
                            rule.name,
                            tuple(int(value) for value in comparable.index[bad][:10]),
                        )
                    )
            if rule.allowed is not None:
                bad = ~non_null.isin(rule.allowed)
                if bad.any():
                    report.issues.append(
                        ContractIssue(
                            "error",
                            "value_not_allowed",
                            f"{rule.name!r} contains values outside its vocabulary",
                            rule.name,
                            tuple(int(value) for value in non_null.index[bad][:10]),
                        )
                    )

        self._validate_availability(frame, report)
        self._validate_cross_field_rules(frame, report)

        if self.unique_by and not missing:
            duplicates = frame.duplicated(list(self.unique_by), keep=False)
            if duplicates.any():
                report.issues.append(
                    ContractIssue(
                        "error",
                        "duplicate_key",
                        f"duplicate rows for key {self.unique_by}",
                        rows=tuple(int(value) for value in frame.index[duplicates][:10]),
                    )
                )
        return report

    def _validate_availability(self, frame: pd.DataFrame, report: ValidationReport) -> None:
        if "feature_stage" not in frame:
            return
        stage_values = frame["feature_stage"].map(_coerce_stage)
        bad_stage = stage_values.isna() & frame["feature_stage"].notna()
        if bad_stage.any():
            report.issues.append(
                ContractIssue(
                    "error",
                    "invalid_stage",
                    "feature_stage contains an unknown information stage",
                    "feature_stage",
                    tuple(int(value) for value in frame.index[bad_stage][:10]),
                )
            )
        for rule in self.rules:
            if rule.name not in frame or rule.name == "feature_stage":
                continue
            unavailable = stage_values.notna() & (stage_values < int(rule.stage)) & frame[rule.name].notna()
            if unavailable.any():
                report.issues.append(
                    ContractIssue(
                        "error",
                        "feature_unavailable_at_stage",
                        f"{rule.name!r} is populated before {rule.stage.name}",
                        rule.name,
                        tuple(int(value) for value in frame.index[unavailable][:10]),
                    )
                )

    def _validate_cross_field_rules(self, frame: pd.DataFrame, report: ValidationReport) -> None:
        datetime_columns = {
            rule.name for rule in self.rules if rule.kind == "datetime" and rule.name in frame
        }
        parsed = {column: pd.to_datetime(frame[column], errors="coerce", utc=True) for column in datetime_columns}
        if {"captured_at", "event_start_at"} <= parsed.keys():
            bad = parsed["captured_at"] > parsed["event_start_at"]
            if bad.any():
                report.issues.append(
                    ContractIssue(
                        "error",
                        "quote_after_event_start",
                        "captured_at must not be later than event_start_at",
                        "captured_at",
                        tuple(int(value) for value in frame.index[bad][:10]),
                    )
                )
        if {"market", "opponent_id"} <= set(frame.columns):
            h2h = frame["market"].astype("string").eq(MarketType.HEAD_TO_HEAD.value)
            missing_opponent = h2h & (frame["opponent_id"].isna() | frame["opponent_id"].astype("string").str.strip().eq(""))
            if missing_opponent.any():
                report.issues.append(
                    ContractIssue(
                        "error",
                        "missing_h2h_opponent",
                        "head-to-head records require opponent_id",
                        "opponent_id",
                        tuple(int(value) for value in frame.index[missing_opponent][:10]),
                    )
                )

    @staticmethod
    def _invalid_type_mask(series: pd.Series, kind: str) -> pd.Series:
        if kind in {"number", "integer"}:
            converted = pd.to_numeric(series, errors="coerce")
            invalid = converted.isna() | ~np.isfinite(converted)
            if kind == "integer":
                invalid |= (converted % 1).abs() > 1e-12
            return invalid
        if kind == "datetime":
            parsed = pd.to_datetime(series, errors="coerce", utc=False)
            invalid = parsed.isna()
            aware = series.map(_is_timezone_aware)
            return invalid | ~aware
        if kind == "boolean":
            return ~series.isin([True, False, 0, 1])
        return series.astype("string").str.strip().eq("")


def _is_timezone_aware(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _coerce_stage(value: object) -> float:
    if isinstance(value, SessionStage):
        return float(value)
    try:
        return float(SessionStage[str(value)].value)
    except (KeyError, TypeError, ValueError):
        return float("nan")


RACE_MODEL_CONTRACT = DatasetContract(
    name="race_model_snapshot",
    unique_by=("event_id", "resultsDriverId", "feature_stage"),
    rules=(
        ColumnRule("event_id", "string", description="Stable race/session key"),
        ColumnRule("grandPrixYear", "integer", minimum=1950, maximum=2200),
        ColumnRule("round", "integer", minimum=0, maximum=99),
        ColumnRule("resultsDriverId", "string"),
        ColumnRule("constructorName", "string", nullable=True),
        ColumnRule("feature_as_of", "datetime"),
        ColumnRule(
            "feature_stage",
            "string",
            allowed=frozenset(stage.name for stage in SessionStage),
        ),
        ColumnRule("schema_version", "string", allowed=frozenset({SCHEMA_VERSION})),
        ColumnRule(
            "resultsStartingGridPositionNumber",
            "number",
            nullable=True,
            minimum=0,
            maximum=40,
            stage=SessionStage.POST_QUALIFYING,
        ),
        ColumnRule(
            "resultsFinalPositionNumber",
            "number",
            nullable=True,
            minimum=1,
            maximum=40,
            stage=SessionStage.POST_RACE,
        ),
    ),
)

EVENT_CONTRACT = DatasetContract(
    name="event_dimension",
    unique_by=("event_id",),
    rules=(
        ColumnRule("event_id", "string"),
        ColumnRule("season", "integer", minimum=1950, maximum=2200),
        ColumnRule("round_number", "integer", minimum=0, maximum=99),
        ColumnRule("session", "string"),
        ColumnRule("scheduled_start_at", "datetime"),
        ColumnRule("circuit_id", "string"),
        ColumnRule("regulation_era", "string"),
    ),
)

FEATURE_SNAPSHOT_CONTRACT = DatasetContract(
    name="feature_snapshot_ledger",
    unique_by=("snapshot_id",),
    rules=(
        ColumnRule("snapshot_id", "string"),
        ColumnRule("event_id", "string"),
        ColumnRule("driver_id", "string"),
        ColumnRule("constructor_id", "string", nullable=True),
        ColumnRule("feature_as_of", "datetime"),
        ColumnRule("feature_stage", "string", allowed=frozenset(stage.name for stage in SessionStage)),
        ColumnRule("schema_version", "string"),
        ColumnRule("source_manifest_id", "string"),
        ColumnRule("snapshot_version", "string"),
    ),
)

ODDS_LEDGER_CONTRACT = DatasetContract(
    name="odds_quote_ledger",
    unique_by=("quote_id",),
    rules=(
        ColumnRule("quote_id", "string"),
        ColumnRule("event_id", "string"),
        ColumnRule(
            "market",
            "string",
            allowed=frozenset(market.value for market in MarketType),
        ),
        ColumnRule("selection_id", "string"),
        ColumnRule("opponent_id", "string", nullable=True),
        ColumnRule("bookmaker", "string"),
        ColumnRule("captured_at", "datetime"),
        ColumnRule("event_start_at", "datetime"),
        ColumnRule("decimal_odds", "number", minimum=1.000001, maximum=10000),
        ColumnRule("line", "number", nullable=True),
    ),
)

FORECAST_LEDGER_CONTRACT = DatasetContract(
    name="forecast_ledger",
    unique_by=("forecast_id",),
    rules=(
        ColumnRule("forecast_id", "string"),
        ColumnRule("event_id", "string"),
        ColumnRule("market", "string", allowed=frozenset(m.value for m in MarketType)),
        ColumnRule("selection_id", "string"),
        ColumnRule("opponent_id", "string", nullable=True),
        ColumnRule("probability", "number", minimum=0, maximum=1),
        ColumnRule("uncertainty", "number", minimum=0, maximum=1),
        ColumnRule("generated_at", "datetime"),
        ColumnRule("stage", "string", allowed=frozenset(stage.name for stage in SessionStage)),
        ColumnRule("model_version", "string"),
        ColumnRule("feature_snapshot_id", "string", nullable=True),
    ),
)

DECISION_LEDGER_CONTRACT = DatasetContract(
    name="paper_decision_ledger",
    unique_by=("bet_id",),
    rules=(
        ColumnRule("bet_id", "string"),
        ColumnRule("quote_id", "string"),
        ColumnRule("forecast_id", "string"),
        ColumnRule("event_id", "string"),
        ColumnRule("market", "string", allowed=frozenset(m.value for m in MarketType)),
        ColumnRule("selection_id", "string"),
        ColumnRule("opponent_id", "string", nullable=True),
        ColumnRule("decided_at", "datetime"),
        ColumnRule("model_probability", "number", minimum=0, maximum=1),
        ColumnRule("fair_market_probability", "number", minimum=0, maximum=1),
        ColumnRule("decimal_odds", "number", minimum=1.000001, maximum=10000),
        ColumnRule("edge", "number", minimum=-1, maximum=1),
        ColumnRule("expected_value", "number", minimum=-1),
        ColumnRule("stake", "number", minimum=0),
        ColumnRule("bankroll_before", "number", minimum=0),
        ColumnRule("reason_code", "string"),
        ColumnRule("status", "string", allowed=frozenset({"placed", "abstained", "rejected"})),
    ),
)

SETTLEMENT_LEDGER_CONTRACT = DatasetContract(
    name="settlement_ledger",
    unique_by=("settlement_id",),
    rules=(
        ColumnRule("settlement_id", "string"),
        ColumnRule("bet_id", "string"),
        ColumnRule("status", "string", allowed=frozenset({"won", "lost", "void"})),
        ColumnRule("profit", "number"),
        ColumnRule("closing_odds", "number", nullable=True, minimum=1.000001),
        ColumnRule("closing_line_value", "number", nullable=True),
        ColumnRule("settled_at", "datetime"),
        ColumnRule("rule_version", "string"),
        ColumnRule("settlement_version", "integer", minimum=1),
        ColumnRule("supersedes_settlement_id", "string", nullable=True),
    ),
)


def add_event_identity(
    frame: pd.DataFrame,
    *,
    season_col: str = "grandPrixYear",
    round_col: str = "round",
    race_col: str = "raceId_results",
    session: str = "R",
) -> pd.DataFrame:
    """Add a stable event key to a legacy wide table without mutating it."""
    result = frame.copy()
    if season_col not in result or round_col not in result:
        raise KeyError(f"{season_col!r} and {round_col!r} are required")
    season = pd.to_numeric(result[season_col], errors="raise").astype(int).astype(str)
    round_value = pd.to_numeric(result[round_col], errors="raise").astype(int).astype(str).str.zfill(2)
    if race_col in result:
        race = result[race_col].astype("string").fillna("").str.strip()
        suffix = race.where(race.ne(""), "R" + round_value)
    else:
        suffix = "R" + round_value
    result["event_id"] = season + "-" + suffix + f"-{session.upper()}"
    return result


def stamp_feature_snapshot(
    frame: pd.DataFrame,
    *,
    as_of: datetime | None = None,
    stage: SessionStage = SessionStage.PRE_RACE,
) -> pd.DataFrame:
    result = frame.copy()
    timestamp = as_of or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    result["feature_as_of"] = timestamp.astimezone(timezone.utc).isoformat()
    result["feature_stage"] = stage.name
    result["schema_version"] = SCHEMA_VERSION
    return result


def mask_fields_unavailable_at_stage(
    frame: pd.DataFrame,
    *,
    contract: DatasetContract = RACE_MODEL_CONTRACT,
    stage: SessionStage,
) -> pd.DataFrame:
    """Return a snapshot with later-stage contract fields explicitly absent."""

    result = frame.copy()
    for rule in contract.rules:
        if rule.name in result and rule.stage > stage:
            result[rule.name] = pd.NA
    return result
