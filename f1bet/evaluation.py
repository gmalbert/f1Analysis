"""Comparable probability ablations and event-clustered slice reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .calibration import EPSILON, probability_metrics


REQUIRED_ABLATION_VARIANTS = (
    "grid_qualifying_only",
    "model_without_market",
    "market_consensus_only",
    "model_plus_market_consensus",
    "no_practice_telemetry",
    "no_weather",
    "no_dnf_model",
    "independent_driver_simulation",
    "correlated_field_simulation",
    "fixed_tyre_curve",
    "adaptive_tyre_residual",
    "raw_probabilities",
    "calibrated_probabilities",
    "flat_stakes",
    "capped_fractional_kelly",
)

REQUIRED_SLICE_DIMENSIONS = (
    "season",
    "circuit_archetype",
    "wet_dry",
    "grid_band",
    "rookie",
    "constructor",
    "data_coverage",
)


@dataclass(frozen=True, slots=True)
class AblationCoverage:
    complete: bool
    missing_variants: tuple[str, ...]
    inconsistent_folds: tuple[str, ...]
    search_trials: int


@dataclass(frozen=True, slots=True)
class SliceEvaluation:
    report: pd.DataFrame
    missing_dimensions: tuple[str, ...]


def _event_clustered_interval(
    frame: pd.DataFrame,
    *,
    value_col: str,
    event_col: str,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    clean = frame[[event_col, value_col]].dropna()
    events = clean[event_col].drop_duplicates().to_numpy()
    if len(events) < 2:
        return float("nan"), float("nan")
    grouped = {
        event: clean.loc[clean[event_col] == event, value_col].to_numpy(dtype=float)
        for event in events
    }
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(events, size=len(events), replace=True)
        estimates[index] = np.concatenate([grouped[event] for event in sampled]).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_probability_variants(
    records: pd.DataFrame,
    probability_columns: Mapping[str, str],
    *,
    outcome_col: str = "outcome",
    event_col: str = "event_id",
    fold_col: str = "fold_id",
    n_bins: int = 10,
    n_bootstrap: int = 1_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Score candidate variants on identical frozen rows and fold IDs."""

    if not probability_columns:
        raise ValueError("at least one probability variant is required")
    required = {outcome_col, event_col, fold_col, *probability_columns.values()}
    missing = required - set(records)
    if missing:
        raise KeyError(f"ablation records missing columns: {sorted(missing)}")
    base = records[list(required)].copy()
    base[outcome_col] = pd.to_numeric(base[outcome_col], errors="raise")
    if not base[outcome_col].isin([0, 1]).all():
        raise ValueError("ablation outcomes must be binary")
    rows: list[dict[str, object]] = []
    for variant, probability_col in probability_columns.items():
        probability = pd.to_numeric(base[probability_col], errors="raise")
        if probability.isna().any() or (~np.isfinite(probability)).any() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{variant!r} probabilities must be complete, finite, and in [0, 1]")
        scored = base[[event_col, fold_col, outcome_col]].copy()
        scored["probability"] = probability.to_numpy()
        scored["brier_row"] = (scored["probability"] - scored[outcome_col]) ** 2
        clipped = np.clip(scored["probability"], EPSILON, 1 - EPSILON)
        scored["log_loss_row"] = -(
            scored[outcome_col] * np.log(clipped)
            + (1 - scored[outcome_col]) * np.log(1 - clipped)
        )
        for fold, group in scored.groupby(fold_col, sort=True, observed=True):
            metrics = probability_metrics(group["probability"], group[outcome_col], n_bins=n_bins)
            brier_low, brier_high = _event_clustered_interval(
                group,
                value_col="brier_row",
                event_col=event_col,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )
            log_low, log_high = _event_clustered_interval(
                group,
                value_col="log_loss_row",
                event_col=event_col,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed + 1,
            )
            rows.append(
                {
                    "variant": variant,
                    "fold_id": fold,
                    **metrics,
                    "brier_ci_low": brier_low,
                    "brier_ci_high": brier_high,
                    "log_loss_ci_low": log_low,
                    "log_loss_ci_high": log_high,
                    "events": int(group[event_col].nunique()),
                }
            )
    return pd.DataFrame(rows)


def validate_ablation_coverage(
    results: pd.DataFrame,
    *,
    required_variants: tuple[str, ...] = REQUIRED_ABLATION_VARIANTS,
) -> AblationCoverage:
    """Require every documented ablation on exactly the same future folds."""

    required_columns = {"variant", "fold_id"}
    missing_columns = required_columns - set(results)
    if missing_columns:
        raise KeyError(f"ablation results missing columns: {sorted(missing_columns)}")
    observed = set(results["variant"].astype(str))
    missing = tuple(sorted(set(required_variants) - observed))
    reference_folds: set[str] | None = None
    inconsistent: list[str] = []
    for variant in required_variants:
        folds = set(results.loc[results["variant"].astype(str).eq(variant), "fold_id"].astype(str))
        if not folds:
            continue
        if reference_folds is None:
            reference_folds = folds
        elif folds != reference_folds:
            inconsistent.append(variant)
    search_trials = int(results["search_trials"].max()) if "search_trials" in results and len(results) else 0
    return AblationCoverage(not missing and not inconsistent, missing, tuple(inconsistent), search_trials)


def _slice_columns(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], tuple[str, ...]]:
    frame = records.copy()
    columns: dict[str, str] = {}
    missing: list[str] = []

    season_col = next((column for column in ("season", "grandPrixYear") if column in frame), None)
    if season_col is None and "event_id" in frame:
        frame["_slice_season"] = pd.to_numeric(
            frame["event_id"].astype(str).str.extract(r"^(\d{4})", expand=False), errors="coerce"
        )
        season_col = "_slice_season"
    columns["season"] = season_col or ""

    columns["circuit_archetype"] = next(
        (column for column in ("circuit_archetype", "circuit_type") if column in frame), ""
    )

    wet_col = next((column for column in ("wet_dry", "is_wet", "is_wet_race") if column in frame), None)
    if wet_col and wet_col != "wet_dry":
        def wet_label(value: object) -> str:
            if pd.isna(value):
                return "unknown"
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "wet"}:
                    return "wet"
                if normalized in {"0", "false", "no", "dry"}:
                    return "dry"
                return "unknown"
            return "wet" if bool(value) else "dry"

        frame["_slice_wet_dry"] = frame[wet_col].map(
            wet_label
        )
        wet_col = "_slice_wet_dry"
    columns["wet_dry"] = wet_col or ""

    grid_col = next(
        (
            column
            for column in ("grid_band", "grid_position", "resultsStartingGridPositionNumber")
            if column in frame
        ),
        None,
    )
    if grid_col and grid_col != "grid_band":
        grid = pd.to_numeric(frame[grid_col], errors="coerce")
        frame["_slice_grid_band"] = pd.cut(
            grid,
            [-np.inf, 0, 3, 10, np.inf],
            labels=["pitlane_or_unknown", "front_3", "grid_4_10", "grid_11_plus"],
        ).astype("string")
        grid_col = "_slice_grid_band"
    columns["grid_band"] = grid_col or ""

    rookie_col = next((column for column in ("rookie", "is_rookie") if column in frame), None)
    if rookie_col is None and "prior_career_starts" in frame:
        frame["_slice_rookie"] = pd.to_numeric(frame["prior_career_starts"], errors="coerce").lt(24)
        rookie_col = "_slice_rookie"
    columns["rookie"] = rookie_col or ""

    columns["constructor"] = next(
        (column for column in ("constructor_id", "constructorId", "constructorName") if column in frame), ""
    )

    coverage_col = next(
        (column for column in ("data_coverage", "source_coverage", "data_coverage_band") if column in frame),
        None,
    )
    if coverage_col and coverage_col != "data_coverage_band":
        coverage = pd.to_numeric(frame[coverage_col], errors="coerce")
        frame["_slice_data_coverage"] = pd.cut(
            coverage,
            [-np.inf, 0.8, 0.95, np.inf],
            labels=["low", "medium", "high"],
        ).astype("string")
        coverage_col = "_slice_data_coverage"
    columns["data_coverage"] = coverage_col or ""

    for dimension in REQUIRED_SLICE_DIMENSIONS:
        column = columns[dimension]
        if not column or frame[column].dropna().empty:
            missing.append(dimension)
    return frame, columns, tuple(missing)


def probability_slice_report(
    records: pd.DataFrame,
    *,
    probability_col: str = "probability",
    outcome_col: str = "outcome",
    event_col: str = "event_id",
    n_bins: int = 10,
    n_bootstrap: int = 1_000,
    random_seed: int = 42,
) -> SliceEvaluation:
    """Report required probability-quality slices with event-clustered CIs."""

    required = {probability_col, outcome_col, event_col}
    missing_columns = required - set(records)
    if missing_columns:
        raise KeyError(f"slice records missing columns: {sorted(missing_columns)}")
    frame, dimensions, missing = _slice_columns(records)
    frame[probability_col] = pd.to_numeric(frame[probability_col], errors="raise")
    frame[outcome_col] = pd.to_numeric(frame[outcome_col], errors="raise")
    frame["_slice_brier"] = (frame[probability_col] - frame[outcome_col]) ** 2
    rows: list[dict[str, object]] = []
    for dimension in REQUIRED_SLICE_DIMENSIONS:
        column = dimensions[dimension]
        if not column or dimension in missing:
            continue
        for value, group in frame.dropna(subset=[column]).groupby(column, sort=True, observed=True):
            metrics = probability_metrics(group[probability_col], group[outcome_col], n_bins=n_bins)
            low, high = _event_clustered_interval(
                group,
                value_col="_slice_brier",
                event_col=event_col,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    **metrics,
                    "brier_ci_low": low,
                    "brier_ci_high": high,
                    "events": int(group[event_col].nunique()),
                }
            )
    return SliceEvaluation(pd.DataFrame(rows), missing)
