"""Coverage, freshness, missingness, and distribution-drift diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


def population_stability_index(
    reference: Iterable[float], current: Iterable[float], bins: int = 10
) -> float:
    ref = np.asarray(list(reference), dtype=float)
    cur = np.asarray(list(current), dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < bins or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_counts = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "rows": len(frame),
            "non_null": [int(frame[column].notna().sum()) for column in frame.columns],
            "missing_rate": [float(frame[column].isna().mean()) for column in frame.columns],
            "unique": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    ).sort_values(["missing_rate", "column"], ascending=[False, True])


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected = list(columns) if columns is not None else sorted(set(reference) & set(current))
    rows: list[dict[str, object]] = []
    for column in selected:
        if pd.api.types.is_numeric_dtype(reference[column]):
            psi = population_stability_index(reference[column], current[column])
            mean_delta = float(pd.to_numeric(current[column], errors="coerce").mean() - pd.to_numeric(reference[column], errors="coerce").mean())
        else:
            psi, mean_delta = float("nan"), float("nan")
        rows.append(
            {
                "column": column,
                "psi": psi,
                "mean_delta": mean_delta,
                "reference_missing": float(reference[column].isna().mean()),
                "current_missing": float(current[column].isna().mean()),
                "missing_delta": float(current[column].isna().mean() - reference[column].isna().mean()),
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report["drift_level"] = pd.cut(
            report["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "watch", "action"]
        ).astype("string").fillna("not_numeric")
    return report


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    source: str
    expected_rows: int
    observed_rows: int
    captured_at: datetime
    newest_record_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.expected_rows < 0 or self.observed_rows < 0:
            raise ValueError("source row counts must be non-negative")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if self.newest_record_at is not None:
            if self.newest_record_at.tzinfo is None or self.newest_record_at.utcoffset() is None:
                raise ValueError("newest_record_at must be timezone-aware")
            if self.newest_record_at > self.captured_at:
                raise ValueError("newest_record_at cannot be later than captured_at")

    @property
    def coverage(self) -> float:
        return min(1.0, self.observed_rows / self.expected_rows) if self.expected_rows > 0 else 0.0

    @property
    def age_hours(self) -> float | None:
        if self.newest_record_at is None:
            return None
        now = self.captured_at.astimezone(timezone.utc)
        newest = self.newest_record_at.astimezone(timezone.utc)
        return (now - newest).total_seconds() / 3600.0


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    abstain: bool
    reason_codes: tuple[str, ...]


def evaluate_abstention(
    *,
    maximum_psi: float | None = None,
    critical_missingness_delta: float = 0.0,
    identity_coverage: float = 1.0,
    confirmed_grid_available: bool = True,
    weather_age_hours: float | None = None,
    maximum_weather_age_hours: float = 6.0,
    artifact_matches: bool = True,
    regulation_history_available: bool = True,
    coherence_issues: Iterable[str] = (),
    odds_snapshot_complete: bool = True,
) -> AbstentionDecision:
    """Apply the automatic abstention conditions from release Gate 7."""

    reasons: list[str] = []
    if maximum_psi is not None and np.isfinite(maximum_psi) and maximum_psi > 0.25:
        reasons.append("material_numeric_drift")
    if critical_missingness_delta > 0.10:
        reasons.append("critical_source_missingness")
    if identity_coverage < 1.0:
        reasons.append("identity_coverage_incomplete")
    if not confirmed_grid_available:
        reasons.append("confirmed_grid_unavailable")
    if weather_age_hours is None or weather_age_hours > maximum_weather_age_hours:
        reasons.append("forecast_weather_stale")
    if not artifact_matches:
        reasons.append("artifact_mismatch")
    if not regulation_history_available:
        reasons.append("regulation_history_unavailable")
    if tuple(coherence_issues):
        reasons.append("probability_coherence_failed")
    if not odds_snapshot_complete:
        reasons.append("odds_snapshot_incomplete")
    return AbstentionDecision(bool(reasons), tuple(reasons))
