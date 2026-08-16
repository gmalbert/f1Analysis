"""Evidence-backed implementation of the validation and release gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import ModelManifest
from .backtest import BacktestResult
from .calibration import calibration_table, probability_metrics
from .contracts import (
    DECISION_LEDGER_CONTRACT,
    FEATURE_SNAPSHOT_CONTRACT,
    FORECAST_LEDGER_CONTRACT,
    ODDS_LEDGER_CONTRACT,
    RACE_MODEL_CONTRACT,
    DatasetContract,
)
from .evaluation import AblationCoverage, SliceEvaluation
from .monitoring import AbstentionDecision
from .validation import WalkForwardFold, assert_strictly_future


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    status: GateStatus
    checks: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    generated_at: str
    gates: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(gate.status is GateStatus.PASS for gate in self.gates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "passed": self.passed,
            "gates": [
                {
                    **asdict(gate),
                    "status": gate.status.value,
                }
                for gate in self.gates
            ],
        }


def _contract_result(name: str, frame: pd.DataFrame | None, contract: DatasetContract) -> GateResult:
    if frame is None:
        return GateResult(name, GateStatus.NOT_EVALUATED, failures=("dataset was not supplied",))
    report = contract.validate(frame)
    failures = tuple(issue.message for issue in report.issues if issue.severity == "error")
    return GateResult(
        name,
        GateStatus.PASS if not failures else GateStatus.FAIL,
        checks=(f"{contract.name} contract",),
        failures=failures,
        metrics={"rows": len(frame), "issues": len(report.issues)},
    )


def data_integrity_gate(
    *,
    race_snapshot: pd.DataFrame | None = None,
    odds_ledger: pd.DataFrame | None = None,
    forecast_ledger: pd.DataFrame | None = None,
    decision_ledger: pd.DataFrame | None = None,
    identity_coverage: float | None = None,
    source_coverage: Mapping[str, float] | None = None,
    source_freshness_hours: Mapping[str, float] | None = None,
    strict_evidence: bool = True,
) -> GateResult:
    supplied = [
        ("race", race_snapshot, RACE_MODEL_CONTRACT),
        ("odds", odds_ledger, ODDS_LEDGER_CONTRACT),
        ("forecast", forecast_ledger, FORECAST_LEDGER_CONTRACT),
        ("decision", decision_ledger, DECISION_LEDGER_CONTRACT),
    ]
    if not any(frame is not None for _, frame, _ in supplied):
        return GateResult("data_integrity", GateStatus.NOT_EVALUATED, failures=("no v2 datasets supplied",))
    checks: list[str] = []
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    for label, frame, contract in supplied:
        if frame is None:
            continue
        result = _contract_result(label, frame, contract)
        checks.extend(result.checks)
        failures.extend(f"{label}: {failure}" for failure in result.failures)
        metrics[f"{label}_rows"] = len(frame)
    if race_snapshot is not None:
        for required in ("schema_version", "feature_as_of", "feature_stage"):
            if required not in race_snapshot:
                failures.append(f"race: missing lineage column {required}")
        if "source_manifest_id" not in race_snapshot:
            failures.append("race: missing source_manifest_id")
        if "snapshot_id" in race_snapshot:
            snapshot_result = _contract_result("snapshot", race_snapshot, FEATURE_SNAPSHOT_CONTRACT)
            checks.extend(snapshot_result.checks)
            failures.extend(f"snapshot: {failure}" for failure in snapshot_result.failures)
        if strict_evidence and identity_coverage is None:
            failures.append("identity coverage was not reported")
        if strict_evidence and source_coverage is None:
            failures.append("source coverage was not reported")
        if strict_evidence and source_freshness_hours is None:
            failures.append("source freshness was not reported")
    if identity_coverage is not None:
        metrics["identity_coverage"] = identity_coverage
        if not np.isfinite(identity_coverage) or not 0 <= identity_coverage <= 1:
            failures.append("identity coverage must be finite and in [0, 1]")
        elif identity_coverage < 1.0:
            failures.append("identity coverage is below 100%")
    if source_coverage is not None:
        metrics["source_coverage"] = dict(source_coverage)
        if strict_evidence and not source_coverage:
            failures.append("source coverage report is empty")
        invalid_coverage = [
            source
            for source, coverage in source_coverage.items()
            if not np.isfinite(coverage) or not 0 <= coverage <= 1
        ]
        if invalid_coverage:
            failures.append(f"invalid source coverage: {sorted(invalid_coverage)}")
        incomplete = [
            source
            for source, coverage in source_coverage.items()
            if np.isfinite(coverage) and 0 <= coverage < 1.0
        ]
        if incomplete:
            failures.append(f"incomplete source coverage: {sorted(incomplete)}")
    if source_freshness_hours is not None:
        metrics["source_freshness_hours"] = dict(source_freshness_hours)
        if strict_evidence and not source_freshness_hours:
            failures.append("source freshness report is empty")
        invalid_freshness = [
            source
            for source, hours in source_freshness_hours.items()
            if not np.isfinite(hours) or hours < 0
        ]
        if invalid_freshness:
            failures.append(f"invalid source freshness: {sorted(invalid_freshness)}")
    if race_snapshot is not None and forecast_ledger is not None and "snapshot_id" in race_snapshot:
        snapshots = race_snapshot[["snapshot_id", "feature_as_of"]].drop_duplicates("snapshot_id")
        linked = forecast_ledger.merge(
            snapshots,
            left_on="feature_snapshot_id",
            right_on="snapshot_id",
            how="left",
            validate="many_to_one",
        )
        missing_snapshot = linked["feature_as_of"].isna()
        if missing_snapshot.any():
            failures.append("forecast references a missing feature snapshot")
        else:
            feature_time = pd.to_datetime(linked["feature_as_of"], utc=True, errors="coerce")
            forecast_time = pd.to_datetime(linked["generated_at"], utc=True, errors="coerce")
            if (feature_time > forecast_time).any():
                failures.append("feature snapshot is later than its forecast")
    if decision_ledger is not None:
        if forecast_ledger is None or odds_ledger is None:
            failures.append("decision lineage requires both forecast and odds ledgers")
        else:
            placed_decisions = decision_ledger.loc[
                decision_ledger["status"].astype("string").eq("placed")
            ]
            lineage = placed_decisions.merge(
                forecast_ledger[["forecast_id", "generated_at"]],
                on="forecast_id",
                how="left",
                validate="many_to_one",
            ).merge(
                odds_ledger[["quote_id", "captured_at"]],
                on="quote_id",
                how="left",
                validate="many_to_one",
            )
            if lineage[["generated_at", "captured_at"]].isna().any().any():
                failures.append("decision references missing forecast or quote lineage")
            else:
                decided = pd.to_datetime(lineage["decided_at"], utc=True, errors="coerce")
                generated = pd.to_datetime(lineage["generated_at"], utc=True, errors="coerce")
                captured = pd.to_datetime(lineage["captured_at"], utc=True, errors="coerce")
                if (generated > decided).any() or (captured > decided).any():
                    failures.append("decision uses a forecast or quote from the future")
                if (captured > generated).any():
                    failures.append("decision quote was captured after the frozen forecast")
    return GateResult(
        "data_integrity",
        GateStatus.FAIL if failures else GateStatus.PASS,
        tuple(checks),
        tuple(failures),
        metrics,
    )


def reproducibility_gate(
    *,
    manifest: ModelManifest | None,
    data_path: str | Path | None = None,
    schema_version: str | None = None,
    feature_names: Iterable[str] | None = None,
    repeated_probabilities: tuple[Iterable[float], Iterable[float]] | None = None,
    tolerance: float = 1e-12,
    offline_tests_passed: bool | None = None,
    require_calibration: bool = True,
) -> GateResult:
    if manifest is None:
        return GateResult("reproducibility", GateStatus.NOT_EVALUATED, failures=("model manifest not supplied",))
    failures: list[str] = []
    checks = ["model manifest", "deterministic seed", "dependency versions"]
    if not manifest.feature_names:
        failures.append("manifest feature order is empty")
    if not manifest.data_sha256:
        failures.append("manifest dataset hash is empty")
    if not manifest.code_revision or manifest.code_revision == "unknown":
        failures.append("manifest code revision is missing or unknown")
    if not manifest.dependency_versions:
        failures.append("manifest dependency versions are missing")
    if require_calibration and not manifest.calibration_method:
        failures.append("calibration method is missing")
    if require_calibration and not (manifest.calibration_start_event and manifest.calibration_end_event):
        failures.append("calibration window is missing")
    if not manifest.hyperparameters:
        failures.append("hyperparameters are missing")
    if data_path is not None and schema_version is not None and feature_names is not None:
        verification = manifest.verify_compatibility(
            data_path=data_path,
            schema_version=schema_version,
            feature_names=feature_names,
        )
        failures.extend(verification.reasons)
    if repeated_probabilities is not None:
        first = np.asarray(list(repeated_probabilities[0]), dtype=float)
        second = np.asarray(list(repeated_probabilities[1]), dtype=float)
        if first.shape != second.shape or not np.allclose(first, second, atol=tolerance, rtol=0):
            failures.append("repeat run probabilities differ beyond tolerance")
    else:
        failures.append("repeat-run probability evidence was not supplied")
    if offline_tests_passed is False:
        failures.append("offline tests did not pass")
    elif offline_tests_passed is None:
        failures.append("offline test result was not supplied")
    return GateResult(
        "reproducibility",
        GateStatus.FAIL if failures else GateStatus.PASS,
        tuple(checks),
        tuple(failures),
        {"random_seed": manifest.random_seed, "search_trials": manifest.search_trials},
    )


def temporal_validation_gate(
    frame: pd.DataFrame | None,
    folds: Iterable[WalkForwardFold] | None,
    *,
    require_embargo: bool = True,
    final_season_untouched: bool | None = None,
    pipeline_fit_within_fold: bool | None = None,
    ablation_coverage: AblationCoverage | None = None,
    required_slices: SliceEvaluation | None = None,
) -> GateResult:
    if frame is None or folds is None:
        return GateResult("temporal_validation", GateStatus.NOT_EVALUATED, failures=("frame and folds are required",))
    fold_list = list(folds)
    failures: list[str] = []
    for fold in fold_list:
        try:
            assert_strictly_future(frame, fold)
        except (AssertionError, ValueError) as exc:
            failures.append(f"fold {fold.fold}: {exc}")
        if require_embargo and not fold.embargoed_events:
            failures.append(f"fold {fold.fold}: at least one embargoed event is required")
    if not fold_list:
        failures.append("no walk-forward folds were generated")
    if final_season_untouched is not True:
        failures.append("untouched final-season evidence was not supplied")
    if pipeline_fit_within_fold is not True:
        failures.append("fold-local preprocessing/calibration evidence was not supplied")
    if ablation_coverage is None:
        failures.append("required ablation coverage was not supplied")
    elif not ablation_coverage.complete:
        failures.extend(f"missing ablation: {name}" for name in ablation_coverage.missing_variants)
        failures.extend(f"inconsistent ablation folds: {name}" for name in ablation_coverage.inconsistent_folds)
    if required_slices is None:
        failures.append("required slice report was not supplied")
    elif required_slices.missing_dimensions:
        failures.append(f"missing validation slices: {list(required_slices.missing_dimensions)}")
    return GateResult(
        "temporal_validation",
        GateStatus.FAIL if failures else GateStatus.PASS,
        ("race-grouped future folds", "event embargo"),
        tuple(failures),
        {
            "folds": len(fold_list),
            "search_trials": ablation_coverage.search_trials if ablation_coverage else None,
        },
    )


def probability_quality_gate(
    records: pd.DataFrame | None,
    *,
    probability_col: str = "probability",
    outcome_col: str = "outcome",
    champion_probability_col: str | None = None,
    market_probability_col: str | None = "fair_market_probability",
    historical_probability_col: str | None = "historical_baseline_probability",
    n_bins: int = 10,
    slice_evaluation: SliceEvaluation | None = None,
    coherence_issues: Iterable[str] | None = None,
    calibration_fitted_on_earlier_data: bool | None = None,
) -> GateResult:
    if records is None:
        return GateResult("probability_quality", GateStatus.NOT_EVALUATED, failures=("probability records not supplied",))
    required = {probability_col, outcome_col, "market"}
    missing = required - set(records)
    if missing:
        return GateResult("probability_quality", GateStatus.FAIL, failures=(f"missing columns: {sorted(missing)}",))
    grouping = ["market"] + (["stage"] if "stage" in records else [])
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for key, group in records.groupby(grouping, dropna=False, sort=True, observed=True):
        label = key if isinstance(key, tuple) else (key,)
        metrics = probability_metrics(group[probability_col], group[outcome_col], n_bins=n_bins)
        row: dict[str, Any] = {column: value for column, value in zip(grouping, label)}
        row.update(metrics)
        if champion_probability_col and champion_probability_col in group:
            champion = probability_metrics(group[champion_probability_col], group[outcome_col], n_bins=n_bins)
            row["champion_brier"] = champion["brier"]
            row["champion_log_loss"] = champion["log_loss"]
            if metrics["brier"] >= champion["brier"] or metrics["log_loss"] >= champion["log_loss"]:
                failures.append(f"{label}: challenger did not improve Brier and log loss")
        elif champion_probability_col:
            failures.append(f"{label}: champion probability is missing")
        if market_probability_col and market_probability_col in group:
            market = probability_metrics(group[market_probability_col], group[outcome_col], n_bins=n_bins)
            row["market_brier"] = market["brier"]
            row["market_log_loss"] = market["log_loss"]
        elif market_probability_col:
            failures.append(f"{label}: de-vigged market consensus is missing")
        if historical_probability_col and historical_probability_col in group:
            historical = probability_metrics(group[historical_probability_col], group[outcome_col], n_bins=n_bins)
            row["historical_brier"] = historical["brier"]
            row["historical_log_loss"] = historical["log_loss"]
        elif historical_probability_col:
            failures.append(f"{label}: historical baseline is missing")
        row["reliability"] = calibration_table(
            group[probability_col], group[outcome_col], n_bins=n_bins
        ).to_dict(orient="records")
        rows.append(row)
    if slice_evaluation is None:
        failures.append("required season/context slice report is missing")
    elif slice_evaluation.missing_dimensions:
        failures.append(f"missing probability slices: {list(slice_evaluation.missing_dimensions)}")
    if coherence_issues is None:
        failures.append("probability coherence evidence was not supplied")
    else:
        failures.extend(str(issue) for issue in coherence_issues)
    if calibration_fitted_on_earlier_data is not True:
        failures.append("earlier-only calibration evidence was not supplied")
    metrics_frame = pd.DataFrame(rows)
    return GateResult(
        "probability_quality",
        GateStatus.FAIL if failures else GateStatus.PASS,
        ("Brier", "log loss", "adaptive reliability", "ECE", "calibration slope/intercept", "ROC AUC"),
        tuple(failures),
        {"groups": metrics_frame.to_dict(orient="records"), "bin_count": n_bins},
    )


def market_replay_gate(
    result: BacktestResult | None,
    *,
    acknowledge_incomplete_season_window: bool = False,
    multiple_bookmakers_available: bool = True,
    maximum_event_concentration: float = 0.35,
) -> GateResult:
    if result is None:
        return GateResult("market_replay", GateStatus.NOT_EVALUATED, failures=("paper replay not supplied",))
    summary = result.summary
    failures: list[str] = []
    if summary.candidates < 300:
        failures.append("fewer than 300 frozen eligible decisions")
    settled = summary.wins + summary.losses
    if settled < 100:
        failures.append("fewer than 100 settled bets")
    if summary.mean_clv is None:
        failures.append("closing-line value is unavailable")
    elif summary.mean_clv <= 0:
        failures.append("mean closing-line value is not positive")
    if summary.mean_clv_ci_low is None or summary.mean_clv_ci_low <= 0:
        failures.append("event-clustered CLV interval does not exclude zero")
    if summary.roi <= 0:
        failures.append("commission-adjusted ROI is not positive")
    evidence_columns = {
        "bookmaker",
        "opening_odds",
        "closing_odds",
        "devig_method",
        "market_snapshot_complete",
        "rule_version",
    }
    missing_evidence = evidence_columns - set(result.ledger)
    if missing_evidence:
        failures.append(f"replay evidence columns are missing: {sorted(missing_evidence)}")
    else:
        if result.ledger[["opening_odds", "closing_odds"]].isna().any().any():
            failures.append("opening/taken/closing price evidence is incomplete")
        complete = result.ledger["market_snapshot_complete"].map(
            lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"1", "true", "yes"}
        )
        if not complete.all():
            failures.append("one or more de-vigged market snapshots are incomplete")
        for odds_column in ("opening_odds", "closing_odds"):
            numeric_odds = pd.to_numeric(result.ledger[odds_column], errors="coerce")
            if numeric_odds.isna().any() or (numeric_odds <= 1).any():
                failures.append(f"{odds_column} contains invalid evidence")
        if result.ledger["devig_method"].astype("string").str.strip().eq("").any():
            failures.append("de-vig method is missing")
        if result.ledger["rule_version"].astype("string").str.strip().eq("").any():
            failures.append("book settlement rule version is missing")
        if multiple_bookmakers_available and result.ledger["bookmaker"].nunique() < 2:
            failures.append("fewer than two bookmakers are represented")
    if "event_id" in result.ledger:
        seasons = result.ledger["event_id"].astype(str).str.extract(r"^(\d{4})", expand=False).dropna().nunique()
        if seasons < 2 and not acknowledge_incomplete_season_window:
            failures.append("paper evidence does not span more than one season")
    if summary.largest_event_exposure_share > maximum_event_concentration:
        failures.append("paper result is dominated by one race")
    if len(result.decisions) != summary.candidates:
        failures.append("not every candidate/abstention was retained")
    return GateResult(
        "market_replay",
        GateStatus.FAIL if failures else GateStatus.PASS,
        ("sample size", "CLV", "ROI", "drawdown", "exposure concentration", "abstentions"),
        tuple(failures),
        {field: getattr(summary, field) for field in summary.__dataclass_fields__},
    )


def risk_gate(policy_checks: Mapping[str, bool] | None = None) -> GateResult:
    if policy_checks is None:
        policy_checks = {
            "quarter_kelly": True,
            "uncertainty_haircut": True,
            "market_shrinkage": True,
            "per_bet_cap": True,
            "per_event_cap": True,
            "per_selection_cap": True,
            "edge_and_ev_gates": True,
            "drawdown_pause": True,
            "paper_only": True,
        }
    failures = tuple(name for name, passed in policy_checks.items() if not passed)
    return GateResult(
        "risk",
        GateStatus.FAIL if failures else GateStatus.PASS,
        tuple(policy_checks),
        failures,
    )


def drift_abstention_gate(decision: AbstentionDecision | None) -> GateResult:
    if decision is None:
        return GateResult("drift_abstention", GateStatus.NOT_EVALUATED, failures=("drift evidence not supplied",))
    return GateResult(
        "drift_abstention",
        GateStatus.FAIL if decision.abstain else GateStatus.PASS,
        ("PSI", "missingness", "identity", "grid", "weather", "artifact", "regulation", "coherence", "odds completeness"),
        decision.reason_codes,
    )


def software_release_gate(
    *,
    offline_tests_passed: bool,
    compile_passed: bool,
    browser_smoke_passed: bool,
    live_requests_during_discovery: bool = False,
    embedded_secrets: bool = False,
    workflow_permissions_reviewed: bool = False,
    action_versions_pinned: bool = False,
    artifacts_have_manifests: bool = False,
    predictions_loadable: bool = False,
) -> GateResult:
    checks = {
        "offline_tests": offline_tests_passed,
        "compile": compile_passed,
        "browser_smoke": browser_smoke_passed,
        "offline_discovery": not live_requests_during_discovery,
        "no_embedded_secrets": not embedded_secrets,
        "workflow_permissions": workflow_permissions_reviewed,
        "pinned_actions": action_versions_pinned,
        "artifact_manifests": artifacts_have_manifests,
        "predictions_loadable": predictions_loadable,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(
        "software_release",
        GateStatus.FAIL if failures else GateStatus.PASS,
        tuple(checks),
        failures,
    )


def build_release_report(*gates: GateResult) -> ReleaseReport:
    return ReleaseReport(datetime.now(timezone.utc).isoformat(), tuple(gates))
