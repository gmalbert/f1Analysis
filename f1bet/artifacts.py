"""Reproducible model manifests and champion/challenger promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import os
import subprocess
from typing import Any, Iterable, Mapping


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    data = Path(path).read_bytes()
    digest.update(data.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_name: str
    model_version: str
    schema_version: str
    trained_at: str
    training_start_event: str
    training_end_event: str
    feature_names: tuple[str, ...]
    target: str
    estimator: str
    hyperparameters: Mapping[str, Any]
    data_sha256: str
    code_revision: str
    metrics: Mapping[str, float]
    calibration_method: str | None = None
    random_seed: int = 42
    notes: tuple[str, ...] = ()
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    calibration_start_event: str | None = None
    calibration_end_event: str | None = None
    source_manifest_ids: tuple[str, ...] = ()
    search_trials: int = 0

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        model_version: str,
        schema_version: str,
        training_start_event: str,
        training_end_event: str,
        feature_names: Iterable[str],
        target: str,
        estimator: str,
        hyperparameters: Mapping[str, Any],
        data_path: str | Path,
        code_revision: str,
        metrics: Mapping[str, float],
        calibration_method: str | None = None,
        random_seed: int = 42,
        notes: Iterable[str] = (),
        dependency_names: Iterable[str] = ("numpy", "pandas", "scikit-learn"),
        calibration_start_event: str | None = None,
        calibration_end_event: str | None = None,
        source_manifest_ids: Iterable[str] = (),
        search_trials: int = 0,
    ) -> "ModelManifest":
        return cls(
            model_name=model_name,
            model_version=model_version,
            schema_version=schema_version,
            trained_at=datetime.now(timezone.utc).isoformat(),
            training_start_event=training_start_event,
            training_end_event=training_end_event,
            feature_names=tuple(feature_names),
            target=target,
            estimator=estimator,
            hyperparameters=dict(hyperparameters),
            data_sha256=sha256_file(data_path),
            code_revision=code_revision,
            metrics=dict(metrics),
            calibration_method=calibration_method,
            random_seed=random_seed,
            notes=tuple(notes),
            dependency_versions=installed_dependency_versions(dependency_names),
            calibration_start_event=calibration_start_event,
            calibration_end_event=calibration_end_event,
            source_manifest_ids=tuple(source_manifest_ids),
            search_trials=search_trials,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["feature_names"] = tuple(payload["feature_names"])
        payload["notes"] = tuple(payload.get("notes", ()))
        payload["source_manifest_ids"] = tuple(payload.get("source_manifest_ids", ()))
        payload.setdefault("dependency_versions", {})
        payload.setdefault("calibration_start_event", None)
        payload.setdefault("calibration_end_event", None)
        payload.setdefault("search_trials", 0)
        return cls(**payload)

    def verify_data(self, path: str | Path) -> bool:
        return sha256_file(path) == self.data_sha256

    def verify_compatibility(
        self,
        *,
        data_path: str | Path,
        schema_version: str,
        feature_names: Iterable[str],
    ) -> "ManifestVerification":
        reasons: list[str] = []
        if not self.verify_data(data_path):
            reasons.append("dataset SHA-256 mismatch")
        if self.schema_version != schema_version:
            reasons.append("schema version mismatch")
        if self.feature_names != tuple(feature_names):
            reasons.append("feature order mismatch")
        return ManifestVerification(not reasons, tuple(reasons))


def installed_dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[str(name)] = importlib_metadata.version(str(name))
        except importlib_metadata.PackageNotFoundError:
            versions[str(name)] = "not-installed"
    return versions


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    valid: bool
    reasons: tuple[str, ...]


def current_code_revision() -> str:
    revision = os.environ.get("GITHUB_SHA", "").strip()
    if revision:
        return revision
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_training_manifest(
    *,
    destination: str | Path,
    model_name: str,
    model_version: str,
    estimator: str,
    preprocessor: Any,
    training_frame: Any,
    data_path: str | Path,
    metrics: Mapping[str, float],
    target: str = "resultsFinalPositionNumber",
    schema_version: str = "legacy-wide-v1",
    hyperparameters: Mapping[str, Any] | None = None,
    random_seed: int = 42,
) -> ModelManifest:
    """Write a complete manifest beside a newly trained legacy model artifact."""

    feature_names = tuple(str(value) for value in getattr(preprocessor, "feature_names_in_", ()))
    if not feature_names:
        raise ValueError("fitted preprocessor does not expose exact feature_names_in_")
    columns = set(getattr(training_frame, "columns", ()))
    if {"grandPrixYear", "round"} - columns:
        raise KeyError("training frame requires grandPrixYear and round")
    order = training_frame[["grandPrixYear", "round"]].dropna().copy()
    order["grandPrixYear"] = order["grandPrixYear"].astype(int)
    order["round"] = order["round"].astype(int)
    order = order.sort_values(["grandPrixYear", "round"], kind="stable")
    if order.empty:
        raise ValueError("training frame has no ordered events")
    first, last = order.iloc[0], order.iloc[-1]
    start_event = getattr(
        preprocessor,
        "f1bet_training_start_event_",
        f"{int(first.grandPrixYear)}-R{int(first['round']):02d}-R",
    )
    end_event = getattr(
        preprocessor,
        "f1bet_training_end_event_",
        f"{int(last.grandPrixYear)}-R{int(last['round']):02d}-R",
    )
    manifest = ModelManifest.create(
        model_name=model_name,
        model_version=model_version,
        schema_version=schema_version,
        training_start_event=start_event,
        training_end_event=end_event,
        feature_names=feature_names,
        target=target,
        estimator=estimator,
        hyperparameters=dict(hyperparameters or {}),
        data_path=data_path,
        code_revision=current_code_revision(),
        metrics=metrics,
        random_seed=random_seed,
        notes=(
            "Legacy finishing-position artifact; not a calibrated market probability model.",
            f"Embargoed events: {tuple(getattr(preprocessor, 'f1bet_embargoed_events_', ()))!r}",
        ),
    )
    manifest.save(destination)
    return manifest


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


def champion_challenger_gate(
    champion: Mapping[str, float],
    challenger: Mapping[str, float],
    *,
    minimum_brier_improvement: float = 0.0,
    minimum_log_loss_improvement: float = 0.0,
    maximum_ece_increase: float = 0.005,
    require_positive_clv: bool = True,
) -> PromotionDecision:
    """Promote only when probability quality improves without calibration regression."""
    reasons: list[str] = []
    for key in ("brier", "log_loss", "ece"):
        if key not in champion or key not in challenger:
            reasons.append(f"missing required metric: {key}")
    if reasons:
        return PromotionDecision(False, tuple(reasons))
    if champion["brier"] - challenger["brier"] <= minimum_brier_improvement:
        reasons.append("Brier score did not improve enough")
    if champion["log_loss"] - challenger["log_loss"] <= minimum_log_loss_improvement:
        reasons.append("log loss did not improve enough")
    if challenger["ece"] - champion["ece"] > maximum_ece_increase:
        reasons.append("calibration error regressed")
    if require_positive_clv and challenger.get("mean_clv", float("-inf")) <= 0:
        reasons.append("challenger lacks positive out-of-sample CLV")
    return PromotionDecision(not reasons, tuple(reasons or ["all promotion gates passed"]))
