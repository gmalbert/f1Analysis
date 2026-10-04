from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from app.config import DATA_DIR, REPO_ROOT
from app.schemas import ViewRequest
from app.services import analysis, data, presentation

from .auth import authorize_research, research_access
from .cache import ViewResponses
from .jobs import BusyQueueError, Jobs
from .metrics import RequestMetrics, metrics_storage

ARTIFACT_SUFFIXES = frozenset(
    {".csv", ".tsv", ".parquet", ".json", ".pkl", ".pickle", ".joblib", ".py", ".txt", ".html", ".png"}
)


def enabled(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes"}


class ArtifactChangedError(RuntimeError):
    """No stable artifact revision was available while a response was rendered."""


def artifact_revision(data_dir: Path, repo_root: Path) -> str:
    """Stat identity, not content integrity; atomic publication must change mtime."""
    roots = (data_dir, repo_root / "fastapi_react" / "backend" / "app")
    paths: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for directory, subdirectories, filenames in root.walk():
            # FastF1's downloaded telemetry is not a presentation source. Avoid
            # enumerating thousands of .ff1pkl blobs during every revision check.
            subdirectories[:] = [name for name in subdirectories if name not in {"f1_cache", "__pycache__"}]
            paths.extend(directory / name for name in filenames)
    paths.extend((repo_root / "raceAnalysis.py", repo_root / "model_artifacts.py"))
    paths.extend((repo_root / "f1bet").glob("*.py"))
    inventory = []
    for path in sorted(paths):
        if path.suffix.lower() not in ARTIFACT_SUFFIXES:
            continue
        # Next Race writes these output downloads during every render. They are
        # not inputs to the presentation; watching them would invalidate itself.
        if path.parent == data_dir and path.match("predictions_*.csv"):
            continue
        try:
            metadata = path.stat()
        except FileNotFoundError:
            # A file removed during enumeration will change the next identity.
            continue
        if not stat.S_ISREG(metadata.st_mode):
            continue
        inventory.append((str(path.relative_to(repo_root)), metadata.st_size, metadata.st_mtime_ns))
    identity = [inventory, enabled("F1_USE_PARQUET")]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def clear_source_caches() -> None:
    # Shared Matplotlib state and model/data caches are only cleared between renders.
    with presentation._RENDER_LOCK, presentation._LOCK, presentation._MODEL_LOCK:
        presentation._CACHE.clear()
        for module in (data, analysis):
            for function in vars(module).values():
                reset = getattr(function, "cache_clear", None)
                if callable(reset):
                    reset()


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    values: dict[str, Any] = Field(default_factory=dict)


def research_values(task: str, values: dict[str, Any]) -> dict[str, Any]:
    """Only task parameters enter the worker, never arbitrary uploads or view state."""
    if task == "leakage-audit":
        key = "Rows to read (0 = all)"
        rows = values.get(key, 1000)
        if set(values) - {key} or type(rows) is not int or not 1 <= rows <= 100000:
            raise ValueError("Choose an audit row limit from 1 through 100000.")
        return {key: rows}
    if task == "bin-comparison":
        key = "Select q values (number of bins)"
        bins = values.get(key, [2])
        if (
            set(values) - {key}
            or not isinstance(bins, list)
            or not 1 <= len(bins) <= 9
            or any(type(q) is not int or not 2 <= q <= 10 for q in bins)
            or len(set(bins)) != len(bins)
        ):
            raise ValueError("Choose one to nine distinct bin counts from 2 through 10.")
        return {key: sorted(bins)}
    raise ValueError("Unsupported research task.")


def execute_research(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Importable Windows-spawn worker using existing calculations outside HTTP rendering."""
    revision = context["revision"]
    if revision != artifact_revision(DATA_DIR, REPO_ROOT):
        raise ArtifactChangedError("Artifacts changed after submission; submit a new job.")
    values = research_values(task, context["values"])
    clear_source_caches()
    if task == "bin-comparison":
        values["_tabs:📊 Model Performance"] = 6
        result = presentation.render_view(5, values, "Run Bin Count Comparison")
    else:
        values["_tabs:Raw Data"] = 1
        result = presentation.render_view(6, values, "Run Leakage Audit")
    if revision != artifact_revision(DATA_DIR, REPO_ROOT):
        raise ArtifactChangedError("Artifacts changed during calculation; submit a new job.")
    return {**result, "source_revision": revision, "task": task}


class Enhancements:
    def __init__(self, *, poll_seconds: float = 1.0) -> None:
        self.guard = threading.RLock()
        self.revision = ""
        self.checked = 0.0
        self.poll_seconds = poll_seconds
        # Resolve dynamically so tests and development instrumentation can wrap rendering.
        self.responses = ViewResponses(
            lambda page, values, action: presentation.render_view(page, values, action)
        )
        self.records, self.record_lock = metrics_storage()
        self.jobs: Jobs | None = None
        self.router = APIRouter(prefix="/api/enhancements", tags=["Analysis enhancements"])
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/research-access", research_access, methods=["GET"])
        protected = [Depends(authorize_research)]
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"], dependencies=protected)
        self.router.add_api_route("/jobs", self.submit, methods=["POST"], status_code=202, dependencies=protected)
        self.router.add_api_route("/jobs/{identity}", self.job_status, methods=["GET"], dependencies=protected)
        self.router.add_api_route("/jobs/{identity}/result", self.job_result, methods=["GET"], dependencies=protected)
        self.router.add_api_route("/jobs/{identity}", self.cancel, methods=["DELETE"], dependencies=protected)

    def current_revision(self, *, force: bool = False) -> str:
        with self.guard:
            if force or not self.revision or time.monotonic() - self.checked >= self.poll_seconds:
                revision = artifact_revision(DATA_DIR, REPO_ROOT)
                if revision != self.revision:
                    clear_source_caches()
                    self.responses.clear()
                    self.revision = revision
                self.checked = time.monotonic()
            return self.revision

    def refresh_sources(self) -> None:
        """Shared API dependency: other data endpoints also observe artifact changes."""
        self.current_revision()

    def render(self, payload: ViewRequest, request: Request) -> Response:
        if payload.action in {"Run Leakage Audit", "Run Bin Count Comparison"}:
            raise HTTPException(409, "Use Research jobs to queue this calculation.")
        # Keep revision checks and rendering together. Recheck the disk after rendering
        # before retaining/returning a response. Never repeat an explicit action.
        with self.guard:
            for _attempt in range(2):
                revision = self.current_revision(force=True)
                result = self.responses.render(
                    payload.page,
                    payload.values,
                    payload.action,
                    revision,
                    request.headers.get("accept-encoding", ""),
                    enabled=enabled("F1_VIEW_RESPONSE_CACHE"),
                )
                if self.current_revision(force=True) == revision:
                    return result
                if payload.action:
                    break
            raise ArtifactChangedError("Source artifacts changed during analysis. Refresh and try again.")

    def status(self) -> dict[str, Any]:
        with self.guard:
            revision = self.current_revision(force=True)
            source = DATA_DIR / "f1ForAnalysis.parquet"
            if not enabled("F1_USE_PARQUET") or not source.exists():
                source = DATA_DIR / "f1ForAnalysis.csv"
            try:
                modified = datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat()
            except FileNotFoundError:
                modified = None
            models = []
            keys = (
                "model_name",
                "model_version",
                "estimator",
                "trained_at",
                "training_end_event",
                "training_start_event",
                "calibration_method",
                "data_sha256",
                "schema_version",
                "notes",
            )
            for path in sorted((DATA_DIR / "models").rglob("*manifest.json")):
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(manifest, dict):
                        raise TypeError("Expected a model manifest object")
                    models.append({key: manifest.get(key) for key in keys})
                except (OSError, ValueError, TypeError):
                    models.append({"model_name": path.stem, "notes": ["Manifest could not be read."]})
            return {
                "revision": revision,
                "build_revision": os.environ.get("F1_BUILD_REVISION", "local-working-tree"),
                "dataset": {"name": source.name, "modified_at": modified},
                "models": models,
            }

    def metrics(self) -> dict[str, Any]:
        with self.record_lock, self.responses.lock:
            return {"requests": list(self.records), "cache_bytes": self.responses.bytes}

    def submit(self, payload: JobRequest) -> dict[str, Any]:
        try:
            values = research_values(payload.task, payload.values)
            with self.guard:
                revision = self.current_revision(force=True)
                if self.jobs is None:
                    self.jobs = Jobs(execute_research)
                identity = self.jobs.submit(payload.task, {"values": values, "revision": revision})
            return self.jobs.status(identity)
        except BusyQueueError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    def job_status(self, identity: str) -> dict[str, Any]:
        try:
            if self.jobs is None:
                raise KeyError(identity)
            return self.jobs.status(identity)
        except KeyError as exc:
            raise HTTPException(404, "Job not found or expired.") from exc

    def job_result(self, identity: str) -> dict[str, Any]:
        self.job_status(identity)
        try:
            if self.jobs is None:
                raise KeyError(identity)
            return self.jobs.result(identity)
        except KeyError as exc:
            raise HTTPException(404, "Job not found or expired.") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    def cancel(self, identity: str) -> dict[str, Any]:
        self.job_status(identity)
        try:
            if self.jobs is None:
                raise KeyError(identity)
            cancelled = self.jobs.cancel(identity)
            return {"cancelled": cancelled, "job": self.jobs.status(identity)}
        except KeyError as exc:
            raise HTTPException(404, "Job not found or expired.") from exc

    def close(self) -> None:
        if self.jobs is not None:
            self.jobs.close()

    def install(self, app: FastAPI) -> None:
        app.include_router(self.router)
        # Install last: timings and byte counts include routing, rendering and gzip.
        app.add_middleware(
            RequestMetrics, records=self.records, lock=self.record_lock, emit_logs=enabled("F1_REQUEST_LOGS")
        )
