from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.config import DATA_DIR, REPO_ROOT
from app.services import analysis, data, presentation

from .cache import ViewResponses, reusable
from .jobs import BusyQueueError, Jobs
from .metrics import BodyLimit, RequestMetrics, metrics_storage


def artifact_revision(data_dir: Path, repo_root: Path) -> str:
    """Cheap identity from atomic files' paths/sizes/mtimes; not a data content hash."""
    paths = list(data_dir.rglob("*")) if data_dir.is_dir() else []
    paths += list((repo_root/"fastapi_react"/"backend"/"app").rglob("*.py"))
    paths += [repo_root/"raceAnalysis.py"]
    inventory = []
    for path in sorted(paths):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet", ".json", ".pkl", ".pickle", ".joblib", ".py"}:
            continue
        stat = path.stat()
        inventory.append((str(path.relative_to(repo_root)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()


def clear_source_caches() -> None:
    # Call with the render lock held, also in the isolated job process.
    with presentation._LOCK:
        presentation._CACHE.clear()
    for module in (data, analysis):
        for function in vars(module).values():
            reset = getattr(function, "cache_clear", None)
            if callable(reset):
                reset()


def execute_research(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Top-level importable worker function, required by Windows process spawning."""
    if context["revision"] != artifact_revision(DATA_DIR, REPO_ROOT):
        raise ValueError("Artifacts changed after the job was queued; submit it again.")
    values = dict(context["values"])
    with presentation._RENDER_LOCK:
        clear_source_caches()
    if task == "bin-comparison":
        q_values = values.get("Select q values (number of bins)", [2])
        if not isinstance(q_values, list) or not q_values or len(q_values) > 9 or any(type(q) is not int or not 2 <= q <= 10 for q in q_values):
            raise ValueError("Choose one to nine q values from 2 through 10.")
        values["Select q values (number of bins)"] = q_values
        values["_tabs:📊 Model Performance"] = 6
        return presentation.render_view(5, values, "Run Bin Count Comparison")
    if task == "leakage-audit":
        rows = values.get("Rows to read (0 = all)", 1000)
        if type(rows) is not int or not 0 <= rows <= 100000:
            raise ValueError("Audit row limit must be from 0 through 100000.")
        values["Rows to read (0 = all)"] = rows
        values["_tabs:Raw Data"] = 1
        return presentation.render_view(6, values, "Run Leakage Audit")
    raise ValueError("Unsupported research task.")


class JobRequest(BaseModel):
    task: str
    values: dict[str, Any] = Field(default_factory=dict)


class Enhancements:
    def __init__(self, *, poll_seconds: float = 1.0) -> None:
        self.guard = threading.RLock()
        self.revision = ""
        self.checked = 0.0
        self.poll_seconds = poll_seconds
        self.responses = ViewResponses(presentation.render_view)
        self.records, self.record_lock = metrics_storage()
        self.jobs = Jobs(execute_research)
        self.router = APIRouter(prefix="/api/enhancements", tags=["Optional enhancements"])
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.router.add_api_route("/jobs", self.submit, methods=["POST"], status_code=202)
        self.router.add_api_route("/jobs/{identity}", self.job_status, methods=["GET"])
        self.router.add_api_route("/jobs/{identity}/result", self.job_result, methods=["GET"])
        self.router.add_api_route("/jobs/{identity}", self.cancel, methods=["DELETE"])

    def current_revision(self) -> str:
        with self.guard:
            if not self.revision or time.monotonic()-self.checked >= self.poll_seconds:
                revision = artifact_revision(DATA_DIR, REPO_ROOT)
                if revision != self.revision:
                    with presentation._RENDER_LOCK:
                        clear_source_caches()
                        self.responses.clear()
                    self.revision = revision
                self.checked = time.monotonic()
            return self.revision

    def render(self, payload: Any, request: Request) -> Response:
        # Global rendering is already serial. Keep revision checking and view
        # rendering together so one request cannot clear another request's data.
        with self.guard:
            revision = self.current_revision()
            return self.responses.render(
                payload.page, payload.values, payload.action, revision,
                request.headers.get("accept-encoding", ""),
                enabled=os.environ.get("F1_VIEW_RESPONSE_CACHE", "0") == "1",
            )

    def status(self) -> dict[str, Any]:
        source = DATA_DIR/"f1ForAnalysis.parquet"
        if os.environ.get("F1_USE_PARQUET", "1").lower() not in {"1", "true", "yes"} or not source.exists():
            source = DATA_DIR/"f1ForAnalysis.csv"
        models = []
        keys = ("model_name", "model_version", "estimator", "trained_at", "training_end_event", "training_start_event", "calibration_method", "data_sha256", "schema_version", "notes")
        for path in sorted((DATA_DIR/"models").rglob("*manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                models.append({key: manifest.get(key) for key in keys})
            except (OSError, ValueError, TypeError):
                models.append({"model_name": path.stem, "notes": ["Manifest could not be read."]})
        return {
            "revision": self.current_revision(),
            "build_revision": os.environ.get("F1_BUILD_REVISION", "local-working-tree"),
            "dataset": {"name": source.name, "modified_at": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat() if source.exists() else None},
            "models": models,
        }

    @staticmethod
    def authorize(token: str | None) -> None:
        expected = os.environ.get("F1_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(503, "Local research jobs and metrics are disabled.")
        if token is None or not secrets.compare_digest(expected, token):
            raise HTTPException(403, "Administrator access is required.")

    def metrics(self, x_f1_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        self.authorize(x_f1_admin_token)
        with self.record_lock:
            return {"requests": list(self.records), "cache_bytes": self.responses.bytes}

    def submit(self, payload: JobRequest, x_f1_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        self.authorize(x_f1_admin_token)
        if payload.task not in {"bin-comparison", "leakage-audit"}:
            raise HTTPException(400, "Unsupported task.")
        if not reusable(1, payload.values, None):
            raise HTTPException(400, "Use ordinary control values; uploaded CSVs and ledger data are not accepted by research jobs.")
        try:
            identity = self.jobs.submit(payload.task, {"values": payload.values, "revision": self.current_revision()})
        except BusyQueueError as exc:
            raise HTTPException(429, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"id": identity, "state": "queued"}

    def job_status(self, identity: str, x_f1_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        self.authorize(x_f1_admin_token)
        try:
            return self.jobs.status(identity)
        except KeyError as exc:
            raise HTTPException(404, "Job not found or expired.") from exc

    def job_result(self, identity: str, x_f1_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        self.job_status(identity, x_f1_admin_token)
        try:
            return self.jobs.result(identity)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    def cancel(self, identity: str, x_f1_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        self.job_status(identity, x_f1_admin_token)
        return {"cancelled": self.jobs.cancel(identity)}

    def install(self, app: FastAPI) -> None:
        app.include_router(self.router)
        app.add_middleware(BodyLimit, max_bytes=int(os.environ.get("F1_MAX_REQUEST_BYTES", str(256*1024*1024))))
        # Install last: request timing includes routing, rendering and gzip.
        app.add_middleware(RequestMetrics, records=self.records, lock=self.record_lock)
