# Current backend implementation

Snapshot of installed main-application source, generated 2026-10-04 03:06:33Z. Use these with the existing repository and its unchanged data/model artifacts and exported reference view modules. The original proposal files are historical candidates. See [implementation policies and evidence](../ENHANCEMENTS.md). Binary marks/fonts and generated WebP images live in frontend/public; the original footer PNG is retained and the optimizer regenerates variants. No production deployment is performed by these files.

## backend/app/main.py

[Editable source](../backend/app/main.py) — SHA-256: `41604827409d4506329aee95d775f96d37d4e4c029765a32f8b06df4dfaaec91`

```python
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.config import DATA_DIR, ENABLE_EXPENSIVE_TOOLS, MODEL_TYPES, REPO_ROOT
from app.enhancements.auth import local_origins, trusted_local_enabled
from app.enhancements.cache import NegotiatedGZipMiddleware
from app.enhancements.metrics import BodyLimit
from app.enhancements.service import ArtifactChangedError, Enhancements, enabled
from app.schemas import (
    AnalyticsRequest,
    BettingValueRequest,
    QueryRequest,
    ToolRunRequest,
    ViewRequest,
)
from app.services.analysis import analytics, current_season, next_race_bundle, tire_strategy
from app.services.betting import governance, value_and_stake
from app.services.data import (
    filter_schema,
    list_data_files,
    model_manifest,
    precomputed,
    query_main,
    query_streamlit_raw_data,
    read_table,
    resolve_data_file,
    streamlit_table_schema,
)
from app.services.presentation import render_view
from app.services.tools import TOOLS, run_tool

enhancements = Enhancements() if enabled("F1_ENHANCEMENTS") else None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if enhancements is not None:
            await run_in_threadpool(enhancements.close)


app = FastAPI(
    title="F1 Analysis API",
    version="1.0.0",
    description="FastAPI backend for the React parity migration of raceAnalysis.py",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    dependencies=[Depends(enhancements.refresh_sources)] if enhancements is not None else [],
    lifespan=lifespan,
)
CODE_DEPLOYED_AT = datetime.now(UTC)
app.add_middleware(NegotiatedGZipMiddleware, minimum_size=1000, compresslevel=5)
app.add_middleware(BodyLimit, max_bytes=int(os.environ.get("F1_MAX_REQUEST_BYTES", str(1024 * 1024))))


@app.post("/api/views", response_model=dict[str, Any])
def view(payload: ViewRequest, request: Request) -> Response:
    try:
        if enhancements is not None:
            return enhancements.render(payload, request)
        # The presentation protocol already normalizes values to JSON primitives.
        # Avoid FastAPI recursively converting millions of table cells again.
        return JSONResponse(render_view(payload.page, payload.values, payload.action))
    except ArtifactChangedError as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("Could not render analysis page %s", payload.page)
        raise _http_error(exc) from exc


app.add_middleware(
    CORSMiddleware,
    allow_origins=local_origins() if trusted_local_enabled() else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Server-Timing", "X-F1-Cache", "X-F1-Revision"],
)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, "Requested resource was not found")
    if isinstance(exc, (KeyError, ValueError)):
        return HTTPException(400, "Invalid request")
    if isinstance(exc, PermissionError):
        return HTTPException(403, "Permission denied")
    return HTTPException(500, "Internal server error")


@app.get("/api/health")
def health() -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    return {
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "data_dir": str(DATA_DIR),
        "dataset_exists": (DATA_DIR / "f1ForAnalysis.csv").exists(),
        "rss_mb": round(process.memory_info().rss / 1024 / 1024, 1),
        "expensive_tools_enabled": ENABLE_EXPENSIVE_TOOLS,
    }


@app.get("/api/brand/logo")
def brand_logo() -> FileResponse:
    """Serve the same Gridlocked mark used by the Streamlit reference."""
    # Match the reference's 450px PNG encoding rather than resizing the
    # original full-resolution asset independently in each browser.
    logo = REPO_ROOT / "fastapi_react" / "frontend" / "public" / "gridlocked-logo.png"
    if not logo.is_file():
        logo = DATA_DIR / "gridlocked-logo-with-text.png"
    if not logo.is_file():
        raise HTTPException(404, "Brand logo is unavailable")
    return FileResponse(logo, media_type="image/png")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    data_files = [path for path in DATA_DIR.iterdir() if path.is_file()] if DATA_DIR.is_dir() else []
    latest_data_file = max(data_files, key=lambda path: path.stat().st_mtime, default=None)
    return {
        "last_updated": (
            datetime.fromtimestamp(latest_data_file.stat().st_mtime).strftime("%Y-%m-%d %I:%M %p")
            if latest_data_file is not None
            else "No data files found"
        ),
        "deployed_at": CODE_DEPLOYED_AT.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tabs": [
            "Data Explorer",
            "Analytics",
            "Current Season",
            "Next Race",
            "Predictive Models",
            "Raw Data",
            "Betting Research",
        ],
        "models": MODEL_TYPES,
        "expensive_tools_enabled": ENABLE_EXPENSIVE_TOOLS,
        "manual_tools": list(TOOLS),
    }


@app.get("/api/data-explorer/schema")
def data_explorer_schema() -> dict[str, Any]:
    try:
        return {"filters": filter_schema()}
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/data-explorer/display-schema")
def data_explorer_display_schema() -> dict[str, Any]:
    try:
        return streamlit_table_schema()
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/raw/analysis-data")
def raw_analysis_data(request: QueryRequest) -> dict[str, Any]:
    try:
        return query_streamlit_raw_data(request.offset, request.limit)
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/data-explorer/query")
def data_explorer_query(request: QueryRequest) -> dict[str, Any]:
    try:
        return query_main(request)
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/analytics")
def analytics_route(request: AnalyticsRequest) -> dict[str, Any]:
    try:
        return analytics(request.filters, request.max_rows)
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/current-season")
def season_route() -> dict[str, Any]:
    try:
        return current_season()
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/next-race")
def next_race_route() -> dict[str, Any]:
    try:
        return next_race_bundle()
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/analytics/tire-strategy")
def tire_strategy_route(
    year: int | None = Query(default=None), event_name: str | None = Query(default=None)
) -> dict[str, Any]:
    """Return the tire-strategy tables and chart data for a year and race."""
    try:
        return tire_strategy(year, event_name)
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"models": MODEL_TYPES}


@app.get("/api/models/manifest")
def model_manifest_route(model_type: str = Query(...)) -> dict[str, Any]:
    try:
        return {"model_type": model_type, "manifest": model_manifest(model_type)}
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/models/precomputed/{name}")
def model_precomputed(name: str) -> dict[str, Any]:
    try:
        return {"name": name, "data": precomputed(name)}
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/raw/files")
def raw_files() -> dict[str, Any]:
    return {"files": list_data_files()}


@app.get("/api/raw/preview")
def raw_preview(path: str = Query(...)) -> dict[str, Any]:
    try:
        target = resolve_data_file(path)
        return {"path": path, **read_table(target)}
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/raw/download")
def raw_download(path: str = Query(...)) -> FileResponse:
    try:
        target = resolve_data_file(path)
        return FileResponse(target, filename=target.name)
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/betting/value")
def betting_value(payload: BettingValueRequest) -> dict[str, Any]:
    try:
        return value_and_stake(payload)
    except Exception as exc:
        raise _http_error(exc) from None


@app.get("/api/betting/governance")
def betting_governance() -> dict[str, Any]:
    try:
        return governance()
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/tools/run")
def tools_run(payload: ToolRunRequest) -> dict[str, Any]:
    try:
        return run_tool(payload.tool, payload.args)
    except Exception as exc:
        raise _http_error(exc) from None


if enhancements is not None:
    enhancements.install(app)
```

## backend/app/config.py

[Editable source](../backend/app/config.py) — SHA-256: `d20f0e1a20ad19ab34bb3d21c479d6da6768fb16500ddd3d2a9172b5a8e6f406`

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_REPO_ROOT = HERE.parents[3]
REPO_ROOT = Path(os.environ.get("F1_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
DATA_DIR = REPO_ROOT / "data_files"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"
MODELS_DIR = DATA_DIR / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENABLE_EXPENSIVE_TOOLS = os.environ.get("ENABLE_EXPENSIVE_TOOLS", "0").strip().lower() in {"1", "true", "yes"}
MAX_TABLE_ROWS = int(os.environ.get("MAX_TABLE_ROWS", "1000"))
CACHE_VERSION = os.environ.get("F1_CACHE_VERSION", "v3.3")

MODEL_TYPES = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Ensemble (XGBoost + LightGBM + CatBoost)",
    "Position Group",
    "Track-Weighted Ensemble",
]
```

## backend/app/schemas.py

[Editable source](../backend/app/schemas.py) — SHA-256: `6dfd4f20709e6868876682a4cd26a4bf91d10c733ece5a085f71bd060337dd70`

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    column: str
    kind: Literal["range", "date_range", "exact", "boolean"]
    value: Any


class QueryRequest(BaseModel):
    filters: list[FilterSpec] = Field(default_factory=list)
    columns: list[str] | None = None
    sort: list[str] = Field(default_factory=list)
    descending: bool = False
    offset: int = 0
    limit: int = Field(default=200, ge=1, le=5000)


class AnalyticsRequest(BaseModel):
    filters: list[FilterSpec] = Field(default_factory=list)
    max_rows: int = Field(default=5000, ge=100, le=50000)


class BettingValueRequest(BaseModel):
    model_probability: float = Field(0.25, gt=0, lt=1)
    decimal_odds: float = Field(2.10, gt=1)
    opposing_odds: float = Field(1.80, gt=1)
    uncertainty: float = Field(0.02, ge=0, le=0.5)
    devig_method: Literal["multiplicative", "additive", "power"] = "multiplicative"
    bankroll: float = Field(10000, gt=0)


class SimulationEntry(BaseModel):
    driver_id: str
    constructor_id: str
    pace_score: float
    dnf_probability: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0)
    race_sensitivity: float = 1.0


class SimulationRequest(BaseModel):
    entries: list[SimulationEntry]
    simulations: int = Field(10000, ge=1000, le=50000)
    seed: int = 42


class RowsPayload(BaseModel):
    rows: list[dict[str, Any]]


class ToolRunRequest(BaseModel):
    tool: str
    args: list[str] = Field(default_factory=list)


class ViewRequest(BaseModel):
    page: int = Field(default=1, ge=1, le=7)
    values: dict[str, Any] = Field(default_factory=dict)
    action: str | None = None
```

## backend/app/enhancements/__init__.py

[Editable source](../backend/app/enhancements/__init__.py) — SHA-256: `3b36599d8e05938210005ee80c59ec0e2ed9298da5a469d5a086315ee8baa922`

```python
"""Bounded view reuse, artifact invalidation, and request diagnostics."""
```

## backend/app/enhancements/cache.py

[Editable source](../backend/app/enhancements/cache.py) — SHA-256: `ad373b1facc2f235a4fe0addef2a6e8bc2039a825525dffea2caaf2d19fe051a`

```python
from __future__ import annotations

import gzip
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import Receive, Scope, Send


def accepts_gzip(header: str) -> bool:
    """Honor explicit gzip exclusions, quality bounds, and wildcard acceptance."""
    choices: dict[str, float] = {}
    for item in header.lower().split(","):
        parts = [part.strip() for part in item.split(";")]
        quality = 1.0
        for part in parts[1:]:
            if part.startswith("q="):
                try:
                    quality = float(part[2:])
                except ValueError:
                    quality = 0.0
        choices[parts[0]] = quality if math.isfinite(quality) and 0 <= quality <= 1 else 0.0
    return choices.get("gzip", choices.get("*", 0.0)) > 0


class NegotiatedGZipMiddleware(GZipMiddleware):
    """Starlette's gzip responder otherwise accepts the literal `gzip;q=0`."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        original_scope = scope
        if scope["type"] == "http":
            headers = scope.get("headers", [])
            accepted = b",".join(value for key, value in headers if key.lower() == b"accept-encoding")
            encoding = b"gzip" if accepts_gzip(accepted.decode("latin-1")) else b"identity"
            scope = {
                **scope,
                "headers": [(key, value) for key, value in headers if key.lower() != b"accept-encoding"]
                + [(b"accept-encoding", encoding)],
            }
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Routing annotates the copied scope; retain its public route template
            # for the outer diagnostic middleware without exposing path values.
            if "route" in scope:
                original_scope["route"] = scope["route"]


def _ordinary(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, str) and len(value) <= 4096


def reusable(page: int, values: dict[str, Any], action: str | None) -> bool:
    """Only cache ordinary public controls on the five nonvolatile analysis pages."""
    if action or page not in {1, 2, 3, 4, 5} or len(values) > 200:
        return False
    for key, value in values.items():
        if len(key) > 200 or any(word in key.lower() for word in ("upload", "csv", "ledger")):
            return False
        if isinstance(value, list):
            if len(value) > 100 or not all(_ordinary(item) for item in value):
                return False
        elif not _ordinary(value):
            return False
    return True


@dataclass(frozen=True)
class Entry:
    body: bytes
    compressed: bytes
    expires: float

    @property
    def size(self) -> int:
        return len(self.body) + len(self.compressed)


class ViewResponses:
    def __init__(
        self,
        renderer: Callable[[int, dict[str, Any], str | None], dict[str, Any]],
        *,
        ttl: float = 20,
        max_bytes: int = 64 * 1024 * 1024,
        max_entries: int = 12,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.renderer, self.ttl, self.max_bytes = renderer, ttl, max_bytes
        self.max_entries, self.clock = max_entries, clock
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.bytes = 0

    def _expire(self) -> None:
        expired = [key for key, entry in self.entries.items() if entry.expires <= self.clock()]
        for key in expired:
            self.bytes -= self.entries.pop(key).size

    def render(
        self,
        page: int,
        values: dict[str, Any],
        action: str | None,
        revision: str,
        encoding: str,
        *,
        enabled: bool = True,
    ) -> Response:
        headers = {"Cache-Control": "no-store", "X-F1-Revision": revision, "Vary": "Accept-Encoding"}
        if not enabled or not reusable(page, values, action):
            if action or not reusable(1, values, None):
                self.clear()
            headers["X-F1-Cache"] = "BYPASS"
            return JSONResponse(self.renderer(page, values, action), headers=headers)
        key = json.dumps([revision, page, values], sort_keys=True, separators=(",", ":"), allow_nan=False)
        # A concurrent identical request waits for the first render, then reuses it.
        with self.lock:
            self._expire()
            entry = self.entries.pop(key, None)
            hit = entry is not None
            if entry is not None:
                self.bytes -= entry.size
            else:
                body = bytes(JSONResponse(self.renderer(page, values, action)).body)
                entry = Entry(body, gzip.compress(body, compresslevel=5, mtime=0), self.clock() + self.ttl)
            if entry.size <= self.max_bytes and self.max_entries > 0:
                self.entries[key] = entry
                self.bytes += entry.size
                while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old.size
            headers["X-F1-Cache"] = "HIT" if hit else "MISS"
            zipped = len(entry.body) >= 1000 and accepts_gzip(encoding)
            if zipped:
                headers["Content-Encoding"] = "gzip"
            return Response(
                entry.compressed if zipped else entry.body, media_type="application/json", headers=headers
            )
```

## backend/app/enhancements/metrics.py

[Editable source](../backend/app/enhancements/metrics.py) — SHA-256: `a51f8a5405c4d84691b18aed692b31336b32ecce59f2b00563893f5aad980774`

```python
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("f1.request")


def configure_request_logging() -> None:
    """Provide structured request records even without an explicit uvicorn config."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


class BodyLimit:
    """Validate aggregate bytes before the application can decode JSON or uploads."""

    def __init__(self, app: ASGIApp, max_bytes: int = 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("F1_MAX_REQUEST_BYTES must be a positive integer.")
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        lengths = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
        declared: int | None = None
        if lengths:
            # Reject duplicates and ambiguous signed/whitespace/comma encodings.
            if len(lengths) != 1 or not lengths[0] or not lengths[0].isdigit():
                await JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)(
                    scope, receive, send
                )
                return
            try:
                declared = int(lengths[0])
            except ValueError:
                declared = self.max_bytes + 1
        if declared is not None and declared > self.max_bytes:
            await self._too_large(scope, receive, send)
            return
        body = bytearray()
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        if declared is not None and declared != size:
            await JSONResponse({"detail": "Content-Length does not match the request body."}, status_code=400)(
                scope, receive, send
            )
            return
        buffered: bytes | None = bytes(body)
        del body

        async def replay() -> Message:
            nonlocal buffered
            if buffered is not None:
                message: Message = {"type": "http.request", "body": buffered, "more_body": False}
                buffered = None
                return message
            return await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(
            {"detail": "The request is too large."}, status_code=413
        )(scope, receive, send)


class RequestMetrics:
    def __init__(
        self,
        app: ASGIApp,
        records: deque[dict[str, Any]],
        lock: Any,
        *,
        emit_logs: bool = True,
    ) -> None:
        self.app, self.records, self.lock, self.emit_logs = app, records, lock, emit_logs
        if emit_logs:
            configure_request_logging()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start, request_id = time.perf_counter(), uuid.uuid4().hex
        status, sent, header_ms = 500, 0, None

        async def measured_send(message: Message) -> None:
            nonlocal status, sent, header_ms
            if message["type"] == "http.response.start":
                status = message["status"]
                header_ms = round(1000 * (time.perf_counter() - start), 2)
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"x-request-id", request_id.encode()),
                        (b"server-timing", ("backend;dur=" + format(header_ms, ".2f")).encode()),
                    ],
                }
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        except Exception:
            if header_ms is None:
                # ServerErrorMiddleware sits outside user middleware. Start its
                # generic 500 here so failures also carry timing and an ID, then
                # re-raise for the server's normal exception logging behavior.
                await JSONResponse({"detail": "Internal server error"}, status_code=500)(
                    scope, receive, measured_send
                )
            raise
        finally:
            record = {
                "request_id": request_id,
                "method": scope["method"],
                "route": getattr(scope.get("route"), "path", "<unmatched>"),
                "status": status,
                "header_ms": header_ms,
                "duration_ms": round(1000 * (time.perf_counter() - start), 2),
                "body_bytes": sent,
            }
            with self.lock:
                self.records.append(record)
            if self.emit_logs:
                log.info("%s", json.dumps(record))


def metrics_storage() -> tuple[deque[dict[str, Any]], Any]:
    return deque(maxlen=500), threading.Lock()
```

## backend/app/enhancements/service.py

[Editable source](../backend/app/enhancements/service.py) — SHA-256: `8dd9ee9459ea0d16158e161e11bad7f03eeeeb7a626ee104c4eda065c05419e7`

```python
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
```

## backend/app/enhancements/jobs.py

[Editable source](../backend/app/enhancements/jobs.py) — SHA-256: `b3209f0ac3c039092c441f55079ddd3d2d4ccdae9bca5309b3da912333f129e8`

```python
from __future__ import annotations

import gzip
import json
import logging
import multiprocessing
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, cast

log = logging.getLogger("f1.jobs")


class BusyQueueError(Exception):
    """No bounded queue slot is available."""


class Jobs:
    """One spawned calculation process; bounded state is local and non-durable."""

    def __init__(
        self,
        execute: Callable[[str, dict[str, Any]], dict[str, Any]],
        *,
        limit: int = 8,
        result_limit: int = 32 * 1024 * 1024,
        ttl: float = 600,
    ) -> None:
        if limit < 1 or result_limit < 1 or ttl <= 0:
            raise ValueError("Job limits must be positive.")
        self.execute, self.limit, self.result_limit, self.ttl = execute, limit, result_limit, ttl
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="f1-research")
        self.worker = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self.lock = threading.RLock()
        self.items: dict[str, dict[str, Any]] = {}
        self.futures: dict[str, Future[None]] = {}
        self.closed = False

    def submit(self, task: str, values: dict[str, Any]) -> str:
        # Serialization isolates caller mutation and bounds retained inputs.
        text = json.dumps(values, allow_nan=False)
        if len(text.encode()) > 64 * 1024:
            raise ValueError("Research job inputs must be below 64 KiB.")
        with self.lock:
            self._expire()
            if self.closed:
                raise BusyQueueError("The local research queue is shutting down.")
            if len(self.items) >= self.limit:
                raise BusyQueueError("The local research queue is full; wait for completed results to expire.")
            identity = uuid.uuid4().hex
            self.items[identity] = {
                "id": identity,
                "task": task,
                "state": "queued",
                "created": time.time(),
                "finished": None,
                "revision": values.get("revision"),
            }
            self.futures[identity] = self.pool.submit(self._run, identity, task, json.loads(text))
            return identity

    def _expire(self) -> None:
        now = time.time()
        for identity, item in list(self.items.items()):
            if item["finished"] is not None and now - item["finished"] > self.ttl:
                self.items.pop(identity)
                self.futures.pop(identity, None)

    def _run(self, identity: str, task: str, values: dict[str, Any]) -> None:
        with self.lock:
            self.items[identity]["state"] = "running"
        try:
            result = self.worker.submit(self.execute, task, values).result()
            if not isinstance(result, dict):
                raise ValueError("Expected a research result object.")
            body = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
            if len(body) > self.result_limit:
                raise ValueError("Research result exceeds the configured limit.")
            compressed = gzip.compress(body, compresslevel=5)
            if len(compressed) > self.result_limit:
                raise ValueError("Compressed research result exceeds the configured limit.")
            with self.lock:
                self.items[identity].update(state="succeeded", body=compressed)
        except Exception:  # Record a worker failure without exposing data or killing the coordinator.
            log.exception("Research job %s failed", identity)
            with self.lock:
                self.items[identity].update(
                    state="failed", error="Research calculation failed; see the server log using this job ID."
                )
        finally:
            with self.lock:
                self.items[identity]["finished"] = time.time()

    def status(self, identity: str) -> dict[str, Any]:
        with self.lock:
            self._expire()
            return {key: value for key, value in self.items[identity].items() if key != "body"}

    def result(self, identity: str) -> dict[str, Any]:
        with self.lock:
            self._expire()
            if self.items[identity]["state"] != "succeeded":
                raise ValueError("The job has not completed successfully.")
            body = self.items[identity]["body"]
        return cast("dict[str, Any]", json.loads(gzip.decompress(body)))

    def cancel(self, identity: str) -> bool:
        with self.lock:
            self._expire()
            if self.items[identity]["state"] != "queued" or not self.futures[identity].cancel():
                return False
            self.items[identity].update(state="cancelled", finished=time.time())
            return True

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for identity in list(self.items):
                self.cancel(identity)
        # Running calculations finish. Shutdown never kills a calculation midway.
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.worker.shutdown(wait=True, cancel_futures=True)
```

## backend/app/services/presentation.py

[Editable source](../backend/app/services/presentation.py) — SHA-256: `64c71f876247ca48a66b6121c1e4c76326c31ba6d1e7f453fcf3f2aca4292f58`

```python
"""Request-isolated Python view data for the native React interface.

The offline-exported views call this small presentation protocol. It carries
values, column configurations, charts and widget state, not Python objects or
executable browser code. Prediction pages only load offline-trained artifacts;
the explicit bin-count experiment retains the reference's opt-in computation.
Administrative audit actions use the shared structured audit implementation.
"""

from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import io
import json
import logging
import pickle
import threading
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from app.config import DATA_DIR, REPO_ROOT

_CACHE: OrderedDict[Any, Any] = OrderedDict()
_LOCK = threading.RLock()
_MODEL_LOCK = threading.RLock()
_RENDER_LOCK = threading.RLock()
_VIEW_FILE = Path(__file__).with_name("reference_views.py")
_CODE = compile(_VIEW_FILE.read_text(encoding="utf-8"), str(_VIEW_FILE), "exec")


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [clean(v) for v in value]
    return scalar(value)


def table_rows(frame: pd.DataFrame) -> list[list[Any]]:
    """Normalize whole numeric columns without rounding values or visiting each cell.

    Mixed/text/date columns retain scalar normalization. An object matrix keeps
    Python integers, booleans and float precision when converted back to rows.
    """
    values = frame.to_numpy(dtype=object, copy=True)
    for index, (_name, series) in enumerate(frame.items()):
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
            values[~np.isfinite(numeric), index] = None
        else:
            values[:, index] = np.fromiter(
                (scalar(value) for value in values[:, index]), dtype=object, count=len(frame)
            )
    rows: list[list[Any]] = values.tolist()
    return rows


class State(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class ColumnConfig:
    def __getattr__(self, kind: str) -> Any:
        def column(label: str | None = None, **kwargs: Any) -> dict[str, Any]:
            return {"label": label, "kind": kind, **clean(kwargs)}

        return column


class Container:
    def __init__(self, ui: Presentation, node: dict[str, Any]):
        self.ui = ui
        self.node = node

    def __enter__(self) -> Container:
        self.ui.stack.append(self.node["children"])
        self.ui.visibility.append(
            True if self.node.get("sidebar") else self.ui.visible and not self.node.get("hidden", False)
        )
        return self

    def __exit__(self, *_args: Any) -> None:
        self.ui.stack.pop()
        self.ui.visibility.pop()

    def __getattr__(self, method: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            with self:
                return getattr(self.ui, method)(*args, **kwargs)

        return call


class Presentation:
    def __init__(self, page: int, values: dict[str, Any], action: str | None = None):
        self.page = page
        self.values = dict(values)
        self.action = action
        self.nodes: list[dict[str, Any]] = []
        self.sidebar_nodes: list[dict[str, Any]] = []
        self.stack = [self.nodes]
        self.visibility = [True]
        self.sidebar = Container(self, {"children": self.sidebar_nodes, "sidebar": True})
        self.column_config = ColumnConfig()
        self.session_state = State()
        self.model_type = values.get("Select Model Type", "XGBoost")
        self.namespace: dict[str, Any] = {}
        self.widgets: dict[str, Any] = {}
        self.root_tabs = False

    @property
    def visible(self) -> bool:
        return self.visibility[-1]

    def add(self, kind: str, **props: Any) -> dict[str, Any]:
        node = {"type": kind, "id": f"n{len(self.stack[-1])}", **props}
        if self.visible:
            self.stack[-1].append(node)
        return node

    def group(self, kind: str, **props: Any) -> Container:
        return Container(self, self.add(kind, children=[], **props))

    def tabs(self, labels: list[str]) -> list[Container]:
        root = not self.root_tabs
        self.root_tabs = True
        node = self.add("tabs", labels=labels, root=root, children=[])
        selected = self.page - 1 if root else int(self.values.get("_tabs:" + labels[0], 0))
        tabs = []
        for i, label in enumerate(labels):
            child = {"type": "tab", "label": label, "children": [], "index": i, "hidden": i != selected}
            node["children"].append(child)
            tabs.append(Container(self, child))
        return tabs

    def columns(self, widths: Any, **kwargs: Any) -> list[Container]:
        widths = [1] * widths if isinstance(widths, int) else list(widths)
        node = self.add("columns", widths=widths, children=[])
        columns = []
        for width in widths:
            child = {"type": "column", "width": width, "children": []}
            node["children"].append(child)
            columns.append(Container(self, child))
        return columns

    def expander(self, label: str, expanded: bool = False, **kwargs: Any) -> Container:
        return self.group("expander", label=label, expanded=expanded)

    def spinner(self, *_args: Any, **_kwargs: Any) -> Container:
        return Container(self, {"children": self.stack[-1]})

    def set_page_config(self, **_kwargs: Any) -> None:
        pass

    def stop(self) -> None:
        raise ValueError("The analysis could not load its required data.")

    def title(self, text: str) -> None:
        self.add("heading", text=text, level=1)

    def header(self, text: str) -> None:
        self.add("heading", text=text, level=2)

    def subheader(self, text: str) -> None:
        self.add("heading", text=text, level=3)

    def caption(self, text: str) -> None:
        self.add("caption", text=str(text))

    def write(self, *args: Any, **_kwargs: Any) -> None:
        for value in args:
            if isinstance(value, pd.DataFrame):
                self.dataframe(value)
            elif isinstance(value, (dict, list, np.ndarray)):
                self.json(value)
            elif value is not None:
                self.markdown(str(value))

    def markdown(self, text: str, unsafe_allow_html: bool = False, **_kwargs: Any) -> None:
        if "<style>" in text:
            return
        self.add("html" if unsafe_allow_html else "markdown", text=text)

    def text(self, text: str) -> None:
        self.add("text", text=str(text))

    def code(self, text: str, **_kwargs: Any) -> None:
        self.add("code", text=str(text))

    def json(self, value: Any, **_kwargs: Any) -> None:
        self.add("json", value=clean(value))

    def divider(self) -> None:
        self.add("divider")

    def info(self, text: str, icon: str | None = None, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="info", icon=icon)

    def warning(self, text: str, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="warning")

    def error(self, text: str, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="error")

    def success(self, text: str, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="success")

    def metric(self, label: str, value: Any, delta: Any = None, **_kwargs: Any) -> None:
        self.add("metric", label=label, value=clean(value), delta=clean(delta))

    def widget(self, kind: str, label: str, default: Any, key: str | None = None, **props: Any) -> Any:
        widget_id = key or label
        value = self.values.get(widget_id, default)
        self.widgets[widget_id] = clean(value)
        self.add(kind, label=label, key=widget_id, value=clean(value), **clean(props))
        return value

    def checkbox(self, label: str, value: bool = False, key: str | None = None, **kwargs: Any) -> bool:
        return bool(self.widget("checkbox", label, value, key, disabled=kwargs.get("disabled", False)))

    def selectbox(
        self, label: str, options: Any, index: int = 0, key: str | None = None, **kwargs: Any
    ) -> Any:
        options = list(options)
        value = self.values.get(key or label, options[index] if options else None)
        if value not in options:
            value = options[0] if options else None
        self.values[key or label] = value
        return self.widget("select", label, value, key, options=options, help=kwargs.get("help"))

    def multiselect(
        self, label: str, options: Any, default: Any = None, key: str | None = None, **kwargs: Any
    ) -> Any:
        return self.widget("multiselect", label, default or [], key, options=list(options))

    def number_input(
        self,
        label: str,
        min_value: Any = None,
        max_value: Any = None,
        value: Any = None,
        step: Any = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        value = value if value is not None else (min_value if min_value is not None else 0)
        return self.widget(
            "number",
            label,
            value,
            key,
            min=min_value,
            max=max_value,
            step=step or (1 if isinstance(value, int) else 0.01),
            format=kwargs.get("format") or ("%d" if isinstance(value, int) else "%.2f"),
        )

    def slider(
        self,
        label: str,
        min_value: Any = None,
        max_value: Any = None,
        value: Any = None,
        step: Any = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        default = value if value is not None else min_value
        result = self.widget(
            "slider",
            label,
            default,
            key,
            min=min_value,
            max=max_value,
            step=step or 1,
            format=kwargs.get("format"),
        )
        is_date = isinstance(min_value, dt.date)
        if isinstance(default, tuple):
            if is_date:
                return tuple(dt.date.fromisoformat(str(v)[:10]) if isinstance(v, str) else v for v in result)
            return tuple(result)
        return result

    def button(self, label: str, key: str | None = None, **kwargs: Any) -> bool:
        disabled = kwargs.get("disabled", False)
        widget_id = key or label
        self.add("button", label=label, key=widget_id, disabled=disabled, help=kwargs.get("help"))
        return self.action == widget_id and not disabled

    def file_uploader(self, label: str, type: Any = None, key: str | None = None, **kwargs: Any) -> Any:
        widget_id = key or label
        csv = self.values.get(widget_id)
        self.add(
            "upload", label=label, key=widget_id, filename=csv.get("name") if isinstance(csv, dict) else None
        )
        if isinstance(csv, dict):
            csv = csv.get("content")
        return io.StringIO(csv) if isinstance(csv, str) else None

    def download_button(
        self, label: str, data: Any, file_name: str = "download.txt", mime: str | None = None, **kwargs: Any
    ) -> None:
        if hasattr(data, "read"):
            data = data.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.add(
            "download",
            label=label,
            filename=file_name,
            mime=mime or "application/octet-stream",
            data=base64.b64encode(data).decode("ascii"),
        )

    def dataframe(
        self,
        frame: Any,
        column_config: dict[str, Any] | None = None,
        column_order: list[str] | None = None,
        hide_index: bool | None = None,
        width: Any = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.visible:
            return
        config = column_config or {}
        styles = {}
        formats = {}
        styler = frame if hasattr(frame, "_compute") and hasattr(frame, "data") else None
        if styler is not None:
            frame = styler.data
            try:
                styler._compute()
                styles = {f"{r}:{c}": dict(style) for (r, c), style in styler.ctx.items()}
                formats = {f"{r}:{c}": fn(frame.iloc[r, c]) for (r, c), fn in styler._display_funcs.items()}
            except Exception:
                logging.getLogger(__name__).exception("Could not apply dataframe styles")
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        # An explicit order is also the displayed column selection. Preserve
        # duplicates: the reference includes positionsGained twice in Explorer.
        cols = list(column_order) if column_order is not None else list(frame.columns)
        cols = [c for c in cols if c in frame and config.get(c, "visible") is not None]
        definitions = []
        for column in cols:
            definition = config.get(column, {})
            definition = definition if isinstance(definition, dict) else {"label": definition}
            series = frame[column]
            inferred = (
                "CheckboxColumn"
                if pd.api.types.is_bool_dtype(series)
                else "NumberColumn"
                if pd.api.types.is_numeric_dtype(series)
                else "DateColumn"
                if pd.api.types.is_datetime64_any_dtype(series)
                else "TextColumn"
            )
            definitions.append(
                {
                    "key": str(column),
                    "label": definition.get("label") or str(column),
                    "kind": inferred,
                    **definition,
                }
            )
        values = table_rows(frame[cols])
        # Preserve original row/column positions for Styler formatting/highlights.
        source_positions = {c: frame.columns.get_loc(c) for c in cols}
        cell_styles = (
            [[styles.get(f"{r}:{source_positions[c]}", {}) for c in cols] for r in range(len(frame))]
            if styles
            else None
        )
        display = (
            [[formats.get(f"{r}:{source_positions[c]}") for c in cols] for r in range(len(frame))]
            if formats
            else None
        )
        self.add(
            "table",
            columns=definitions,
            rows=values,
            index=clean(list(frame.index)),
            index_name=frame.index.name,
            hide_index=bool(hide_index),
            width=width,
            height=height or min(400, 35 * (len(frame) + 1) + 3),
            styles=cell_styles,
            display=display,
        )

    def chart(
        self,
        kind: str,
        data: Any,
        x: str | None = None,
        y: Any = None,
        x_label: str | None = None,
        y_label: str | None = None,
        color: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.visible:
            return
        import altair as alt

        from app.services.chart_builder import ChartType, generate_chart

        alt.data_transformers.disable_max_rows()
        chart = generate_chart(
            {"scatter": ChartType.SCATTER, "line": ChartType.LINE, "bar": ChartType.VERTICAL_BAR}[kind],
            data,
            x_from_user=x,
            y_from_user=y,
            x_axis_label=x_label,
            y_axis_label=y_label,
            color_from_user=color,
            size_from_user=kwargs.get("size"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            stack=kwargs.get("stack"),
            sort_from_user=kwargs.get("sort", False),
        )
        self.altair_chart(chart, width=kwargs.get("width"))

    def scatter_chart(self, data: Any, **kwargs: Any) -> None:
        self.chart("scatter", data, **kwargs)

    def line_chart(self, data: Any, **kwargs: Any) -> None:
        self.chart("line", data, **kwargs)

    def bar_chart(self, data: Any, **kwargs: Any) -> None:
        self.chart("bar", data, **kwargs)

    def altair_chart(self, chart: Any, **kwargs: Any) -> None:
        if not self.visible:
            return
        import altair as alt

        with alt.theme.enable("none"):
            spec = chart.to_dict()
        self.add("vega", spec=clean(spec), width=kwargs.get("width"))

    def plotly_chart(self, figure: Any, **kwargs: Any) -> None:
        self.add("plotly", spec=json.loads(figure.to_json()))

    def pyplot(self, figure: Any, **kwargs: Any) -> None:
        if not self.visible:
            import matplotlib.pyplot as plt

            plt.close(figure)
            return
        stream = io.BytesIO()
        figure.savefig(stream, format="png", bbox_inches="tight", dpi=200)
        import matplotlib.pyplot as plt

        plt.close(figure)
        self.add(
            "image",
            src="data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii"),
            width="stretch",
        )

    def image(self, image: Any, width: Any = None, **kwargs: Any) -> None:
        if not self.visible:
            return
        path = Path(image)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            self.warning(f"Image not found: {path.name}")
            return
        if path.name == "gridlocked-logo-with-text.png":
            self.add("image", src="/api/brand/logo", width=width)
            return
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        self.add(
            "image",
            src=f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii"),
            width=width,
        )

    def cache_data(self, func: Any = None, **_kwargs: Any) -> Any:
        def decorate(fn: Any) -> Any:
            @wraps(fn)
            def cached(*args: Any, **kwargs: Any) -> Any:
                key = (fn.__name__, hashlib.sha256(pickle.dumps((args, kwargs), protocol=5)).hexdigest())
                with _LOCK:
                    if key not in _CACHE:
                        _CACHE[key] = fn(*args, **kwargs)
                        if len(_CACHE) > 64:
                            _CACHE.popitem(last=False)
                    return copy.deepcopy(_CACHE[key])

            return cached

        return decorate(func) if func else decorate

    cache_resource = cache_data

    def load_model(self, name: str, model_type: str | None, fingerprint: dict[str, Any], version: str) -> Any:
        from model_artifacts import artifact_matches_fingerprint

        directory = {
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "CatBoost": "catboost",
            "Ensemble (XGBoost + LightGBM + CatBoost)": "ensemble",
            "Position Group": "position_group",
            "Track-Weighted Ensemble": "track_weighted",
        }
        dirs = (
            [directory[model_type], ""]
            if model_type in directory
            else ["xgboost", "lightgbm", "catboost", "ensemble", ""]
        )
        stale = None
        for folder in dirs:
            path = DATA_DIR / "models" / folder / f"{name}.pkl"
            if not path.is_file():
                continue
            key = ("model", str(path), path.stat().st_mtime_ns)
            with _MODEL_LOCK:
                if key not in _CACHE:
                    namespace = self.namespace

                    class Unpickler(pickle.Unpickler):
                        def __init__(self, source: Any, namespace: dict[str, Any]):
                            super().__init__(source)
                            self.view_namespace = namespace

                        def find_class(self, module: str, cls: str) -> Any:
                            if module in {"raceAnalysis", "__main__"} and cls in self.view_namespace:
                                return self.view_namespace[cls]
                            return super().find_class(module, cls)

                    with path.open("rb") as source:
                        _CACHE[key] = Unpickler(source, namespace).load()
                artifact = dict(_CACHE[key])
            if artifact.get("cache_version") != version:
                continue
            if (
                name == "position_model"
                and model_type
                and artifact.get("model_type")
                not in self.namespace["_MODEL_TYPE_ARTIFACT_LABELS"].get(model_type, {model_type})
            ):
                continue
            artifact["_artifact_path"] = str(path)
            manifest_name = {
                "position_model": "manifest.json",
                "dnf_model": "dnf_manifest.json",
                "safetycar_model": "safetycar_manifest.json",
            }.get(name)
            manifest_path = path.parent / manifest_name if manifest_name else None
            if manifest_path and manifest_name and not manifest_path.exists():
                manifest_path = DATA_DIR / "models" / manifest_name
            if manifest_path and manifest_path.exists():
                from f1bet.artifacts import ModelManifest

                try:
                    manifest = ModelManifest.load(manifest_path)
                    if name == "position_model":
                        feature_names = tuple(
                            str(v) for v in getattr(artifact.get("preprocessor"), "feature_names_in_", ())
                        )
                        if (
                            manifest.schema_version != "legacy-wide-v1"
                            or manifest.feature_names != feature_names
                        ):
                            continue
                    artifact["_manifest_status"] = (
                        "verified" if manifest.data_sha256 == fingerprint.get("data_sha256") else "stale"
                    )
                except (ValueError, KeyError, TypeError):
                    continue
            else:
                artifact["_manifest_status"] = "legacy-missing"
            match = artifact_matches_fingerprint(artifact, fingerprint)
            artifact["_artifact_status"] = "current" if match else "legacy" if match is None else "stale"
            if match is not False:
                return artifact
            stale = stale or artifact
        return stale

    def dnf_diagnostics(self, data: pd.DataFrame) -> np.ndarray:
        path = Path(__file__).with_name("dnf_diagnostics.json")
        if not path.is_file():
            raise ValueError("Export DNF diagnostic probabilities offline before serving Next Race.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(
            (DATA_DIR / "f1ForAnalysis.csv").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if payload["data_sha256"] != digest or len(payload["probabilities"]) != len(data):
            raise ValueError("Re-export DNF diagnostics for the current analysis dataset.")
        return np.array(payload["probabilities"])


def render_view(page: int, values: dict[str, Any], action: str | None = None) -> dict[str, Any]:
    ui = Presentation(page, values, action)
    namespace = {
        "ui": ui,
        "__name__": "react_reference_views",
        "__file__": str(REPO_ROOT / "raceAnalysis.py"),
        "repository_data_dir": DATA_DIR,
        "repository_root": REPO_ROOT,
    }
    namespace["view_namespace"] = lambda: namespace
    ui.namespace = namespace
    # Each request owns its variables, widgets and model selection. Cached data
    # is immutable to the caller: source view mutations receive a private copy.
    # Matplotlib and Altair maintain process-wide rendering state.
    with _RENDER_LOCK:
        exec(_CODE, namespace)  # noqa: S102 - fixed, checked-in module; no user-supplied code.
    root = next((n for n in ui.nodes if n["type"] == "tabs" and n.get("root")), None)
    page_nodes = root["children"][page - 1]["children"] if root else []
    shell = ui.nodes[: ui.nodes.index(root)] if root else []
    return {
        "shell": shell,
        "tabs": root["labels"] if root else [],
        "nodes": page_nodes,
        "sidebar": ui.sidebar_nodes,
        "widgets": ui.widgets,
    }
```

## backend/logging.json

[Editable source](../backend/logging.json) — SHA-256: `4b713b74d25ef3bfe4f4c0ddf115165c85c5950a935783f91f919df5dcc13d2a`

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "text": {"format": "%(levelname)s %(name)s %(message)s"},
    "json_record": {"format": "%(message)s"}
  },
  "handlers": {
    "console": {"class": "logging.StreamHandler", "formatter": "text", "stream": "ext://sys.stderr"},
    "requests": {"class": "logging.StreamHandler", "formatter": "json_record", "stream": "ext://sys.stderr"}
  },
  "loggers": {
    "f1.request": {"handlers": ["requests"], "level": "INFO", "propagate": false},
    "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": false},
    "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": false}
  },
  "root": {"handlers": ["console"], "level": "INFO"}
}
```

## backend/requirements.txt

[Editable source](../backend/requirements.txt) — SHA-256: `9cdcfa9c3e6f0519f048301ec7620a010a4164dd738adde9eca6a2f5598288d0`

```text
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.8
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn==1.8.0
xgboost>=3.1.1
lightgbm==4.6.0
catboost>=1.2
pyarrow>=16
duckdb>=1.1
psutil>=6
python-multipart>=0.0.9
altair>=6.0,<7
matplotlib>=3.9
plotly>=6.0
```

## backend/requirements-dev.txt

[Editable source](../backend/requirements-dev.txt) — SHA-256: `7d8179f31a876fe9e262128013337f2adb5aa304f6d3e99ed41af0ba617bb19e`

```text
# Development-only requirements for the FastAPI backend.
# Install with:
#   pip install -r requirements.txt -r requirements-dev.txt
-r requirements.txt

# Lint / format
ruff>=0.6
mypy>=1.10

# Tests + coverage
pytest>=8
pytest-cov>=5
httpx>=0.27

# Security audit
pip-audit>=2.7
```

## backend/Dockerfile

[Editable source](../backend/Dockerfile) — SHA-256: `51ca8125587d19bb5fb6e425aca8d8bf885206b72feb863077c24914b4a11804`

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    F1_REPO_ROOT=/repo

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY fastapi_react/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY fastapi_react/backend/app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

## backend/app/enhancements/auth.py

[Editable source](../backend/app/enhancements/auth.py) — SHA-256: `05398b7acb52478ffd12df277b8cf820acad404ce58c8a772bd154ac70c73274`

```python
"""Explicit local access, with administrator authentication for hosted sessions."""

from __future__ import annotations

import ipaddress
import os
import secrets
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, Request

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_LOCAL_ORIGINS = tuple(
    f"http://{host}:{port}"
    for host in ("127.0.0.1", "localhost", "[::1]")
    for port in (5173, 5174, 8000)
)
FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto")


def trusted_local_enabled() -> bool:
    return os.environ.get("F1_TRUSTED_LOCAL", "0").strip().lower() in {"1", "true", "yes"}


def local_origins() -> list[str]:
    configured = os.environ.get("F1_LOCAL_ORIGINS")
    origins = configured.split(",") if configured is not None else list(DEFAULT_LOCAL_ORIGINS)
    result = []
    for entry in origins:
        origin = entry.strip()
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in LOCAL_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("F1_LOCAL_ORIGINS must contain only explicit localhost HTTP origins.")
        # Accessing port also rejects malformed or out-of-range port numbers.
        if parsed.port is not None and parsed.port < 1:
            raise ValueError("F1_LOCAL_ORIGINS ports must be positive.")
        if origin not in result:
            result.append(origin)
    return result


def is_trusted_local(request: Request) -> bool:
    if not trusted_local_enabled() or request.client is None:
        return False
    try:
        if not ipaddress.ip_address(request.client.host).is_loopback:
            return False
    except ValueError:
        return False
    # Local mode is for direct local use, never a public reverse proxy. The
    # launcher also disables Uvicorn's rewriting of the connection peer.
    if any(header in request.headers for header in FORWARDING_HEADERS):
        return False
    origins = local_origins()
    hosts = request.headers.getlist("host")
    if len(hosts) != 1 or hosts[0].lower() not in {urlsplit(origin).netloc.lower() for origin in origins}:
        return False
    browser_origins = request.headers.getlist("origin")
    if len(browser_origins) > 1 or (browser_origins and browser_origins[0] not in origins):
        return False
    # Cross-site navigations can omit Origin. Same-origin browser traffic and
    # local command-line clients (which omit both headers) remain supported.
    return request.headers.get("sec-fetch-site", "").lower() != "cross-site"


def research_access(request: Request) -> dict[str, str | bool]:
    local = is_trusted_local(request)
    return {"mode": "local" if local else "token", "token_required": not local}


def authorize_research(request: Request, x_f1_admin_token: str | None = Header(default=None)) -> None:
    if is_trusted_local(request):
        return
    expected = os.environ.get("F1_ADMIN_TOKEN")
    if expected:
        if x_f1_admin_token is not None and secrets.compare_digest(expected.encode(), x_f1_admin_token.encode()):
            return
        raise HTTPException(403, "Administrator access is required.")
    if trusted_local_enabled():
        raise HTTPException(403, "Research access is limited to this application's trusted local session.")
    raise HTTPException(503, "Administrator access is not configured. Set F1_ADMIN_TOKEN on the server.")
```

## start-local.ps1

[Editable source](../start-local.ps1) — SHA-256: `35db4cd5cc8d4276bdcc2772272805e1ed753be531fa3fce4055bd9a73f34119`

```powershell
# Launch the API for direct use on this computer. React continues to use port 5174.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repoDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Project Python is missing: $pythonExecutable. Create the project .venv first."
}

$previousLocalMode = $env:F1_TRUSTED_LOCAL
try {
    $env:F1_TRUSTED_LOCAL = '1'
    Push-Location (Join-Path $PSScriptRoot 'backend')
    try {
        & $pythonExecutable -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers --log-config logging.json
        if ($LASTEXITCODE -ne 0) { throw "The local API exited with code $LASTEXITCODE." }
    } finally { Pop-Location }
} finally {
    $env:F1_TRUSTED_LOCAL = $previousLocalMode
}
```

## docker-compose.yml

[Editable source](../docker-compose.yml) — SHA-256: `cb675ac9dd34e80eee81175862e1df4e262000d6ef7e4c11744930b3df85df73`

```yaml
services:
  backend:
    build:
      context: ..
      dockerfile: fastapi_react/backend/Dockerfile
    environment:
      F1_REPO_ROOT: /repo
      ENABLE_EXPENSIVE_TOOLS: "0"
      F1_TRUSTED_LOCAL: "0"
      OMP_NUM_THREADS: "1"
      OPENBLAS_NUM_THREADS: "1"
      MKL_NUM_THREADS: "1"
      NUMEXPR_NUM_THREADS: "1"
    volumes:
      - ..:/repo:ro
    restart: unless-stopped
    mem_limit: 1600m

  frontend:
    build:
      context: ..
      dockerfile: fastapi_react/frontend/Dockerfile
    depends_on:
      - backend
    ports:
      - "8080:80"
    restart: unless-stopped
    mem_limit: 128m
```
