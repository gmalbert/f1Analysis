# Complete backend implementation

All source files below are complete, tested candidate files. The proposed `main.py` preserves existing routes and adds flagged enhancement integration, including clean job-executor shutdown. Existing calculations remain in the current services.

## Copy map

Paths on the right are relative to `fastapi_react/backend/`.

| Supplied source | Destination |
| --- | --- |
| `code/backend/main.py` | `app/main.py` |
| `code/backend/__init__.py`, `cache.py`, `metrics.py`, `jobs.py`, `service.py` | Corresponding files under `app/enhancements/` |
| `code/backend/testing_worker.py` | `app/enhancements/testing_worker.py` for tests only |
| `code/backend/test_enhancements.py` | `test_enhancements.py` |
| `code/deployment/logging.json` | `logging.json` |

Do not copy the generated `main.py` or `test_enhancements.py` inside the enhancements package. The small test worker is required to exercise Windows spawned-process jobs in tests; production routes never dispatch it.

## Flags

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `F1_ENHANCEMENTS` | `0` | Install status/jobs/metrics routes, middleware, revision management, and the alternate view response path |
| `F1_VIEW_RESPONSE_CACHE` | `0` | Enable B1 server reuse when enhancements are installed |
| `F1_MAX_REQUEST_BYTES` | `268435456` | Aggregate accepted request-body limit in bytes |
| `F1_ADMIN_TOKEN` | Unset | Enable/authorize new local jobs and metrics; unset returns 503 |
| `F1_BUILD_REVISION` | `local-working-tree` | Source revision label in context export |
| `F1_REPO_ROOT` | Existing config default | Override repository path for an isolated/nested preview |

Do not enable `F1_RESEARCH_MODE` merely to use these two new explicit task routes. The proposal retains the existing general research-mode setting and dispatches only the operations documented in B4.

## Start a local integrated checkout

Run from `fastapi_react/backend`, with the application's existing data/model artifacts available:

```powershell
$env:F1_ENHANCEMENTS = '1'
$env:F1_VIEW_RESPONSE_CACHE = '1'
$env:F1_BUILD_REVISION = (git rev-parse HEAD)
../../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --log-config logging.json
```

To test administrator routes locally, provide a freshly generated administrator token through the process environment, then enter it only in the preview password field. Keep it out of command history, checked-in files, share URLs, and diagnostics. The complete package functions without an administrator token: only the new metrics/jobs operations remain disabled.

## API contracts

All paths below are under `/api/enhancements`.

| Method/path | Request/result |
| --- | --- |
| `GET /status` | Public revision, build revision, dataset name/time, recorded model-manifest metadata |
| `GET /metrics` | Admin header required; recent records and cache byte count |
| `POST /jobs` | Admin header; `{"task":"leakage-audit","values":{}}` or `bin-comparison`; 202 with job ID |
| `GET /jobs/{id}` | Admin header; current state and timestamps |
| `GET /jobs/{id}/result` | Admin header; original view-node result on success |
| `DELETE /jobs/{id}` | Admin header; `{"cancelled":true/false}`; queued jobs only |

The header name is `X-F1-Admin-Token`. Wrong/missing tokens return 403 when a token is configured; disabled admin operations return 503. Unknown jobs return 404, unsupported/invalid inputs 400, full queue 429, and unfinished/failed result requests 409. Results expire after ten minutes and restart loses job IDs.

`POST /api/views` retains its original request contract. It adds revision/cache headers on the flagged path and remains the source of all complete view tables.

## Backend verification

From the backend directory:

```powershell
$env:F1_ENHANCEMENTS = '0'
../../.venv/Scripts/python.exe -m compileall -q app
../../.venv/Scripts/python.exe -m ruff check app test_enhancements.py
../../.venv/Scripts/python.exe -m mypy app
../../.venv/Scripts/python.exe -m pytest
```

The existing suite is run with the global flag off to preserve baseline route expectations; the new service tests instantiate and exercise the enhancement integration directly. Follow with a real API probe against a process started with enhancements on.

## Full source

Every file below also exists separately in [code/backend](code/backend). The code uses the existing installed FastAPI/Starlette/Pydantic stack and Python standard library. It does not add a queue, Redis, or database dependency.

## code/backend/__init__.py

[Separate source file](code/backend/__init__.py)

```python
"""Optional enhancements; install under app/enhancements only after review."""
```

## code/backend/cache.py

[Separate source file](code/backend/cache.py)

```python
from __future__ import annotations

import gzip
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.responses import JSONResponse, Response


def accepts_gzip(header: str) -> bool:
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
        choices[parts[0]] = quality
    return choices.get("gzip", choices.get("*", 0.0)) > 0


def reusable(page: int, values: dict[str, Any], action: str | None) -> bool:
    if action or page not in {1, 2, 3, 4, 5}:
        return False
    for key, value in values.items():
        if any(word in key.lower() for word in ("upload", "csv", "ledger")):
            return False
        if isinstance(value, dict) or (isinstance(value, str) and len(value) > 4096):
            return False
        if isinstance(value, list) and (
            len(value) > 100 or any(isinstance(item, (dict, list)) for item in value)
        ):
            return False
    return True


@dataclass
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

    def render(
        self, page: int, values: dict[str, Any], action: str | None,
        revision: str, encoding: str, *, enabled: bool = True,
    ) -> Response:
        headers = {"Cache-Control": "no-store", "X-F1-Revision": revision}
        if not enabled or not reusable(page, values, action):
            if action:
                self.clear()
            headers["X-F1-Cache"] = "BYPASS"
            return JSONResponse(self.renderer(page, values, action), headers=headers)
        key = json.dumps([revision, page, values], sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.lock:
            entry = self.entries.get(key)
            hit = entry is not None and entry.expires > self.clock()
            if entry is not None:
                self.entries.pop(key)
                self.bytes -= entry.size
            if not hit:
                body = bytes(JSONResponse(self.renderer(page, values, action)).body)
                entry = Entry(body, gzip.compress(body, compresslevel=5, mtime=0), self.clock()+self.ttl)
            if entry is None:
                raise RuntimeError("Could not construct the view cache entry")
            if entry.size <= self.max_bytes:
                self.entries[key] = entry
                self.bytes += entry.size
                while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old.size
            headers["X-F1-Cache"] = "HIT" if hit else "MISS"
            headers["Vary"] = "Accept-Encoding"
            zipped = len(entry.body) >= 1000 and accepts_gzip(encoding)
            if zipped:
                headers["Content-Encoding"] = "gzip"
            return Response(entry.compressed if zipped else entry.body, media_type="application/json", headers=headers)
```

## code/backend/jobs.py

[Separate source file](code/backend/jobs.py)

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
    pass


class Jobs:
    """Bounded local queue with a separate calculation process. State expires on restart."""

    def __init__(
        self, execute: Callable[[str, dict[str, Any]], dict[str, Any]],
        *, limit: int = 8, result_limit: int = 32*1024*1024, ttl: float = 600,
    ) -> None:
        self.execute, self.limit, self.result_limit, self.ttl = execute, limit, result_limit, ttl
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="f1-research")
        self.worker = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self.lock = threading.RLock()
        self.items: dict[str, dict[str, Any]] = {}
        self.futures: dict[str, Future[None]] = {}

    def submit(self, task: str, values: dict[str, Any]) -> str:
        # Serialize before queueing: isolate caller mutation and bound retained inputs.
        text = json.dumps(values, allow_nan=False)
        if len(text.encode()) > 64*1024:
            raise ValueError("Research job inputs must be below 64 KiB; uploaded CSVs are not accepted.")
        with self.lock:
            self._expire()
            if len(self.items) >= self.limit:
                raise BusyQueueError("The local research queue is full.")
            identity = uuid.uuid4().hex
            self.items[identity] = {"id": identity, "task": task, "state": "queued", "created": time.time(), "finished": None}
            self.futures[identity] = self.pool.submit(self._run, identity, task, json.loads(text))
            return identity

    def _expire(self) -> None:
        now = time.time()
        for identity, item in list(self.items.items()):
            if item["finished"] and now-item["finished"] > self.ttl:
                self.items.pop(identity)
                self.futures.pop(identity, None)

    def _run(self, identity: str, task: str, values: dict[str, Any]) -> None:
        with self.lock:
            self.items[identity]["state"] = "running"
        try:
            result = self.worker.submit(self.execute, task, values).result()
            body = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
            if len(body) > self.result_limit:
                raise ValueError("Research result exceeds the configured limit.")
            compressed = gzip.compress(body, compresslevel=5)
            with self.lock:
                self.items[identity].update(state="succeeded", body=compressed)
        except Exception:  # A worker records failure without killing the queue.
            log.exception("Research job %s failed", identity)
            with self.lock:
                self.items[identity].update(state="failed", error="Research calculation failed; see the server log using this job ID.")
        finally:
            with self.lock:
                self.items[identity]["finished"] = time.time()

    def status(self, identity: str) -> dict[str, Any]:
        with self.lock:
            self._expire()
            item = self.items[identity]
            return {key: value for key, value in item.items() if key != "body"}

    def result(self, identity: str) -> dict[str, Any]:
        with self.lock:
            if self.items[identity]["state"] != "succeeded":
                raise ValueError("The job has not completed successfully.")
            body = self.items[identity]["body"]
        return cast("dict[str, Any]", json.loads(gzip.decompress(body)))

    def cancel(self, identity: str) -> bool:
        with self.lock:
            if self.items[identity]["state"] != "queued":
                return False
            if not self.futures[identity].cancel():
                return False
            self.items[identity].update(state="cancelled", finished=time.time())
            return True

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.worker.shutdown(wait=True, cancel_futures=True)
```

## code/backend/main.py

[Separate source file](code/backend/main.py)

```python
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.config import DATA_DIR, ENABLE_EXPENSIVE_TOOLS, MODEL_TYPES, REPO_ROOT
from app.enhancements.service import Enhancements
from app.schemas import (
    AnalyticsRequest,
    BettingValueRequest,
    QueryRequest,
    RowsPayload,
    SimulationRequest,
    ToolRunRequest,
    ViewRequest,
)
from app.services.analysis import analytics, current_season, next_race_bundle, tire_strategy
from app.services.betting import backtest, calibration, governance, simulate, value_and_stake
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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if enhancements is not None:
            await run_in_threadpool(enhancements.jobs.close)


app = FastAPI(
    lifespan=lifespan,
    title="F1 Analysis API",
    version="1.0.0",
    description="FastAPI backend for the React parity migration of raceAnalysis.py",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
enhancements = Enhancements() if os.environ.get('F1_ENHANCEMENTS', '0') == '1' else None
CODE_DEPLOYED_AT = datetime.now(UTC)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)


@app.post("/api/views", response_model=dict[str, Any])
def view(payload: ViewRequest, request: Request) -> Response:
    try:
        # The presentation protocol already normalizes values to JSON primitives.
        # Avoid FastAPI recursively converting millions of table cells again.
        if enhancements is not None:
            return enhancements.render(payload, request)
        return JSONResponse(render_view(payload.page, payload.values, payload.action))
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("Could not render analysis page %s", payload.page)
        raise _http_error(exc) from exc


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/api/betting/simulate")
def betting_simulate(payload: SimulationRequest) -> dict[str, Any]:
    try:
        return simulate(payload)
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/betting/backtest")
def betting_backtest(payload: RowsPayload) -> dict[str, Any]:
    try:
        return backtest(payload.rows)
    except Exception as exc:
        raise _http_error(exc) from None


@app.post("/api/betting/calibration")
def betting_calibration(payload: RowsPayload) -> dict[str, Any]:
    try:
        return calibration(payload.rows)
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

## code/backend/metrics.py

[Separate source file](code/backend/metrics.py)

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


class RequestMetrics:
    def __init__(self, app: ASGIApp, records: deque[dict[str, Any]], lock: Any) -> None:
        self.app, self.records, self.lock = app, records, lock

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
                header_ms = 1000*(time.perf_counter()-start)
                message = {**message, "headers": [*message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                    (b"server-timing", ("backend;dur="+format(header_ms, ".2f")).encode()),
                ]}
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            record = {
                "request_id": request_id, "method": scope["method"],
                "route": getattr(scope.get("route"), "path", "<unmatched>"),
                "status": status, "header_ms": header_ms,
                "duration_ms": round(1000*(time.perf_counter()-start), 2), "body_bytes": sent,
            }
            with self.lock:
                self.records.append(record)
            log.info("%s", json.dumps(record))


class BodyLimit:
    """Bound the entire request before JSON parsing; preserve valid body bytes."""

    def __init__(self, app: ASGIApp, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = self.max_bytes+1
        chunks: list[bytes] = []
        size = 0
        if declared <= self.max_bytes:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                chunks.append(chunk)
                size += len(chunk)
                if size > self.max_bytes or not message.get("more_body", False):
                    break
        if declared > self.max_bytes or size > self.max_bytes:
            await JSONResponse({"detail": "The request is too large. Reduce uploaded CSV data."}, status_code=413)(scope, receive, send)
            return
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def metrics_storage() -> tuple[deque[dict[str, Any]], Any]:
    return deque(maxlen=500), threading.Lock()
```

## code/backend/service.py

[Separate source file](code/backend/service.py)

```python
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
```

## code/backend/test_enhancements.py

[Separate source file](code/backend/test_enhancements.py)

```python
import gzip
import json
import time

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.enhancements.cache import ViewResponses, accepts_gzip
from app.enhancements.jobs import BusyQueueError, Jobs
from app.enhancements.metrics import BodyLimit, RequestMetrics, metrics_storage
from app.enhancements.testing_worker import fake_work


def test_cache_revision_precision_encoding_expiry_actions_and_uploads():
    calls, clock = [], [10.]
    def render(page, values, action):
        calls.append((page, values, action))
        return {"page":page, "integer":2**60+1,"float":1.0000000000000002,"text":"x"*2000,"values":values}
    cache = ViewResponses(render, clock=lambda:clock[0])
    first = cache.render(1,{"year":2026},None,"r1","gzip")
    second = cache.render(1,{"year":2026},None,"r1","gzip")
    assert second.headers["x-f1-cache"] == "HIT"
    assert len(calls) == 1
    assert json.loads(gzip.decompress(first.body))["integer"] == 2**60+1
    identity = cache.render(1,{"year":2026},None,"r1","gzip;q=0")
    assert "content-encoding" not in identity.headers
    assert json.loads(identity.body)["float"] == 1.0000000000000002
    cache.render(1,{"year":2026},None,"r2","gzip")
    assert len(calls) == 2
    clock[0] += 21
    cache.render(1,{"year":2026},None,"r2","gzip")
    assert len(calls) == 3
    assert cache.render(1,{}, "action","r2","gzip").headers["x-f1-cache"] == "BYPASS"
    assert cache.render(1,{"f1bet_field_upload":"csv"},None,"r2","gzip").headers["x-f1-cache"] == "BYPASS"
    assert cache.render(6,{},None,"r2","gzip").headers["x-f1-cache"] == "BYPASS"
    assert not accepts_gzip("gzip;q=0,*;q=1")


def test_cache_memory_bound():
    cache = ViewResponses(lambda *_: {"text":"x"*3000}, max_bytes=1000)
    cache.render(1,{},None,"r","gzip")
    assert cache.bytes == 0
    assert not cache.entries


def test_body_limit_and_timing_preserve_valid_json_and_reject_oversize():
    async def echo(request):
        return JSONResponse(await request.json())
    app = Starlette(routes=[Route("/echo",echo,methods=["POST"])])
    records, lock = metrics_storage()
    app.add_middleware(BodyLimit,max_bytes=128)
    app.add_middleware(RequestMetrics,records=records,lock=lock)
    with TestClient(app) as client:
        result = client.post("/echo",json={"year":2026})
        assert result.json() == {"year":2026}
        assert result.headers["server-timing"].startswith("backend;dur=")
        assert len(result.headers["x-request-id"]) == 32
        assert client.post("/echo",json={"text":"x"*200}).status_code == 413
    assert [record["status"] for record in records] == [200,413]
    assert all("values" not in record for record in records)


def test_isolated_jobs_results_capacity_and_queued_cancellation():
    jobs = Jobs(fake_work,limit=2)
    try:
        first = jobs.submit("test",{"value":7,"delay":1})
        second = jobs.submit("test",{"value":8})
        with pytest.raises(BusyQueueError):
            jobs.submit("test",{"value":9})
        assert jobs.cancel(second)
        deadline = time.monotonic()+30
        while jobs.status(first)["state"] in {"queued","running"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert jobs.result(first) == {"task":"test","value":7}
        assert jobs.status(second)["state"] == "cancelled"
        jobs.ttl = -1
        with pytest.raises(KeyError):
            jobs.status(first)
        jobs.ttl = 600
        with pytest.raises(ValueError, match="below 64 KiB"):
            jobs.submit("test",{"value":"x"*70000})
        failed = jobs.submit("fail",{"value":0})
        deadline = time.monotonic()+30
        while jobs.status(failed)["state"] in {"queued","running"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert jobs.status(failed)["state"] == "failed"
        with pytest.raises(ValueError, match="not completed successfully"):
            jobs.result(failed)
    finally:
        jobs.close()


def test_service_status_refresh_auth_and_metrics(tmp_path, monkeypatch):
    from fastapi import FastAPI

    from app.enhancements import service
    root = tmp_path
    dataset = root/"data_files"
    models = dataset/"models"
    models.mkdir(parents=True)
    (dataset/"f1ForAnalysis.csv").write_text("year\n2026\n")
    (root/"raceAnalysis.py").write_text("# source")
    (models/"manifest.json").write_text(json.dumps({"model_name":"position","notes":["recorded"],"trained_at":"today"}))
    monkeypatch.setattr(service,"DATA_DIR",dataset)
    monkeypatch.setattr(service,"REPO_ROOT",root)
    monkeypatch.delenv("F1_ADMIN_TOKEN",raising=False)
    enhancement = service.Enhancements(poll_seconds=0)
    app = FastAPI()
    enhancement.install(app)
    try:
        with TestClient(app) as client:
            first = client.get("/api/enhancements/status").json()
            assert first["dataset"]["name"] == "f1ForAnalysis.csv"
            (dataset/"f1ForAnalysis.csv").write_text("year\n2025\n2026\n")
            assert client.get("/api/enhancements/status").json()["revision"] != first["revision"]
            assert client.get("/api/enhancements/metrics").status_code == 503
            monkeypatch.setenv("F1_ADMIN_TOKEN","test-only")
            assert client.get("/api/enhancements/metrics").status_code == 403
            assert client.get("/api/enhancements/metrics",headers={"X-F1-Admin-Token":"test-only"}).status_code == 200
            assert client.post("/api/enhancements/jobs",json={"task":"unsupported"},headers={"X-F1-Admin-Token":"test-only"}).status_code == 400
            assert client.post("/api/enhancements/jobs",json={"task":"leakage-audit","values":{"Uploaded CSV":"year\n2026"}},headers={"X-F1-Admin-Token":"test-only"}).status_code == 400
            assert client.get("/api/enhancements/jobs/missing",headers={"X-F1-Admin-Token":"test-only"}).status_code == 404
    finally:
        enhancement.jobs.close()


def test_research_dispatch_uses_only_existing_opt_in_actions(monkeypatch):
    from app.enhancements import service
    monkeypatch.setattr(service,"artifact_revision",lambda *_:"revision")
    monkeypatch.setattr(service,"clear_source_caches",lambda:None)
    monkeypatch.setattr(service.presentation,"render_view",lambda page,values,action:{"page":page,"values":values,"action":action})
    context = {"revision":"revision","values":{}}
    bins = service.execute_research("bin-comparison",context)
    assert bins["action"] == "Run Bin Count Comparison"
    assert bins["values"]["Select q values (number of bins)"] == [2]
    assert service.execute_research("leakage-audit",context)["action"] == "Run Leakage Audit"
    with pytest.raises(ValueError, match="Unsupported research task"):
        service.execute_research("unknown",context)
    with pytest.raises(ValueError, match="q values"):
        service.execute_research("bin-comparison",{"revision":"revision","values":{"Select q values (number of bins)":[1]}})
    with pytest.raises(ValueError, match="Audit row limit"):
        service.execute_research("leakage-audit",{"revision":"revision","values":{"Rows to read (0 = all)":-1}})
    with pytest.raises(ValueError, match="Artifacts changed"):
        service.execute_research("leakage-audit",{"revision":"stale","values":{}})
```

## code/backend/testing_worker.py

[Separate source file](code/backend/testing_worker.py)

```python
"""Deterministic isolated worker used by proposal checks, never by app routes."""
import time
from typing import Any


def fake_work(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    time.sleep(payload.get("delay", 0.01))
    if task == "fail":
        raise ValueError("Intentional test failure.")
    return {"task": task, "value": payload["value"]}
```
