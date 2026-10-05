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
