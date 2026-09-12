from __future__ import annotations

import os
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import DATA_DIR, ENABLE_EXPENSIVE_TOOLS, MODEL_TYPES, REPO_ROOT
from app.schemas import (
    AnalyticsRequest,
    BettingValueRequest,
    QueryRequest,
    RowsPayload,
    SimulationRequest,
    ToolRunRequest,
)
from app.services.analysis import analytics, current_season, next_race_bundle
from app.services.betting import backtest, calibration, governance, simulate, value_and_stake
from app.services.data import (
    filter_schema,
    list_data_files,
    model_manifest,
    precomputed,
    query_main,
    read_table,
    resolve_data_file,
)
from app.services.tools import TOOLS, run_tool

app = FastAPI(
    title="F1 Analysis API",
    version="1.0.0",
    description="FastAPI backend for the React parity migration of raceAnalysis.py",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

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


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "tabs": [
            "Data Explorer", "Analytics", "Current Season", "Next Race",
            "Predictive Models", "Raw Data", "Betting Research",
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
