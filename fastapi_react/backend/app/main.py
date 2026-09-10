from __future__ import annotations

import os
import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import DATA_DIR, ENABLE_EXPENSIVE_TOOLS, MODEL_TYPES, REPO_ROOT
from .schemas import AnalyticsRequest, BettingValueRequest, QueryRequest, RowsPayload, SimulationRequest, ToolRunRequest
from .services.analysis import analytics, current_season, next_race_bundle
from .services.betting import backtest, calibration, governance, simulate, value_and_stake
from .services.data import filter_schema, list_data_files, model_manifest, precomputed, query_main, read_table, resolve_data_file
from .services.tools import TOOLS, run_tool

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
        return HTTPException(404, str(exc))
    if isinstance(exc, (KeyError, ValueError)):
        return HTTPException(400, str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    return HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.get("/api/health")
def health():
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
def meta():
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
def data_explorer_schema():
    try:
        return {"filters": filter_schema()}
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/data-explorer/query")
def data_explorer_query(request: QueryRequest):
    try:
        return query_main(request)
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/analytics")
def analytics_route(request: AnalyticsRequest):
    try:
        return analytics(request.filters, request.max_rows)
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/current-season")
def season_route():
    try:
        return current_season()
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/next-race")
def next_race_route():
    try:
        return next_race_bundle()
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/models")
def models():
    return {"models": MODEL_TYPES}


@app.get("/api/models/manifest")
def model_manifest_route(model_type: str = Query(...)):
    try:
        return {"model_type": model_type, "manifest": model_manifest(model_type)}
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/models/precomputed/{name}")
def model_precomputed(name: str):
    try:
        return {"name": name, "data": precomputed(name)}
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/raw/files")
def raw_files():
    return {"files": list_data_files()}


@app.get("/api/raw/preview")
def raw_preview(path: str = Query(...)):
    try:
        target = resolve_data_file(path)
        return {"path": path, **read_table(target)}
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/raw/download")
def raw_download(path: str = Query(...)):
    try:
        target = resolve_data_file(path)
        return FileResponse(target, filename=target.name)
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/betting/value")
def betting_value(payload: BettingValueRequest):
    try:
        return value_and_stake(payload)
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/betting/simulate")
def betting_simulate(payload: SimulationRequest):
    try:
        return simulate(payload)
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/betting/backtest")
def betting_backtest(payload: RowsPayload):
    try:
        return backtest(payload.rows)
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/betting/calibration")
def betting_calibration(payload: RowsPayload):
    try:
        return calibration(payload.rows)
    except Exception as exc:
        raise _http_error(exc)


@app.get("/api/betting/governance")
def betting_governance():
    try:
        return governance()
    except Exception as exc:
        raise _http_error(exc)


@app.post("/api/tools/run")
def tools_run(payload: ToolRunRequest):
    try:
        return run_tool(payload.tool, payload.args)
    except Exception as exc:
        raise _http_error(exc)
