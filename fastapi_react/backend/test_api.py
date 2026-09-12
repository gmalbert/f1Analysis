"""Backend API and service tests.

These tests exercise the FastAPI surface and the underlying service
modules. They run against the real repository data (data_files/) so
they double as a smoke test for the migration.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import analysis, betting, tools
from app.services import data as data_svc

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "repo_root" in body
    assert "data_dir" in body
    assert "rss_mb" in body


def test_meta_contains_parity_tabs() -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    tabs = response.json()["tabs"]
    for expected in (
        "Data Explorer", "Analytics", "Current Season", "Next Race",
        "Predictive Models", "Raw Data", "Betting Research",
    ):
        assert expected in tabs
    assert "models" in response.json()
    assert "manual_tools" in response.json()


# ----- Data Explorer --------------------------------------------------------

def test_data_explorer_schema_returns_filters() -> None:
    response = client.get("/api/data-explorer/schema")
    assert response.status_code == 200
    body = response.json()
    assert "filters" in body
    assert isinstance(body["filters"], list)
    assert body["filters"], "schema should return at least one filterable column"


def test_data_explorer_query_unfiltered() -> None:
    response = client.post("/api/data-explorer/query", json={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert "total" in body
    assert "columns" in body
    assert "rows" in body
    assert body["total"] > 0
    assert len(body["rows"]) <= 5


def test_data_explorer_query_with_filters() -> None:
    response = client.post(
        "/api/data-explorer/query",
        json={
            "filters": [
                {"column": "grandPrixYear", "kind": "range", "value": [2020, 2025]},
            ],
            "limit": 10,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] > 0
    for row in body["rows"]:
        assert 2020 <= int(row["grandPrixYear"]) <= 2025


def test_data_explorer_query_bad_limit_returns_422() -> None:
    response = client.post("/api/data-explorer/query", json={"limit": 0})
    assert response.status_code == 422


# ----- Analytics ------------------------------------------------------------

def test_analytics_endpoint_returns_payload() -> None:
    response = client.post(
        "/api/analytics",
        json={"filters": [], "max_rows": 200},
    )
    assert response.status_code == 200
    body = response.json()
    assert "rows_considered" in body
    assert "charts" in body
    assert "regressions" in body


def test_analytics_service_smoke() -> None:
    payload = analysis.analytics([], 100)
    assert payload["rows_considered"] > 0
    assert "charts" in payload
    assert "regressions" in payload


# ----- Current season / Next race ------------------------------------------

def test_current_season_endpoint() -> None:
    response = client.get("/api/current-season")
    assert response.status_code == 200
    body = response.json()
    assert "year" in body
    assert "rows" in body
    assert "columns" in body


def test_next_race_endpoint() -> None:
    response = client.get("/api/next-race")
    # 200 if a next race is detected, 404 if the season is over
    assert response.status_code in (200, 404)


# ----- Models ---------------------------------------------------------------

def test_models_endpoint_lists_all_types() -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    expected = {"XGBoost", "LightGBM", "CatBoost", "Position Group", "Track-Weighted Ensemble"}
    assert expected.issubset(set(body["models"]))


def test_models_manifest_unknown_returns_400() -> None:
    response = client.get("/api/models/manifest", params={"model_type": "nope"})
    assert response.status_code == 400


def test_models_precomputed_unknown_returns_400() -> None:
    response = client.get("/api/models/precomputed/no-such-artifact")
    assert response.status_code == 400


# ----- Raw data -------------------------------------------------------------

def test_raw_files_returns_list() -> None:
    response = client.get("/api/raw/files")
    assert response.status_code == 200
    body = response.json()
    assert "files" in body
    assert isinstance(body["files"], list)


def test_raw_preview_csv() -> None:
    files = client.get("/api/raw/files").json()["files"]
    csv_files = [f for f in files if f["suffix"] in {".csv", ".tsv"}]
    assert csv_files, "expected at least one CSV/TSV in data_files"
    target = csv_files[0]["path"]
    response = client.get("/api/raw/preview", params={"path": target})
    assert response.status_code == 200
    body = response.json()
    assert "columns" in body
    assert "rows" in body


def test_raw_preview_path_traversal_blocked() -> None:
    response = client.get("/api/raw/preview", params={"path": "../raceAnalysis.py"})
    assert response.status_code in (400, 403, 404)


def test_raw_download_path_traversal_blocked() -> None:
    response = client.get("/api/raw/download", params={"path": "../raceAnalysis.py"})
    assert response.status_code in (400, 403, 404)


# ----- Betting --------------------------------------------------------------

def test_betting_value_endpoint() -> None:
    response = client.post("/api/betting/value", json={})
    assert response.status_code == 200
    body = response.json()
    assert "market_probability" in body
    assert "raw_ev" in body
    assert "adjusted_probability" in body
    assert "stake" in body
    assert "reason_code" in body


def test_betting_simulation_endpoint() -> None:
    payload = {
        "entries": [
            {"driver_id": "verstappen", "constructor_id": "red_bull",
             "pace_score": 0.95, "dnf_probability": 0.02, "uncertainty": 0.01},
            {"driver_id": "norris", "constructor_id": "mclaren",
             "pace_score": 0.93, "dnf_probability": 0.03, "uncertainty": 0.01},
        ],
        "simulations": 2000,
        "seed": 7,
    }
    response = client.post("/api/betting/simulate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "columns" in body
    assert "rows" in body


def test_betting_backtest_endpoint() -> None:
    # f1bet backtest requires a full ledger-shaped row; supply a minimal one
    rows = [{
        "event_id": "race-1",
        "selection_id": "sel-A",
        "market": "win",
        "forecast_at": "2024-01-01T00:00:00Z",
        "quote_at": "2024-01-01T00:00:00Z",
        "event_start_at": "2024-01-01T03:00:00Z",
        "probability": 0.6,
        "uncertainty": 0.02,
        "fair_market_probability": 0.5,
        "decimal_odds": 2.0,
        "outcome": 1,
    }]
    response = client.post("/api/betting/backtest", json={"rows": rows})
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "ledger" in body
    assert "decisions" in body
    assert "sensitivity" in body


def test_betting_calibration_endpoint() -> None:
    rows = [
        {"probability": 0.1, "outcome": 0},
        {"probability": 0.3, "outcome": 1},
        {"probability": 0.7, "outcome": 1},
        {"probability": 0.9, "outcome": 1},
    ]
    response = client.post("/api/betting/calibration", json={"rows": rows})
    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body
    assert "reliability" in body


def test_betting_governance_endpoint() -> None:
    response = client.get("/api/betting/governance")
    assert response.status_code == 200


# ----- Tools gate -----------------------------------------------------------

def test_tools_disabled_by_default() -> None:
    response = client.post("/api/tools/run", json={"tool": "monte_carlo", "args": []})
    assert response.status_code == 403


def test_tools_unknown_tool() -> None:
    # Even with the gate, the unknown-tool path is reached via a ValueError/KeyError
    # after the gate; we can't easily reach it without enabling expensive tools.
    # So just check the route exists via the 403 path.
    response = client.post("/api/tools/run", json={"tool": "nope", "args": []})
    assert response.status_code == 403


# ----- Service-level unit tests --------------------------------------------

def test_filter_schema_columns_are_strings() -> None:
    schema = data_svc.filter_schema()
    assert all("column" in item for item in schema)
    assert all("kind" in item for item in schema)


def test_records_handles_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = data_svc.records(df, limit=2)
    assert len(out) == 2
    assert out[0] == {"a": 1, "b": "x"}


def test_records_handles_none_and_nan() -> None:
    df = pd.DataFrame({"a": [None, 1.0], "b": [float("nan"), "z"]})
    out = data_svc.records(df)
    assert out[0]["a"] is None
    assert out[0]["b"] is None
    assert out[1] == {"a": 1.0, "b": "z"}


def test_resolve_data_file_blocks_traversal() -> None:
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        data_svc.resolve_data_file("../raceAnalysis.py")


def test_resolve_data_file_blocks_absolute_outside_data() -> None:
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        data_svc.resolve_data_file("C:/Windows/System32/drivers/etc/hosts")


def test_resolve_data_file_resolves_known() -> None:
    files = data_svc.list_data_files()
    assert files, "data_files should not be empty"
    target = data_svc.resolve_data_file(files[0]["path"])
    assert target.exists()


def test_list_data_files_shape() -> None:
    files = data_svc.list_data_files()
    assert isinstance(files, list)
    if files:
        assert "path" in files[0]
        assert "size" in files[0]
        assert "suffix" in files[0]


def test_model_manifest_unknown_type_raises() -> None:
    with pytest.raises(KeyError):
        data_svc.model_manifest("nope")


def test_precomputed_unknown_raises() -> None:
    with pytest.raises(KeyError):
        data_svc.precomputed("no-such-artifact")


# ----- Tools service gate ---------------------------------------------------

def test_tools_gate_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import ENABLE_EXPENSIVE_TOOLS
    assert ENABLE_EXPENSIVE_TOOLS is False
    with pytest.raises(PermissionError):
        tools.run_tool("monte_carlo", [])


def test_tools_unknown_raises_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "ENABLE_EXPENSIVE_TOOLS", True)
    with pytest.raises(KeyError):
        tools.run_tool("not-a-real-tool", [])


def test_tools_directory_constant_includes_known_scripts() -> None:
    for key in ("monte_carlo", "rfe", "boruta", "shap", "permutation", "temporal_leakage"):
        assert key in tools.TOOLS


# ----- Betting service unit tests ------------------------------------------

def test_betting_value_service_smoke() -> None:
    from app.schemas import BettingValueRequest
    out = betting.value_and_stake(BettingValueRequest())
    assert "raw_ev" in out
    assert "stake" in out
    assert "market_probability" in out


def test_betting_calibration_service_smoke() -> None:
    out = betting.calibration([
        {"probability": 0.1, "outcome": 0},
        {"probability": 0.7, "outcome": 1},
    ])
    assert "metrics" in out
    assert "reliability" in out
