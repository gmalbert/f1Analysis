"""Contracts for bounded reuse, artifact publication, and diagnostic privacy."""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from app.enhancements import metrics, service
from app.enhancements.cache import NegotiatedGZipMiddleware, ViewResponses, accepts_gzip, reusable
from app.enhancements.metrics import RequestMetrics, metrics_storage
from app.schemas import ViewRequest


@pytest.fixture(autouse=True)
def clean_shared_response_cache() -> Iterator[None]:
    from app.main import enhancements

    yield
    if enhancements is not None:
        enhancements.responses.clear()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("gzip", True),
        ("GZIP; q=0.5", True),
        ("*;q=1", True),
        ("gzip;q=0,*;q=1", False),
        ("br", False),
        ("gzip;q=bad", False),
        ("gzip;q=nan", False),
        ("gzip;q=2", False),
        ("gzip;q=-1", False),
        ("", False),
    ],
)
def test_encoding_quality(header: str, expected: bool) -> None:
    assert accepts_gzip(header) is expected


def test_cache_precision_expiry_revision_and_bypasses() -> None:
    calls: list[tuple[int, dict[str, Any], str | None]] = []
    clock = [10.0]

    def render(page: int, values: dict[str, Any], action: str | None) -> dict[str, Any]:
        calls.append((page, values, action))
        return {"integer": 2**60 + 1, "float": 1.0000000000000002, "text": "x" * 2000, "values": values}

    cache = ViewResponses(render, clock=lambda: clock[0])
    first = cache.render(1, {"year": 2026}, None, "r1", "gzip")
    assert cache.render(1, {"year": 2026}, None, "r1", "gzip").headers["x-f1-cache"] == "HIT"
    assert len(calls) == 1
    assert json.loads(gzip.decompress(first.body))["integer"] == 2**60 + 1
    identity = cache.render(1, {"year": 2026}, None, "r1", "gzip;q=0")
    assert "content-encoding" not in identity.headers
    assert json.loads(identity.body)["float"] == 1.0000000000000002
    cache.render(1, {"year": 2026}, None, "r2", "gzip")
    assert len(calls) == 2
    clock[0] += 21
    cache.render(1, {"year": 2026}, None, "r2", "gzip")
    assert len(calls) == 3
    for page, values, action in [
        (1, {}, "action"),
        (1, {"f1bet_csv_upload": "csv"}, None),
        (6, {}, None),
        (7, {}, None),
    ]:
        assert cache.render(page, values, action, "r2", "gzip").headers["x-f1-cache"] == "BYPASS"
    assert cache.render(1, {}, None, "r2", "gzip", enabled=False).headers["x-f1-cache"] == "BYPASS"
    assert not reusable(1, {"nested": {"key": "value"}}, None)
    assert not reusable(1, {"large": "x" * 4097}, None)
    assert not reusable(1, {"number": float("nan")}, None)
    assert not reusable(1, {"list": [[1]]}, None)
    assert not reusable(1, {"list": list(range(101))}, None)


def test_cache_bounds_eviction_and_concurrent_request_deduplication() -> None:
    calls = [0]
    started = threading.Event()
    release = threading.Event()

    def render(_page: int, values: dict[str, Any], _action: str | None) -> dict[str, Any]:
        calls[0] += 1
        started.set()
        if calls[0] == 1:
            assert release.wait(5)
        return {"text": "x" * 2000, "values": values}

    cache = ViewResponses(render, max_entries=2, max_bytes=10000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.render, 1, {}, None, "r", "gzip")
        assert started.wait(5)
        second = pool.submit(cache.render, 1, {}, None, "r", "gzip")
        release.set()
        results = [first.result(5), second.result(5)]
    assert calls[0] == 1
    assert {result.headers["x-f1-cache"] for result in results} == {"MISS", "HIT"}
    cache.render(1, {"id": 2}, None, "r", "gzip")
    cache.render(1, {"id": 3}, None, "r", "gzip")
    assert len(cache.entries) == 2
    assert cache.bytes <= cache.max_bytes
    assert cache.bytes == sum(entry.size for entry in cache.entries.values())
    bounded = ViewResponses(render, max_bytes=100)
    bounded.render(1, {}, None, "r", "gzip")
    assert bounded.bytes == 0
    assert not bounded.entries
    cache.clear()
    assert cache.bytes == 0


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dataset = tmp_path / "data_files"
    (dataset / "models").mkdir(parents=True)
    (dataset / "f1ForAnalysis.csv").write_text("year\n2026\n", encoding="utf-8")
    (tmp_path / "raceAnalysis.py").write_text("# source", encoding="utf-8")
    monkeypatch.setattr(service, "DATA_DIR", dataset)
    monkeypatch.setattr(service, "REPO_ROOT", tmp_path)
    return tmp_path


def test_revision_clears_response_presentation_and_source_caches(
    artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enhancement = service.Enhancements(poll_seconds=0)
    monkeypatch.setattr(service.presentation, "render_view", lambda *_: {"year": 2026})
    calls = [0]

    @lru_cache(maxsize=1)
    def cached_loader() -> int:
        calls[0] += 1
        return calls[0]

    monkeypatch.setattr(service.data, "fixture_loader", cached_loader, raising=False)
    first = enhancement.current_revision()
    service.presentation._CACHE["test-marker"] = "stale"
    assert cached_loader() == 1
    assert cached_loader() == 1
    enhancement.responses.render(1, {}, None, first, "identity")
    assert enhancement.responses.bytes > 0
    (artifact_root / "data_files" / "f1ForAnalysis.csv").write_text("year\n2025\n2026\n", encoding="utf-8")
    assert enhancement.current_revision() != first
    assert not enhancement.responses.entries
    assert enhancement.responses.bytes == 0
    assert "test-marker" not in service.presentation._CACHE
    assert cached_loader() == 2
    # A new/deleted source and an encoding-source selection change are revisions too.
    old = enhancement.current_revision()
    added = artifact_root / "data_files" / "new.json"
    added.write_text("{}", encoding="utf-8")
    assert enhancement.current_revision() != old
    old = enhancement.current_revision()
    added.unlink()
    assert enhancement.current_revision() != old
    old = enhancement.current_revision()
    monkeypatch.setenv("F1_USE_PARQUET", "0")
    assert enhancement.current_revision() != old


def test_revision_retries_changed_read_but_never_repeats_an_action(
    artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enhancement = service.Enhancements()
    calls = [0]
    dataset = artifact_root / "data_files" / "f1ForAnalysis.csv"

    def changing_renderer(_page: int, _values: dict[str, Any], _action: str | None) -> dict[str, Any]:
        calls[0] += 1
        if calls[0] == 1:
            dataset.write_text("year\n2025\n2026\n", encoding="utf-8")
        return {"render_number": calls[0]}

    monkeypatch.setattr(service.presentation, "render_view", changing_renderer)
    request = Request({"type": "http", "headers": []})
    result = enhancement.render(ViewRequest(page=1), request)
    assert json.loads(result.body)["render_number"] == 2
    assert len(enhancement.responses.entries) == 1
    assert result.headers["x-f1-revision"] == enhancement.current_revision()
    calls[0] = 0
    dataset.write_text("year\n2026\n", encoding="utf-8")
    with pytest.raises(service.ArtifactChangedError, match="Source artifacts changed"):
        enhancement.render(ViewRequest(page=1, action="explicit-action"), request)
    assert calls[0] == 1
    assert not enhancement.responses.entries


def test_continuously_changing_artifacts_return_retryable_503(
    artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app, enhancements

    assert enhancements is not None
    dataset = artifact_root / "data_files" / "f1ForAnalysis.csv"
    calls = [0]

    def renderer(*_args: Any) -> dict[str, Any]:
        calls[0] += 1
        dataset.write_text("year\n" + "2026\n" * calls[0], encoding="utf-8")
        return {"render": calls[0]}

    monkeypatch.setattr(service.presentation, "render_view", renderer)
    result = TestClient(app).post("/api/views", json={"page": 1})
    assert result.status_code == 503
    assert result.headers["retry-after"] == "1"
    assert calls[0] == 2
    assert not enhancements.responses.entries


def test_actual_api_cache_opt_out_and_encoding_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app, enhancements

    assert enhancements is not None
    monkeypatch.setattr(service.presentation, "render_view", lambda *_: {"text": "x" * 2000, "year": 2026})
    enhancements.responses.clear()
    with TestClient(app) as client:
        response = client.post("/api/views", json={"page": 1}, headers={"Accept-Encoding": "gzip"})
        assert response.headers["x-f1-cache"] == "MISS"
        assert response.headers["content-encoding"] == "gzip"
        identity = client.post("/api/views", json={"page": 1}, headers={"Accept-Encoding": "gzip;q=0,*;q=1"})
        assert identity.headers["x-f1-cache"] == "HIT"
        assert "content-encoding" not in identity.headers
        assert identity.json() == response.json()
        assert identity.headers["cache-control"] == "no-store"
        assert "Accept-Encoding" in identity.headers["vary"]
        monkeypatch.setenv("F1_VIEW_RESPONSE_CACHE", "0")
        bypass = client.post("/api/views", json={"page": 1}, headers={"Accept-Encoding": "gzip;q=0"})
        assert bypass.headers["x-f1-cache"] == "BYPASS"
        assert "content-encoding" not in bypass.headers
        assert "server-timing" in bypass.headers
        # Existing non-view routes use the same quality-aware middleware.
        assert (
            "content-encoding" not in client.get("/api/meta", headers={"Accept-Encoding": "gzip;q=0"}).headers
        )


def test_status_manifests_and_metrics_authorization(
    artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("F1_TRUSTED_LOCAL", raising=False)
    models = artifact_root / "data_files" / "models"
    (models / "manifest.json").write_text(
        json.dumps({"model_name": "position", "trained_at": "recorded-time"}), encoding="utf-8"
    )
    (models / "bad_manifest.json").write_text("[]", encoding="utf-8")
    enhancement = service.Enhancements()
    app = FastAPI()
    enhancement.install(app)
    monkeypatch.delenv("F1_ADMIN_TOKEN", raising=False)
    with TestClient(app) as client:
        status = client.get("/api/enhancements/status").json()
        assert status["dataset"]["name"] == "f1ForAnalysis.csv"
        assert any(manifest.get("trained_at") == "recorded-time" for manifest in status["models"])
        assert any(manifest.get("notes") == ["Manifest could not be read."] for manifest in status["models"])
        assert client.get("/api/enhancements/metrics").status_code == 503
        monkeypatch.setenv("F1_ADMIN_TOKEN", "test-only-admin")
        assert client.get("/api/enhancements/metrics").status_code == 403
        assert (
            client.get("/api/enhancements/metrics", headers={"X-F1-Admin-Token": "wrong"}).status_code == 403
        )
        response = client.get("/api/enhancements/metrics", headers={"X-F1-Admin-Token": "test-only-admin"})
        assert response.status_code == 200
        assert response.json()["cache_bytes"] == 0
        assert response.json()["requests"]
        assert client.post("/api/enhancements/jobs", json={"task": "leakage-audit"}).status_code == 403
    (artifact_root / "data_files" / "f1ForAnalysis.csv").unlink()
    assert enhancement.status()["dataset"]["modified_at"] is None


def test_bounded_request_records_unique_ids_redaction_and_compressed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[str] = []
    monkeypatch.setattr(metrics.log, "info", lambda pattern, value: logged.append(pattern % value))

    async def echo(_request: Request) -> JSONResponse:
        return JSONResponse({"text": "x" * 2000})

    app = FastAPI()
    app.add_api_route("/echo/{identity}", echo, methods=["POST"])
    records, lock = metrics_storage()
    app.add_middleware(NegotiatedGZipMiddleware, minimum_size=1000, compresslevel=5)
    app.add_middleware(RequestMetrics, records=records, lock=lock)
    with TestClient(app) as client:
        result = client.post(
            "/echo/private-value?token=private-query",
            json={"secret": "private-body"},
            headers={"Authorization": "private-header"},
        )
        assert result.headers["server-timing"].startswith("backend;dur=")
        assert len(result.headers["x-request-id"]) == 32
        assert records[-1]["body_bytes"] == int(result.headers["content-length"])
        assert records[-1]["body_bytes"] < len(result.content)
        for _ in range(500):
            client.post("/echo/ignored", json={})
    assert len(records) == 500
    assert len({record["request_id"] for record in records}) == 500
    assert all(record["route"] == "/echo/{identity}" for record in records)
    assert all(record["status"] == 200 and record["header_ms"] >= 0 for record in records)
    serialized = "\n".join(logged)
    assert all(
        value not in serialized
        for value in ["private-value", "private-query", "private-body", "private-header"]
    )


def test_metrics_record_failures_and_preserve_non_http_scopes() -> None:
    async def fail(_request: Request) -> JSONResponse:
        raise RuntimeError("failure")

    async def websocket(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_text("connected")
        await socket.close()

    app = FastAPI()
    app.add_api_route("/fail", fail, methods=["GET"])
    app.add_api_websocket_route("/socket", websocket)
    records, lock = metrics_storage()
    app.add_middleware(RequestMetrics, records=records, lock=lock, emit_logs=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        failure = client.get("/fail")
        assert failure.status_code == 500
        assert len(failure.headers["x-request-id"]) == 32
        assert failure.headers["server-timing"].startswith("backend;dur=")
        with client.websocket_connect("/socket") as connection:
            assert connection.receive_text() == "connected"
    assert len(records) == 1
    assert records[0]["status"] == 500
    assert records[0]["route"] == "/fail"
    assert records[0]["duration_ms"] >= 0
    assert records[0]["body_bytes"] == len(failure.content)


def test_revision_scan_cost_is_bounded_to_analysis_sources(artifact_root: Path) -> None:
    dataset = artifact_root / "data_files"
    telemetry = dataset / "f1_cache"
    telemetry.mkdir()
    first = service.artifact_revision(dataset, artifact_root)
    (telemetry / "downloaded.json").write_text("{}", encoding="utf-8")
    assert service.artifact_revision(dataset, artifact_root) == first
    (dataset / "predictions_race_2026.csv").write_text("driver,prediction\nname,1", encoding="utf-8")
    assert service.artifact_revision(dataset, artifact_root) == first
    (dataset / "supported.tsv").write_text("year\t2026", encoding="utf-8")
    assert service.artifact_revision(dataset, artifact_root) != first
    # The poll avoids a second stat scan on legacy endpoints within its window.
    enhancement = service.Enhancements(poll_seconds=60)
    enhancement.refresh_sources()
    checked = enhancement.checked
    time.sleep(0.001)
    enhancement.refresh_sources()
    assert enhancement.checked == checked
