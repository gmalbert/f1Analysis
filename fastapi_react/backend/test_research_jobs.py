from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.enhancements import service
from app.enhancements.jobs import BusyQueueError, Jobs
from app.enhancements.metrics import BodyLimit


def fixture_work(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Importable test worker; never uses datasets or training routines."""
    time.sleep(payload.get("delay", 0.01))
    if task == "fail":
        raise ValueError("Intentional failure")
    return {"pid": os.getpid(), "value": payload.get("value", "ok")}


def finished(queue: Jobs, identity: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = queue.status(identity)
        if state["state"] not in {"queued", "running"}:
            return state
        time.sleep(0.02)
    raise AssertionError("Fixture job did not finish")


def test_spawned_worker_queue_cancel_capacity_failures_and_expiry() -> None:
    queue = Jobs(fixture_work, limit=3)
    try:
        first = queue.submit("work", {"delay": 0.5, "value": "original"})
        while queue.status(first)["state"] == "queued":
            time.sleep(0.01)
        second = queue.submit("work", {"value": "cancelled"})
        assert queue.cancel(second)
        third = queue.submit("fail", {})
        with pytest.raises(BusyQueueError):
            queue.submit("work", {})
        assert not queue.cancel(first)
        with pytest.raises(ValueError, match="not completed"):
            queue.result(first)
        assert finished(queue, first)["state"] == "succeeded"
        output = queue.result(first)
        assert output["pid"] != os.getpid()
        assert output["value"] == "original"
        assert finished(queue, third)["state"] == "failed"
        assert queue.status(second)["state"] == "cancelled"
        queue.ttl = 0.5
        time.sleep(0.51)
        with pytest.raises(KeyError):
            queue.status(first)
    finally:
        queue.close()
        queue.close()
    with pytest.raises(BusyQueueError):
        queue.submit("work", {})


def test_job_input_snapshot_and_size_bounds() -> None:
    queue = Jobs(fixture_work, result_limit=128)
    try:
        values = {"value": "original"}
        identity = queue.submit("work", values)
        values["value"] = "changed"
        assert finished(queue, identity)["state"] == "succeeded"
        assert queue.result(identity)["value"] == "original"
        large = queue.submit("work", {"value": "x" * 200})
        assert finished(queue, large)["state"] == "failed"
        with pytest.raises(ValueError, match="64 KiB"):
            queue.submit("work", {"value": "x" * 70000})
        with pytest.raises(ValueError, match="JSON compliant"):
            queue.submit("work", {"value": float("nan")})
    finally:
        queue.close()
    with pytest.raises(ValueError, match="positive"):
        Jobs(fixture_work, limit=0)


@pytest.mark.parametrize(("task", "values"), [
    ("unknown", {}), ("leakage-audit", {"Rows to read (0 = all)": 0}),
    ("leakage-audit", {"Rows to read (0 = all)": True}),
    ("leakage-audit", {"uploaded_csv": "private"}),
    ("bin-comparison", {"Select q values (number of bins)": [2, 2]}),
    ("bin-comparison", {"Select q values (number of bins)": [[2]]}),
    ("bin-comparison", {"Select q values (number of bins)": []}),
    ("bin-comparison", {"Select q values (number of bins)": [11]}),
])
def test_research_task_allowlist(task: str, values: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"Unsupported|Choose"):
        service.research_values(task, values)


def test_worker_dispatch_and_artifact_revision_before_and_after(monkeypatch: pytest.MonkeyPatch) -> None:
    revisions = iter(["r1", "r1", "r1", "r1", "r2", "r1", "r2"])
    monkeypatch.setattr(service, "artifact_revision", lambda *_: next(revisions))
    monkeypatch.setattr(service, "clear_source_caches", lambda: None)
    calls: list[Any] = []
    monkeypatch.setattr(service.presentation, "render_view", lambda *args: calls.append(args) or {"nodes": []})
    context = {"revision": "r1", "values": {}}
    assert service.execute_research("bin-comparison", context)["source_revision"] == "r1"
    assert calls[-1] == (5, {"Select q values (number of bins)": [2], "_tabs:📊 Model Performance": 6}, "Run Bin Count Comparison")
    service.execute_research("leakage-audit", context)
    assert calls[-1][0] == 6
    assert calls[-1][1]["Rows to read (0 = all)"] == 1000
    with pytest.raises(service.ArtifactChangedError, match="after submission"):
        service.execute_research("leakage-audit", context)
    with pytest.raises(service.ArtifactChangedError, match="during calculation"):
        service.execute_research("leakage-audit", context)


def test_authenticated_job_routes_and_synchronous_action_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("F1_TRUSTED_LOCAL", raising=False)
    enhancement = service.Enhancements()
    enhancement.jobs = Jobs(fixture_work)
    monkeypatch.setattr(enhancement, "current_revision", lambda **_: "fixture-r1")
    app = FastAPI()
    enhancement.install(app)
    monkeypatch.delenv("F1_ADMIN_TOKEN", raising=False)
    headers = {"X-F1-Admin-Token": "fixture-admin"}
    try:
        with TestClient(app) as client:
            assert client.post("/api/enhancements/jobs", json={"task": "leakage-audit"}).status_code == 503
            monkeypatch.setenv("F1_ADMIN_TOKEN", "fixture-admin")
            assert client.post("/api/enhancements/jobs", json={"task": "leakage-audit"}).status_code == 403
            assert client.post("/api/enhancements/jobs", json={"task": "leakage-audit", "values": {"upload": "private"}}, headers=headers).status_code == 400
            response = client.post("/api/enhancements/jobs", json={"task": "leakage-audit"}, headers=headers)
            assert response.status_code == 202
            identity = response.json()["id"]
            assert client.get(f"/api/enhancements/jobs/{identity}").status_code == 403
            finished(enhancement.jobs, identity)
            assert client.get(f"/api/enhancements/jobs/{identity}/result", headers=headers).json()["pid"] != os.getpid()
            assert not client.delete(f"/api/enhancements/jobs/{identity}", headers=headers).json()["cancelled"]
            assert client.get("/api/enhancements/jobs/missing", headers=headers).status_code == 404
        from app.main import app as main_app
        with TestClient(main_app) as client:
            for action in ("Run Leakage Audit", "Run Bin Count Comparison"):
                assert client.post("/api/views", json={"page": 6, "values": {}, "action": action}).status_code == 409
    finally:
        enhancement.close()


def test_body_limit_streaming_declared_malformed_disconnect_and_non_http() -> None:
    calls: list[Any] = []
    async def app(scope: Any, receive: Any, send: Any) -> None:
        calls.append(scope["type"])
        if scope["type"] == "http":
            calls.append(await receive())
    async def run(headers: list[Any], messages: list[Any], kind: str = "http") -> list[Any]:
        events: list[Any] = []
        iterator = iter(messages)
        async def receive() -> Any:
            return next(iterator)
        async def send(event: Any) -> None:
            events.append(event)
        await BodyLimit(app, max_bytes=4)({"type": kind, "headers": headers}, receive, send)
        return events
    for headers in [[(b"content-length", b"5")], [(b"content-length", b"-1")], [(b"content-length", b"x")], [(b"content-length", b"1"), (b"content-length", b"1")]]:
        assert asyncio.run(run(headers, []))[0]["status"] in {400, 413}
    assert not calls
    chunks = [{"type": "http.request", "body": b"ab", "more_body": True}, {"type": "http.request", "body": b"cde"}]
    assert asyncio.run(run([], chunks))[0]["status"] == 413
    assert not calls
    assert asyncio.run(run([(b"content-length", b"4")], [{"type": "http.request", "body": b"abc"}]))[0]["status"] == 400
    assert asyncio.run(run([], [{"type": "http.disconnect"}])) == []
    assert not calls
    asyncio.run(run([], [{"type": "http.request", "body": b"ab", "more_body": True}, {"type": "http.request", "body": b"cd"}]))
    assert calls[-1]["body"] == b"abcd"
    asyncio.run(run([], [], "lifespan"))
    assert calls[-1] == "lifespan"
    with pytest.raises(ValueError, match="positive"):
        BodyLimit(app, max_bytes=0)


def test_global_body_rejection_has_cors_timing_and_request_identifier() -> None:
    from app.main import app
    with TestClient(app) as client:
        response = client.post("/api/views", content=b"", headers={"Content-Length": str(1024 * 1024 + 1), "Origin": "http://local-test"})
    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "*"
    assert len(response.headers["x-request-id"]) == 32
    assert "backend;dur=" in response.headers["server-timing"]
