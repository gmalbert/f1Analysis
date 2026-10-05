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
