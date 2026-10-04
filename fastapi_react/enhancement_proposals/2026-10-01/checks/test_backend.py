import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("proposal_backend", ROOT/"code/backend/__init__.py", submodule_search_locations=[str(ROOT/"code/backend")])
module = importlib.util.module_from_spec(spec)
sys.modules["proposal_backend"] = module
spec.loader.exec_module(module)
# Windows workers can import the test package by the same name.
sys.path.insert(0, str(ROOT/"code"))
# Import under its actual package name for process-picklable worker functions.
from backend.testing_worker import fake_work
from proposal_backend.cache import ViewResponses, accepts_gzip
from proposal_backend.jobs import BusyQueueError, Jobs
from proposal_backend.metrics import BodyLimit, RequestMetrics, metrics_storage


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
