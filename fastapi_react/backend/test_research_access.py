from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.enhancements.auth import local_origins
from app.enhancements.service import Enhancements


class FixtureJobs:
    """Route fixture: exercise access checks without starting calculations."""

    def __init__(self) -> None:
        self.state = "queued"
        self.submissions = 0

    def submit(self, _task: str, _context: dict[str, Any]) -> str:
        self.submissions += 1
        return "fixture"

    def status(self, identity: str) -> dict[str, Any]:
        return {"id": identity, "state": self.state}

    def result(self, _identity: str) -> dict[str, Any]:
        return {"source_revision": "fixture-r1", "nodes": []}

    def cancel(self, _identity: str) -> bool:
        self.state = "cancelled"
        return True


@pytest.fixture
def research_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.delenv("F1_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("F1_TRUSTED_LOCAL", raising=False)
    monkeypatch.delenv("F1_LOCAL_ORIGINS", raising=False)
    enhancement = Enhancements()
    monkeypatch.setattr(enhancement, "current_revision", lambda **_: "fixture-r1")
    monkeypatch.setattr(enhancement, "jobs", FixtureJobs())
    app = FastAPI()
    enhancement.install(app)
    return app


def protected_responses(client: TestClient) -> list[int]:
    return [
        client.get("/api/enhancements/metrics").status_code,
        client.post("/api/enhancements/jobs", json={"task": "leakage-audit"}).status_code,
        client.get("/api/enhancements/jobs/fixture").status_code,
        client.get("/api/enhancements/jobs/fixture/result").status_code,
        client.delete("/api/enhancements/jobs/fixture").status_code,
    ]


@pytest.mark.parametrize(("base", "peer", "origin"), [
    ("http://127.0.0.1:8000", "127.0.0.1", "http://127.0.0.1:5174"),
    ("http://localhost:8000", "127.0.0.1", "http://localhost:5174"),
    ("http://127.0.0.1:8000", "::1", "http://[::1]:5174"),
    ("http://127.0.0.1:8000", "127.0.0.1", "http://127.0.0.1:8000"),
    ("http://127.0.0.1:8000", "127.0.0.1", None),
])
def test_opt_in_direct_local_access_on_every_route(
    research_app: FastAPI, monkeypatch: pytest.MonkeyPatch, base: str, peer: str, origin: str | None,
) -> None:
    monkeypatch.setenv("F1_TRUSTED_LOCAL", "1")
    headers = {"Origin": origin} if origin else {}
    if peer == "::1":
        # Starlette's HTTPX transport cannot parse an IPv6 base URL. The raw
        # peer, browser Origin and actual Host still exercise IPv6 access.
        headers["Host"] = "[::1]:8000"
    with TestClient(research_app, base_url=base, client=(peer, 43000), headers=headers) as client:
        assert client.get("/api/enhancements/research-access").json() == {"mode": "local", "token_required": False}
        assert protected_responses(client) == [200, 202, 200, 200, 200]


@pytest.mark.parametrize(("peer", "headers"), [
    ("192.0.2.10", {"Origin": "http://127.0.0.1:5174"}),
    ("testclient", {}),
    ("127.0.0.1", {"Origin": "https://example.com"}),
    ("127.0.0.1", {"Origin": "null"}),
    ("127.0.0.1", {"Origin": "http://localhost:9000"}),
    ("127.0.0.1", {"Host": "example.com:8000"}),
    ("127.0.0.1", {"Host": "localhost.example.com:8000"}),
    ("127.0.0.1", {"Host": "127.0.0.1:9000"}),
    ("127.0.0.1", {"Sec-Fetch-Site": "cross-site"}),
    ("127.0.0.1", {"Forwarded": "for=127.0.0.1"}),
    ("127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}),
    ("127.0.0.1", {"X-Forwarded-Host": "localhost:8000"}),
    ("127.0.0.1", {"X-Forwarded-Proto": "http"}),
])
def test_remote_cross_site_rebinding_and_forwarded_requests_cannot_bypass(
    research_app: FastAPI, monkeypatch: pytest.MonkeyPatch, peer: str, headers: dict[str, str],
) -> None:
    monkeypatch.setenv("F1_TRUSTED_LOCAL", "1")
    with TestClient(research_app, base_url="http://127.0.0.1:8000", client=(peer, 43000), headers=headers) as client:
        assert client.get("/api/enhancements/research-access").json() == {"mode": "token", "token_required": True}
        assert protected_responses(client) == [403] * 5


def test_duplicate_origin_or_host_cannot_bypass(research_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1_TRUSTED_LOCAL", "1")
    with TestClient(research_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 43000)) as client:
        for headers in [
            [("Origin", "http://127.0.0.1:5174"), ("Origin", "https://example.com")],
            [("Host", "127.0.0.1:8000"), ("Host", "example.com")],
        ]:
            assert client.get("/api/enhancements/metrics", headers=headers).status_code == 403


def test_hosted_default_requires_credentials_even_on_loopback(
    research_app: FastAPI, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(research_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 43000)) as client:
        assert client.get("/api/enhancements/research-access").json() == {"mode": "token", "token_required": True}
        assert protected_responses(client) == [503] * 5
        monkeypatch.setenv("F1_ADMIN_TOKEN", "fixture-admin")
        assert protected_responses(client) == [403] * 5
        client.headers["X-F1-Admin-Token"] = "incorrect"
        assert protected_responses(client) == [403] * 5
        client.headers["X-F1-Admin-Token"] = "fixture-admin"
        assert protected_responses(client) == [200, 202, 200, 200, 200]


@pytest.mark.parametrize("origin", [
    "https://example.com", "http://localhost.example.com:5174", "http://localhost:5174/path",
    "http://localhost:5174?query=yes", "http://localhost:5174#fragment", "http://user@localhost:5174",
    "http://localhost:0", "http://localhost:65536", "null", "",
])
def test_configured_origins_must_remain_local(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    monkeypatch.setenv("F1_LOCAL_ORIGINS", origin)
    with pytest.raises(ValueError, match=r"F1_LOCAL_ORIGINS|Port"):
        local_origins()


def test_explicit_custom_local_port(research_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1_TRUSTED_LOCAL", "1")
    monkeypatch.setenv("F1_LOCAL_ORIGINS", "http://127.0.0.1:8000,http://localhost:9000")
    with TestClient(research_app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 43000)) as client:
        assert client.get("/api/enhancements/metrics", headers={"Origin": "http://localhost:9000"}).status_code == 200
        assert client.get("/api/enhancements/metrics", headers={"Origin": "http://localhost:5174"}).status_code == 403
