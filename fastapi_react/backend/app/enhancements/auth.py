"""Explicit local access, with administrator authentication for hosted sessions."""

from __future__ import annotations

import ipaddress
import os
import secrets
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, Request

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_LOCAL_ORIGINS = tuple(
    f"http://{host}:{port}"
    for host in ("127.0.0.1", "localhost", "[::1]")
    for port in (5173, 5174, 8000)
)
FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto")


def trusted_local_enabled() -> bool:
    return os.environ.get("F1_TRUSTED_LOCAL", "0").strip().lower() in {"1", "true", "yes"}


def local_origins() -> list[str]:
    configured = os.environ.get("F1_LOCAL_ORIGINS")
    origins = configured.split(",") if configured is not None else list(DEFAULT_LOCAL_ORIGINS)
    result = []
    for entry in origins:
        origin = entry.strip()
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in LOCAL_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("F1_LOCAL_ORIGINS must contain only explicit localhost HTTP origins.")
        # Accessing port also rejects malformed or out-of-range port numbers.
        if parsed.port is not None and parsed.port < 1:
            raise ValueError("F1_LOCAL_ORIGINS ports must be positive.")
        if origin not in result:
            result.append(origin)
    return result


def is_trusted_local(request: Request) -> bool:
    if not trusted_local_enabled() or request.client is None:
        return False
    try:
        if not ipaddress.ip_address(request.client.host).is_loopback:
            return False
    except ValueError:
        return False
    # Local mode is for direct local use, never a public reverse proxy. The
    # launcher also disables Uvicorn's rewriting of the connection peer.
    if any(header in request.headers for header in FORWARDING_HEADERS):
        return False
    origins = local_origins()
    hosts = request.headers.getlist("host")
    if len(hosts) != 1 or hosts[0].lower() not in {urlsplit(origin).netloc.lower() for origin in origins}:
        return False
    browser_origins = request.headers.getlist("origin")
    if len(browser_origins) > 1 or (browser_origins and browser_origins[0] not in origins):
        return False
    # Cross-site navigations can omit Origin. Same-origin browser traffic and
    # local command-line clients (which omit both headers) remain supported.
    return request.headers.get("sec-fetch-site", "").lower() != "cross-site"


def research_access(request: Request) -> dict[str, str | bool]:
    local = is_trusted_local(request)
    return {"mode": "local" if local else "token", "token_required": not local}


def authorize_research(request: Request, x_f1_admin_token: str | None = Header(default=None)) -> None:
    if is_trusted_local(request):
        return
    expected = os.environ.get("F1_ADMIN_TOKEN")
    if expected:
        if x_f1_admin_token is not None and secrets.compare_digest(expected.encode(), x_f1_admin_token.encode()):
            return
        raise HTTPException(403, "Administrator access is required.")
    if trusted_local_enabled():
        raise HTTPException(403, "Research access is limited to this application's trusted local session.")
    raise HTTPException(503, "Administrator access is not configured. Set F1_ADMIN_TOKEN on the server.")
