from __future__ import annotations

import gzip
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import Receive, Scope, Send


def accepts_gzip(header: str) -> bool:
    """Honor explicit gzip exclusions, quality bounds, and wildcard acceptance."""
    choices: dict[str, float] = {}
    for item in header.lower().split(","):
        parts = [part.strip() for part in item.split(";")]
        quality = 1.0
        for part in parts[1:]:
            if part.startswith("q="):
                try:
                    quality = float(part[2:])
                except ValueError:
                    quality = 0.0
        choices[parts[0]] = quality if math.isfinite(quality) and 0 <= quality <= 1 else 0.0
    return choices.get("gzip", choices.get("*", 0.0)) > 0


class NegotiatedGZipMiddleware(GZipMiddleware):
    """Starlette's gzip responder otherwise accepts the literal `gzip;q=0`."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        original_scope = scope
        if scope["type"] == "http":
            headers = scope.get("headers", [])
            accepted = b",".join(value for key, value in headers if key.lower() == b"accept-encoding")
            encoding = b"gzip" if accepts_gzip(accepted.decode("latin-1")) else b"identity"
            scope = {
                **scope,
                "headers": [(key, value) for key, value in headers if key.lower() != b"accept-encoding"]
                + [(b"accept-encoding", encoding)],
            }
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Routing annotates the copied scope; retain its public route template
            # for the outer diagnostic middleware without exposing path values.
            if "route" in scope:
                original_scope["route"] = scope["route"]


def _ordinary(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, str) and len(value) <= 4096


def reusable(page: int, values: dict[str, Any], action: str | None) -> bool:
    """Only cache ordinary public controls on the five nonvolatile analysis pages."""
    if action or page not in {1, 2, 3, 4, 5} or len(values) > 200:
        return False
    for key, value in values.items():
        if len(key) > 200 or any(word in key.lower() for word in ("upload", "csv", "ledger")):
            return False
        if isinstance(value, list):
            if len(value) > 100 or not all(_ordinary(item) for item in value):
                return False
        elif not _ordinary(value):
            return False
    return True


@dataclass(frozen=True)
class Entry:
    body: bytes
    compressed: bytes
    expires: float

    @property
    def size(self) -> int:
        return len(self.body) + len(self.compressed)


class ViewResponses:
    def __init__(
        self,
        renderer: Callable[[int, dict[str, Any], str | None], dict[str, Any]],
        *,
        ttl: float = 20,
        max_bytes: int = 64 * 1024 * 1024,
        max_entries: int = 12,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.renderer, self.ttl, self.max_bytes = renderer, ttl, max_bytes
        self.max_entries, self.clock = max_entries, clock
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.bytes = 0

    def _expire(self) -> None:
        expired = [key for key, entry in self.entries.items() if entry.expires <= self.clock()]
        for key in expired:
            self.bytes -= self.entries.pop(key).size

    def render(
        self,
        page: int,
        values: dict[str, Any],
        action: str | None,
        revision: str,
        encoding: str,
        *,
        enabled: bool = True,
    ) -> Response:
        headers = {"Cache-Control": "no-store", "X-F1-Revision": revision, "Vary": "Accept-Encoding"}
        if not enabled or not reusable(page, values, action):
            if action or not reusable(1, values, None):
                self.clear()
            headers["X-F1-Cache"] = "BYPASS"
            return JSONResponse(self.renderer(page, values, action), headers=headers)
        key = json.dumps([revision, page, values], sort_keys=True, separators=(",", ":"), allow_nan=False)
        # A concurrent identical request waits for the first render, then reuses it.
        with self.lock:
            self._expire()
            entry = self.entries.pop(key, None)
            hit = entry is not None
            if entry is not None:
                self.bytes -= entry.size
            else:
                body = bytes(JSONResponse(self.renderer(page, values, action)).body)
                entry = Entry(body, gzip.compress(body, compresslevel=5, mtime=0), self.clock() + self.ttl)
            if entry.size <= self.max_bytes and self.max_entries > 0:
                self.entries[key] = entry
                self.bytes += entry.size
                while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old.size
            headers["X-F1-Cache"] = "HIT" if hit else "MISS"
            zipped = len(entry.body) >= 1000 and accepts_gzip(encoding)
            if zipped:
                headers["Content-Encoding"] = "gzip"
            return Response(
                entry.compressed if zipped else entry.body, media_type="application/json", headers=headers
            )
