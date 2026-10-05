from __future__ import annotations

import gzip
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.responses import JSONResponse, Response


def accepts_gzip(header: str) -> bool:
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
        choices[parts[0]] = quality
    return choices.get("gzip", choices.get("*", 0.0)) > 0


def reusable(page: int, values: dict[str, Any], action: str | None) -> bool:
    if action or page not in {1, 2, 3, 4, 5}:
        return False
    for key, value in values.items():
        if any(word in key.lower() for word in ("upload", "csv", "ledger")):
            return False
        if isinstance(value, dict) or (isinstance(value, str) and len(value) > 4096):
            return False
        if isinstance(value, list) and (
            len(value) > 100 or any(isinstance(item, (dict, list)) for item in value)
        ):
            return False
    return True


@dataclass
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

    def render(
        self, page: int, values: dict[str, Any], action: str | None,
        revision: str, encoding: str, *, enabled: bool = True,
    ) -> Response:
        headers = {"Cache-Control": "no-store", "X-F1-Revision": revision}
        if not enabled or not reusable(page, values, action):
            if action:
                self.clear()
            headers["X-F1-Cache"] = "BYPASS"
            return JSONResponse(self.renderer(page, values, action), headers=headers)
        key = json.dumps([revision, page, values], sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.lock:
            entry = self.entries.get(key)
            hit = entry is not None and entry.expires > self.clock()
            if entry is not None:
                self.entries.pop(key)
                self.bytes -= entry.size
            if not hit:
                body = bytes(JSONResponse(self.renderer(page, values, action)).body)
                entry = Entry(body, gzip.compress(body, compresslevel=5, mtime=0), self.clock()+self.ttl)
            if entry is None:
                raise RuntimeError("Could not construct the view cache entry")
            if entry.size <= self.max_bytes:
                self.entries[key] = entry
                self.bytes += entry.size
                while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old.size
            headers["X-F1-Cache"] = "HIT" if hit else "MISS"
            headers["Vary"] = "Accept-Encoding"
            zipped = len(entry.body) >= 1000 and accepts_gzip(encoding)
            if zipped:
                headers["Content-Encoding"] = "gzip"
            return Response(entry.compressed if zipped else entry.body, media_type="application/json", headers=headers)
