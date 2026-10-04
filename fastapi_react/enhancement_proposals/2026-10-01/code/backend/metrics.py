from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("f1.request")


class RequestMetrics:
    def __init__(self, app: ASGIApp, records: deque[dict[str, Any]], lock: Any) -> None:
        self.app, self.records, self.lock = app, records, lock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start, request_id = time.perf_counter(), uuid.uuid4().hex
        status, sent, header_ms = 500, 0, None

        async def measured_send(message: Message) -> None:
            nonlocal status, sent, header_ms
            if message["type"] == "http.response.start":
                status = message["status"]
                header_ms = 1000*(time.perf_counter()-start)
                message = {**message, "headers": [*message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                    (b"server-timing", ("backend;dur="+format(header_ms, ".2f")).encode()),
                ]}
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            record = {
                "request_id": request_id, "method": scope["method"],
                "route": getattr(scope.get("route"), "path", "<unmatched>"),
                "status": status, "header_ms": header_ms,
                "duration_ms": round(1000*(time.perf_counter()-start), 2), "body_bytes": sent,
            }
            with self.lock:
                self.records.append(record)
            log.info("%s", json.dumps(record))


class BodyLimit:
    """Bound the entire request before JSON parsing; preserve valid body bytes."""

    def __init__(self, app: ASGIApp, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = self.max_bytes+1
        chunks: list[bytes] = []
        size = 0
        if declared <= self.max_bytes:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                chunks.append(chunk)
                size += len(chunk)
                if size > self.max_bytes or not message.get("more_body", False):
                    break
        if declared > self.max_bytes or size > self.max_bytes:
            await JSONResponse({"detail": "The request is too large. Reduce uploaded CSV data."}, status_code=413)(scope, receive, send)
            return
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def metrics_storage() -> tuple[deque[dict[str, Any]], Any]:
    return deque(maxlen=500), threading.Lock()
