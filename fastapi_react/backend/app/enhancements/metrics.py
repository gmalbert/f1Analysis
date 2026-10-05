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


def configure_request_logging() -> None:
    """Provide structured request records even without an explicit uvicorn config."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


class BodyLimit:
    """Validate aggregate bytes before the application can decode JSON or uploads."""

    def __init__(self, app: ASGIApp, max_bytes: int = 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("F1_MAX_REQUEST_BYTES must be a positive integer.")
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        lengths = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
        declared: int | None = None
        if lengths:
            # Reject duplicates and ambiguous signed/whitespace/comma encodings.
            if len(lengths) != 1 or not lengths[0] or not lengths[0].isdigit():
                await JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)(
                    scope, receive, send
                )
                return
            try:
                declared = int(lengths[0])
            except ValueError:
                declared = self.max_bytes + 1
        if declared is not None and declared > self.max_bytes:
            await self._too_large(scope, receive, send)
            return
        body = bytearray()
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        if declared is not None and declared != size:
            await JSONResponse({"detail": "Content-Length does not match the request body."}, status_code=400)(
                scope, receive, send
            )
            return
        buffered: bytes | None = bytes(body)
        del body

        async def replay() -> Message:
            nonlocal buffered
            if buffered is not None:
                message: Message = {"type": "http.request", "body": buffered, "more_body": False}
                buffered = None
                return message
            return await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(
            {"detail": "The request is too large."}, status_code=413
        )(scope, receive, send)


class RequestMetrics:
    def __init__(
        self,
        app: ASGIApp,
        records: deque[dict[str, Any]],
        lock: Any,
        *,
        emit_logs: bool = True,
    ) -> None:
        self.app, self.records, self.lock, self.emit_logs = app, records, lock, emit_logs
        if emit_logs:
            configure_request_logging()

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
                header_ms = round(1000 * (time.perf_counter() - start), 2)
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"x-request-id", request_id.encode()),
                        (b"server-timing", ("backend;dur=" + format(header_ms, ".2f")).encode()),
                    ],
                }
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        except Exception:
            if header_ms is None:
                # ServerErrorMiddleware sits outside user middleware. Start its
                # generic 500 here so failures also carry timing and an ID, then
                # re-raise for the server's normal exception logging behavior.
                await JSONResponse({"detail": "Internal server error"}, status_code=500)(
                    scope, receive, measured_send
                )
            raise
        finally:
            record = {
                "request_id": request_id,
                "method": scope["method"],
                "route": getattr(scope.get("route"), "path", "<unmatched>"),
                "status": status,
                "header_ms": header_ms,
                "duration_ms": round(1000 * (time.perf_counter() - start), 2),
                "body_bytes": sent,
            }
            with self.lock:
                self.records.append(record)
            if self.emit_logs:
                log.info("%s", json.dumps(record))


def metrics_storage() -> tuple[deque[dict[str, Any]], Any]:
    return deque(maxlen=500), threading.Lock()
