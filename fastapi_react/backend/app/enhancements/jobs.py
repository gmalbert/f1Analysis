from __future__ import annotations

import gzip
import json
import logging
import multiprocessing
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, cast

log = logging.getLogger("f1.jobs")


class BusyQueueError(Exception):
    """No bounded queue slot is available."""


class Jobs:
    """One spawned calculation process; bounded state is local and non-durable."""

    def __init__(
        self,
        execute: Callable[[str, dict[str, Any]], dict[str, Any]],
        *,
        limit: int = 8,
        result_limit: int = 32 * 1024 * 1024,
        ttl: float = 600,
    ) -> None:
        if limit < 1 or result_limit < 1 or ttl <= 0:
            raise ValueError("Job limits must be positive.")
        self.execute, self.limit, self.result_limit, self.ttl = execute, limit, result_limit, ttl
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="f1-research")
        self.worker = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self.lock = threading.RLock()
        self.items: dict[str, dict[str, Any]] = {}
        self.futures: dict[str, Future[None]] = {}
        self.closed = False

    def submit(self, task: str, values: dict[str, Any]) -> str:
        # Serialization isolates caller mutation and bounds retained inputs.
        text = json.dumps(values, allow_nan=False)
        if len(text.encode()) > 64 * 1024:
            raise ValueError("Research job inputs must be below 64 KiB.")
        with self.lock:
            self._expire()
            if self.closed:
                raise BusyQueueError("The local research queue is shutting down.")
            if len(self.items) >= self.limit:
                raise BusyQueueError("The local research queue is full; wait for completed results to expire.")
            identity = uuid.uuid4().hex
            self.items[identity] = {
                "id": identity,
                "task": task,
                "state": "queued",
                "created": time.time(),
                "finished": None,
                "revision": values.get("revision"),
            }
            self.futures[identity] = self.pool.submit(self._run, identity, task, json.loads(text))
            return identity

    def _expire(self) -> None:
        now = time.time()
        for identity, item in list(self.items.items()):
            if item["finished"] is not None and now - item["finished"] > self.ttl:
                self.items.pop(identity)
                self.futures.pop(identity, None)

    def _run(self, identity: str, task: str, values: dict[str, Any]) -> None:
        with self.lock:
            self.items[identity]["state"] = "running"
        try:
            result = self.worker.submit(self.execute, task, values).result()
            if not isinstance(result, dict):
                raise ValueError("Expected a research result object.")
            body = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
            if len(body) > self.result_limit:
                raise ValueError("Research result exceeds the configured limit.")
            compressed = gzip.compress(body, compresslevel=5)
            if len(compressed) > self.result_limit:
                raise ValueError("Compressed research result exceeds the configured limit.")
            with self.lock:
                self.items[identity].update(state="succeeded", body=compressed)
        except Exception:  # Record a worker failure without exposing data or killing the coordinator.
            log.exception("Research job %s failed", identity)
            with self.lock:
                self.items[identity].update(
                    state="failed", error="Research calculation failed; see the server log using this job ID."
                )
        finally:
            with self.lock:
                self.items[identity]["finished"] = time.time()

    def status(self, identity: str) -> dict[str, Any]:
        with self.lock:
            self._expire()
            return {key: value for key, value in self.items[identity].items() if key != "body"}

    def result(self, identity: str) -> dict[str, Any]:
        with self.lock:
            self._expire()
            if self.items[identity]["state"] != "succeeded":
                raise ValueError("The job has not completed successfully.")
            body = self.items[identity]["body"]
        return cast("dict[str, Any]", json.loads(gzip.decompress(body)))

    def cancel(self, identity: str) -> bool:
        with self.lock:
            self._expire()
            if self.items[identity]["state"] != "queued" or not self.futures[identity].cancel():
                return False
            self.items[identity].update(state="cancelled", finished=time.time())
            return True

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for identity in list(self.items):
                self.cancel(identity)
        # Running calculations finish. Shutdown never kills a calculation midway.
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.worker.shutdown(wait=True, cancel_futures=True)
