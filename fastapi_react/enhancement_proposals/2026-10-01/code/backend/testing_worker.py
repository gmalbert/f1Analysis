"""Deterministic isolated worker used by proposal checks, never by app routes."""
import time
from typing import Any


def fake_work(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    time.sleep(payload.get("delay", 0.01))
    if task == "fail":
        raise ValueError("Intentional test failure.")
    return {"task": task, "value": payload["value"]}
