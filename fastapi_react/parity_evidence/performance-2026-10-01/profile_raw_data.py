"""Separate presentation, framework encoding, JSON and gzip costs for the raw view."""

import cProfile
import gzip
import hashlib
import io
import json
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fastapi_react" / "backend"))

from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from app.services.presentation import render_view

OUT = Path(__file__).parent
PHASE = "after" if "--after" in sys.argv else "before"
VALUES = {
    "filter_results_main": True,
    "range_filter_grandPrixYear": [2017, 2026],
    "show_raw_data_debug": True,
}
render_view(6, {"filter_results_main": True})
samples = []
for run in range(3):
    start = time.perf_counter()
    payload = render_view(6, VALUES)
    rendered = time.perf_counter()
    encoded = jsonable_encoder(payload)
    converted = time.perf_counter()
    body = JSONResponse(encoded).body
    serialized = time.perf_counter()
    direct = JSONResponse(payload).body
    assert json.loads(direct) == json.loads(body)
    compressed = gzip.compress(body, compresslevel=9, mtime=0)
    finished = time.perf_counter()
    samples.append({"run": run + 1, "render_ms": (rendered - start) * 1000,
                    "framework_encoder_ms": (converted - rendered) * 1000,
                    "json_ms": (serialized - converted) * 1000,
                    "gzip9_and_equivalence_ms": (finished - serialized) * 1000,
                    "raw_bytes": len(body), "gzip9_bytes": len(compressed)})
    print(samples[-1], flush=True)
compression = []
for level in [1, 3, 4, 5, 6, 9]:
    times = []
    for _ in range(3):
        start = time.perf_counter()
        compressed = gzip.compress(body, compresslevel=level, mtime=0)
        times.append((time.perf_counter() - start) * 1000)
    compression.append({"level": level, "bytes": len(compressed), "times_ms": times})
profiler = cProfile.Profile()
profiler.enable()
render_view(6, VALUES)
profiler.disable()
profile_text = io.StringIO()
pstats.Stats(profiler, stream=profile_text).sort_stats("cumulative").print_stats(35)
(OUT / f"raw-render-profile-{PHASE}.txt").write_text(profile_text.getvalue(), encoding="utf-8")

def tables(nodes):
    for node in nodes:
        if node.get("type") == "table":
            yield node
        yield from tables(node.get("children", []))

table_data = list(tables(payload["nodes"]))
canonical = JSONResponse(table_data).body
result = {"samples": samples, "compression": compression,
          "table_sha256": hashlib.sha256(canonical).hexdigest(),
          "tables": [{"rows": len(table["rows"]), "columns": len(table["columns"])}
                     for table in table_data]}
(OUT / f"raw-profile-{PHASE}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
