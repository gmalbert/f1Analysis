"""Compare the current complete raw table with the pre-optimization digest."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fastapi_react" / "backend"))

from starlette.responses import JSONResponse

from app.services.presentation import render_view

OUT = Path(__file__).parent


def tables(nodes):
    for node in nodes:
        if node.get("type") == "table":
            yield node
        yield from tables(node.get("children", []))


payload = render_view(
    6,
    {
        "filter_results_main": True,
        "range_filter_grandPrixYear": [2017, 2026],
        "show_raw_data_debug": True,
    },
)
table_data = list(tables(payload["nodes"]))
digest = hashlib.sha256(JSONResponse(table_data).body).hexdigest()
baseline = json.loads((OUT / "raw-profile-before.json").read_text(encoding="utf-8"))
if digest != baseline["table_sha256"]:
    raise AssertionError("The raw table differs from its pre-optimization values or metadata")
result = {
    "table_sha256": digest,
    "matches_baseline": True,
    "tables": [
        {"rows": len(table["rows"]), "columns": len(table["columns"])} for table in table_data
    ],
}
(OUT / "raw-checksum-final.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result), flush=True)
