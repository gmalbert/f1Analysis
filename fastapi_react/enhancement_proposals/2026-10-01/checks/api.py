"""Verify the installed preview routes with real data, without running training."""
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from starlette.responses import JSONResponse

HERE = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:"+os.environ.get("PROPOSAL_API_PORT","9008")


def post(payload):
    request = urllib.request.Request(BASE+"/api/views",json.dumps(payload).encode(),{"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=120) as response:
        return json.loads(response.read()),dict(response.headers)


def tables(nodes):
    for node in nodes:
        if node.get("type") == "table":
            yield node
        yield from tables(node.get("children",[]))


payload = {"page":1,"values":{"filter_results_main":False,"_proposal_check":1}}
_, first = post(payload)
_, second = post(payload)
assert second["x-f1-cache"] == "HIT"
assert second["server-timing"].startswith("backend;dur=")
raw, headers = post({"page":6,"values":{"filter_results_main":True,"range_filter_grandPrixYear":[2017,2026],"show_raw_data_debug":True}})
table_data = list(tables(raw["nodes"]))
digest = hashlib.sha256(JSONResponse(table_data).body).hexdigest()
baseline = json.loads((HERE.parents[1]/"parity_evidence/performance-2026-10-01/raw-profile-before.json").read_text(encoding="utf-8"))
assert digest == baseline["table_sha256"], "The raw table changed"
request = urllib.request.Request(BASE+"/api/enhancements/jobs",json.dumps({"task":"leakage-audit","values":{}}).encode(),{"Content-Type":"application/json"})
try:
    urllib.request.urlopen(request)
    raise AssertionError("An unauthenticated job was accepted")
except urllib.error.HTTPError as error:
    assert error.code in {403,503}
result = {"cache_hit":True,"server_timing":True,"raw_table_sha256":digest,"raw_table_matches_baseline":True,
          "rows":len(table_data[0]["rows"]),"columns":len(table_data[0]["columns"]),"unauthenticated_jobs_rejected":True}
(HERE/"validation-api.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
