# Complete verification source

This chapter contains every test and preview script supplied with the proposal. The current application's existing tests are retained; these files add module and feature checks and adapt the changed request boundary.

## Test layers

- `Enhancements.test.jsx` exercises the semantic table/driver view, presets/links, caching and private-data exclusions.
- `test_enhancements.py` integrates the four standalone backend contracts and two service/dispatch tests into the current pytest suite.
- `checks/test_backend.py` can exercise proposal cache, body limit, metrics, and spawned-job behavior without installing the candidate in the main application.
- `checks/frontend.mjs` checks safe view encoding, input exclusions, request deduplication, cancellation, timeout feedback, and bounded client reuse.
- `checks/api.py` probes a running flagged API for response reuse, timing, raw-data identity, and guarded routes.
- `checks/browser.mjs` builds no app code; it serves existing baseline/proposed dist directories, exercises five flows, collects errors, and writes eight real screenshots.
- `prepare_preview.py` generates complete integration files and a named isolated staging copy.

The backend dispatcher test deliberately mocks expensive source calculations. The spawned process queue is tested with a small top-level importable worker. These tests prove the queue contract and dispatch choices, not the scientific validity or runtime cost of a newly trained model.

See [deployment and validation](06_DEPLOYMENT_AND_VALIDATION.md) for commands, environment flags, recorded results, and limits. See [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) for file identities.

## Full verification files

## checks/api.py

[Separate source file](checks/api.py)

```python
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
```

## checks/browser.mjs

[Separate source file](checks/browser.mjs)

```javascript
import {chromium} from 'playwright';
import {createServer,request} from 'node:http';
import {mkdir,readFile,stat,writeFile} from 'node:fs/promises';
import {dirname,extname,resolve,sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url)), pack = resolve(here,'..');
const repo = resolve(pack,'../../..'), dist = resolve(repo,'fastapi_react/.runtime/enhancement-preview/frontend/dist');
const backendPort = Number(process.env.PROPOSAL_API_PORT || 9008);
const images = resolve(pack,'images');await mkdir(images,{recursive:true});
const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp','.woff2':'font/woff2'};
function preview(directory) {return createServer(async(req,res) => {
  try {
    const url = new URL(req.url,'http://localhost');
    if(url.pathname.startsWith('/api/')) {
      const upstream = request({host:'127.0.0.1',port:backendPort,path:req.url,method:req.method,headers:{...req.headers,host:'127.0.0.1:'+backendPort}},
        response => {res.writeHead(response.statusCode,response.headers);response.pipe(res);});
      upstream.on('error',error => {res.writeHead(502);res.end(error.message);});req.pipe(upstream);return;
    }
    let path = resolve(directory,'.'+decodeURIComponent(url.pathname));
    if(path !== directory && !path.startsWith(directory+sep)){res.writeHead(403);res.end();return;}
    if(!(await stat(path).catch(() => null))?.isFile())path = resolve(directory,'index.html');
    const body = await readFile(path);
    res.writeHead(200,{'content-type':mime[extname(path)] || 'application/octet-stream'});res.end(body);
  }catch(error){res.writeHead(500);res.end(error.message);}
});}
const server = preview(dist), currentServer = preview(resolve(repo,'fastapi_react/frontend/dist'));
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
await new Promise(resolve => currentServer.listen(0,'127.0.0.1',resolve));
const base = 'http://127.0.0.1:'+server.address().port;
const currentBase = 'http://127.0.0.1:'+currentServer.address().port;
const browser = await chromium.launch(), checks = [], errors = [];
function track(page) {
  page.on('pageerror',error => errors.push(error.message));
  page.on('console',message => {if(message.type() === 'error')errors.push(message.text());});
  page.on('response',response => {if(response.status() >= 400)errors.push(response.status()+' '+response.url());});
}
async function ready(page) {
  await page.waitForTimeout(150);
  await page.locator('main[aria-busy=false]').waitFor({timeout:120000});
  await page.evaluate(async() => {await document.fonts.ready;await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));});
}
async function screenshot(page,name) {await page.screenshot({path:resolve(images,name+'.png')});}
try {
  for(const viewport of [{name:'desktop',width:1280,height:900},{name:'mobile',width:390,height:844}]) {
    const current = await browser.newPage({viewport});track(current);
    await current.goto(currentBase);await ready(current);await screenshot(current,'current-'+viewport.name);await current.close();
    const page = await browser.newPage({viewport});track(page);
    await page.goto(base);await ready(page);
    await page.getByText('Analysis tools',{exact:true}).click();
    await page.getByRole('checkbox',{name:'Improve readability',exact:true}).check();
    await page.waitForFunction(() => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() === '#b4232d');
    await page.getByText('Analysis tools',{exact:true}).click();await screenshot(page,'proposed-'+viewport.name);
    if(viewport.name === 'desktop') {
      await page.getByRole('checkbox',{name:'Filter Results',exact:true}).check();await ready(page);
      const region = page.getByRole('region',{name:'Table display',exact:true}).first();
      await region.scrollIntoViewIfNeeded();
      await region.getByRole('button',{name:'Accessible table',exact:true}).click();
      await region.getByRole('table').first().waitFor();
      await region.getByRole('button',{name:'Next rows',exact:true}).click();
      if(!await region.getByRole('caption').first().textContent().then(text => text.includes('51')))throw new Error('Accessible row paging failed.');
      await region.screenshot({path:resolve(images,'proposed-accessible-table.png')});
      checks.push('Accessible semantic table, all-field selector, row paging and formatting');
      await region.getByRole('button',{name:'Compare drivers',exact:true}).click();
      await region.getByRole('checkbox',{name:'Max Verstappen',exact:true}).check();
      await region.getByRole('checkbox',{name:'Lewis Hamilton',exact:true}).check();
      await region.getByRole('table').last().screenshot({path:resolve(images,'proposed-driver-comparison.png')});
      checks.push('Descriptive driver comparison on current filtered rows');
      await page.evaluate(() => window.scrollTo(0,0));
      await page.getByText('Analysis tools',{exact:true}).click();
      await page.getByRole('checkbox',{name:'Reuse recent views',exact:true}).check();await ready(page);
      await page.getByRole('textbox',{name:'View name',exact:true}).fill('Filtered history');
      await page.getByRole('button',{name:'Save view',exact:true}).click();
      await page.getByText('View saved on this device.',{exact:true}).waitFor();
      await screenshot(page,'proposed-analysis-tools');
      checks.push('Saved view and optional client cache controls');
      await page.getByRole('button',{name:'Find section (Ctrl/⌘ K)',exact:true}).click();
      await page.getByRole('dialog').getByRole('textbox').fill('Models');
      await screenshot(page,'proposed-command-palette');
      await page.getByRole('dialog').getByRole('button',{name:'Predictive Models',exact:true}).click();await ready(page);
      if(await page.getByRole('dialog').count() && await page.getByRole('dialog').isVisible())throw new Error('Command dialog did not close.');
      checks.push('Native modal search, keyboard-capable navigation and model loading');
      const evidence = page.waitForEvent('download');
      await page.getByRole('button',{name:'Download analysis context',exact:true}).click();
      const context = JSON.parse(await readFile(await(await evidence).path(),'utf8'));
      if(context.schema !== 'f1-analysis-context-v1' || !context.provenance.revision)throw new Error('Analysis context export is incomplete.');
      checks.push('Reproducible analysis context JSON with data/model provenance');
    }
    await page.close();
  }
  if(errors.length)throw new Error(JSON.stringify(errors));
}catch(error){errors.push(error.stack || error.message);process.exitCode=1;}
finally {
  await writeFile(resolve(pack,'validation-browser.json'),JSON.stringify({generated_at:new Date().toISOString(),checks,errors,preview_backend:'isolated port '+backendPort,baseline:'Existing production bundle in a temporary static preview',live_app_changed:false},null,2)+'\n');
  await browser.close();await new Promise(resolve => server.close(resolve));await new Promise(resolve => currentServer.close(resolve));
}
```

## checks/frontend.mjs

[Separate source file](checks/frontend.mjs)

```javascript
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// Load pure modules as data URLs; the implementation is identical to the source files.
const root = new URL('../code/frontend/',import.meta.url);
const preferencesText = await readFile(new URL('preferences.js',root),'utf8');
const preferencesURL = 'data:text/javascript;base64,'+Buffer.from(preferencesText).toString('base64');
const preferences = await import(preferencesURL);
const clientText = (await readFile(new URL('viewClient.js',root),'utf8')).replace("'./preferences.js'",JSON.stringify(preferencesURL));
const {createViewClient} = await import('data:text/javascript;base64,'+Buffer.from(clientText).toString('base64'));
const storage = {value:null,getItem(){return this.value;},setItem(_key,value){this.value=value;}};
const values = {filter_results_main:true,range_filter_grandPrixYear:[2017,2026],filter_resultsDriverName:'Max Verstappen',f1bet_field_upload:{content:'private'},bankroll:10000};
preferences.savePreset('Recent seasons',2,values,storage);
assert.equal(preferences.readPresets(storage)[0].values.f1bet_field_upload,undefined);
assert.equal(preferences.readPresets(storage)[0].values.bankroll,undefined);
const url = preferences.shareUrl(2,values,'http://localhost/#/Analytics');
assert.deepEqual(preferences.readSharedView(new URL(url).hash).values,preferences.safeValues(values));
assert.throws(() => preferences.validateView({version:99,page:1}));
assert.equal(preferences.hasUpload({f1bet_field_upload:'csv'}),true);
let revision = 'r1', posts = 0;
const fetcher = async url => {
  if(url.endsWith('/status'))return {ok:true,json:async() => ({revision})};
  posts++; await new Promise(resolve => setTimeout(resolve,10));
  return {ok:true,json:async() => ({nodes:[],value:posts})};
};
const client = createViewClient({fetcher});
const payload = {page:1,values:{year:2026}};
await Promise.all([client.load(payload,{enabled:true}),client.load(payload,{enabled:true})]);
assert.equal(posts,1,'Identical requests must share one API post');
await client.load(payload,{enabled:true});assert.equal(posts,1);
revision = 'r2';await client.load(payload,{enabled:true});assert.equal(posts,2);
await client.load({...payload,action:'explicit'},{enabled:true});assert.equal(posts,3);
await client.load(payload,{enabled:true});assert.equal(posts,4,'Actions must invalidate retained views');
const cancelled = new AbortController();cancelled.abort();
await assert.rejects(client.load(payload,{signal:cancelled.signal}),{name:'AbortError'});
const tiny = createViewClient({fetcher,maxBytes:1});
await tiny.load(payload,{enabled:true});assert.equal(tiny.retainedBytes(),0);
const timeoutClient = createViewClient({normalTimeout:5,fetcher:(_url,{signal}) => new Promise((_resolve,reject) => signal.addEventListener('abort',() => reject(new DOMException('Cancelled','AbortError'))))});
await assert.rejects(timeoutClient.load(payload),/timed out/);
console.log('Frontend proposal contracts passed: presets, share links, privacy, request deduplication, revision invalidation, actions, cancellation and cache budget.');
```

## checks/service_tests.py

[Separate source file](checks/service_tests.py)

```python
def test_service_status_refresh_auth_and_metrics(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from app.enhancements import service
    root = tmp_path
    dataset = root/"data_files"
    models = dataset/"models"
    models.mkdir(parents=True)
    (dataset/"f1ForAnalysis.csv").write_text("year\n2026\n")
    (root/"raceAnalysis.py").write_text("# source")
    (models/"manifest.json").write_text(json.dumps({"model_name":"position","notes":["recorded"],"trained_at":"today"}))
    monkeypatch.setattr(service,"DATA_DIR",dataset)
    monkeypatch.setattr(service,"REPO_ROOT",root)
    monkeypatch.delenv("F1_ADMIN_TOKEN",raising=False)
    enhancement = service.Enhancements(poll_seconds=0)
    app = FastAPI()
    enhancement.install(app)
    try:
        with TestClient(app) as client:
            first = client.get("/api/enhancements/status").json()
            assert first["dataset"]["name"] == "f1ForAnalysis.csv"
            (dataset/"f1ForAnalysis.csv").write_text("year\n2025\n2026\n")
            assert client.get("/api/enhancements/status").json()["revision"] != first["revision"]
            assert client.get("/api/enhancements/metrics").status_code == 503
            monkeypatch.setenv("F1_ADMIN_TOKEN","test-only")
            assert client.get("/api/enhancements/metrics").status_code == 403
            assert client.get("/api/enhancements/metrics",headers={"X-F1-Admin-Token":"test-only"}).status_code == 200
            assert client.post("/api/enhancements/jobs",json={"task":"unsupported"},headers={"X-F1-Admin-Token":"test-only"}).status_code == 400
            assert client.post("/api/enhancements/jobs",json={"task":"leakage-audit","values":{"Uploaded CSV":"year\n2026"}},headers={"X-F1-Admin-Token":"test-only"}).status_code == 400
            assert client.get("/api/enhancements/jobs/missing",headers={"X-F1-Admin-Token":"test-only"}).status_code == 404
    finally:
        enhancement.jobs.close()


def test_research_dispatch_uses_only_existing_opt_in_actions(monkeypatch):
    from app.enhancements import service
    monkeypatch.setattr(service,"artifact_revision",lambda *_:"revision")
    monkeypatch.setattr(service,"clear_source_caches",lambda:None)
    monkeypatch.setattr(service.presentation,"render_view",lambda page,values,action:{"page":page,"values":values,"action":action})
    context = {"revision":"revision","values":{}}
    bins = service.execute_research("bin-comparison",context)
    assert bins["action"] == "Run Bin Count Comparison"
    assert bins["values"]["Select q values (number of bins)"] == [2]
    assert service.execute_research("leakage-audit",context)["action"] == "Run Leakage Audit"
    with pytest.raises(ValueError, match="Unsupported research task"):
        service.execute_research("unknown",context)
    with pytest.raises(ValueError, match="q values"):
        service.execute_research("bin-comparison",{"revision":"revision","values":{"Select q values (number of bins)":[1]}})
    with pytest.raises(ValueError, match="Audit row limit"):
        service.execute_research("leakage-audit",{"revision":"revision","values":{"Rows to read (0 = all)":-1}})
    with pytest.raises(ValueError, match="Artifacts changed"):
        service.execute_research("leakage-audit",{"revision":"stale","values":{}})
```

## checks/test_backend.py

[Separate source file](checks/test_backend.py)

```python
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("proposal_backend", ROOT/"code/backend/__init__.py", submodule_search_locations=[str(ROOT/"code/backend")])
module = importlib.util.module_from_spec(spec)
sys.modules["proposal_backend"] = module
spec.loader.exec_module(module)
# Windows workers can import the test package by the same name.
sys.path.insert(0, str(ROOT/"code"))
# Import under its actual package name for process-picklable worker functions.
from backend.testing_worker import fake_work
from proposal_backend.cache import ViewResponses, accepts_gzip
from proposal_backend.jobs import BusyQueueError, Jobs
from proposal_backend.metrics import BodyLimit, RequestMetrics, metrics_storage


def test_cache_revision_precision_encoding_expiry_actions_and_uploads():
    calls, clock = [], [10.]
    def render(page, values, action):
        calls.append((page, values, action))
        return {"page":page, "integer":2**60+1,"float":1.0000000000000002,"text":"x"*2000,"values":values}
    cache = ViewResponses(render, clock=lambda:clock[0])
    first = cache.render(1,{"year":2026},None,"r1","gzip")
    second = cache.render(1,{"year":2026},None,"r1","gzip")
    assert second.headers["x-f1-cache"] == "HIT"
    assert len(calls) == 1
    assert json.loads(gzip.decompress(first.body))["integer"] == 2**60+1
    identity = cache.render(1,{"year":2026},None,"r1","gzip;q=0")
    assert "content-encoding" not in identity.headers
    assert json.loads(identity.body)["float"] == 1.0000000000000002
    cache.render(1,{"year":2026},None,"r2","gzip")
    assert len(calls) == 2
    clock[0] += 21
    cache.render(1,{"year":2026},None,"r2","gzip")
    assert len(calls) == 3
    assert cache.render(1,{}, "action","r2","gzip").headers["x-f1-cache"] == "BYPASS"
    assert cache.render(1,{"f1bet_field_upload":"csv"},None,"r2","gzip").headers["x-f1-cache"] == "BYPASS"
    assert cache.render(6,{},None,"r2","gzip").headers["x-f1-cache"] == "BYPASS"
    assert not accepts_gzip("gzip;q=0,*;q=1")


def test_cache_memory_bound():
    cache = ViewResponses(lambda *_: {"text":"x"*3000}, max_bytes=1000)
    cache.render(1,{},None,"r","gzip")
    assert cache.bytes == 0
    assert not cache.entries


def test_body_limit_and_timing_preserve_valid_json_and_reject_oversize():
    async def echo(request):
        return JSONResponse(await request.json())
    app = Starlette(routes=[Route("/echo",echo,methods=["POST"])])
    records, lock = metrics_storage()
    app.add_middleware(BodyLimit,max_bytes=128)
    app.add_middleware(RequestMetrics,records=records,lock=lock)
    with TestClient(app) as client:
        result = client.post("/echo",json={"year":2026})
        assert result.json() == {"year":2026}
        assert result.headers["server-timing"].startswith("backend;dur=")
        assert len(result.headers["x-request-id"]) == 32
        assert client.post("/echo",json={"text":"x"*200}).status_code == 413
    assert [record["status"] for record in records] == [200,413]
    assert all("values" not in record for record in records)


def test_isolated_jobs_results_capacity_and_queued_cancellation():
    jobs = Jobs(fake_work,limit=2)
    try:
        first = jobs.submit("test",{"value":7,"delay":1})
        second = jobs.submit("test",{"value":8})
        with pytest.raises(BusyQueueError):
            jobs.submit("test",{"value":9})
        assert jobs.cancel(second)
        deadline = time.monotonic()+30
        while jobs.status(first)["state"] in {"queued","running"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert jobs.result(first) == {"task":"test","value":7}
        assert jobs.status(second)["state"] == "cancelled"
        jobs.ttl = -1
        with pytest.raises(KeyError):
            jobs.status(first)
        jobs.ttl = 600
        with pytest.raises(ValueError, match="below 64 KiB"):
            jobs.submit("test",{"value":"x"*70000})
        failed = jobs.submit("fail",{"value":0})
        deadline = time.monotonic()+30
        while jobs.status(failed)["state"] in {"queued","running"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert jobs.status(failed)["state"] == "failed"
        with pytest.raises(ValueError, match="not completed successfully"):
            jobs.result(failed)
    finally:
        jobs.close()
```

## prepare_preview.py

[Separate source file](prepare_preview.py)

```python
"""Generate complete proposed replacements and an isolated validation checkout."""

from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
APP = REPO/"fastapi_react"
STAGE = APP/".runtime"/"enhancement-preview"
CODE = HERE/"code"


def once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("The integration anchor changed: "+old[:80])
    return source.replace(old, new, 1)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


source = (APP/"frontend/src/App.jsx").read_text(encoding="utf-8")
source = once(source, "import { api } from './api';", """import { viewClient } from './enhancements/viewClient';
import { FeatureBar, LoadingFeedback, readOptions } from './enhancements/FeatureBar';
import { ResearchJobs } from './enhancements/ResearchJobs';
import { readSharedView, safeValues } from './enhancements/preferences';""")
source = once(source, "const BASE_TITLE =", "const FEATURES_ENABLED = import.meta.env.VITE_F1_ENHANCEMENTS === '1';\nconst BASE_TITLE =")
source = once(source, "function readPage() {", """function sharedView() {
  try {return FEATURES_ENABLED ? readSharedView() : null;} catch {return null;}
}

function readPage() {""")
source = once(source, "  const route = decodeURIComponent(location.hash.replace('#/', ''));", """  const shared = sharedView();
  if (shared) return shared.page;
  let route;
  try {route = decodeURIComponent(location.hash.replace('#/', '').split('?')[0]);} catch {return 1;}""")
source = once(source, "function readValues() {\n  try {", "function readValues() {\n  const shared = sharedView();\n  if (shared) return shared.values;\n  try {")
source = once(source, "  const [page, setPage] = useState(readPage);", "  const [options, setOptions] = useState(readOptions);\n  const [page, setPage] = useState(readPage);")
source = once(source, "    const update = () => setPage(readPage());", """    const update = () => {
      const shared = sharedView();
      if (shared) {setValues(shared.values);setRequest(null);}
      setPage(readPage());
    };""")
source = once(source, "  useEffect(() => {\n    const current = ++generation.current;", """  useEffect(() => {
    document.documentElement.dataset.enhancements = FEATURES_ENABLED && options.design ? 'on' : 'off';
  }, [options.design]);

  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;""")
source = once(source, "    api.post('/api/views', {page, values, action: request?.key})", "    viewClient.load({page, values, action: request?.key}, {signal: controller.signal, enabled: FEATURES_ENABLED && options.cache})")
source = once(source, "      .catch(err => {if (current === generation.current) setError(err.message);})", "      .catch(err => {if (err.name !== 'AbortError' && current === generation.current) setError(err.message);})")
source = once(source, "  }, [page, values, request]);", "    return () => controller.abort();\n  }, [page, values, request, options.cache]);")
source = once(source, "JSON.stringify(next)", "JSON.stringify(FEATURES_ENABLED ? safeValues(next) : next)")
source = once(source, "values: next}", "values: FEATURES_ENABLED ? safeValues(next) : next}")
source = once(source, "  function navigate(index) {", """  function restore(view) {
    setValues(view.values); setPage(view.page); setRequest(null);
    try {sessionStorage.setItem('f1analysis.view-values', JSON.stringify(view.values));} catch { /* Optional storage. */ }
    location.hash = '/' + encodeURIComponent(routes[view.page-1]);
  }

  function navigate(index) {""")
source = once(source, '      <header className="parity-header">', """      {FEATURES_ENABLED && <details><summary>Analysis tools</summary><FeatureBar page={page} values={values} options={options} setOptions={setOptions} restore={restore} navigate={navigate}/></details>}
      <header className="parity-header">""")
source = once(source, '      <main id="main-content"', '      {FEATURES_ENABLED && <LoadingFeedback busy={busy} hasResults={data?.page === page}/>}\n      <main id="main-content"')
source = once(source, '        {busy && <span className="sr-only"', '        {FEATURES_ENABLED && <ResearchJobs values={values}/>}\n        {busy && !FEATURES_ENABLED && <span className="sr-only"')
source = once(source, '<img src="/betting-oracle-logo.png" alt="Betting Oracle Logo" />', """{FEATURES_ENABLED ? <picture><source type="image/webp" srcSet="/betting-oracle-logo-60.webp 1x, /betting-oracle-logo-120.webp 2x"/><img src="/betting-oracle-logo.png" alt="Betting Oracle Logo" loading="lazy" decoding="async"/></picture> : <img src="/betting-oracle-logo.png" alt="Betting Oracle Logo"/>}""")
write(CODE/"frontend/App.jsx", source)

app_test = (APP/"frontend/src/App.test.jsx").read_text(encoding="utf-8")
app_test = once(app_test, "import {api} from './api';", "import {viewClient} from './enhancements/viewClient';")
app_test = once(app_test, "vi.mock('./api',()=>({api:{post:vi.fn()}}));", "vi.mock('./enhancements/viewClient',()=>({viewClient:{load:vi.fn(),clear:vi.fn()}}));")
app_test = app_test.replace("api.post", "viewClient.load")
app_test = once(app_test, "async(_,payload)", "async(payload)")
app_test = once(app_test, "toHaveBeenLastCalledWith('/api/views',expect.objectContaining({page:2,values:{filter_results_main:true}}))", "toHaveBeenLastCalledWith(expect.objectContaining({page:2,values:{filter_results_main:true}}),expect.objectContaining({enabled:false}))")
write(CODE/"frontend/App.test.jsx", app_test)

main_js = (APP/"frontend/src/main.jsx").read_text(encoding="utf-8")
main_js = once(main_js, 'import "./parity.css";', 'import "./parity.css";\nimport "./enhancements/enhancements.css";')
write(CODE/"frontend/main.jsx", main_js)
vite = (APP/"frontend/vite.config.js").read_text(encoding="utf-8").replace("sourcemap: true", "sourcemap: false")
vite = once(vite, "// Vite fails the build if any individual chunk exceeds this budget.", "// This setting warns; scripts/check-budgets.mjs enforces the gzip budget.")
write(CODE/"deployment/vite.config.js", vite)
package = __import__("json").loads((APP/"frontend/package.json").read_text(encoding="utf-8"))
package["scripts"]["prebuild"] = "node scripts/optimize-assets.mjs"
package["scripts"]["build"] = "vite build && node scripts/check-budgets.mjs"
write(CODE/"deployment/package.json", __import__("json").dumps(package, indent=2)+"\n")

presentation = (APP/"frontend/src/components/Presentation.jsx").read_text(encoding="utf-8")
presentation = once(presentation, "import {ViewTable} from './ViewTable';", """import {EnhancedTable} from '../enhancements/EnhancedTable';
import {SafePlotlyChart} from '../enhancements/SafePlotlyChart';""")
presentation = presentation.replace("<ViewTable ", "<EnhancedTable ")
presentation, count = re.subn(r"\nfunction PlotlyChart\(\{ node \}\) \{.*?\n\}\n", "\n", presentation, count=1, flags=re.S)
if count != 1:
    raise ValueError("The Plotly integration anchor changed")
presentation = once(presentation, "<PlotlyChart key={key}", "<SafePlotlyChart key={key}")
write(CODE/"frontend/Presentation.jsx", presentation)

main_py = (APP/"backend/app/main.py").read_text(encoding="utf-8")
main_py = once(main_py, "import os", "import os\nfrom contextlib import asynccontextmanager\nfrom collections.abc import AsyncIterator")
main_py = once(main_py, "from fastapi import FastAPI, HTTPException, Query", "from fastapi import FastAPI, HTTPException, Query, Request")
main_py = once(main_py, "from fastapi.responses import FileResponse, JSONResponse", "from fastapi.responses import FileResponse, JSONResponse, Response\nfrom starlette.concurrency import run_in_threadpool\nfrom app.enhancements.service import Enhancements")
main_py = once(main_py, "app = FastAPI(", """@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if enhancements is not None:
            await run_in_threadpool(enhancements.jobs.close)


app = FastAPI(
    lifespan=lifespan,""")
main_py = once(main_py, "CODE_DEPLOYED_AT = datetime.now(UTC)", "enhancements = Enhancements() if os.environ.get('F1_ENHANCEMENTS', '0') == '1' else None\nCODE_DEPLOYED_AT = datetime.now(UTC)")
main_py = once(main_py, "def view(payload: ViewRequest) -> JSONResponse:", "def view(payload: ViewRequest, request: Request) -> Response:")
main_py = once(main_py, "        return JSONResponse(render_view(payload.page, payload.values, payload.action))", "        if enhancements is not None:\n            return enhancements.render(payload, request)\n        return JSONResponse(render_view(payload.page, payload.values, payload.action))")
main_py += "\n\nif enhancements is not None:\n    enhancements.install(app)\n"
write(CODE/"backend/main.py", main_py)
subprocess.run([sys.executable, "-m", "ruff", "check", str(CODE/"backend/main.py"), "--config", str(APP/"backend/pyproject.toml"), "--select", "I", "--fix"], check=True)
tests = (HERE/"checks/test_backend.py").read_text(encoding="utf-8")
start = tests.index("ROOT = Path(")
end = tests.index("def test_cache_revision")
tests = tests[:start]+"""from app.enhancements.testing_worker import fake_work
from app.enhancements.cache import ViewResponses, accepts_gzip
from app.enhancements.jobs import BusyQueueError, Jobs
from app.enhancements.metrics import BodyLimit, RequestMetrics, metrics_storage


"""+tests[end:]
tests = tests.replace("import importlib.util\n","").replace("import sys\n","").replace("from pathlib import Path\n","")
tests += "\n\n"+(HERE/"checks/service_tests.py").read_text(encoding="utf-8")
write(CODE/"backend/test_enhancements.py", tests)
subprocess.run([sys.executable, "-m", "ruff", "check", str(CODE/"backend/test_enhancements.py"), "--config", str(APP/"backend/pyproject.toml"), "--select", "I", "--fix"], check=True)

# No production files are edited. Re-running refreshes this named staging copy.
STAGE.mkdir(parents=True, exist_ok=True)
shutil.copytree(APP/"frontend/src", STAGE/"frontend/src", dirs_exist_ok=True)
shutil.copytree(APP/"backend/app", STAGE/"backend/app", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
for name in ("App.jsx", "App.test.jsx", "main.jsx"):
    shutil.copy2(CODE/"frontend"/name, STAGE/"frontend/src"/name)
shutil.copy2(CODE/"frontend/Presentation.jsx", STAGE/"frontend/src/components/Presentation.jsx")
for path in (CODE/"frontend").iterdir():
    if path.name not in {"App.jsx", "App.test.jsx", "main.jsx", "Presentation.jsx"}:
        target = STAGE/"frontend/src/enhancements"/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
shutil.copytree(CODE/"backend", STAGE/"backend/app/enhancements", dirs_exist_ok=True, ignore=shutil.ignore_patterns("main.py","test_enhancements.py"))
shutil.copy2(CODE/"backend/main.py", STAGE/"backend/app/main.py")
shutil.copy2(CODE/"backend/test_enhancements.py", STAGE/"backend/test_enhancements.py")
# main.py is an integration replacement, not an enhancements package module.
if (STAGE/"backend/app/enhancements/main.py").exists():
    (STAGE/"backend/app/enhancements/main.py").unlink()
for name in ("jsconfig.json", "tsconfig.json", "eslint.config.js", "index.html"):
    shutil.copy2(APP/"frontend"/name, STAGE/"frontend"/name)
for name in ("vite.config.js", "package.json"):
    shutil.copy2(CODE/"deployment"/name, STAGE/"frontend"/name)
shutil.copytree(APP/"frontend/public", STAGE/"frontend/public", dirs_exist_ok=True)
for name in ("optimize-assets.mjs", "check-budgets.mjs"):
    target = STAGE/"frontend/scripts"/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CODE/"deployment"/name, target)
for name in ("pyproject.toml", "test_api.py", "test_presentation.py"):
    shutil.copy2(APP/"backend"/name, STAGE/"backend"/name)
dependencies = APP/"frontend/node_modules"
linked = STAGE/"frontend/node_modules"
if not dependencies.is_dir():
    raise ValueError("Install the main frontend dependencies first.")
if linked.exists():
    if linked.resolve() != dependencies.resolve():
        raise ValueError("The preview node_modules must resolve to the main frontend dependencies.")
else:
    command = "New-Item -ItemType Junction -Path '{}' -Target '{}' | Out-Null".format(
        str(linked).replace("'", "''"), str(dependencies).replace("'", "''"),
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)
print(STAGE)
```
