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
