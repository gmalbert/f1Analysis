# F1 Analysis — FastAPI + React

The React application reproduces the existing raceAnalysis.py application's
fields, formatting and workflows. The current local parity results and evidence
are in [PARITY_REPORT.md](PARITY_REPORT.md) and
[PARITY_CHECKLIST.md](PARITY_CHECKLIST.md).

The [enhancement guide](enhancement_proposals/2026-10-01/README.md) contains
17 optional design, feature, backend, and deployment proposals, real preview
screenshots, complete implementation files, and validation/rollback instructions.
All 17 enhancements (D1–D4, F1–F6, B1–B5 and O1–O2) are now implemented in the main application. See
[ENHANCEMENTS.md](ENHANCEMENTS.md) for current behavior and verification.

## Architecture

React/Vite renders a native interface from POST /api/views. FastAPI returns a
declarative tree of headings, values, column configurations, charts, controls
and downloads. The checked-in Python views were exported offline from
raceAnalysis.py and f1bet/streamlit_page.py; production does not import or start
Streamlit. This preserves the original Python calculations and model feature
ordering while allowing React to own rendering and browser state.

Data and model artifacts remain in the repository's data_files/ directory.
The API prefers the same Parquet analysis artifact as the reference, with CSV
fallback. Tables include all rows rather than a 50-row preview. An explicit
column_order selects only the listed fields; otherwise the source's configured
visible fields are exposed. Data Explorer and Data & Debug have different
reference column selections, which are preserved.

Source chart-builder helpers are adapted offline into small runtime-independent
modules with their original license headers. Vega uses the reference theme and
encodings; Matplotlib images use server-safe rendering. The same Glide canvas
grid provides selection/copy, scrolling, sorting, search, column resizing,
visibility, pinning, formatting, CSV download and fullscreen.

## Sections

1. 📊 Data Explorer
2. 📈 Analytics & Visualizations
3. 🏎️ Schedule
4. 🏁 Next Race
5. 🤖 Predictive Models
6. 💾 Data & Debug
7. 📐 Betting Research

Models include all seven original nested panels and six estimator choices.
Betting Research exposes the Value & stake calculator. Field simulation, Paper
replay and Calibration uploads and their compatibility API endpoints are disabled
in the public React application. Offline research functions remain available.
Downloads retain the original CSV contracts. Filters and selected panels persist
across navigation. API request bodies default to a 1 MiB limit, matching Nginx;
`F1_MAX_REQUEST_BYTES` overrides the backend limit only.

## Local development

From the repository root in PowerShell, use the project's `.venv` with the backend
requirements installed, then start the API:

    .\.venv\Scripts\python.exe -m pip install -r fastapi_react/backend/requirements.txt -r fastapi_react/backend/requirements-dev.txt
    .\fastapi_react\start-local.ps1

The local launcher enables research tools and diagnostics without an administrator
token. It binds the API to `127.0.0.1:8000`, disables forwarded-header trust, and
limits browser access to the configured local app addresses. The React research
form automatically omits the token field. Calculations still start only when you
press **Queue calculation**. Stop this API with Ctrl+C.

For another local port, see `F1_LOCAL_ORIGINS` in the
[backend guide](backend/ENHANCEMENTS.md). For hosted use, leave
`F1_TRUSTED_LOCAL=0` (the default) and configure `F1_ADMIN_TOKEN` on the server.
Do not use the local launcher behind a public proxy.

In a second terminal:

    cd fastapi_react/frontend
    npm ci
    npm run dev -- --host 127.0.0.1 --port 5174 --strictPort

Open http://127.0.0.1:5174. API docs are at http://127.0.0.1:8000/api/docs.
Vite proxies /api to the backend. The reference used for local comparison is
http://127.0.0.1:8502.

The frontend .npmrc retains legacy peer resolution for Glide's published React
peer range. React 19 behavior is verified by unit and browser tests. The
postinstall script applies one bounds guard to Glide 6.0.3 in both module builds;
it fails clearly if a future package version needs a different patch.

## Docker

From the repository root:

    cd fastapi_react
    docker compose up --build

Open http://localhost:8080. Compose explicitly keeps trusted local mode off;
research jobs and diagnostics require server-configured administrator credentials.
Compose mounts the repository read-only at /repo
and sets F1_REPO_ROOT=/repo. This avoids copying large data/model artifacts into
the image. Docker deployment was not rerun as part of the latest local parity
verification.

## Resource policy

Page loads use existing model artifacts; they do not train models. The reference's
research-only training controls remain disabled. The bin-count comparison and
temporal leakage audit now open an explicit research queue, which runs calculations
in a separate process. The local launcher needs no token; hosted use retains
administrator authentication. The
shared audit implementation returns findings without rewriting the dataset.

The separate compatibility API's expensive tools remain disabled by default
(ENABLE_EXPENSIVE_TOOLS=0). Its allow-listed commands can be enabled for an
isolated development environment. Numerical-library thread limits in Docker
remain appropriate for a small server.

## APIs and compatibility

POST /api/views powers the active React shell. GET /api/brand/logo serves the
reference mark. Existing /api/data-explorer, /api/analytics, /api/current-season,
/api/next-race, /api/models, /api/raw, /api/betting and /api/tools endpoints remain
available for compatibility. The raw-file API enforces data-directory boundaries;
its former React-only file-browser page is not part of the reference UI.

## Updating the reference export

Run from fastapi_react/backend after reviewing an intentional source change:

    python export_reference_views.py
    python export_chart_helpers.py
    python export_dnf_diagnostics.py

The first tool exports view declarations/calculations. The second adapts the
installed reference version's pure chart builder and retains its licenses.
The third freezes the original DNF diagnostic calculation, recording the dataset
hash so a stale snapshot fails clearly. Streamlit is needed for development
export/comparison tooling, not production requirements.

Review the generated diff and rerun the source oracle, filters, browser workflows,
screenshots and build. Generated exports must stay aligned with the reference
source and installed reference version; changes are not silently auto-exported
during page requests.

## Verification

Backend, from fastapi_react/backend:

    python -m compileall -q app
    python -m ruff check .
    python -m mypy --explicit-package-bases app
    python -m pytest

Frontend, from fastapi_react/frontend:

    npm run lint
    npm run typecheck
    npm test
    npm run build
    npm audit --omit=dev

See [parity_evidence/README.md](parity_evidence/README.md) for actual Streamlit
comparisons, CSV exports and Playwright capture/interaction commands.
The report documents screenshot tolerances and the retained reference contrast
findings. Local parity evidence does not certify another branch's CI or a
production deployment.
