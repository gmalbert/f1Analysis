# F1 Analysis — FastAPI + React Migration

This folder is an **independent FastAPI + React implementation** of the existing `raceAnalysis.py` Streamlit site.

It is intentionally isolated under `fastapi_react/`. The existing Streamlit application, generator, model artifacts, data files, workflows, and `f1bet` package remain untouched and continue to be the reference implementation while parity is tested.

## Architecture

```text
Browser
  |
  v
React + Vite
  |
  v
FastAPI
  |
  +-- existing data_files/
  +-- existing data_files/precomputed/
  +-- existing f1bet/ package
  +-- existing workflow-generated artifacts
```

The migration does **not** rewrite modeling logic in JavaScript.

The backend reads the existing repository's data and imports the existing `f1bet` pure-Python package. Expensive model training and precomputation remain external to normal web requests.

## User-facing areas

The React application maps the current site into these primary sections:

1. Data Explorer
2. Analytics & Visualizations
3. Current Season
4. Next Race
5. Predictive Models & Advanced Options
6. Raw Data
7. Probability & Betting Research

Betting Research includes:

- Value & stake
- Field simulation
- Paper replay
- Calibration
- Release gates

Predictive Models includes:

- Performance
- Feature Importance
- Feature Selection
- Position Analysis
- Hyperparameters
- Historical Validation
- Debug / manual tools

## Quickest test: Docker

From the repository root:

```bash
cd fastapi_react
docker compose up --build
```

Open:

```text
http://localhost:8080
```

FastAPI documentation is available at:

```text
http://localhost:8080/api/docs
```

For direct backend development, use:

```text
http://localhost:8000/docs
```

if you run Uvicorn separately.

Stop the stack:

```bash
docker compose down
```

## Run without Docker

### Backend

From `fastapi_react/`:

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend

In another terminal:

```bash
cd fastapi_react/frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The Vite development server proxies `/api` to `http://127.0.0.1:8000`.

## Repository-root detection

When run directly from this repository, the backend automatically locates the repository root.

Docker explicitly sets:

```text
F1_REPO_ROOT=/repo
```

and mounts the repository read-only at `/repo`.

You can override the root manually:

```bash
F1_REPO_ROOT=/path/to/f1Analysis
```

## Resource policy

Normal production operation is artifact-first.

The backend does not train models during page loads.

Heavy/manual tools are disabled by default:

```text
ENABLE_EXPENSIVE_TOOLS=0
```

For an isolated development/test machine only, they can be enabled:

```text
ENABLE_EXPENSIVE_TOOLS=1
```

The Docker configuration also limits numerical-library parallelism:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

This is deliberate for inexpensive VPS hosting.

## Backend API

Major endpoints include:

```text
GET  /api/health
GET  /api/meta

GET  /api/data-explorer/schema
POST /api/data-explorer/query

POST /api/analytics
GET  /api/current-season
GET  /api/next-race

GET  /api/models
GET  /api/models/precomputed/{name}

GET  /api/raw/files
GET  /api/raw/preview
GET  /api/raw/download

POST /api/betting/value
POST /api/betting/simulate
POST /api/betting/backtest
POST /api/betting/calibration
GET  /api/betting/governance

POST /api/tools/run
```

## Data Explorer

Unlike the Streamlit implementation, React does not need to create 2,200+ sidebar widgets at page execution time.

The backend returns a schema describing each field as:

```text
range
date_range
boolean
exact
```

The frontend provides searchable dynamic filters.

This preserves access to the wide-table filtering capability without forcing every possible filter control to render at once.

## Raw Data

The Raw Data page can inspect existing files under `data_files/`.

Path traversal outside `data_files/` is rejected by the backend.

Large files are previewed with bounded row counts. The original file remains downloadable through the API.

## Betting Research

The FastAPI routes call the existing `f1bet` package directly for:

- de-vigging
- expected value
- stake proposals
- correlated field simulation
- backtesting
- risk sensitivity
- calibration
- feature availability
- contract validation
- release evidence

There is no independent JavaScript implementation of these calculations.

## Testing parity

Use `PARITY_CHECKLIST.md` as the acceptance checklist.

The existing Streamlit application remains the authoritative output until each item has been compared with the React version using the same repository data/artifacts.

## Important deployment note

The included Docker Compose configuration intentionally mounts the parent repository read-only.

That makes this folder easy to test without copying the large F1 datasets or model artifacts into another directory.

For a final production image, the next optimization should be to build/copy only the subset of artifacts needed by the live site rather than mounting the full repository.
