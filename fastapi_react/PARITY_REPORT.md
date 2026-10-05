# FastAPI + React Parity Report

Verified locally on 2026-10-01 against Streamlit http://127.0.0.1:8502,
React http://127.0.0.1:5174 and FastAPI http://127.0.0.1:8000.
The application parity gates below pass in the current working tree.
These results describe local validation recorded with the parity completion
commit. They do not describe GitHub PR checks or a deployed release.

## Implementation

React now renders the original seven sections and nested panels. FastAPI executes
offline-exported Python view calculations through a request-isolated presentation
protocol (POST /api/views). React renders the fields, charts, controls and
downloads natively. Production requests do not import or run Streamlit.

Model and betting calculations remain in Python. All six model selections load
the appropriate artifacts, including custom ensembles. Shared caches return
private copies; model metadata checks and feature order are preserved.
Matplotlib uses a server-safe backend and a rendering lock. Training never runs
implicitly during page loads. The explicit bin-count experiment and administrative
audit run only when their buttons are pressed. Research training controls remain
disabled, matching the reference setting.

The joined Parquet snapshot has 4,629 rows and 544 source columns. Data & Debug
displays 530 columns under the reference's hide rules. Data Explorer's explicit
column order displays 34 columns, including its repeated positionsGained column.
Explicit column orders select only those fields; extra fields are not appended.
Other tables expose the same configured visible fields as Streamlit.

Typography, logo encoding, spacing, sidebar filters, responsive tabs, metric
precision, dates, localized times and conditional cell styles follow the reference.
Tables use the same Glide grid library, with search, selection/copy, sorting,
resizing, visibility, pinning, number-format controls, CSV export and fullscreen.
Charts retain the native builder's series and encodings, with data views, PNG
export, spec copying and fullscreen.

Two compatibility repairs were necessary:

- Glide 6.0.3 could access a missing header during layout changes. A narrow guard
  is applied reproducibly by the frontend postinstall script.
- The shared audit script lacked the run_audit callable used by the original UI.
  It now returns the structured heuristic report for both applications.

## Verified results

| Check | Result |
|---|---|
| Python py_compile | All 22 backend Python files and shared audit script pass |
| Ruff | Pass |
| Strict mypy | Pass, 12 source files |
| Backend tests | 59 passed; 87.36% coverage |
| ESLint and TypeScript | Pass |
| Frontend tests | 57 passed; 73.74% line coverage; existing thresholds unchanged |
| Vite production build | Pass |
| Production npm audit | Zero vulnerabilities |
| Actual Streamlit table/model comparisons | 77 checks, zero failures |
| Boolean/category/year/date filter comparisons | Four checks, zero failures |
| Data Explorer CSV | 4,629 rows × 34 columns; contents match |
| Next Race CSV | Date/time fields and contents match |
| Playwright application workflows | 11 workflows; zero page, console or HTTP errors |
| Explicit experiment/audit workflows | Both pass; zero browser errors |
| Playwright viewport captures | 24 states; zero browser errors |
| Visual comparisons | 24/24 within unchanged tolerances |

The initial JavaScript chunk is approximately 219 KB gzip. Vega and Plotly load
separately. Vite prints advisory chunk-size warnings; the build succeeds.
The large Plotly chunk loads only for views requesting Plotly.
Generated views and vendored chart code retain provenance and are excluded from
handwritten-code lint/type/coverage measurements. They are compiled and exercised
by integration tests and the actual Streamlit comparison oracle.

## Evidence and limits

- parity_evidence/reference-table-parity.json: actual reference table values,
  labels, displayed ordering and index visibility, plus six-model comparisons.
- parity_evidence/reference-filter-parity.json: filtered values, labels and
  number formats using real Streamlit widget selections.
- parity_evidence/download-parity.json and download-next-race-parity.json:
  parsed CSV contents from both live applications. The reference's download
  fallback avoids an operating-system save dialog.
- parity_evidence/interaction-results.json and experiment-results.json:
  browser workflows and errors.
- parity_evidence/visual/parity-2026-10-01/: paired captures, diff images,
  diff/summary.json and React browser-errors.json.

Visual tolerances remain 2% desktop and 3% tablet/mobile. They are not pixel
identity. Residual differences include changing timestamps, framework chrome
and browser control rendering. Captures use actual Source fonts and normal
animations. The reference footer's remote image is served from the identical
bundled asset during capture. No screenshot regions or content are masked.
Raw Data is enabled on desktop/tablet and unchecked on mobile in both apps;
the mobile capture does not establish full-table visual parity.

The refreshed axe audit reports color-contrast findings on all seven sections
because reference accent/caption styling is preserved. It does not establish
WCAG AA compliance. No other axe violation categories were reported.
This is separate from the zero-error Playwright runtime result.

No production deployment, comparative load benchmark, commit, push or PR update
was performed. Previous PR #128 and benchmark results in the historical handoff
apply to earlier code and are not current evidence.
