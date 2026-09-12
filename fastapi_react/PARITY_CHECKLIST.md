# FastAPI + React Parity Checklist

The purpose of this file is to prevent the migration from being declared complete merely because the main prediction page works.

The current `raceAnalysis.py` Streamlit application is the reference implementation.

## Acceptance rule

For each item:

1. Run Streamlit and React against the same checkout and same generated artifacts.
2. Apply the same input/filter selection.
3. Compare values, ordering, empty states, and downloads.
4. Mark the item verified only when outputs match or an intentional UI-only difference is documented.

---

## 1. Application shell

- [x] Independent `fastapi_react/` folder
- [x] Existing Streamlit code untouched
- [x] React navigation
- [x] FastAPI API
- [x] Docker Compose test deployment
- [x] API health/RSS reporting
- [x] Single backend worker by default
- [x] Numerical thread limits for small hosts
- [ ] Visual comparison against deployed Streamlit styling

## 2. Data Explorer

- [x] Read `data_files/f1ForAnalysis.csv` using tab separator
- [x] Searchable field/filter schema
- [x] Numeric range filters
- [x] Date range filters
- [x] Boolean / 0-1 filters
- [x] Exact categorical filters
- [x] Null-preserving filter semantics
- [x] Row count
- [x] Sorting
- [x] Bounded table response
- [x] Primary race/driver/constructor/result fields
- [ ] Verify every Streamlit exclusion/friendly-label rule against the current app
- [ ] Compare filtered outputs for a representative sample of fields

## 3. Analytics & Visualizations

- [x] Active years vs final position
- [x] Positions gained over time
- [x] Last practice vs final position
- [x] Starting grid vs final position
- [x] Average practice position vs final position
- [x] Average pit-stop time vs final position
- [x] Practice/final-position linear regression
- [x] Grid/final-position linear regression
- [x] Correlation matrix
- [x] Driver performance over time
- [x] Constructor performance over time
- [x] DNF reasons
- [ ] Verify every additional tire/pit-stop visualization currently rendered by Streamlit
- [ ] Match friendly chart axis labels
- [ ] Compare numerical regression output

## 4. Current Season

- [x] Current/latest season detection
- [x] Schedule table
- [x] Race count
- [x] Circuit/race metadata returned when present
- [ ] Match Streamlit next-race row highlighting exactly
- [ ] Confirm F1DB schedule enrichment produces identical columns

## 5. Next Race

- [x] Next-race detection
- [x] Race details
- [x] Historical results at the same Grand Prix
- [x] Driver historical performance
- [x] Constructor historical performance
- [x] Race-control / safety-car table when available
- [x] Weather table when available
- [x] Select and display committed prediction artifact
- [ ] Verify prediction-artifact selection against every Streamlit filename/fallback rule
- [ ] Port exact fastest-pit-stop/stationary-time presentation
- [ ] Port all tire-strategy blocks if present in current Streamlit build
- [ ] Compare all active-driver prediction rows and model outputs with Streamlit

## 6. Predictive Models

Model types:

- [x] XGBoost
- [x] LightGBM
- [x] CatBoost
- [x] Ensemble
- [x] Position Group
- [x] Track-Weighted Ensemble

Advanced areas:

- [x] Performance area
- [x] Feature Importance area
- [x] Feature Selection area
- [x] Position Analysis area
- [x] Hyperparameters area
- [x] Historical Validation area
- [x] Debug/runtime area

Precomputed artifacts:

- [x] Monte Carlo results
- [x] Monte Carlo run log
- [x] SHAP results
- [x] RFE results
- [x] Boruta results
- [x] Permutation importance
- [x] Bayesian HPO results
- [x] Grid HPO results
- [x] Historical validation
- [x] Position MAE detail

Manual research tools:

- [x] FastAPI execution gate exists
- [x] Disabled by default
- [x] Environment-variable enable switch
- [ ] Verify exact current script filenames for every manual tool
- [ ] Compare model metrics and feature-importance ordering for every model type
- [ ] Port any model-specific diagnostic tables not represented by a committed artifact

## 7. Raw Data

- [x] Recursive `data_files/` browser
- [x] Search by filename/path
- [x] Tab-separated CSV preview
- [x] Conventional CSV fallback
- [x] JSON preview
- [x] Text/Markdown/log preview
- [x] Binary file metadata
- [x] Original-file download
- [x] Path traversal protection
- [ ] Compare exact set/order of raw-data tables exposed by Streamlit

## 8. Betting Research

### Value & stake

- [x] Model probability
- [x] Selection decimal odds
- [x] Opposing decimal odds
- [x] Probability uncertainty
- [x] Multiplicative de-vig
- [x] Additive de-vig
- [x] Power de-vig
- [x] De-vigged market probability
- [x] Raw expected value
- [x] Conservative probability
- [x] Paper stake
- [x] Decision reason code

### Field simulation

- [x] CSV field input
- [x] Default simulation template in UI
- [x] Existing `RaceEntry` model
- [x] Existing correlated `simulate_race` engine
- [x] Simulation count
- [x] Probability output table
- [ ] Add one-click CSV output download

### Paper replay

- [x] CSV ledger input
- [x] Existing `run_backtest`
- [x] Summary
- [x] Placed paper-bet ledger
- [x] All decisions/abstentions
- [x] Risk sensitivity

### Calibration

- [x] CSV input
- [x] Probability/outcome validation
- [x] Probability metrics
- [x] Market/stage grouping
- [x] Adaptive reliability table
- [ ] Add reliability line visualization

## 9. Operational parity

- [x] Existing generator remains authoritative
- [x] Existing data files remain authoritative
- [x] Existing `f1bet` implementation reused
- [x] Expensive tasks kept out of normal page requests
- [x] Streamlit and React implementations can coexist
- [x] Docker test environment does not alter source data (`/repo` is read-only)
- [ ] Benchmark memory against Streamlit
- [ ] Benchmark first-page latency
- [ ] Benchmark repeated navigation
- [ ] Test two simultaneous users
- [ ] Test five simultaneous users

## 10. Final cutover gate

Do not remove or replace the Streamlit deployment until:

- [ ] All functional items above are verified
- [ ] Prediction output matches for all six model types
- [ ] Current-season schedule matches
- [ ] Next-race selection matches
- [ ] Major analytical figures match
- [ ] Betting smoke test gives the same expected calculator output
- [ ] No endpoint performs unintended request-time training
- [ ] Memory usage is measured under representative load
- [ ] Production deployment has rollback instructions
- [ ] Cross-cutting quality gates in sections 11–14 pass, or each failure is documented with rationale in `fastapi_react/PARITY_REPORT.md`

## 11. Accessibility

The React app must meet a basic WCAG 2.1 AA bar. Streamlit's accessibility is itself imperfect; this section defines what the React app must do regardless of what Streamlit provides.

### Keyboard navigation

- [ ] All interactive elements reachable via Tab in DOM order
- [ ] Visible focus indicator on every focusable element (contrast ≥3:1)
- [ ] Logical reading order matches visual order
- [ ] No keyboard traps
- [ ] Modal dialogs trap focus and restore it on close
- [ ] Skip-to-main-content link on every page

### Semantic structure

- [ ] One `<h1>` per page
- [ ] Heading levels do not skip
- [ ] Navigation, main, and footer use landmark elements
- [ ] Document `<title>` updates per route
- [ ] Data tables use `<table>` with `<thead>`, `<tbody>`, and `<th scope>`

### Labels and ARIA

- [ ] Every form control has an associated `<label>` or `aria-label`
- [ ] Icon-only buttons have `aria-label` describing their action
- [ ] Charts have a text alternative (data table or `aria-label` summary)
- [ ] Loading regions marked with `aria-busy="true"`
- [ ] Error messages announced via `aria-live` (polite by default; assertive for blocking errors)

### Color and contrast

- [ ] Body text contrast ≥4.5:1 against background
- [ ] Large text contrast ≥3:1
- [ ] Non-text UI elements (icons, chart axes, focus rings) contrast ≥3:1
- [ ] Information not conveyed by color alone
- [ ] Both light and dark themes pass the above checks

### Evidence

An item is satisfied only when both:

- An automated a11y check (axe-core, pa11y, or equivalent) reports no violations on the relevant page, **and**
- A manual keyboard pass-through is recorded in `PARITY_REPORT.md` for at least Home, Data Explorer, Models, and Betting Research.

## 12. Per-page error, empty, and loading states

Every page must explicitly handle three states. Silent fallbacks (blank canvas, stuck spinner, swallowed errors) are parity failures.

| Page | Loading state | Empty state | Error state |
|------|---------------|-------------|-------------|
| Data Explorer | Skeleton rows + schema-fetch indicator | "No rows match the current filters" + reset button | Inline alert with retry; filter selection preserved |
| Analytics | Chart skeletons per panel | "No data for the selected years / drivers" | Inline alert per panel; other panels still render |
| Current Season | Skeleton schedule table | "No race data for the current year" | Inline alert with retry |
| Next Race | Skeleton race header + sub-tables | "No upcoming race detected" + link to current season | Inline alert with retry; historical tables still render if available |
| Models | Skeleton metrics tiles | "No trained model for the selected type" + link to docs | Inline alert with retry; precomputed artifacts still listed |
| Raw Data | Skeleton file tree | "No files in data_files/" | Inline alert with retry |
| Betting Research | Spinner during calculation | "Provide a value to compute" placeholder | Inline alert with friendly message in production, traceback in dev |

### Cross-cutting requirements

- [ ] Loading skeletons never block the entire page; long operations show progress
- [ ] Every error message is user-actionable (retry, change input, or open docs)
- [ ] No uncaught exceptions in the browser console during normal navigation
- [ ] 4xx responses are distinguished from 5xx in the UI text

## 13. Visual diff via paired screenshots

Compare the Streamlit app to the React app page-by-page using the same dataset and the same filter selections.

### Capture setup

- [ ] Commit a pinned `data_files/` snapshot (or document the exact commit hash) used for both runs
- [ ] Capture at two viewports: 1280×800 (desktop) and 768×1024 (tablet)
- [ ] Disable animations, defer non-essential fonts, and use a fixed system font for both runs
- [ ] Capture Streamlit pages first, then React pages, against the same checkout

### Per-page captures

- [ ] Home / shell
- [ ] Data Explorer — unfiltered, with one numeric filter, with one date filter, with one categorical filter
- [ ] Analytics — each chart panel listed in section 3 of this checklist
- [ ] Current Season — full schedule; single race selected
- [ ] Next Race — header, predictions table, historical results
- [ ] Models — each model-type dropdown selection
- [ ] Raw Data — file tree, CSV preview, JSON preview
- [ ] Betting Research — value & stake, simulation, replay, calibration

### Diff and acceptance

- [ ] Generate a pixel-diff per page (Playwright `toHaveScreenshot`, ImageMagick `compare`, or equivalent)
- [ ] Tolerance: ≤2% differing pixels at the desktop viewport, ≤3% at the tablet viewport
- [ ] Differences above tolerance are either fixed or explicitly recorded as intentional UI-only differences in `PARITY_REPORT.md`
- [ ] Screenshots and diffs are stored under `fastapi_react/parity_evidence/visual/` and referenced from the checklist

## 14. Code quality

The migrated code is held to a higher bar than the existing Streamlit code, because it is new and fully reviewable.

### Backend (Python under `fastapi_react/backend/`)

- [ ] Lint passes with no errors (e.g., `ruff check` with project config)
- [ ] Type check passes (e.g., `mypy --strict` or `pyright`); any relaxed settings are documented in `PARITY_REPORT.md`
- [ ] `pytest` runs and reports ≥80% line coverage for the backend (e.g., `pytest-cov`)
- [ ] No `print()` calls in non-test code
- [ ] No bare `except:` clauses
- [ ] Public functions and route handlers have docstrings
- [ ] `pip-audit` (or equivalent) reports no high/critical vulnerabilities, or each is documented with rationale

### Frontend (JavaScript/React under `fastapi_react/frontend/`)

- [ ] `eslint` passes with React + Hooks + JSX-a11y rule sets
- [ ] Type check passes. Either the codebase is migrated to TypeScript with `tsc --noEmit` clean, **or** JSX uses `// @ts-check` with a `jsconfig.json` that resolves to a typed stub; the chosen path is recorded in `PARITY_REPORT.md`
- [ ] Component and page tests run (e.g., `vitest` + `@testing-library/react`) and report ≥80% line coverage
- [ ] No `console.log` in production builds (Vite strips them or an ESLint rule forbids them)
- [ ] Production bundle: main chunk < 500 KB gzipped; any chunk above the budget is documented in `PARITY_REPORT.md`
- [ ] `npm audit` (or equivalent) reports no high/critical vulnerabilities, or each is documented with rationale

### Cross-cutting

- [ ] A CI workflow runs lint + type check + tests on every PR that touches `fastapi_react/`
- [ ] Pre-commit hook (or equivalent) runs at least the fast checks locally
- [ ] `requirements.txt` and `package.json` are pinned (or backed by a lockfile) so the test environment is reproducible
- [ ] No `TODO`/`FIXME` without a linked issue or follow-up note in `PARITY_REPORT.md`
