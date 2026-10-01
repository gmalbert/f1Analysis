## Local completion update — 2026-10-01

This update supersedes the earlier continuation status below. Application parity
is implemented in the local working tree through the native React presentation
protocol. The original Python calculations and all seven sections/nested panels
are preserved, including six model choices, original column selection/formatting,
charts, uploads, exports, responsive controls and explicit experiments.

Verification: 77 actual Streamlit table/model comparisons, four filtered-data
comparisons and two live CSV export comparisons pass. All 24 visual pairs pass
the unchanged tolerances. Playwright application/experiment/capture reports
contain zero browser errors. Python compilation passes, backend tests are
59/59 (87.36% coverage), and frontend tests are 57/57. Lint, type checks,
production build and production npm audit pass.

The missing shared run_audit callable was repaired for both apps. A narrow Glide
header bounds fix is reproduced on dependency installation. Runtime views use
checked-in offline exports and do not import or start Streamlit.

Current implementation and reproducible evidence commands are documented in
fastapi_react/README.md, PARITY_REPORT.md, PARITY_CHECKLIST.md and
parity_evidence/README.md. The report states evidence limits, including retained
reference color-contrast findings and the unchecked mobile Raw Data capture.

The parity completion is recorded in a local commit at the user's request.
No push, PR modification or production deployment was performed. The
historical branch/PR/CI and benchmark details below are not current verification
of this working tree. They should be refreshed only when release work is requested.

---

# F1 Analysis — FastAPI + React Parity Migration
## Complete Engineering Handoff
## Continuation Status — 2026-10-01

### Latest Raw-Data/Display Update

The local Data & Debug implementation now follows the Streamlit raw-data source
path: Parquet-first `f1ForAnalysis`, then pit-stop, constructor-standing,
driver-standing, weather, and qualifying joins. On the current snapshot that
produces 4,629 rows and 544 merged columns. The Streamlit
`columns_to_display` configuration hides fields and supplies labels; React now
renders the same 530 visible columns with headers such as `Year`, `Grand Prix`,
`Driver`, and `Number of Stops`. IDs configured hidden by Streamlit (including
`driverId` and `raceId_results`) are not displayed. The raw-data file browser is
separate from the dataset view.

The Models page no longer renders raw JSON in Performance, artifacts, or Debug;
metrics remain visible and artifacts render summaries/tables. Local backend
gates pass (47 tests, 86.33% coverage), frontend gates pass (49 tests, 83.18%
coverage), and axe reports zero violations. The latest local React/FastAPI
benchmark measured 845 ms first-page latency and 257.5 MB RSS after five
concurrent users.

The latest 24-pair local screenshot set is
`fastapi_react/parity_evidence/visual/local-validation-2026-10-01-final/`.
Raw Data differs by 11.19% desktop, 21.47% tablet, and 19.26% mobile. Overall
0/24 pairs pass; visual parity remains a release blocker. The mobile Raw Data
capture uses the default unchecked state on both apps because Streamlit's
checkbox does not reliably toggle in the mobile viewport.

This update supersedes the frozen run-in-progress snapshot below. PR #128 is
still open and draft on `chatgpt/fastapi-react-parity` at
`124c1a7b2d0bd5d9ff8b43ef84663f83dc66f92e`. Run #115 (`36796824643`) has
completed and failed its final visual evidence gate: backend and frontend
jobs passed, but only 6 of 24 screenshot pairs were within tolerance, and axe
reported color-contrast violations on six sections. The run's evidence
artifact was downloaded and inspected.

The checked-out workspace is a different branch,
`react/updates-to-design`, at `31d790c8bb7dd8a84392adcc49d6bfa14fafedbc`,
with a dirty worktree. Do not assume local changes or results are on PR #128.
The local apps are running at React `http://127.0.0.1:5174`, FastAPI
`http://127.0.0.1:8000`, and Streamlit `http://127.0.0.1:8502`.

Local backend and frontend gates pass, local axe is at zero violations. The
CSV fallback contains 4,651 rows; the shared API defaults to the 4,629-row
Parquet artifact used by Streamlit. The latest local screenshot run has 0 of
24 pairs within tolerance (desktop 3.57-15.12%, tablet 5.27-22.21%, mobile
9.40-19.89%). Neither local nor PR evidence meets the visual release gate.
Keep the PR draft; do not merge.

Current evidence and remaining acceptance work are recorded in
`fastapi_react/PARITY_CHECKLIST.md` and `fastapi_react/PARITY_REPORT.md`.

**Repository:** `gmalbert/f1Analysis`<br>
**Reference implementation:** root `raceAnalysis.py` Streamlit application<br>
**Migration target:** `fastapi_react/`<br>
**Working branch:** `chatgpt/fastapi-react-parity`<br>
**Pull request:** PR #128 — `Bring FastAPI/React migration to Streamlit feature and design parity`<br>
**Base branch:** `main`<br>
**Current handoff head:** `124c1a7b2d0bd5d9ff8b43ef84663f83dc66f92e`<br>
**PR state at handoff:** open, draft, not merged<br>
**Do not merge automatically.** Finish parity evidence, update documentation, then mark ready for review. Merge only if explicitly requested.

---

# 1. Executive Summary

The `fastapi_react` migration is substantially implemented and is now in the final parity-verification phase.

The core application architecture is complete:

- React frontend
- FastAPI backend
- seven major application sections matching the Streamlit tab structure
- shared filtering
- prediction / analysis / model artifacts
- betting-research workflows
- accessibility audit
- backend and frontend code-quality gates
- visual screenshot capture for React and local Streamlit
- screenshot comparison
- operational benchmark
- CI evidence artifact upload

The remaining work is not a broad rewrite. Run #115 and its artifact have been
inspected, and the checklist/report have been updated. The current work is
still final parity verification: reconcile local branch changes with the PR,
compare remaining functionality and layout differences, then rerun CI on the
actual PR head. Do not mark the PR ready until its gates pass.

The original CI snapshot below is historical, not a live status:

- **Dependency and Code Security:** success
- **Streamlit API Compatibility:** success
- **F1Bet Offline Release Gates:** success
- **FastAPI + React parity checks:** in progress
  - Frontend job had already completed successfully.
  - Backend job was still running.
  - Visual evidence job had not yet started because it depends on backend + frontend.

Current parity workflow run:

- Workflow: `FastAPI + React parity checks`
- Run number: **#115**
- Run ID: **36796824643**
- Head: `124c1a7b2d0bd5d9ff8b43ef84663f83dc66f92e`

Run #115 is complete and failed the final visual evidence gate; its artifact
results are summarized in the Continuation Status above and in
`fastapi_react/PARITY_REPORT.md`.

---

# 2. User's Original Requirement

The user asked for the `fastapi_react` implementation to achieve **full feature and design parity** with the existing Streamlit application in `raceAnalysis.py`.

The standard throughout this effort has therefore been:

> `raceAnalysis.py` is the authoritative behavioral and visual reference.

Do not rely on older migration notes or stale parity documentation when they conflict with current Streamlit source behavior.

The user's explicit direction was to continue until the migration was finished. The objective is not merely "a working React app." It is parity with the current Streamlit application, including:

- section hierarchy
- navigation
- labels
- filters
- table columns
- ordering
- prediction output
- analytics
- model tools / artifacts
- betting research
- visual layout
- responsive behavior
- accessibility
- code quality
- evidence and acceptance gates

---

# 3. Current Git / PR State

## Branch

```text
chatgpt/fastapi-react-parity
```

## Pull request

```text
#128
Bring FastAPI/React migration to Streamlit feature and design parity
```

PR URL:

```text
https://github.com/gmalbert/f1Analysis/pull/128
```

## Handoff head

```text
124c1a7b2d0bd5d9ff8b43ef84663f83dc66f92e
```

## PR state

At snapshot:

```text
state: open
draft: true
base: main
head: chatgpt/fastapi-react-parity
```

Do not merge the PR merely because functional tests are green. The visual acceptance run is part of the definition of done.

# 3A. Repository Scope of the Migration

All substantive FastAPI/React application development for this migration is contained under:

```text
fastapi_react/
```

That includes:

```text
fastapi_react/backend/
fastapi_react/frontend/
fastapi_react/parity_evidence/
fastapi_react/PARITY_CHECKLIST.md
fastapi_react/PARITY_REPORT.md
fastapi_react/README.md
fastapi_react/docker-compose.yml
```

There is one intentional migration-related file outside that folder:

```text
.github/workflows/fastapi-react.yml
```

That workflow runs the migration's CI and parity gates, including:

- backend lint/type/tests/coverage
- frontend lint/type/tests/coverage/build
- dependency/security checks
- accessibility audit
- operational benchmark
- React screenshot capture
- local Streamlit screenshot capture
- visual diff enforcement
- parity evidence artifact upload

The repository should therefore be understood as:

```text
f1Analysis/
├── .github/
│   └── workflows/
│       └── fastapi-react.yml       ← CI/parity workflow for the migration
│
├── fastapi_react/                  ← all FastAPI/React application development
│   ├── backend/
│   ├── frontend/
│   ├── parity_evidence/
│   ├── PARITY_CHECKLIST.md
│   ├── PARITY_REPORT.md
│   ├── README.md
│   ├── docker-compose.yml
│   └── ...
│
├── raceAnalysis.py                 ← authoritative Streamlit reference
├── data_files/                     ← existing shared data/model artifacts
└── ...
```

Important scope rules:

- `raceAnalysis.py` is the reference implementation and should not be modified merely to make the React version easier to match.
- `data_files/` remains the authoritative shared data/model artifact source.
- The migration does not introduce a separate duplicate data-generation or model-training pipeline elsewhere in the repository.
- The FastAPI/React implementation should consume the same existing artifacts the Streamlit app uses.
- For normal continuation work, changes should stay inside `fastapi_react/` unless the change is specifically to the migration CI workflow in `.github/workflows/fastapi-react.yml`.
- Do not scatter new migration files across unrelated root-level directories unless there is a concrete repository-level reason.

This scope boundary is deliberate and should be preserved during handoff continuation.

---

# 4. CI Snapshot at Handoff

At the exact handoff snapshot for head `124c1a7b...`:

| Workflow | Run | Status |
|---|---:|---|
| Dependency and Code Security | #147 | success |
| Streamlit API Compatibility | #136 | success |
| F1Bet Offline Release Gates | #129 | success |
| FastAPI + React parity checks | #115 | in progress |

For parity run #115:

### Frontend job

**Job:** `Frontend (eslint, tsc, vitest, build, audit)`<br>
**Status:** completed / success

Already green:

- npm install
- ESLint
- TypeScript check
- Vitest + coverage
- Vite build / bundle budget
- production dependency audit

### Backend job

**Job:** `Backend (ruff, mypy, pytest)`<br>
At snapshot it was still installing dependencies. Expected gates are:

- Ruff
- strict mypy
- pytest
- backend line coverage >=80%
- pip-audit

### Visual job

Runs only after frontend and backend pass.

It performs:

1. checkout
2. install backend
3. install Streamlit reference runtime
4. install frontend + Playwright Chromium
5. launch FastAPI
6. launch Vite
7. launch local `raceAnalysis.py` Streamlit
8. capture React screenshots
9. run axe accessibility audit
10. run operational benchmark
11. capture local Streamlit screenshots
12. diff screenshots
13. upload parity evidence even if an evidence gate failed
14. enforce final evidence outcomes

---

# 5. Most Important Rule for the Next Developer

## Do not start with another broad code audit.

Run #115 has already been inspected. It completed with a failed visual
evidence gate; do not wait for it or treat the snapshot below as live state.
Its artifact summary reported 6/24 pairs passing and six sections with axe
color-contrast violations. The detailed result is summarized in
`fastapi_react/PARITY_REPORT.md`.

The previous full evidence run already exposed the major visual defects. Those defects were subsequently fixed in the current branch. The whole point of the current run is to tell us which differences remain after those fixes.

The correct continuation sequence is:

```text
1. Keep the local `react/updates-to-design` worktree intact; it is dirty and is not the PR branch.
2. If visual-parity job succeeds:
   proceed to final verification/documentation.
3. If visual-parity job fails:
   inspect its uploaded evidence artifact.
4. Read visual/diff/summary.json.
5. Open the failing React / Streamlit / diff PNG triplets.
6. Fix real layout/content/style differences.
7. Do not weaken the 2% / 3% page tolerances.
8. Repeat until green.
```

Do not make speculative CSS changes without looking at the newest evidence.

---

# 6. Authoritative Streamlit Shell

The reference page is configured in `raceAnalysis.py`.

Relevant behavior:

```python
st.set_page_config(
    page_title="Gridlocked - Formula 1 Betting & Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

Logo:

```text
data_files/gridlocked-logo-with-text.png
```

Reference logo display width:

```text
450 px
```

Reference title:

```text
F1 Races from {raceNoEarlierThan} to {current_year}
```

Metadata captions:

```text
Last updated: ...
Code deployed at: ...
```

Browser title must remain exactly:

```text
Gridlocked - Formula 1 Betting & Analytics
```

Do not append route/page names to the browser title unless the Streamlit reference changes.

---

# 7. Exact Main Navigation Contract

The Streamlit top-level tabs, in order:

1. `📊 Data Explorer`
2. `📈 Analytics & Visualizations`
3. `🏎️ Schedule`
4. `🏁 Next Race`
5. `🤖 Predictive Models`
6. `💾 Data & Debug`
7. `📐 Betting Research`

React maps them internally as:

| Internal key | Visible label |
|---|---|
| Data Explorer | 📊 Data Explorer |
| Analytics | 📈 Analytics & Visualizations |
| Current Season | 🏎️ Schedule |
| Next Race | 🏁 Next Race |
| Predictive Models | 🤖 Predictive Models |
| Raw Data | 💾 Data & Debug |
| Betting Research | 📐 Betting Research |

Hash navigation is used, for example:

```text
#/Data%20Explorer
#/Analytics
#/Current%20Season
#/Next%20Race
#/Predictive%20Models
#/Raw%20Data
#/Betting%20Research
```

---

# 8. App Shell Implementation

Primary file:

```text
fastapi_react/frontend/src/App.jsx
```

Implemented:

- exact seven tabs
- Streamlit-like shell
- logo from FastAPI brand endpoint
- metadata from `/api/meta`
- persistent global sidebar after Data Explorer filters are enabled
- semantic navigation
- `role="tablist"`
- buttons use `role="tab"`
- `aria-selected`
- skip-to-main-content link
- footer
- fixed Streamlit page title
- active tab scrolling so later tabs remain visible in a horizontally scrollable tab strip

Important recent change:

The active top tab is scrolled into view using a ref. It is guarded for environments where `scrollIntoView` is unavailable.

Do not accidentally reintroduce the literal `\n` bug that briefly appeared during editing; the final source was corrected before handoff.

---

# 9. Global CSS / Streamlit Visual System

Primary file:

```text
fastapi_react/frontend/src/styles.css
```

The React app intentionally approximates Streamlit's light theme rather than inventing a new design system.

Core values:

```text
page background: white
secondary background: #f0f2f6
text: #31333f
Streamlit accent: #ff4b4b
```

Implemented visual contracts include:

- wide block container
- Streamlit-like heading sizes
- 450px logo
- horizontal main tab strip
- red active tab underline
- red active sub-tab text
- fixed sidebar
- Streamlit-style input backgrounds
- table borders / sticky header
- yellow next-race highlighting
- Streamlit-ish cards/headings
- footer styling
- responsive behavior
- visible focus indicators
- reduced-motion support

Recent global visual corrections include:

- narrow-width sidebar preserves Streamlit's persistent left-column layout instead of overlaying the main page
- sidebar filter labels/controls constrained to the sidebar content width
- checkboxes retain flex layout
- Streamlit-like table header styling
- Streamlit-like subheader scale
- Streamlit-like active sub-tab styling
- input/select secondary-background treatment
- checkbox alignment
- top-tab active-item scroll behavior

---

# 10. Shared UI Components

Primary file:

```text
fastapi_react/frontend/src/components/UI.jsx
```

Major components:

- `Card`
- `Status`
- `DataTable`
- `JsonBlock`
- `Metric`
- `Tabs`

## DataTable accessibility

There was an important accessibility evolution:

### Earlier state

Every table wrapper was being assigned a region landmark and repeated aria label behavior. Axe reported duplicate landmark problems.

### Current state

Only explicitly labeled tables receive:

```jsx
role="region"
aria-label="..."
```

All `.table-wrap` containers are keyboard focusable:

```jsx
tabIndex={0}
```

This fixes Safari/axe's `scrollable-region-focusable` rule without creating duplicate landmarks.

Do not revert this.

## Tabs semantics

Sub-tabs use:

```text
role="tablist"
role="tab"
aria-selected
```

Arrow-key behavior is implemented:

- ArrowRight
- ArrowLeft
- Home
- End

Unit tests were updated accordingly.

---

# 11. Accessibility Status

A completed evidence run before the latest visual changes reported:

```text
violation_count: 0
```

That is important. Earlier runs had:

- `aria-required-parent`
- color contrast violations
- duplicate `landmark-unique`
- `scrollable-region-focusable`

Those were corrected.

The current visual workflow still runs axe on the following app states:

- Data Explorer
- Analytics
- Schedule
- Next Race
- Predictive Models
- Data & Debug
- Betting Research

Do not reintroduce:

- duplicate table region landmarks
- non-focusable scrolling tables
- low-contrast captions
- improper tab semantics

---

# 12. Data Explorer

Frontend:

```text
fastapi_react/frontend/src/pages/DataExplorer.jsx
fastapi_react/frontend/src/components/FilterSidebar.jsx
```

Backend:

```text
fastapi_react/backend/app/services/data.py
```

## Reference behavior

Heading:

```text
Data Explorer
```

Description:

```text
Filter and explore F1 race data from multiple perspectives.
```

Toggle:

```text
Filter Results
```

When enabled, Streamlit creates its global sidebar:

```text
Select filters to apply:
```

and constructs controls dynamically from the loaded/merged dataset.

## Critical discovery: Streamlit does not build filters from raw f1ForAnalysis alone

This was one of the largest parity issues found in visual evidence.

`raceAnalysis.py` loads the primary dataset and then merges:

```text
constructor_standings.csv
driver_standings.csv
```

before it creates `column_names`.

Therefore the Streamlit sidebar includes fields that do **not** necessarily exist in the un-enriched raw `f1ForAnalysis.csv`.

Examples visibly missing in the old React screenshot included:

```text
Current Year Points (Driver)
Best Champ Pos.
Best Race Result
Best Starting Grid Pos.
Constructor Rank
Driver Rank
```

The backend was fixed to reproduce Streamlit's enrichment.

## Current backend enrichment

`load_main_data()` now reconstructs several canonical columns from legacy merge-suffixed fields where necessary, including:

```text
bestChampionshipPosition
bestStartingGridPosition
bestRaceResult
totalChampionshipWins
totalRaceStarts
totalRaceWins
totalRaceLaps
totalPodiums
totalPoints
totalChampionshipPoints
totalFastestLaps
totalRaceEntries
```

It also uses `constructor_standings.csv` to map constructor-standing data via `constructorId_results`.

It uses `driver_standings.csv` to map driver ranking via `resultsDriverId`.

This is intentionally done without a many-to-many dataframe merge so the FastAPI dataset keeps one row per race/driver.

## Filter labels

Friendly labels come from `column_rename_for_filter` in `raceAnalysis.py`.

Notable examples:

```text
Points -> Current Year Points (Driver)
bestChampionshipPosition -> Best Champ Pos.
bestStartingGridPosition -> Best Starting Grid Pos.
bestRaceResult -> Best Race Result
championship_position -> Current Championship Position
constructorTotalRaceStarts -> Constructor Total Starts
constructorTotalRaceWins -> Constructor Total Wins
```

## Filter order

Streamlit does:

```python
column_names = data.columns.tolist()
column_names.sort()
```

The API therefore sorts raw field names alphabetically and applies friendly labels only for presentation.

Do **not** sort by display label.

## Exclusion behavior

The backend parses the authoritative Streamlit definitions from `raceAnalysis.py` using AST:

- `column_rename_for_filter`
- `exclusionList`
- `suffixes_to_exclude`
- `selected_columns`

This avoids hardcoding a stale separate filter definition.

## Filter semantics

Implemented:

- numeric range
- date range
- boolean / 0-1
- exact categorical
- null-preserving semantics

Streamlit intentionally retains null rows while applying a filter:

```python
(filtered_data[column] >= lo & <= hi) | filtered_data[column].isna()
```

React/FastAPI follows that behavior.

## Filter state

State is stored in session storage:

```text
f1analysis.filters
```

The sidebar remains present when navigating to Analytics after filters have been enabled, matching Streamlit's global sidebar behavior.

## Initial defaults

When Filter Results is enabled:

- range/date controls start at full min/max
- those ranges are immediately included in the active filter state
- exact and boolean controls default to no restriction

## Query ordering

Data Explorer query sorts:

```text
grandPrixYear descending
resultsFinalPositionNumber ascending
```

Backend accepts mixed sort direction via:

```text
ascending: list[bool] | None
```

## Result columns

The React table intentionally uses the same major column order as Streamlit's `st.dataframe` call.

## Inner tabs

Data Explorer contains:

```text
Data
Data & Debug
```

The source Streamlit Data & Debug pane in this location is effectively empty, so the blank React pane is intentional.

## Regression test added

A backend test asserts the presence and labels of standings-backed filters, including:

```text
Points
bestChampionshipPosition
bestRaceResult
bestStartingGridPosition
constructorRank
driverRank
```

Do not remove this test.

## One remaining Data Explorer verification item

React query currently requests:

```text
limit: 5000
```

The Streamlit dataframe can display the full filtered dataset.

The current dataset is likely below this threshold, but **verify row count before final signoff**.

If the dataset is >5000:

- increase the allowed request max
- avoid silently truncating the main Data Explorer result

Do not claim perfect parity while a real row-cap difference exists.

---

# 13. Analytics & Visualizations

Frontend:

```text
fastapi_react/frontend/src/pages/Analytics.jsx
fastapi_react/frontend/src/components/Charts.jsx
```

Backend:

```text
fastapi_react/backend/app/services/analysis.py
```

Analytics only displays the main analytical content after Data Explorer filtering has been enabled, matching the Streamlit workflow.

If no filters are active:

```text
Please filter results in the Data Explorer tab first to view analytics.
```

## Implemented analysis areas

The backend currently supports:

- active years vs final position
- positions gained over time
- last practice vs final position
- starting grid vs final position
- average practice vs final position
- average pit stop vs final position
- practice/final regression
- grid/final regression
- correlation matrix
- driver performance over time
- constructor performance over time
- driver vs constructor performance
- DNF reasons
- DNF by driver
- DNF by race
- DNF by constructor
- track turns vs final position
- season summary
- driver consistency
- model manifest metrics
- feature importance
- tire strategy
- historical validation summaries
- top-3 MAE
- top-3 predictions
- first-30 predictions

## Constructor dominance

The Streamlit reference uses both:

```text
wins
podiums
```

The React migration now uses `MultiBarPanel` to render both series instead of wins alone.

## Driver performance

Driver performance now uses a true multi-series line chart rather than collapsing drivers into one line.

## DNF reasons

Both a bar representation and pie representation are available in parity with source behavior.

## Tire strategy

Tire analysis is present as a major Analytics section.

The API supplies race rows, degradation rows, and historical rows.

## Visual note

The old visual evidence showed Analytics being vertically displaced largely because the sidebar field set was incomplete.

That has now been corrected at the data/schema level.

Shared Streamlit subheader sizing was also corrected after the old evidence.

Use run #115 evidence before adjusting Analytics spacing again.

---

# 14. Schedule / Current Season

Frontend:

```text
fastapi_react/frontend/src/pages/CurrentSeason.jsx
```

Backend:

```text
/api/current-season
```

Reference heading:

```text
{current_year} Season
```

Reference description:

```text
Complete schedule and information for the {current_year} Formula 1 season.
```

Reference count:

```text
Total number of races: ...
```

## Exact column order

```text
round
fullName
date
time
circuitType
courseLength
laps
turns
distance
totalRacesHeld
```

## Exact visible labels

Recently corrected to match `schedule_columns_to_display`:

```text
Round
Name
Date
Time
Type
Lap Length (km)
Number of Laps
Number of Turns
Distance (km)
Races Held
```

Do not revert to generic labels such as:

```text
Grand Prix
Circuit Type
Course Length
Laps
Turns
Distance
Total Races Held
```

because those were visibly different from Streamlit.

## Next-race highlighting

Streamlit uses:

```text
#ffe599
```

React uses the same color.

---

# 15. Next Race

Frontend:

```text
fastapi_react/frontend/src/pages/NextRace.jsx
```

Backend logic:

```text
fastapi_react/backend/app/services/analysis.py
```

Reference order is important.

Major blocks:

1. heading / description
2. Show Next Race checkbox
3. Next Race details
4. Past Results
5. Predictive Results for Active Drivers
6. Predictive DNF
7. Predicted Safety Car
8. Flags and Safety Cars
9. Driver Performance
10. Constructor Performance
11. Fastest Individual Pit Stop per Constructor
12. Weather

## Next-race detail columns

Order:

```text
date
time
fullName
courseLength
turns
laps
```

Visible labels now match Streamlit:

```text
Date
Time
Grand Prix
Lap Length (km)
Number of Turns
Number of Laps
```

## Position predictions

React uses committed prediction artifacts and renders:

```text
Constructor
Driver
Predicted Final Position
Predicted Position Std.
Predicted Position Low
Predicted Position High
Historical MAE by Rank
MAE Low
MAE High
```

Intervals use:

- global model MAE
- position/rank-specific historical MAE where available

## DNF prediction parity

This was substantially upgraded.

Backend helper:

```text
_load_dnf_model()
```

loads the committed model:

```text
data_files/models/dnf_model.pkl
```

The DNF feature order comes from:

```text
data_files/models/dnf_manifest.json
```

Important feature set includes:

```text
grandPrixName
constructorName
resultsDriverName
driverTotalRaceEntries
driverTotalRaceStarts
driverTotalChampionshipWins
driverTotalRaceWins
driverTotalPodiums
yearsActive
constructorTotalRaceStarts
constructorTotalRaceWins
constructorTotalPolePositions
averagePracticePosition
lastFPPositionNumber
resultsStartingGridPositionNumber
numberOfStops
trackRace
streetRace
turns
average_temp
average_humidity
average_wind_speed
total_precipitation
driverDNFCount
driverDNFAvg
driver_dnf_rate_5_races
recent_dnf_rate_3_races
constructor_dnf_rate_3_races
constructor_dnf_rate_5_races
total_experience
driverAge
```

If legacy DNF prediction rows are not available, React now builds predictions from the saved artifact using active-driver position-prediction identities.

## DNF diagnostics

Streamlit prints:

```text
Logistic Regression DNF Probabilities:
Min: ...
Max: ...
Mean: ...
```

Backend now computes historical saved-model diagnostic probabilities and React renders them.

This was a specific parity gap that has already been closed.

## Safety-car artifact parity

This was a known risk earlier in the migration and has been fixed.

Current `_load_safety_car_model()` searches the same model-directory hierarchy expected by Streamlit, including:

```text
models/xgboost/safetycar_model.pkl
models/lightgbm/safetycar_model.pkl
models/catboost/safetycar_model.pkl
models/ensemble/safetycar_model.pkl
models/safetycar_model.pkl
```

Relevant commits included:

```text
2d321829... Match Streamlit safety-car artifact loading
7377ebb... Test safety-car artifact search parity
```

Do not undo the multi-directory search by reverting to root-only lookup.

---

# 16. Predictive Models

Frontend:

```text
fastapi_react/frontend/src/pages/Models.jsx
```

Reference heading:

```text
Predictive Models & Advanced Options
```

Reference model selector includes six model choices:

1. `XGBoost`
2. `LightGBM`
3. `CatBoost`
4. `Ensemble (XGBoost + LightGBM + CatBoost)`
5. `Position Group`
6. `Track-Weighted Ensemble`

## Advanced tabs

Exact set:

1. `📊 Model Performance`
2. `🔍 Feature Analysis`
3. `🎯 Feature Selection`
4. `🏎️ Position-Specific Analysis`
5. `⚙️ Hyperparameters`
6. `📈 Historical Validation`
7. `🛠️ Debug & Experiments`

## Model Performance

Implemented major source areas include:

- Predictive Data Model Metrics
- Mean Error and MAE per Driver
- Error Metrics per Driver
- Predictive Results with Features
- Feature Importances
- MAE by Position Groups
- MAE by Individual Positions
- Position Group Summary
- Prediction Error Distribution by Position Groups

## Feature Analysis

Includes:

- Feature Analysis
- Permutation Importance
- low-importance features
- high-importance features
- High-Cardinality Features
- Safety Car Feature Importance
- Correlation Matrix
- Feature Importances

## Feature Selection

Includes precomputed sources for:

- Monte Carlo
- Monte Carlo run log
- SHAP
- RFE
- Boruta
- Permutation

Also includes major selection summaries and export/detail sections.

## Position-Specific Analysis

Includes:

- Position Group MAE Summary
- Overall Model MAE
- group rows
- winner examples
- position detail
- historical position analysis where artifacts exist

## Hyperparameters

Includes:

- Bayesian HPO precomputed artifact
- Grid HPO precomputed artifact

## Historical Validation

Includes current historical validation artifact data.

## Debug & Experiments

Includes corresponding research/debug controls, gated as appropriate.

## Expensive tools

Hosted mode keeps expensive research operations disabled.

FastAPI uses environment gating rather than running training on normal page requests.

Do not add request-time model training to restore a UI control. The Streamlit app's data/model artifacts are authoritative and the React app should consume committed/precomputed output whenever possible.

---

# 17. Data & Debug

Frontend:

```text
fastapi_react/frontend/src/pages/RawData.jsx
```

The source Streamlit code literally renders:

```python
st.write("Tab 6 START")
```

before:

```python
st.header("Data & Debug Tools")
```

React intentionally preserves:

```text
Tab 6 START
```

This may look like a debug artifact, but it is currently source parity.

Do not remove it unless `raceAnalysis.py` itself is changed.

## Sub-tabs

```text
Raw Data
Temporal Leakage Audit
Hyperparameter Tuning
```

## Raw Data

Current React behavior allows viewing the unfiltered dataset and paging it.

## Temporal Leakage Audit

Admin / research control.

Hosted mode warns that research controls are disabled.

## Hyperparameter Tuning

Also research-gated.

---

# 18. Betting Research

Frontend:

```text
fastapi_react/frontend/src/pages/BettingResearch.jsx
```

Backend:

```text
fastapi_react/backend/app/services/betting.py
```

This is a React port of the current `f1bet` research UI / logic rather than a separate rewritten betting engine.

Sub-tabs:

1. `Value & stake`
2. `Field simulation`
3. `Paper replay`
4. `Calibration`

## Value & stake

Supports:

- model probability
- selection decimal odds
- opposing decimal odds
- uncertainty
- multiplicative de-vig
- additive de-vig
- power de-vig
- de-vigged market probability
- raw EV
- conservative probability
- paper stake
- decision reason code

## Field simulation

Supports:

- CSV input
- default template
- coherent field simulation
- simulation count
- probability output table
- CSV download

## Paper replay

Supports:

- ledger CSV
- backtest
- summary
- placed paper bets
- decisions / abstentions
- sensitivity

## Calibration

Supports:

- CSV input
- probability/outcome metrics
- reliability table
- calibration line visualization

---

# 19. Backend Data / API Architecture

Key files:

```text
fastapi_react/backend/app/main.py
fastapi_react/backend/app/schemas.py
fastapi_react/backend/app/services/data.py
fastapi_react/backend/app/services/analysis.py
fastapi_react/backend/app/services/betting.py
fastapi_react/backend/app/services/tools.py
```

Data root is inherited from the main repository.

The migration does **not** maintain an independent duplicate data pipeline.

This is important: the original generator/artifact workflow remains authoritative.

FastAPI should read the same committed/generated files the Streamlit app uses.

---

# 20. Data Loading Behavior

Primary dataset:

```text
data_files/f1ForAnalysis.csv
```

Current API uses tab-separated parsing.

`load_main_data()` is cached via `lru_cache`.

Several source-derived aliases/enrichments are applied before filters are constructed.

This is necessary because Streamlit's in-memory dataset is richer than raw `f1ForAnalysis.csv`.

Do not simplify `load_main_data()` back to just:

```python
pd.read_csv(...)
```

without recreating Streamlit's effective dataset contract.

---

# 21. Visual Parity Infrastructure

Directory:

```text
fastapi_react/parity_evidence/
```

Important scripts:

```text
capture_react.mjs
capture_streamlit.mjs
diff_screenshots.mjs
audit_accessibility.mjs
benchmark.mjs
```

## React capture

Runs against:

```text
http://127.0.0.1:5173
```

## Streamlit capture

Runs local `raceAnalysis.py` against:

```text
http://127.0.0.1:8501
```

This is deliberate.

The public Streamlit Community Cloud app can enter auth/sleep/wake behavior and is not a deterministic visual oracle.

The CI therefore launches the same repository checkout locally and compares React against local Streamlit using the same data and source revision.

---

# 22. Screenshot Viewports

Current capture includes:

```text
desktop: 1280 x 800
tablet: 768 x 1024
mobile: 390 x 844
```

This is stricter than the older checklist, which only mentioned desktop/tablet.

Do not remove mobile from the evidence run.

---

# 23. Screenshot Page States

The visual workflow currently captures eight states per viewport:

```text
home
data-explorer
analytics
current-season
next-race
models
raw-data
betting-research
```

That produces:

```text
8 pages x 3 viewports = 24 screenshot pairs
```

React screenshots are written under:

```text
fastapi_react/parity_evidence/visual/react/
```

Streamlit screenshots:

```text
fastapi_react/parity_evidence/visual/streamlit/
```

Diff images:

```text
fastapi_react/parity_evidence/visual/diff/
```

Diff summary:

```text
fastapi_react/parity_evidence/visual/diff/summary.json
```

---

# 24. Dynamic Text Normalization

Both capture scripts normalize dynamic metadata so timestamp drift does not create false visual failures.

Normalized values include:

```text
Last updated: 2026-09-30 09:00 PM
Code deployed at: 2026-09-30 21:00:00 UTC
```

Both sides also force a system font during screenshot capture.

This is intentional to reduce machine/font nondeterminism.

---

# 25. Visual-Diff Acceptance Threshold

The acceptance thresholds remain:

```text
desktop <= 2%
tablet <= 3%
mobile <= 3%
```

These thresholds must **not** be loosened merely to make CI green.

---

# 26. Antialias/Subpixel Correction in Diff Engine

The old raw comparator counted a large amount of one-pixel glyph/vector rasterization drift as a visual failure.

For example, visually equivalent Home screenshots were still failing because every character edge differed slightly.

The current comparator therefore:

- keeps the same 2%/3% page thresholds
- compares each React pixel to the nearest color in a **3x3 Streamlit neighborhood**
- treats only that one-pixel offset as allowable rasterization drift
- still marks larger/contiguous layout/content differences

This is not intended as a threshold relaxation.

It is intended to stop counting font/vector antialias placement as a real design defect.

Relevant code:

```text
fastapi_react/parity_evidence/diff_screenshots.mjs
```

Do not increase neighborhood radius beyond 1 pixel without a strong reason.

Do not increase page tolerance.

---

# 27. Previous Full Visual Evidence Run

The important completed evidence run immediately before the newest changes was:

```text
FastAPI + React parity checks #101
run ID: 36780523871
```

Its functional result was excellent:

- backend: success
- frontend: success
- accessibility: success
- operational benchmark: success
- local Streamlit capture: success
- screenshot diff executed
- evidence artifact uploaded

But visual diff failed:

```text
24 / 24 pairs above threshold
```

The old raw percentages included:

### Desktop

Roughly:

```text
2.4% to 6.5%
```

depending on page.

### Tablet / mobile

Some states were much higher, reaching approximately:

```text
10% to 22%
```

The screenshots revealed that these were **not all equivalent defects**.

Some were rasterization noise, but there were also real layout problems.

---

# 28. Major Visual Defects Found in the Previous Evidence

## 28.1 Missing sidebar filters

This was a real feature/layout defect.

React's Analytics sidebar omitted Streamlit controls including:

```text
Current Year Points
Best Champ Pos.
Best Race Result
Best Starting Grid Pos.
```

Because the sidebar had fewer controls, every downstream vertical location differed.

Root cause:

React filter schema was being based on raw data rather than Streamlit's post-standing-merge dataset.

This is now fixed in backend data enrichment.

## 28.2 Sidebar responsive behavior

On tablet/mobile, React dropped the main-content left margin while keeping the sidebar fixed.

Result:

```text
sidebar overlayed main content
```

Streamlit maintains the page alongside its sidebar.

React CSS was corrected to preserve sidebar/main geometry at narrow widths.

## 28.3 Sidebar control width

React sliders/selects were extending too far to the right edge of the sidebar.

Filter label/control layout is now explicitly block-width constrained, except checkbox rows, which retain flex layout.

## 28.4 Schedule header labels

React had generic column labels instead of the exact `st.column_config` labels.

Fixed.

## 28.5 Next-race header labels

Also fixed to use exact Streamlit labels.

## 28.6 Table headers

Old React screenshots had darker/bolder headers than Streamlit.

Global table header style was adjusted toward Streamlit's regular-weight gray presentation.

## 28.7 Sub-tabs

Old React active sub-tabs were dark/bold.

Streamlit's active tabs are red and regular-weight.

Adjusted globally.

## 28.8 Streamlit subheaders

Shared `Card` h2 presentation was visibly smaller than `st.subheader`.

Adjusted upward in size/spacing.

## 28.9 Top tab scrolling

For later tabs, Streamlit horizontally scrolls the main tab strip enough to keep the active tab visible.

React now attempts to scroll the active tab into view.

## 28.10 Checkbox geometry

Browser-native checkbox margins created a small but repeatable mismatch.

Shared filter/dataset checkbox inputs now explicitly remove margin.

---

# 29. Why the Current Run Matters

The old artifact cannot tell you whether the current branch is still visually failing, because significant parity changes were made after it.

Therefore:

**Do not make decisions based solely on run #101 percentages.**

Use run #115.

The old artifact remains useful to understand why specific fixes were added.

---

# 30. CI Workflow Design

Workflow file:

```text
.github/workflows/fastapi-react.yml
```

## Concurrency

The parity workflow now uses branch-level concurrency:

```yaml
concurrency:
  group: fastapi-react-parity-${{ github.ref }}
  cancel-in-progress: true
```

This was added because many quick parity commits were creating a large queue of obsolete visual runs.

## Backend job

Runs:

```text
ruff
mypy strict
pytest with >=80% coverage
pip-audit
```

## Frontend job

Runs:

```text
npm ci
eslint
tsc --noEmit
vitest --coverage
vite build
npm audit --omit=dev
```

Frontend line coverage is now enforced at:

```text
>=80%
```

This is no longer the older ~60–70% state described in stale `PARITY_REPORT.md`.

Relevant earlier commits included:

```text
d259a9c... Expand chart coverage tests
5fc8351... Cover complete next-race rendering paths
c56bdb8... Raise model page coverage across all advanced tabs
477ad0a... Enforce 80 percent frontend line coverage
```

## Visual evidence job

The workflow intentionally uses `continue-on-error` for individual evidence stages so one failed gate does not prevent collection of later evidence.

It then has a final enforcement step.

This is crucial.

Before this change, a failed accessibility gate prevented:

- benchmark
- Streamlit screenshots
- diff
- artifact upload

Now evidence still gets collected.

Do not remove this pattern.

## Artifact upload

Evidence upload uses `if: always()`.

This allows inspection even when visual diff fails.

---

# 31. Operational Benchmark

Script:

```text
fastapi_react/parity_evidence/benchmark.mjs
```

A recent completed run reported the benchmark step as successful.

The benchmark is intended to help verify that the migration does not introduce an operational regression and that request paths are not training models unexpectedly.

FastAPI health exposes process RSS.

Keep expensive research/training outside normal request paths.

---

# 32. Code-Quality State

## Backend

Expected / previously green:

```text
Ruff
strict mypy
pytest
>=80% line coverage
pip-audit runtime requirements
```

## Frontend

Current workflow had already completed successfully at handoff head:

```text
ESLint
tsc --noEmit
Vitest
>=80% line coverage
Vite build
bundle budget
production npm audit
```

## Security workflows

At handoff:

```text
Dependency and Code Security: green
F1Bet Offline Release Gates: green
Streamlit API Compatibility: green
```

---

# 33. Important Source-Specific Quirks That Must Be Preserved

These are easy for another developer to "clean up" incorrectly.

## `Tab 6 START`

It exists in Streamlit. React intentionally shows it.

## Browser title

Keep exactly:

```text
Gridlocked - Formula 1 Betting & Analytics
```

## Sidebar order

Sort raw column names, not friendly labels.

## Null-preserving filters

Streamlit intentionally preserves null rows when filtering.

## Current-season labels

Use the exact column-config labels.

## DNF diagnostic Min / Max / Mean

These are real Streamlit output and now exist in React.

## Safety-car artifact search

Search model-type subdirectories and root.

## Analytics requires Data Explorer filters

This workflow relationship is intentional.

---

# 34. Files Changed by PR #128

At the handoff snapshot, major modified/added paths included:

## Workflow

```text
.github/workflows/fastapi-react.yml
```

## Backend

```text
fastapi_react/backend/app/main.py
fastapi_react/backend/app/schemas.py
fastapi_react/backend/app/services/analysis.py
fastapi_react/backend/app/services/data.py
fastapi_react/backend/test_api.py
```

## Frontend shell/components

```text
fastapi_react/frontend/src/App.jsx
fastapi_react/frontend/src/App.test.jsx
fastapi_react/frontend/src/components/Charts.jsx
fastapi_react/frontend/src/components/Charts.test.jsx
fastapi_react/frontend/src/components/FilterSidebar.jsx
fastapi_react/frontend/src/components/UI.jsx
fastapi_react/frontend/src/components/UI.test.jsx
fastapi_react/frontend/src/styles.css
```

## Frontend pages

```text
fastapi_react/frontend/src/pages/Analytics.jsx
fastapi_react/frontend/src/pages/Analytics.test.jsx
fastapi_react/frontend/src/pages/BettingResearch.jsx
fastapi_react/frontend/src/pages/BettingResearch.test.jsx
fastapi_react/frontend/src/pages/CurrentSeason.jsx
fastapi_react/frontend/src/pages/CurrentSeason.test.jsx
fastapi_react/frontend/src/pages/DataExplorer.jsx
fastapi_react/frontend/src/pages/DataExplorer.test.jsx
fastapi_react/frontend/src/pages/Models.jsx
fastapi_react/frontend/src/pages/Models.test.jsx
fastapi_react/frontend/src/pages/NextRace.jsx
fastapi_react/frontend/src/pages/NextRace.test.jsx
fastapi_react/frontend/src/pages/RawData.jsx
fastapi_react/frontend/src/pages/RawData.test.jsx
fastapi_react/frontend/vite.config.js
```

## Parity tooling

```text
fastapi_react/parity_evidence/audit_accessibility.mjs
fastapi_react/parity_evidence/benchmark.mjs
fastapi_react/parity_evidence/capture_react.mjs
fastapi_react/parity_evidence/capture_streamlit.mjs
fastapi_react/parity_evidence/diff_screenshots.mjs
```

---

# 35. Current Documentation Is Stale

Files:

```text
fastapi_react/PARITY_CHECKLIST.md
fastapi_react/PARITY_REPORT.md
```

These files were stale at the start of this continuation and have now been
rewritten against the completed run #115 artifact and the newer local
verification. They intentionally distinguish PR evidence from local evidence.
Neither document declares parity complete: visual comparison, feature-level
output comparisons, and PR-head verification remain open.

---

# 36. What the Final PARITY_REPORT Must Eventually Contain

After the final green run, replace stale text with actual evidence.

At minimum include:

- final PR head SHA
- final parity workflow run ID / number
- backend lint/type/test/coverage result
- frontend lint/type/test/coverage result
- accessibility violation count
- benchmark result
- screenshot capture viewports
- visual page states
- per-page diff ratios
- statement that every required pair is within threshold
- any explicitly intentional UI difference
- confirmation that request-time training is not happening
- confirmation that Streamlit remains available for rollback until cutover

Do not say "perfect parity" without showing the evidence.

---

# 37. What the Final PARITY_CHECKLIST Must Eventually Do

After evidence is complete:

- check items actually verified
- remove stale "deferred" language
- add mobile viewport to the visual section
- reflect the actual CI-driven process
- reflect 80% frontend line coverage
- reflect zero automated accessibility violations if still true
- record any manual keyboard verification honestly
- distinguish implementation from evidence where manual checks remain

---

# 38. Remaining Work — Updated 2026-10-01

Current priorities, superseding the frozen run-in-progress steps below:

1. Preserve the dirty local worktree; reconcile applicable changes with the actual PR branch without overwriting existing work.
2. Resolve functional output differences and visual layout/content deviations on the PR head.
3. Resolve its six-section axe contrast failures and record the required keyboard pass.
4. Rerun all 24 visual pairs without changing the 2%/3% tolerances.
5. Compare Streamlit and React/FastAPI operational benchmarks under the same workload.
6. Refresh CI and documentation on the exact PR head; mark ready only after all gates pass.
7. Do not merge unless explicitly requested.

The priority notes below describe the handoff-time sequence for historical context.

## Priority 1 — Inspect CI run #115

Find:

```text
FastAPI + React parity checks
run ID 36796824643
run #115
head 124c1a7b...
```

Check whether:

- backend job passes
- visual evidence job starts
- accessibility stays at zero
- benchmark passes
- Streamlit screenshot capture passes
- visual diff passes

## Priority 2 — If visual diff fails, download the evidence artifact

Expected artifact name:

```text
fastapi-react-parity-evidence
```

Inspect:

```text
visual/diff/summary.json
```

Sort failures by:

1. highest `diff_ratio`
2. page
3. viewport

For each failing page, inspect:

```text
visual/react/{viewport}-{page}.png
visual/streamlit/{viewport}-{page}.png
visual/diff/{viewport}-{page}.png
```

## Priority 3 — Fix real differences, not diff noise

Focus in this order:

1. missing content
2. wrong column/header labels
3. sidebar geometry
4. content width
5. vertical layout
6. tab visibility/scroll position
7. widget sizes
8. typography
9. minor borders/background

The 3x3 diff neighborhood already handles one-pixel raster differences.

Do not compensate for a content mismatch by modifying the comparator.

## Priority 4 — Re-run until all 24 pairs are within threshold

Required:

```text
desktop <= .02
tablet <= .03
mobile <= .03
```

## Priority 5 — Verify Data Explorer total-row cap

Determine full dataset row count.

If >5000, remove the silent truncation risk.

## Priority 6 — Validate any remaining source-specific model-tab details

If screenshots reveal missing content in Predictive Models, compare directly against the corresponding `raceAnalysis.py` block.

Do not assume the old checklist identifies the missing block correctly.

## Priority 7 — Final documentation

Rewrite:

```text
PARITY_CHECKLIST.md
PARITY_REPORT.md
```

against current source/evidence.

## Priority 8 — PR state

When all acceptance gates are genuinely satisfied:

- mark PR #128 ready for review

Do **not** merge unless explicitly asked.

---

# 39. How to Investigate a Visual Failure Efficiently

Suppose the new summary says:

```json
{
  "viewport": "mobile",
  "page": "analytics",
  "diff_ratio": 0.08,
  "tolerance": 0.03
}
```

Do not immediately change general mobile CSS.

Instead:

1. Open the React mobile Analytics screenshot.
2. Open the Streamlit mobile Analytics screenshot.
3. Compare top-left anchored geometry.
4. Ask whether the difference begins:
   - at logo/title shell
   - tab strip
   - sidebar
   - first page heading
   - first chart
5. Inspect diff PNG for contiguous blocks.
6. Trace only the first divergence.

A vertical displacement early in the page causes every later pixel to differ, so fixing the first divergence can collapse the entire diff.

This is exactly what happened with the old Analytics sidebar.

---

# 40. Do Not Blindly Copy Streamlit DOM/CSS

The objective is visible and behavioral parity, not reproducing Streamlit's internal DOM.

React should keep:

- semantic HTML
- accessible tabs
- keyboard-focusable scroll regions
- explicit error states
- tests
- stable API boundaries

When Streamlit's DOM is inaccessible or semantically weak, use an accessible React equivalent that renders similarly.

---

# 41. Accessibility Guardrails During Final Styling

While fixing pixel parity, preserve:

```text
0 automated axe violations
```

In particular:

- do not hide focus outlines
- do not remove `tabIndex=0` from scrollable tables
- do not re-add identical generic region landmarks
- do not lower caption/footer contrast
- do not replace semantic tabs with plain buttons lacking tab relationships
- keep labels on form controls

Visual parity is not permission to regress accessibility.

---

# 42. Performance / Architecture Guardrails

Do not reintroduce expensive model computation into user-request handlers.

Preferred pattern remains:

```text
GitHub Actions / preprocessing
      ↓
committed/generated artifacts
      ↓
FastAPI lightweight read/transform
      ↓
React rendering
```

Use saved models for inference only where the Streamlit page itself performs corresponding inference and it is operationally safe.

---

# 43. Testing Guardrails

Do not lower:

```text
backend coverage threshold
frontend line coverage threshold
visual page tolerances
```

Do not disable a failing accessibility rule merely to obtain green CI.

If a test reveals that Streamlit behavior changed, update the implementation against current `raceAnalysis.py`.

---

# 44. Known Good Historical Milestones

Several key commits from this parity effort are useful landmarks:

```text
888c09c... Match constructor dominance wins and podiums chart
50c952b... Render Streamlit constructor wins and podiums series
8e39c22... Preserve Streamlit Data Debug marker
04d7704... Expose saved-model DNF diagnostics for Next Race parity
71a6744... Render Streamlit DNF diagnostic statistics
c0082b3... Cancel superseded parity workflow runs
60a2244... Fix keyboard access for scrollable tables
d259a9c... Expand chart coverage tests
5fc8351... Cover complete next-race rendering paths
c56bdb8... Raise model page coverage across all advanced tabs
477ad0a... Enforce 80 percent frontend line coverage
e4e15c6... Always upload parity evidence artifacts
4636d03... Match Streamlit shell spacing and responsive typography
525828a... Keep parity evidence flowing through failed gates
2d32182... Match Streamlit safety-car artifact loading
7377ebb... Test safety-car artifact search parity
```

Later branch work also added:

- standings-backed filter enrichment
- narrow viewport sidebar correction
- exact Schedule / Next Race labels
- Streamlit sub-tab styling
- active tab scrolling
- table header styling
- subheader scale
- antialias-aware screenshot comparison
- sidebar content-width constraints
- standings filter regression test
- checkbox geometry normalization

The handoff head includes those later changes.

---

# 45. Previous Accessibility Failure Evolution

This is useful if a future change causes axe regressions.

## Earlier failure class 1

```text
aria-required-parent
```

Cause:

top-level tab buttons did not sit under correct tablist semantics.

Fix:

semantic tablist wrapper.

## Earlier failure class 2

```text
color-contrast
```

Affected:

- captions
- footer subtitle
- footer link

Fix:

higher-contrast values.

## Earlier failure class 3

```text
landmark-unique
```

Cause:

generic DataTable wrappers repeatedly used region landmarks / identical labels.

Fix:

landmark props only when an explicit `ariaLabel` is supplied.

## Earlier failure class 4

```text
scrollable-region-focusable
```

Cause:

overflowing table wrappers without keyboard focus.

Fix:

all `.table-wrap` containers receive `tabIndex={0}`.

Latest completed accessibility evidence:

```text
0 violations
```

---

# 46. Old Visual Comparison Lessons

The raw old diff made visually close pages appear worse than they were because glyph edges differed by a pixel.

However, visual screenshots also exposed real issues.

Therefore the correct lesson is **not** "the comparator was wrong."

The correct lesson is:

```text
Use antialias-aware comparison,
then inspect the remaining contiguous differences manually.
```

The updated comparator follows that principle.

---

# 47. Filter Sidebar Visual Contract

When filters are active, Streamlit's sidebar is a major part of the screenshot.

Important properties:

- fixed left column
- light secondary background
- approximately 300px wide
- heading near top
- controls vertically stacked
- sliders/selects remain within the padded content width
- main content begins to the right of sidebar
- behavior remains similar on tablet/mobile

Because Analytics requires active filters, a sidebar mismatch can make **every Analytics screenshot fail**.

Treat it as a global visual component.

---

# 48. Current Top-Level UI Text Worth Comparing Literally

## Data Explorer

```text
Data Explorer
Filter and explore F1 race data from multiple perspectives.
Filter Results
```

## Analytics

```text
Analytics & Visualizations
Comprehensive charts, regressions, and analysis of filtered data.
```

## Schedule

```text
{year} Season
Complete schedule and information for the {year} Formula 1 season.
Total number of races: ...
```

## Next Race

```text
Next Race
Details, predictions, and analysis for the upcoming race.
Show Next Race
Next Race:
Past Results:
Predictive Results for Active Drivers
Predictive DNF
Predicted Safety Car
```

## Predictive Models

```text
Predictive Models & Advanced Options
Advanced machine learning models, hyperparameter tuning, and feature selection tools.
Select Model Type
```

## Data & Debug

```text
Tab 6 START
Data & Debug Tools
```

## Betting

```text
Probability & Betting Research
```

---

# 49. Source Navigation Locations

Useful approximate source areas in `raceAnalysis.py` from this work:

## Filter definitions / labels

Around the block containing:

```text
column_rename_for_filter
```

## Dataset construction

Around:

```text
load_data(...)
get_shared_dataset(...)
```

and standing merges.

## `column_names`

The current code constructs/sorts:

```python
column_names = data.columns.tolist()
column_names.sort()
```

## Data Explorer

Around:

```python
with tab1:
```

## Analytics

Around:

```python
with tab2:
```

## Schedule

Around:

```python
with tab3:
```

## Next Race

Around:

```python
with tab4:
```

## Data & Debug

Around:

```python
with tab6:
```

For exact behavior always search the live branch source rather than relying on these approximate line numbers, because the file is actively developed.

---

# 50. Current Streamlit Schedule Column Config

Exact reference labels from `schedule_columns_to_display`:

```python
'round'          -> "Round"
'fullName'       -> "Name"
'date'           -> "Date"
'time'           -> "Time"
'courseLength'   -> "Lap Length (km)"
'laps'           -> "Number of Laps"
'turns'          -> "Number of Turns"
'distance'       -> "Distance (km)"
'totalRacesHeld' -> "Races Held"
'circuitType'    -> "Type"
```

This was already corrected in React.

---

# 51. Current Streamlit Next Race Column Config

Exact reference labels:

```python
'date'         -> "Date"
'time'         -> "Time"
'fullName'     -> "Grand Prix"
'courseLength' -> "Lap Length (km)"
'turns'        -> "Number of Turns"
'laps'         -> "Number of Laps"
```

Already corrected in React.

---

# 52. Troubleshooting CI

If a new parity run appears stuck:

1. Check whether it is waiting on backend/frontend `needs`.
2. Confirm no newer run canceled it via concurrency.
3. Inspect workflow run jobs.
4. If visual job ran, always check artifact presence even if job conclusion is failure.
5. The artifact upload is intentionally `if: always()`.

The workflow should no longer lose visual evidence solely because one evidence stage failed.

---

# 53. If Backend CI Fails on the Handoff Head

Because the latest backend enrichment was added shortly before handoff, if run #115 fails backend, inspect first:

```text
fastapi_react/backend/app/services/data.py
fastapi_react/backend/test_api.py
```

Likely categories:

- Ruff formatting/style
- mypy inference around Pandas mappings
- schema regression test expectation
- unexpected absent standings key

Do not revert the standings enrichment simply to get tests green; fix the implementation/type issue because the enrichment reflects actual Streamlit behavior.

---

# 54. If Frontend CI Fails in a Later Run

The handoff-head frontend job was already green.

If future CSS/JS edits break it, likely checks are:

- ESLint JSX-a11y
- TypeScript `checkJs`
- Vitest line coverage >=80%
- Vite build

Keep new feature branches covered by tests rather than lowering coverage.

---

# 55. Final Acceptance Definition

The migration should be considered complete only when all of the following are true:

- seven top-level sections match current Streamlit behavior
- shared filtering matches Streamlit
- Data Explorer output is not unintentionally truncated
- Analytics major content/data series match
- Current Season matches
- Next Race prediction/DNF/safety-car/etc. blocks match
- Predictive Models major tabs and artifacts match
- Data & Debug matches
- Betting Research matches
- backend code-quality gates pass
- frontend code-quality gates pass
- frontend line coverage >=80%
- accessibility audit reports zero violations
- operational benchmark succeeds
- React capture succeeds
- local Streamlit capture succeeds
- every visual screenshot pair is within configured tolerance
- `PARITY_CHECKLIST.md` is updated honestly
- `PARITY_REPORT.md` records actual evidence
- PR is ready for review
- no merge occurs without explicit user direction

---

# 56. Recommended Immediate Command/Tool Sequence for the Next Agent

If working through the GitHub connector:

1. Fetch PR #128.
2. Read current head SHA.
3. Fetch workflow runs for that head.
4. Find `FastAPI + React parity checks`.
5. Fetch its jobs.
6. If completed:
   - fetch visual job logs
   - fetch workflow artifacts
   - download `fastapi-react-parity-evidence`
7. Inspect:
   - `accessibility.json`
   - `benchmarks.json`
   - `visual/diff/summary.json`
8. For each failed visual pair:
   - compare React image
   - compare Streamlit image
   - compare diff image
9. Patch the earliest real divergence.
10. Commit.
11. Let concurrency cancel obsolete visual runs.
12. Repeat.

---

# 57. What Not to Do

Do **not**:

- rewrite the entire migration
- start a second React app
- replace FastAPI
- alter the authoritative data-generation pipeline
- remove source-parity quirks because they look ugly
- delete `Tab 6 START` on aesthetic grounds
- simplify Streamlit-derived filters back to raw CSV only
- switch back to root-only safety-car model lookup
- weaken accessibility
- lower coverage
- loosen visual tolerance
- increase pixel neighborhood arbitrarily
- mark stale checklist items complete without evidence
- mark the PR ready simply because unit tests pass
- merge the PR without explicit instruction

---

# 58. Suggested Final PR Review Checklist

Before marking ready:

```text
[ ] Current head CI fully green
[ ] visual diff all 24 pairs within tolerance
[ ] accessibility violation_count == 0
[ ] backend coverage >= 80%
[ ] frontend line coverage >= 80%
[ ] dependency/security workflows green
[ ] Streamlit compatibility workflow green
[ ] F1Bet release gates green
[ ] Data Explorer full row-count parity checked
[ ] PARITY_CHECKLIST rewritten/current
[ ] PARITY_REPORT rewritten/current
[ ] PR body reflects actual completed work/evidence
[ ] no debug-only accidental code in React/FastAPI
[ ] no request-time training added
[ ] PR remains unmerged until user explicitly requests merge
```

---

# 59. Handoff Bottom Line

The project is in **late-stage parity verification**, not early migration.

The key engineering work is already present.

The last completed evidence run proved:

- functionality and tests were strong
- accessibility reached zero violations
- benchmark worked
- local Streamlit capture worked
- screenshot evidence pipeline worked

It also identified the real design defects.

Those major defects were then addressed:

- Streamlit standings-backed sidebar filters
- sidebar narrow-screen layout
- sidebar control width
- Schedule labels
- Next Race labels
- tab styling
- tab scrolling
- table header styling
- subheader sizing
- checkbox geometry
- one-pixel raster handling

Run #115 has already been inspected. The newest local visual run is recorded
in `fastapi_react/parity_evidence/visual/local-validation-2026-10-01-final/`;
its 24 pairs still exceed tolerance, and it is from the separate dirty local
branch rather than PR #128.

Do not restart the audit from zero.

Do not treat local evidence as a result for the PR head. Do not declare
victory until the PR-head visual evidence and all other gates pass.

---

# 60. Current Snapshot Reference

At the moment this handoff was generated:

```text
Repository: gmalbert/f1Analysis
PR: #128
Branch: chatgpt/fastapi-react-parity
Head: 124c1a7b2d0bd5d9ff8b43ef84663f83dc66f92e
PR state: open / draft

Dependency and Code Security #147: success
Streamlit API Compatibility #136: success
F1Bet Offline Release Gates #129: success
FastAPI + React parity checks #115: completed, failure

Frontend parity job: success
Backend parity job: success
Visual evidence job: failed final evidence enforcement
```

Run #115 completed with 6/24 visual pairs passing, 18/24 failing, and
color-contrast violations on six sections. PR #128 remains open and draft.

The local workspace is separate: branch `react/updates-to-design`, HEAD
`31d790c8bb7dd8a84392adcc49d6bfa14fafedbc`, dirty worktree. Local apps are
React `http://127.0.0.1:5174`, FastAPI `http://127.0.0.1:8000`, and Streamlit
`http://127.0.0.1:8502`. Local axe reports zero violations. The latest local
raw-table comparison uses the checked table on desktop/tablet and matching
unchecked states on mobile; it measures 11.19%, 21.47%, and 19.26%, respectively.
The local screenshot run has 0/24 pairs within tolerance. These local results
do not certify the PR head.

The preceding status block is historical; refresh live checks before any PR
state change.
