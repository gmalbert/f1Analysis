"""Assemble the enhancement guide, complete code appendices, and source inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
documents: dict[str, str] = {}

documents["README.md"] = r"""
# F1 Analysis enhancement guide

Prepared October 1–2, 2026. This package contains **17 proposed enhancements**, eight actual browser screenshots, complete implementation files, and executable verification scripts. The proposals were integrated and tested in an isolated copy of the application. They have **not been enabled in the main application**.

The earlier raw-data optimization is already implemented in the working tree. Its measured response-header wait fell from a median **8.148 seconds to 1.982 seconds**, and the table retained all **4,629 rows and 561 columns**. Gzip transfer increased approximately 5.3% with the faster compression setting. See the [raw-data optimization report](../../parity_evidence/performance-2026-10-01/RAW_DATA_OPTIMIZATION.md) for measurement conditions, samples, and the tradeoff. Those measurements describe that completed change, not these new proposals.

## Read the guide

| Document | Contents |
| --- | --- |
| [01 — Design](01_DESIGN.md) | Readability, responsive layout, controls, loading, and before/proposed screenshots |
| [02 — Features](02_FEATURES.md) | Saved views, links, section search, accessible tables, comparisons, and reproducibility exports |
| [03 — Backend](03_BACKEND.md) | Cache identity and limits, timing, research job isolation, and request limits |
| [04 — Frontend implementation](04_FRONTEND_IMPLEMENTATION.md) | Copy map, activation, and every frontend source file in full |
| [05 — Backend implementation](05_BACKEND_IMPLEMENTATION.md) | Copy map, API contracts, flags, and every backend source file in full |
| [06 — Deployment and validation](06_DEPLOYMENT_AND_VALIDATION.md) | Asset savings, build budget, full deployment files, measured checks, rollout, and rollback |
| [07 — Verification source](07_VERIFICATION_CODE.md) | Complete test files, preview generator, browser screenshot script, and API probe |

## Recommended order

Priority means implementation order, not a promise of performance gains. Some improvements intentionally change the appearance of the parity site; keep them under the optional enhancement profile so the original display remains available.

| ID | Priority | Proposal | Code entry point |
| --- | --- | --- | --- |
| D1 | First | Improve contrast, focus, and control target sizes | §enhancements.css§ |
| D2 | Next | Make the header, navigation, and mobile spacing more compact | §enhancements.css§ |
| D3 | First | Keep table tools visible and usable | §enhancements.css§, §EnhancedTable.jsx§ |
| D4 | First | Show understandable loading and failure states; cancel stale requests | §FeatureBar.jsx§, §viewClient.js§, §SafePlotlyChart.jsx§ |
| F1 | Next | Save named analysis views locally | §preferences.js§, §FeatureBar.jsx§ |
| F2 | Next | Share a link that restores safe analysis settings | §preferences.js§, proposed §App.jsx§ |
| F3 | Later | Search and open sections with a keyboard command palette | §FeatureBar.jsx§ |
| F4 | First | Offer a semantic HTML table alongside the canvas grid | §EnhancedTable.jsx§ |
| F5 | Later | Compare up to four drivers using descriptive historical data | §EnhancedTable.jsx§ |
| F6 | Next | Export analysis context and recorded model provenance | §FeatureBar.jsx§, §service.py§ |
| B1 | Next | Reuse bounded, short-lived view responses and deduplicate requests | §viewClient.js§, §cache.py§ |
| B2 | First | Invalidate caches when source artifacts change | §service.py§ |
| B3 | First | Add request timing, identifiers, and bounded diagnostic records | §metrics.py§, §logging.json§ |
| B4 | Later | Run explicit administrator research tasks in a separate process | §jobs.py§, §ResearchJobs.jsx§ |
| B5 | First | Bound aggregate request bodies before JSON parsing | §metrics.py§, §nginx.conf§ |
| O1 | First | Resize the footer image and serve efficient responsive assets | §optimize-assets.mjs§, proposed §App.jsx§ |
| O2 | First | Enforce a build budget and configure production HTTP caching | §check-budgets.mjs§, §vite.config.js§, §nginx.conf§ |

Start with D1/D3/D4/F4/B2/B3/B5/O1/O2, then add B1 together with B2, then the analysis workflow features. B4 should remain a separate decision because it changes resource use and administrator operations. The supplied integrated preview includes all proposals behind flags to make that decision concrete and reviewable.

## What was verified

- Full integrated backend suite: **66 passed**, **87.90% coverage**; existing 80% threshold retained.
- Full integrated frontend suite: **63 passed**, **73.78% statement/line coverage**; existing thresholds retained.
- Backend Python compilation, Ruff, and strict mypy; frontend ESLint and TypeScript checks.
- Production Vite build and a real gzip main-entry budget check.
- Five Playwright interaction flows, desktop/mobile captures, and **zero captured page errors, console errors, or failed HTTP responses** in those flows.
- Real API cache hit, request-timing headers, guarded administrator routes, and the unchanged raw-table checksum.

The included [validation JSON files](06_DEPLOYMENT_AND_VALIDATION.md#recorded-results) preserve the browser, API, bundle, and quality-check evidence. Browser checks cover the flows listed in the validation chapter; they are not a complete accessibility audit or a production load test.

## Review boundary

The full replacement files are tied to the repository state recorded in [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json). Review a diff before installing them over a newer checkout. Existing computations, model inputs, displayed data, CSV contracts, and all-field raw responses stay in the original backend services. No new package dependency is needed by the supplied implementation.

Saved and shared settings use a narrow allowlist. Uploaded CSV bodies, betting inputs, administrator tokens, and ledgers are excluded. Share links are readable encodings, not encrypted secrets. Proposed jobs and metrics require an administrator token; existing direct research endpoints retain their current authorization behavior.

The production Nginx configuration is supplied for review and deployment; it was not run against a production server. Long-term durable jobs, multi-instance coordination, and new trained probability models are outside this implementation. The existing model manifest's calibration limitations remain visible in provenance exports.
"""

documents["01_DESIGN.md"] = r"""
# Design proposals

The reference Streamlit layout remains the parity baseline. These proposals are optional changes to improve everyday use after conversion. The generated frontend exposes an **Analysis tools** drawer when §VITE_F1_ENHANCEMENTS=1§. **Improve readability** defaults off; enabling it applies §data-enhancements="on"§ to the document root. Number and word typography continues to use Source Sans Pro, and years continue to display without thousands separators.

## D1 — Contrast, keyboard focus, and target sizes

**Reason.** The existing [accessibility evidence](../../parity_evidence/accessibility.json) recorded insufficient contrast for some captions and active navigation text. Pale secondary text makes dense numerical analysis harder to read.

**Behavior.** Remove reduced caption opacity. Use light-theme accent §#b4232d§ and muted text §#596273§, and dark-theme accent §#ffb4ab§ and muted text §#c5cbd7§. Add visible three-pixel focus outlines. Use 44-pixel minimum toolbar/button targets where this stylesheet controls them; preserve compact table data cells.

**Implementation.** [enhancements.css](code/frontend/enhancements.css), imported after the parity stylesheet in the supplied [main.jsx](code/frontend/main.jsx). Theme selectors distinguish light and dark modes and override the existing theme variables with sufficient specificity.

**Acceptance.** Check focus order, visible outlines, captions, nav selection, disabled controls, and both themes. Run the existing axe script after integration. WCAG AA generally requires 4.5:1 for ordinary text; 44-pixel targets are an additional usability choice, not a claim about the AA target-size requirement. The preview is not certified as WCAG conformant. [W3C contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

## D2 — Compact branding and responsive navigation

**Reason.** Large branding and top spacing postpone the first useful table, especially on a phone. Dense horizontal tools also need deliberate mobile behavior.

**Behavior.** Keep the same branding image but cap its width at 280 pixels on desktop and 210 pixels on mobile. Use 40-pixel desktop top padding and 56 pixels on mobile, a 30-pixel mobile title, sticky section navigation, and narrower mobile sidebar spacing. Keep the same headings, filters, data, and theme.

**Implementation.** The optional CSS profile in [enhancements.css](code/frontend/enhancements.css). The profile does not rewrite the source view tree or change the analysis.

**Acceptance.** At 1280×900 and 390×844, ensure the title, navigation, filter controls, tables, and sidebar remain reachable without page-level horizontal overflow. Test both themes and zoom separately. Sticky controls reduce usable vertical space on small screens; disabling the readability profile restores the base layout.

## Actual desktop screenshots

The “current” images use the existing production bundle in a temporary local static preview. The “proposed” images use the separately built enhancement preview. Both are real Chromium captures, not generated mockups. The screenshot script proxies the same local API, but the views shown are examples rather than a pixel-diff parity test.

Existing desktop:

![Existing React desktop layout](images/current-desktop.png)

Proposed desktop, with readability enabled:

![Proposed React desktop layout](images/proposed-desktop.png)

## Actual mobile screenshots

Existing mobile:

![Existing React mobile layout](images/current-mobile.png)

Proposed mobile, with readability enabled:

![Proposed React mobile layout](images/proposed-mobile.png)

## D3 — Visible table and chart controls

**Reason.** Hover-only controls are difficult to discover and unreliable for touch and keyboard use. Users need to locate search, column selection, export, and display controls before interacting with a dense table.

**Behavior.** The profile makes existing table toolbars visible, increases control targets, and positions the column picker within the table area. The additional display selector offers **Data grid** and **Accessible table**. The grid remains the default, with its existing sorting, pinning, selection/copy, numeric formatting, fullscreen, and CSV export.

**Implementation.** [enhancements.css](code/frontend/enhancements.css) and [EnhancedTable.jsx](code/frontend/EnhancedTable.jsx). The original §ViewTable§ supplies the grid behavior; the new component delegates to it unless the semantic mode is selected.

**Acceptance.** Open/close the column picker by keyboard and touch; check it does not obscure unrelated content. Confirm the original grid's exports and interactions still work. The semantic view provides its own search, columns, and paging, while the grid retains the richer spreadsheet-style controls.

## D4 — Loading, errors, and asynchronous chart cleanup

**Reason.** A blank or frozen-looking panel gives no indication that a large analysis is still running. Changing filters quickly can also deliver obsolete responses after the latest request.

**Behavior.** Display a polite loading status outside the §main[aria-busy]§ region. Show elapsed seconds visually without announcing every increment. Keep a readable request failure state. A generation guard prevents stale rendering; AbortController releases requests no longer used by the current view. Default analysis requests time out after 120 seconds, action requests after 600 seconds.

**Implementation.** [LoadingFeedback in FeatureBar.jsx](code/frontend/FeatureBar.jsx), [viewClient.js](code/frontend/viewClient.js), and the effect cleanup in the supplied [App.jsx](code/frontend/App.jsx). [SafePlotlyChart.jsx](code/frontend/SafePlotlyChart.jsx) catches asynchronous chart errors, observes container resizing, and purges charts during disposal.

**Acceptance.** Rapidly change sections/filters, interrupt a slow request, and simulate an HTTP failure. Confirm the latest view wins and no stale chart updates occur after unmounting. Browser cancellation does not preempt Python already executing on the server. Timeouts are meaningful UI feedback, not server-side execution limits. See [React effect cleanup](https://react.dev/reference/react/useEffect) and [MDN AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController).

## Design integration and rollback

Use the [frontend copy map](04_FRONTEND_IMPLEMENTATION.md#copy-map) and review the complete files below it. Set §VITE_F1_ENHANCEMENTS=1§ at build time to expose the tools. The profile remains a user choice. To return to the base interface, disable the readability checkbox; to remove the optional tools, rebuild with §VITE_F1_ENHANCEMENTS=0§. Asset optimization and lifecycle cleanup remain in the candidate files even with the UI flag off.
"""

documents["02_FEATURES.md"] = r"""
# Feature proposals

All features in this chapter have complete code in the [frontend appendix](04_FRONTEND_IMPLEMENTATION.md). The prototype uses the existing declarative views and calculations. It does not add new predictions, change the raw-data schema, or change current CSV formats.

## F1 — Named saved views

**User flow.** Open **Analysis tools**, enter a name, and save the current section and safe filter settings. Select a saved view to restore it, or delete it. Saving the same name replaces that entry.

**Implementation.** §preferences.js§ uses versioned §f1analysis.saved-views.v1§ local storage, at most 20 entries, and names limited to 80 characters. §FeatureBar.jsx§ provides the controls; the supplied §App.jsx§ restores the state and navigation together.

**Limits and checks.** Views belong to the current browser, not an account, and may disappear when browser storage is cleared. Uploaded data and betting/financial settings are excluded. Storage errors produce feedback rather than losing the current analysis. Tests cover validation and restoration; also check persistence after a normal reload and invalid/corrupt storage.

## F2 — Shareable analysis links

**User flow.** Choose **Copy link** to share a section with its selected safe settings. Opening it restores the encoded view before the initial request.

**Implementation.** A version-1 JSON object is encoded as Unicode-safe base64url in the hash query. Allowed values are §filter_results_main§, filter/range/checkbox keys, §_tabs:*§, model selection, and tire year/race selectors. Primitive arrays are limited to 20 items. The link token is limited to 6,000 characters. Invalid links fall back safely in the application.

**Limits and checks.** The token is readable; users should share only settings they intend to disclose. It omits uploads, ledgers, betting amounts, and administrator tokens. It records settings, not a frozen copy of the dataset: opening it against changed data can produce changed results. Test Unicode section/filter values, excluded keys, malformed tokens, old versions, oversized links, and reload behavior.

## F3 — Section search

**User flow.** Press Ctrl+K or Cmd+K, type part of a section name, and open a matching section. Escape closes the dialog.

**Implementation.** §FeatureBar.jsx§ uses the native §dialog§ element, filtered section buttons, and an Enter action. This searches section names, not every data value or chart label.

**Limits and checks.** Native focus handling supports modal behavior, but keyboard focus return and assistive technology behavior should still be checked in target browsers. Avoid overriding shortcuts while the dialog is being dismissed. The browser check covers searching and opening Predictive Models. [MDN dialog documentation](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/dialog).

![Section search in the proposed preview](images/proposed-command-palette.png)

## F4 — Semantic table mode

**User flow.** Choose **Accessible table** above a data grid. Search the table, choose columns, and move between 50-row pages. Switch to **Data grid** for the original spreadsheet-style interaction.

**Implementation.** §EnhancedTable.jsx§ renders a real §table§, caption, column headers, and body cells. It starts with eight visible columns to keep a phone-sized table manageable. All supplied columns remain selectable, and search covers all fields in the underlying rows.

**Data guarantee.** Paging changes the displayed slice, not the API response or stored rows. The raw payload still contains 4,629×561 cells. Year formatting keeps years without grouping; numeric and textual cells inherit the same font. Existing column formatting metadata remains in use.

**Limits and checks.** Semantic mode does not duplicate every sorting/pinning feature of the grid. Wide column selections still need horizontal scrolling within the table. An all-field client search across a large table consumes CPU; assess it on the target phone hardware. Check captions, headers, page totals, empty searches, column selection, and screen-reader navigation. [W3C table guidance](https://www.w3.org/WAI/tutorials/tables/).

![Semantic table and paging controls](images/proposed-accessible-table.png)

## F5 — Historical driver comparison

**User flow.** In a suitable driver table, open the comparison controls and select up to four drivers. Compare average start, finish, position gain, and DNF percentage over the current supplied table.

**Implementation.** §EnhancedTable.jsx§ detects supported driver columns and calculates descriptive summaries from existing rows. It reports row sample counts and uses only known finish/status rows for the relevant denominators.

**Limits and checks.** The input may contain multiple rows per race, so the sample count is a row count, not a guaranteed unique-race count. The comparison reflects the current filtered table and any missing data. It is historical description, not a forecast, model confidence interval, or calibrated betting probability. Check empty selections, missing fields, unknown finish status, the four-driver cap, and deterministic results for a fixed table.

![Historical driver comparison example](images/proposed-driver-comparison.png)

## F6 — Reproducibility context and printing

**User flow.** Choose **Export context** to download JSON describing the current analysis settings and source provenance. Existing CSV and chart downloads stay available. **Print current view** opens the browser print flow.

**Implementation.** §FeatureBar.jsx§ requests §GET /api/enhancements/status§ and exports safe settings, section/page, UTC export time, artifact revision, build revision, dataset modification time, and selected recorded model-manifest fields. Print styling hides unnecessary controls.

**Limits and checks.** The revision is based on file metadata rather than a content checksum; model fields are recorded manifest provenance, not independent verification of current model quality. Existing manifest notes about legacy finishing-position models and absent probability calibration must be preserved. Printing the semantic table prints its current page, not every raw row. Exporting context does not replace or reformat an existing CSV contract.

**Acceptance.** Validate the downloaded JSON schema, safe-key exclusion, model notes, revision and UTC time. Verify original CSV exports separately. The supplied browser test downloads and inspects the JSON.

![Saved views, links, context export, and optional display settings](images/proposed-analysis-tools.png)

## Feature defaults

Set §VITE_F1_ENHANCEMENTS=1§ when building to expose these tools. The drawer starts collapsed; semantic mode is opt-in per table; readability and response caching default off. Named views are local to a browser. Section settings stored during the enhancement mode use the safe-key allowlist, so private uploaded content is not retained through that mechanism.

Research jobs are a separate optional administrator feature described in [B4](03_BACKEND.md#b4--isolated-local-research-jobs). They require the backend flag and a token; the token field never writes the token to saved views, share links, or browser storage.
"""

documents["03_BACKEND.md"] = r"""
# Backend proposals

The completed serialization/compression change is the first performance improvement. These additional proposals focus on repeated work, stale-data correctness, operational visibility, and isolation of explicit research actions. The complete implementation appears in [05 — Backend implementation](05_BACKEND_IMPLEMENTATION.md).

## B1 — Bounded response reuse and request deduplication

**Reason.** Revisiting an unchanged section can repeat full Python rendering, JSON serialization, and gzip compression. A browser can also issue duplicate requests for the same view while effects are mounting.

**Server behavior.** With §F1_ENHANCEMENTS=1§ and §F1_VIEW_RESPONSE_CACHE=1§, reuse only pages 1–5 with no action or uploaded/private structured values. Cache keys contain the complete values, page, and source revision. Retain at most 12 entries or 64 MiB across plain and gzip response bytes, with a 20-second TTL. Precompress gzip once per miss at level five; handle §gzip;q=0§ correctly.

**Browser behavior.** The optional cache setting enables §viewClient.js§. It first requests the current revision on every reusable navigation, then reuses a matching response for up to 15 seconds. Limit storage to six entries and 12,000,000 serialized bytes. Concurrent identical reusable requests share one fetch; aborting one subscriber does not cancel another active subscriber.

**Exclusions.** Raw Data/page 6, Betting Research/page 7, actions, uploads, CSV and ledger keys bypass reuse. Actions clear retained responses. Upload/private value changes clear the client cache. Server responses keep §Cache-Control: no-store§: this is explicit application reuse, not a shared browser/proxy HTTP cache.

**Limits.** Server state is process-local. The browser budget estimates serialized data, not actual JS heap. A miss still needs the full rendering computation. Python view rendering is already serialized and the candidate keeps a guarded revision/render boundary; this cache does not make misses parallel. No cold-start speedup or throughput improvement is claimed without a load test.

**Acceptance.** Same settings hit; changed settings miss; artifact revision invalidates; TTL/byte/entry limits evict; private values bypass; actions invalidate; response content matches. The real API probe observed a cache HIT. Measure first visit and repeated visit separately before changing defaults.

## B2 — Revision identity and source-cache invalidation

**Reason.** Caching a response safely also requires detecting new data/model artifacts. Existing data loaders and presentation caches can otherwise keep old contents.

**Behavior.** §artifact_revision§ scans eligible data/model and backend source paths, sizes, and nanosecond modification times. Poll at most once per second. When the fingerprint changes, clear presentation and data/analysis loader caches under the existing render lock, plus the response cache. Responses expose §X-F1-Revision§. The status route also exposes the revision for the client and context exports.

**Limits.** This is a metadata fingerprint, not a cryptographic content identity, despite using SHA-256 to summarize the inventory. Replacing content while deliberately retaining identical size/mtime can defeat it. Publish artifacts atomically with a changed mtime. Restart workers after code changes: fingerprinting Python files does not reload an already compiled view module. A file modified during a render can still require an operationally coordinated publish; the scan is not a database snapshot.

**Acceptance.** Change a sample artifact and confirm status revision and both cache layers change. Test CSV fallback and missing/unreadable manifests. Model manifests are exported as recorded provenance; recalculating every dataset/model content hash on every navigation would reintroduce avoidable work.

## B3 — Request timing, IDs, and bounded diagnostics

**Behavior.** Pure ASGI middleware attaches §X-Request-ID§ and §Server-Timing: backend;dur=...§ to responses. Retain 500 recent records containing route, status, time, and response-body bytes. Use structured request log lines and a token-protected metrics endpoint.

**Scope.** Middleware is installed outside the gzip layer, so header timing includes application processing and compression until response headers are sent. It excludes the network/browser and is not a breakdown of dataset load versus chart rendering versus JSON serialization. Byte counts describe emitted response bodies; on gzip responses these are compressed bytes.

**Privacy and logging.** Records do not include query strings, request bodies, headers, or tokens. Use the supplied §logging.json§ through Uvicorn's §--log-config§ option to enable INFO-level structured logs. Worker exception logs contain tracebacks and need normal server-log access controls.

**Acceptance.** Check the headers on success/failure, bounded record retention, request-ID uniqueness, and token restrictions. Detailed metrics require §F1_ADMIN_TOKEN§; status metadata remains a public local API response. See [MDN Server-Timing](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Server-Timing) and [Starlette middleware](https://starlette.dev/middleware/).

## B4 — Isolated local research jobs

**Reason.** An explicit audit or model bin comparison can run much longer than normal navigation. Executing it in the web process can block other analyses behind the same rendering lock.

**Behavior.** A one-thread coordinator submits work to a separate spawned calculation process with one worker. Expose only the existing **Bin Count Comparison** and **Temporal Leakage Audit** actions. Defaults are q=2 and 1,000 audit rows. The worker dispatches the original presentation action; it does not turn on general §F1_RESEARCH_MODE§ or automatic training.

**Queue contract.** Retain at most eight jobs, input JSON below 64 KiB, plain results below 32 MiB, and compressed results for ten minutes after completion. Reject uploaded CSV/ledger/private structured inputs. Jobs pass through queued/running/succeeded/failed states. Queued jobs can be cancelled; a running calculation finishes. Status/result/delete requests require the same administrator token as submission.

**Correctness.** Pin the artifact revision at submission and check it in the worker. Changed artifacts cause failure and require resubmission. Clear worker source caches before executing. Bin q values must be integers 2–10, no more than nine values; audit rows must be 0–100,000, where zero means all source rows.

**Operational limits.** Use **one API worker and one instance** for this local implementation so polling reaches the process holding the job IDs. Jobs and results disappear on restart. The separate worker duplicates dataset/model memory. Graceful shutdown waits for running work; it does not provide a hard stop deadline. This is a bounded local queue, not a durable distributed worker system.

**Authorization boundary.** §F1_ADMIN_TOKEN§ protects only the new job and metrics endpoints. Existing direct research actions retain their current behavior. The token input is held in React memory and never persisted; serve an authenticated/encrypted origin before exposing administrative controls outside a trusted local setup.

**Acceptance.** Unit checks exercise the actual spawned queue with a tiny importable test worker, successful results, failure, queue bounds, and cancellation. Dispatcher checks mock the expensive source actions and confirm the two allowed operations. No real retraining/bin experiment was executed for this proposal. Test the actual calculation separately in a controlled session before using it for a long run. [Python executor documentation](https://docs.python.org/3.13/library/concurrent.futures.html).

## B5 — Aggregate request-size limit

**Behavior.** With the backend enhancement flag enabled, the ASGI body limiter rejects request bodies exceeding 256 MiB before FastAPI JSON parsing. §F1_MAX_REQUEST_BYTES§ changes the limit. The supplied Nginx configuration uses the aligned §client_max_body_size 256m§ proxy cap.

**Compatibility.** The current frontend's individual-file 200 MB limit is unchanged. Several files or JSON escaping overhead can exceed the aggregate API limit even when each file individually passes the UI check. A rejected oversized body returns 413.

**Limits.** This buffers accepted bodies before parsing; choose a lower cap if the service memory cannot accommodate it. It protects the configured API process, not a complete ingress-denial-of-service strategy. Normal private uploads still bypass application caches.

**Acceptance.** Check accepted small bodies, over-limit payloads with and without Content-Length, and preserved request bytes. The module contract tests exercise the body limiter and the integrated service route tests cover its installation.

## Request architecture

§§§mermaid
flowchart TD
    Browser["React view client"] --> Status["Source revision status"]
    Browser --> Metrics["Timing and request-size middleware"]
    Metrics --> Cache["Optional bounded response cache"]
    Cache --> Render["Existing serialized view renderer"]
    Render --> Sources["Existing data and model artifacts"]
    Status --> Sources
    Admin["Administrator job controls"] --> Queue["Token-protected local queue"]
    Queue --> Process["Separate spawned calculation process"]
    Process --> Sources
§§§

The cache accelerates reuse; it does not reduce the underlying raw response fields. The isolated process removes research computation from the web process's rendering lock, while OS CPU/RAM remain shared resources.
"""

documents["04_FRONTEND_IMPLEMENTATION.md"] = r"""
# Complete frontend implementation

These are complete source files, not pseudocode. They were integrated in the isolated preview and passed the build, ESLint, type checks, the 63-test frontend suite, and the browser checks. Use a review branch and compare against [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) before replacing files in a checkout that has moved on.

## Copy map

Paths on the right are relative to §fastapi_react/frontend/§.

| Supplied source | Destination |
| --- | --- |
| §code/frontend/App.jsx§ | §src/App.jsx§ |
| §code/frontend/App.test.jsx§ | §src/App.test.jsx§ |
| §code/frontend/main.jsx§ | §src/main.jsx§ |
| §code/frontend/Presentation.jsx§ | §src/components/Presentation.jsx§ |
| §preferences.js§, §viewClient.js§, §FeatureBar.jsx§, §EnhancedTable.jsx§, §SafePlotlyChart.jsx§, §ResearchJobs.jsx§, §enhancements.css§, §enhancements-env.d.ts§, §Enhancements.test.jsx§ | Corresponding files under §src/enhancements/§ |
| §code/deployment/optimize-assets.mjs§ | §scripts/optimize-assets.mjs§ |
| §code/deployment/check-budgets.mjs§ | §scripts/check-budgets.mjs§ |
| §code/deployment/vite.config.js§ | §vite.config.js§ |
| §code/deployment/package.json§ | §package.json§ |

The supplied package file retains the existing dependency list and adds asset optimization before build and budget enforcement after build. Keep the current lockfile; these changes do not introduce a new dependency. The existing Glide patch postinstall script remains.

## Integration behavior

1. §main.jsx§ imports the enhancement CSS after the current parity CSS.
2. §App.jsx§ mounts optional tools, restores safe shared settings, uses cancellable requests, adds the loading status, and uses the responsive footer image with the original PNG fallback.
3. §Presentation.jsx§ routes table nodes through §EnhancedTable§ and Plotly nodes through §SafePlotlyChart§. The original table grid remains available.
4. §FeatureBar.jsx§ owns saved-view/link/context/command controls and persisted nonprivate options.
5. §viewClient.js§ defaults to uncached requests. Its reuse mode requires the backend status route from the backend integration.
6. §App.test.jsx§ adjusts the existing test mock to the new request boundary. §Enhancements.test.jsx§ adds new contracts; it does not remove the existing suite.

## Activation

From a normal frontend checkout with dependencies installed:

§§§powershell
$env:VITE_F1_ENHANCEMENTS = '1'
npm run lint
npm run typecheck
npm test
npm run build
§§§

Vite embeds this flag at build time; changing the server environment after building will not toggle the compiled bundle. **Improve readability** and **Reuse recent views** are off by default. The backend enhancement status route must be installed before enabling client cache/context export or local jobs. For production, serve the built §dist§ directory using the configuration in the deployment chapter.

To disable optional controls, rebuild with §VITE_F1_ENHANCEMENTS=0§. The candidate's request cleanup, safer Plotly lifecycle, and optimized footer loading are present regardless of this UI flag. Restore the prior tracked files to revert those implementation changes as well.

## Full source

Every source file below also exists separately in [code/frontend](code/frontend). Tests are included so installation retains useful checks. Deployment script/configuration source is in the [deployment chapter](06_DEPLOYMENT_AND_VALIDATION.md).
"""

documents["05_BACKEND_IMPLEMENTATION.md"] = r"""
# Complete backend implementation

All source files below are complete, tested candidate files. The proposed §main.py§ preserves existing routes and adds flagged enhancement integration, including clean job-executor shutdown. Existing calculations remain in the current services.

## Copy map

Paths on the right are relative to §fastapi_react/backend/§.

| Supplied source | Destination |
| --- | --- |
| §code/backend/main.py§ | §app/main.py§ |
| §code/backend/__init__.py§, §cache.py§, §metrics.py§, §jobs.py§, §service.py§ | Corresponding files under §app/enhancements/§ |
| §code/backend/testing_worker.py§ | §app/enhancements/testing_worker.py§ for tests only |
| §code/backend/test_enhancements.py§ | §test_enhancements.py§ |
| §code/deployment/logging.json§ | §logging.json§ |

Do not copy the generated §main.py§ or §test_enhancements.py§ inside the enhancements package. The small test worker is required to exercise Windows spawned-process jobs in tests; production routes never dispatch it.

## Flags

| Environment variable | Default | Purpose |
| --- | --- | --- |
| §F1_ENHANCEMENTS§ | §0§ | Install status/jobs/metrics routes, middleware, revision management, and the alternate view response path |
| §F1_VIEW_RESPONSE_CACHE§ | §0§ | Enable B1 server reuse when enhancements are installed |
| §F1_MAX_REQUEST_BYTES§ | §268435456§ | Aggregate accepted request-body limit in bytes |
| §F1_ADMIN_TOKEN§ | Unset | Enable/authorize new local jobs and metrics; unset returns 503 |
| §F1_BUILD_REVISION§ | §local-working-tree§ | Source revision label in context export |
| §F1_REPO_ROOT§ | Existing config default | Override repository path for an isolated/nested preview |

Do not enable §F1_RESEARCH_MODE§ merely to use these two new explicit task routes. The proposal retains the existing general research-mode setting and dispatches only the operations documented in B4.

## Start a local integrated checkout

Run from §fastapi_react/backend§, with the application's existing data/model artifacts available:

§§§powershell
$env:F1_ENHANCEMENTS = '1'
$env:F1_VIEW_RESPONSE_CACHE = '1'
$env:F1_BUILD_REVISION = (git rev-parse HEAD)
../../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --log-config logging.json
§§§

To test administrator routes locally, provide a freshly generated administrator token through the process environment, then enter it only in the preview password field. Keep it out of command history, checked-in files, share URLs, and diagnostics. The complete package functions without an administrator token: only the new metrics/jobs operations remain disabled.

## API contracts

All paths below are under §/api/enhancements§.

| Method/path | Request/result |
| --- | --- |
| §GET /status§ | Public revision, build revision, dataset name/time, recorded model-manifest metadata |
| §GET /metrics§ | Admin header required; recent records and cache byte count |
| §POST /jobs§ | Admin header; §{"task":"leakage-audit","values":{}}§ or §bin-comparison§; 202 with job ID |
| §GET /jobs/{id}§ | Admin header; current state and timestamps |
| §GET /jobs/{id}/result§ | Admin header; original view-node result on success |
| §DELETE /jobs/{id}§ | Admin header; §{"cancelled":true/false}§; queued jobs only |

The header name is §X-F1-Admin-Token§. Wrong/missing tokens return 403 when a token is configured; disabled admin operations return 503. Unknown jobs return 404, unsupported/invalid inputs 400, full queue 429, and unfinished/failed result requests 409. Results expire after ten minutes and restart loses job IDs.

§POST /api/views§ retains its original request contract. It adds revision/cache headers on the flagged path and remains the source of all complete view tables.

## Backend verification

From the backend directory:

§§§powershell
$env:F1_ENHANCEMENTS = '0'
../../.venv/Scripts/python.exe -m compileall -q app
../../.venv/Scripts/python.exe -m ruff check app test_enhancements.py
../../.venv/Scripts/python.exe -m mypy app
../../.venv/Scripts/python.exe -m pytest
§§§

The existing suite is run with the global flag off to preserve baseline route expectations; the new service tests instantiate and exercise the enhancement integration directly. Follow with a real API probe against a process started with enhancements on.

## Full source

Every file below also exists separately in [code/backend](code/backend). The code uses the existing installed FastAPI/Starlette/Pydantic stack and Python standard library. It does not add a queue, Redis, or database dependency.
"""

documents["06_DEPLOYMENT_AND_VALIDATION.md"] = r"""
# Deployment, measurements, and validation

## O1 — Responsive footer assets

The existing footer PNG is roughly 1.13 MB although it displays at a small size. The supplied Sharp script creates resized losslessly encoded WebP files at 60- and 120-pixel heights, retaining the source PNG as fallback. The image uses width/height attributes, lazy loading, and asynchronous decoding.

The generated preview assets are **4,938 bytes at 1×** and **15,096 bytes at 2×**. This is a measured asset-size reduction, not a measured whole-page load-time improvement. Resizing intentionally reduces source resolution; “lossless” describes the encoding of the resized output. The visible branding source stays the same. [Sharp resize documentation](https://sharp.pixelplumbing.com/api-resize/).

Copy §optimize-assets.mjs§ into §frontend/scripts/§ and use the supplied package build scripts. It writes responsive assets into §public/§ before Vite copies them to §dist/§. The proposed §App.jsx§ references them with a PNG fallback.

## O2 — Build budget and production delivery

**Budget.** §check-budgets.mjs§ actually fails if the main entry exceeds 500,000 gzip bytes. The current Vite §chunkSizeWarningLimit§ is a warning, not a build failure. The supplied Vite config disables public sourcemaps; Nginx also denies §.map§ requests.

**Recorded bundle.** The final preview's main entry is **699,541 bytes plain / 226,969 bytes using the budget script's gzip settings**. All JavaScript chunks together are **2,015,252 gzip bytes**. Plotly alone is about **1.48 MB gzip**, loaded as a separate chunk. These are output artifact sizes; actual browser bandwidth depends on which routes/charts are visited, HTTP compression, cache state, and source maps. Vite's printed gzip estimate differs slightly because its compression settings differ.

**Serving policy.** Enable gzip level five, one-year immutable caching for hashed §/assets/§ files, revalidation for §index.html§, one-hour caching for unversioned images/fonts, no shared API caching, a 600-second API read timeout, and a 256 MiB aggregate ingress cap. The configuration proxies to §backend:8000§; adapt that hostname to the deployed service topology.

**Limits.** Nginx deployment was not executed here. The supplied port-80 configuration assumes the hosting platform or an upstream ingress terminates TLS; provide that before exposing administrator tokens. Keep a single API worker/instance if using the local job queue. Do not describe static gzip or response reuse as a proven production throughput gain without measurements. [Vite build documentation](https://vite.dev/guide/build.html), [Nginx gzip documentation](https://nginx.org/en/docs/http/ngx_http_gzip_module.html).

## Recorded results

| Check | Result | Evidence/meaning |
| --- | --- | --- |
| Integrated backend pytest | 66 passed, 87.90% coverage | Existing 80% minimum retained |
| Integrated frontend Vitest | 63 passed, 73.78% statement/line coverage | Existing coverage thresholds retained |
| Backend compilation, Ruff, mypy | Passed | Mypy checked 18 source files |
| Frontend ESLint and TypeScript | Passed | No warnings/errors under the existing configuration |
| Vite production build | Passed | 1,112 modules transformed; existing large lazy chunks produce a nonfatal size warning |
| Enforced main gzip budget | Passed | [validation-budgets.json](validation-budgets.json) |
| Browser flows/screenshots | Five flows passed; zero captured errors | [validation-browser.json](validation-browser.json) |
| Real API reuse/timing/auth/raw identity | Passed | [validation-api.json](validation-api.json) |
| Standalone module contracts | Four Python tests and Node contracts passed | Source in the verification chapter |

The [complete quality-check record](validation-quality.json) includes the test counts, coverage, lint/type results, and explicit §py_compile§ verification of 22 integrated Python files.

The browser flows cover semantic table paging/search, driver comparison, saved-view/cache controls, section search/navigation, and context JSON download. They capture desktop at 1280×900 and mobile at 390×844. Error collection includes uncaught page exceptions, console errors, and HTTP status codes of 400 or higher in the visited flows.

The API probe verifies a genuine §X-F1-Cache: HIT§, timing headers, guarded administrator routes, and the raw table's canonical content checksum:

§§§text
0389ff31e162ebc06710cbbfc77c9ea8d3028ecce30dd502f4de5f044598415e
§§§

This matches the prior optimization baseline across all 4,629 rows and 561 columns. It is a data-content check; the stat-based source revision used by the proposed cache has a different purpose.

One backend test warning comes from the installed Starlette/httpx test-client deprecation. The production build retains warnings for the existing large lazy Plotly/Vega chunks. Neither is a browser console failure; the explicit main-entry budget passes. A full accessibility audit, actual expensive research run, production deployment, and multi-user load benchmark remain unperformed.

## Recreate the isolated preview

The package includes a generator that produces full integrated replacements and copies them to §fastapi_react/.runtime/enhancement-preview/§. It checks integration anchors and stops if the baseline has changed unexpectedly. It does not edit main application source files. Generated full replacements in §code/§ are tied to that baseline.

From the repository root:

§§§powershell
.venv/Scripts/python.exe fastapi_react/enhancement_proposals/2026-10-01/prepare_preview.py
§§§

The generator copies application source, frontend configs/public assets/build scripts, and backend test configuration. It creates a node_modules junction to the main frontend's installed dependencies and refuses to replace an unexpected dependency path. Install the main frontend dependencies first. **Do not run §npm ci§ in this staging copy**, because it shares the main checkout's dependency tree. Use the installed CLI entry points, or use a separate ordinary checkout with its own node_modules for dependency installation.

Build from the staged frontend:

§§§powershell
$env:VITE_F1_ENHANCEMENTS = '1'
node scripts/optimize-assets.mjs
node node_modules/vite/bin/vite.js build --config vite.config.js
node scripts/check-budgets.mjs
§§§

The optimizer/budget scripts use the current working directory. Run them in the staged frontend so they write/read its §public§/§dist§. The deployment appendix contains the exact scripts.

Start the staged API in a separate terminal. From its §backend§ directory, set §F1_REPO_ROOT§ to the absolute main repository path, §F1_ENHANCEMENTS=1§, and §F1_VIEW_RESPONSE_CACHE=1§. Run the main repository's virtualenv Uvicorn on an unused port, for example 9008. Never stop a preexisting listener just to claim this port.

Then, from the main repository root:

§§§powershell
$env:PROPOSAL_API_PORT = '9008'
node fastapi_react/enhancement_proposals/2026-10-01/checks/browser.mjs
.venv/Scripts/python.exe fastapi_react/enhancement_proposals/2026-10-01/checks/api.py
node fastapi_react/enhancement_proposals/2026-10-01/checks/frontend.mjs
.venv/Scripts/python.exe -m pytest fastapi_react/enhancement_proposals/2026-10-01/checks/test_backend.py --no-cov
§§§

The browser script launches temporary static preview servers itself and closes them afterward. The “current” screenshots read the main frontend's existing §dist§ build; build that baseline separately if it is missing. The API probe accepts §PROPOSAL_API_PORT§ as well. Use the ordinary frontend/backend validation commands in their implementation chapters to run the full integrated suites.

## Rollout

1. Review complete replacements against the recorded source snapshot and the current checkout. Install in a review branch with recoverable original files.
2. Run Python compilation, lint/type checks, both full unit suites, the Vite build, and the main-entry budget.
3. Start the integrated backend with enhancements enabled but response cache off. Verify full raw content, exports, uploads, and normal analysis flows.
4. Build the frontend with tools enabled. Check light/dark themes, keyboard access, mobile layout, years, numeric fonts, original CSV downloads, and the existing parity workflows.
5. Enable server/client reuse only after revision invalidation checks pass. Measure cold/warm results separately and watch retained memory.
6. Enable administrator jobs only for a trusted local session. Test a small real audit before a long computation and observe the separate process's memory/CPU.
7. Deploy the reviewed Nginx/static configuration, then verify real cache headers, compression, SPA fallback, request limits, and denied source maps.

## Rollback

Turn off §F1_ENHANCEMENTS§ and §F1_VIEW_RESPONSE_CACHE§, restart the API, and rebuild the frontend with §VITE_F1_ENHANCEMENTS=0§. A user can immediately disable the readability/cache choices in the tools drawer. Running job shutdown waits for completion; plan restarts accordingly. Restore the original tracked files to remove lifecycle and asset implementation changes as well. Job IDs/results are process-local and will not survive the restart.

## Full deployment source

The files below are complete. The supplied package file changes scripts, not dependency versions. The Nginx file should replace the frontend serving configuration only after adapting the upstream host and reviewing the actual hosting setup.
"""

documents["07_VERIFICATION_CODE.md"] = r"""
# Complete verification source

This chapter contains every test and preview script supplied with the proposal. The current application's existing tests are retained; these files add module and feature checks and adapt the changed request boundary.

## Test layers

- §Enhancements.test.jsx§ exercises the semantic table/driver view, presets/links, caching and private-data exclusions.
- §test_enhancements.py§ integrates the four standalone backend contracts and two service/dispatch tests into the current pytest suite.
- §checks/test_backend.py§ can exercise proposal cache, body limit, metrics, and spawned-job behavior without installing the candidate in the main application.
- §checks/frontend.mjs§ checks safe view encoding, input exclusions, request deduplication, cancellation, timeout feedback, and bounded client reuse.
- §checks/api.py§ probes a running flagged API for response reuse, timing, raw-data identity, and guarded routes.
- §checks/browser.mjs§ builds no app code; it serves existing baseline/proposed dist directories, exercises five flows, collects errors, and writes eight real screenshots.
- §prepare_preview.py§ generates complete integration files and a named isolated staging copy.

The backend dispatcher test deliberately mocks expensive source calculations. The spawned process queue is tested with a small top-level importable worker. These tests prove the queue contract and dispatch choices, not the scientific validity or runtime cost of a newly trained model.

See [deployment and validation](06_DEPLOYMENT_AND_VALIDATION.md) for commands, environment flags, recorded results, and limits. See [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) for file identities.

## Full verification files
"""

appendices = {
    "04_FRONTEND_IMPLEMENTATION.md": sorted((HERE/"code/frontend").glob("*")),
    "05_BACKEND_IMPLEMENTATION.md": sorted((HERE/"code/backend").glob("*")),
    "06_DEPLOYMENT_AND_VALIDATION.md": sorted((HERE/"code/deployment").glob("*")),
    "07_VERIFICATION_CODE.md": sorted((HERE/"checks").glob("*")) + [HERE/"prepare_preview.py"],
}
languages = {".py": "python", ".jsx": "jsx", ".js": "javascript", ".mjs": "javascript",
             ".css": "css", ".json": "json", ".ts": "typescript", ".conf": "nginx"}
fence = chr(96) * 3
for name, content in documents.items():
    content = content.strip().replace("§", chr(96)) + "\n"
    for path in appendices.get(name, []):
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        local = path.relative_to(HERE).as_posix()
        content += f"\n## {local}\n\n[Separate source file]({local})\n\n"
        content += fence + languages.get(path.suffix, "text") + "\n" + source.rstrip() + "\n" + fence + "\n"
    (HERE/name).write_text(content, encoding="utf-8")

baseline = [
    "fastapi_react/frontend/src/App.jsx", "fastapi_react/frontend/src/App.test.jsx",
    "fastapi_react/frontend/src/main.jsx", "fastapi_react/frontend/src/components/Presentation.jsx",
    "fastapi_react/frontend/src/components/ViewTable.jsx",
    "fastapi_react/frontend/package.json", "fastapi_react/frontend/vite.config.js",
    "fastapi_react/backend/app/main.py", "fastapi_react/backend/app/services/presentation.py",
    "fastapi_react/backend/pyproject.toml",
]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
snapshot = {
    "prepared": "2026-10-02",
    "purpose": "Baseline identities and complete candidate source identities, not a dataset manifest.",
    "baseline": {name: digest(REPO/name) for name in baseline},
    "candidate": {path.relative_to(HERE).as_posix(): digest(path)
                  for path in sorted((HERE/"code").rglob("*"))
                  if path.is_file() and "__pycache__" not in path.parts},
}
(HERE/"SOURCE_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2)+"\n", encoding="utf-8")
print(f"Wrote {len(documents)} Markdown documents and source inventory.")
