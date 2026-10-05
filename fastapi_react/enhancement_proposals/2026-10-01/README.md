# F1 Analysis enhancement guide

**Implementation update:** All 17 items (D1–D4, F1–F6, B1–B5 and O1–O2) are installed in the main
application. See [the implementation report](../../ENHANCEMENTS.md).
The text and preview evidence below describe the original planning snapshot.

Prepared October 1–2, 2026. This package contains the original **17 proposed enhancements**, eight actual browser screenshots, complete candidate files, and executable verification scripts. The original preview was tested in an isolated copy. The main application now implements the complete list; its current source and verification are linked above. Copying these historical candidate files over the current implementation would lose subsequent fixes.

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
| D1 | First | Improve contrast, focus, and control target sizes | `enhancements.css` |
| D2 | Next | Make the header, navigation, and mobile spacing more compact | `enhancements.css` |
| D3 | First | Keep table tools visible and usable | `enhancements.css`, `EnhancedTable.jsx` |
| D4 | First | Show understandable loading and failure states; cancel stale requests | `FeatureBar.jsx`, `viewClient.js`, `SafePlotlyChart.jsx` |
| F1 | Next | Save named analysis views locally | `preferences.js`, `FeatureBar.jsx` |
| F2 | Next | Share a link that restores safe analysis settings | `preferences.js`, proposed `App.jsx` |
| F3 | Later | Search and open sections with a keyboard command palette | `FeatureBar.jsx` |
| F4 | First | Offer a semantic HTML table alongside the canvas grid | `EnhancedTable.jsx` |
| F5 | Later | Compare up to four drivers using descriptive historical data | `EnhancedTable.jsx` |
| F6 | Next | Export analysis context and recorded model provenance | `FeatureBar.jsx`, `service.py` |
| B1 | Next | Reuse bounded, short-lived view responses and deduplicate requests | `viewClient.js`, `cache.py` |
| B2 | First | Invalidate caches when source artifacts change | `service.py` |
| B3 | First | Add request timing, identifiers, and bounded diagnostic records | `metrics.py`, `logging.json` |
| B4 | Later | Run explicit administrator research tasks in a separate process | `jobs.py`, `ResearchJobs.jsx` |
| B5 | First | Bound aggregate request bodies before JSON parsing | `metrics.py`, `nginx.conf` |
| O1 | First | Resize the footer image and serve efficient responsive assets | `optimize-assets.mjs`, proposed `App.jsx` |
| O2 | First | Enforce a build budget and configure production HTTP caching | `check-budgets.mjs`, `vite.config.js`, `nginx.conf` |

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
