# Implemented application enhancements

All 17 requested enhancements (D1–D4, F1–F6, B1–B5 and O1–O2) are installed in the main application. Original data, calculations, year formatting, number/text typography and CSV contracts remain in use.

Open **Analysis tools** to change readability/cache preferences, save or restore named views, copy a view link, export context, print, or search sections. Readability and response reuse default on. Ctrl+K/Cmd+K opens the native section search; arrows, Home/End, Enter and Escape are supported.

Tables default to the original Interactive grid. Accessible table provides semantic headers, all-field search, selectable columns and 50-row paging without reducing the underlying data. Compare drivers appears only when a driver table has useful comparison metrics, and accepts up to four drivers. Repeated race records show descriptive averages, known-record DNF rates and a Records included count. Tables already summarized to one record per driver show their actual displayed metrics without a redundant row count.

The race tire comparison names the selected Grand Prix and year, and shows degradation in seconds per lap, starting compound, stints, stint lengths, soft-tire lap percentage and laps. Its chart follows the same selected drivers; clearing or closing the comparison restores the full field. The annual tire summary names the season and retains the source Races count. These are descriptive source values. See [live driver comparison checks and screenshots](parity_evidence/enhancements/DRIVER_COMPARISON_RESULTS.md).

Saved/shared/persisted settings use a narrow allowlist. Uploaded CSVs, ledgers, betting inputs, passwords and tokens are excluded. Unicode links are readable encodings and do not freeze the dataset. Views are saved on the current browser.

Context JSON records safe settings, UTC export time, section, displayed analysis revision and recorded dataset/model/build provenance. Export is disabled while loading, after a failed request, when metadata is unavailable, or when source/display revisions differ. Printing uses the current displayed table page; existing full CSV downloads remain available.

Loading feedback is outside the busy content region. Cancelled/stale requests cannot replace current results, and invalidated pending responses cannot refill the client cache. Plotly failures and asynchronous disposal are handled.

## Backend policy

The browser retains up to six responses/12,000,000 serialized bytes for 15 seconds, checking the source revision before reuse. Serialized size does not bound actual JavaScript heap.

The server retains up to 12 responses/64 MiB of plain-plus-gzip bytes for 20 seconds. Ordinary page1–5 reads are eligible; raw data, betting, actions and private inputs bypass reuse. HTTP responses remain no-store. Gzip exclusions are honored.

Artifact changes invalidate presentation/model caches, source loaders and responses. Revision identity uses paths/sizes/mtimes, not content integrity. Publish inputs atomically with changed mtimes. Generated Next Race CSV outputs and downloaded FastF1 telemetry caches are excluded. Python source changes require restart.

Request IDs, Server-Timing, structured records and a bounded 500-record diagnostic history are enabled. Records exclude queries, path parameters, bodies and tokens. The local launcher grants diagnostics access on this computer; hosted access requires an administrator token. See [backend policies and flags](backend/ENHANCEMENTS.md).

## Verification

The complete suites pass 125 backend tests (90.50% coverage) and 98 frontend tests, with the existing coverage thresholds retained. All 22 Python application sources compile through py_compile. Ruff, strict mypy, ESLint and TypeScript checks pass. The production frontend builds with the existing nonfatal warnings about lazy chart chunks.

The [acceptance checklist](parity_evidence/enhancements/README.md) maps requirements to browser and unit evidence. [Main acceptance](parity_evidence/enhancements/RESULTS.md) and [structured evidence](parity_evidence/enhancements/results.json) record 16 passing flows, exact styles, real exports, screenshots and controlled loading/failure fixtures. [Driver comparison acceptance](parity_evidence/enhancements/DRIVER_COMPARISON_RESULTS.md) adds four live-data checks for real metrics, matching chart selection, restored full-field data, annual race counts and mobile layout. The latest [additional results](parity_evidence/enhancements/QUEUED_RESULTS.md) cover five hosted-form, size-limit, asset and budget flows. [Trusted local acceptance](parity_evidence/enhancements/LOCAL_ACCESS_RESULTS.md) covers live authorization, the local form without a token, queued cancellation, results and continued browsing. These latest Playwright reports contain zero unexpected errors; deliberately cancelled obsolete requests are recorded separately. The command palette also handles delayed native close events when rapidly reopening after Escape. Earlier production-browser and HTTP-caching evidence is retained separately.

Run from the repository root with React5174/API8000 available:

```powershell
node fastapi_react/parity_evidence/enhancements/verify.mjs
node fastapi_react/parity_evidence/enhancements/verify-driver-comparison.mjs
```

Start the backend with `--log-config logging.json` from its directory to suppress separate raw-URL access lines. `F1_VIEW_RESPONSE_CACHE=0` disables server reuse; `F1_REQUEST_LOGS=0` disables structured log emission at startup. `F1_ENHANCEMENTS=0` removes backend enhancement routes/dependencies. Align this with frontend `VITE_F1_ENHANCEMENTS=0` when opting out of the tool/profile/cache integration.

## Research jobs and request limits

Research jobs appears below the main results on Predictive Models and Data & Debug. Existing Run Bin Count Comparison and Run Leakage Audit buttons open that form. The trusted local launcher removes the token field and focuses the task selector; hosted mode focuses the administrator token field. Calculations start only after you press Queue calculation. Any hosted token stays in component memory and never enters saved/shared views, exports or browser storage.

Run `.\fastapi_react\start-local.ps1` from the repository root to enable direct trusted local access without a token. Local mode checks the loopback peer, local Host and browser Origin on every protected request and rejects forwarded or cross-site requests. It is off by default and explicitly off in Docker. Hosted mode requires F1_ADMIN_TOKEN for every submit/status/result/cancel request. One spawned calculation process works separately from HTTP rendering. Eight queue/result slots, 64 KiB inputs, 32 MiB uncompressed results and ten-minute completed-result expiry bound retention. Only queued jobs can be cancelled. Running jobs finish; status polling can be retried and navigation stays available. State is lost on restart. Deploy this local queue with one API worker; several workers need an external shared queue and result store.

Audit inputs are bounded to 1–100000 rows, default 1000. Bin comparison accepts one to nine distinct q values from 2–10, default [2], and uses the original experiment. Arbitrary controls/uploads are rejected. Jobs pin and recheck source revision before and after calculation. Output is a read-only snapshot with existing table/chart/download rendering and its recorded revision. Bin experiments do not replace production model artifacts. Browser acceptance uses fixtures; it does not train real models.

Aggregate request bodies are bounded before JSON parsing, including streamed bodies and misleading Content-Length headers. F1_MAX_REQUEST_BYTES defaults to 1048576 (1 MiB); Nginx also limits requests to 1m. Public betting upload workflows are disabled. Adjust both limits together when changing the request policy. Buffered requests and JSON decoding allocate additional memory beyond the payload bytes.

![Administrator research interface using a controlled fixture](parity_evidence/enhancements/screenshots/research-jobs.png)

![Trusted local research interface using a controlled result fixture](parity_evidence/enhancements/screenshots/trusted-local-research.png)

## Production assets and HTTP caching

The original footer PNG remains as a compatibility fallback. Every build produces lossless transparent WebP variants at heights 60/120 pixels. A picture source selects 1x/2x; lazy loading, asynchronous decoding and fixed original proportions retain the 60-pixel display size. Measured files are 4938/15096 bytes versus the original 1127638 bytes, reductions of 99.56%/98.66% for that image. This is an asset-byte comparison, not a measured reduction in total page loading time.

![Responsive footer](parity_evidence/enhancements/screenshots/responsive-footer.png)

`npm run build` enforces 500000 gzip bytes for the entry and its static JavaScript dependencies, using gzip level 5 to match hosting. The measured initial total is approximately 231 KB; all lazy JavaScript together is approximately 2.02 MB. Lazy chart bundles remain substantial. The checker fails for missing entry/dependencies, oversized initial JavaScript, or published .map files. [Recorded build measurements](parity_evidence/enhancements/build-budget.json) contain exact bytes. Vite's warnings alone do not enforce this budget; CI now runs the complete build script.

The Docker Nginx configuration compresses text assets, caches hashed /assets files for a year with immutable, revalidates HTML/SPAs, gives unversioned images/fonts a one-hour cache, and marks API responses no-store. API paths take priority over filename matching, including downloads ending in .png/.map. Maps and the hidden build manifest return 404. These policies follow the [Nginx headers](https://nginx.org/en/docs/http/ngx_http_headers_module.html) and [location routing](https://nginx.org/en/docs/http/ngx_http_core_module.html#location) contracts.

Actual syntax, HTTP headers and a production-browser smoke test passed using temporary localhost Nginx, with only its port, root and upstream adapted for Windows. [HTTP evidence](parity_evidence/enhancements/hosting-results.json) is recorded; CI runs the same header checker against its Nginx container. This config applies when hosted through Nginx; the Vite development server remains at 5174. No production deployment was performed.

## Complete source and original proposals

The current complete integration source is included in [frontend/deployment code](implementation/FRONTEND_AND_HOSTING.md) and [backend code](implementation/BACKEND.md), with file hashes and links to editable originals. The original [eight-document proposal guide](enhancement_proposals/2026-10-01/README.md) remains a historical planning snapshot with its initial screenshots and candidate code; use the current implementation for rollout.
