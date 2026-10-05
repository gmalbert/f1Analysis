# Implemented analysis backend enhancements

B1–B5 are active in the main FastAPI application. These changes preserve the presentation protocol, existing API routes, calculations, table values, and the earlier raw-data serialization optimization. Explicit research actions now use the administrator queue rather than the synchronous view route.

## B1: bounded view responses and request deduplication

`app/enhancements/cache.py` retains up to 12 ordinary view responses for 20 seconds, with a combined 64 MiB limit for their original JSON and precompressed gzip bodies. Cache identity includes the artifact revision, page, and all supplied control values. Concurrent requests for identical controls reuse the first completed render. The server stores both encodings so gzip cache hits avoid repeated JSON serialization and compression.

Pages 1 through 5 are eligible. Raw Data, Betting Research, explicit actions, uploaded CSV controls, ledger controls, nested objects, nonfinite numeric controls, and oversized control values bypass reuse. Actions and uploads clear previously retained responses. Expired responses are pruned on subsequent cache access, and oversized results are served without retention. These limits bound retained response bytes; temporary rendering objects and request buffers are separate allocations.

View responses use `Cache-Control: no-store`, `Vary: Accept-Encoding`, `X-F1-Cache: HIT|MISS|BYPASS`, and `X-F1-Revision`. The quality-aware gzip middleware respects `gzip;q=0`, including an explicit exclusion combined with a wildcard. Both cached and uncached routes use the same negotiation behavior. Original integer and floating-point values are retained without rounding.

## B2: artifact revision and source cache invalidation

`app/enhancements/service.py` calculates a revision from the relative paths, sizes, and nanosecond modification times of presentation inputs, model artifacts, relevant source files, and the effective `F1_USE_PARQUET` setting. It watches CSV/TSV, Parquet, JSON, pickle/joblib, text, HTML, image, and Python files. It excludes Python bytecode and downloaded FastF1 telemetry directories because those files are not inputs to the React presentation.

The root-level `data_files/predictions_*.csv` files are also excluded. The existing Next Race presentation writes these generated download outputs on every render. Treating them as inputs would make that view continually invalidate itself. Its JSON precomputed predictions, data, model artifacts, and schedules remain watched. The separate legacy API's CSV prediction reader does not retain those files in an LRU cache. The exclusion is covered by a contract test, and all seven existing presentation pages pass integration tests.

Every API request uses a shared dependency to check for changes at most once per second. A view request and the public status endpoint force a fresh check. Changes clear the presentation's shared data/model cache under `_RENDER_LOCK`, `_LOCK`, and `_MODEL_LOCK`, all cached data and analysis loaders, and the retained response cache.

View rendering checks the revision again before returning or retaining a result. A safe read retries once if input files changed during rendering. An explicit action is never repeated. Persistently changing inputs produce HTTP 503 with `Retry-After: 1` and a readable message, rather than labeling an inconsistent response as a valid cache hit.

Publish datasets and model files atomically with a changed modification time. This revision is a cheap stat identity, not a content-integrity hash. Preserving the old size and mtime can evade detection. Multi-file publication is not a filesystem snapshot; use coordinated atomic publication and avoid editing files in place while requests are running. Python source changes still require a worker restart because the exported reference view is compiled at import time.

`GET /api/enhancements/status` supplies the current revision, build identifier, selected dataset name/time, and recorded model manifest fields for frontend reproducibility exports. Manifest timestamps, hashes, calibration notes, and metrics are recorded provenance, rather than claims of freshly validated model quality.

## B3: request timing and bounded diagnostics

`app/enhancements/metrics.py` is a pure ASGI middleware outside routing and gzip. Every completed HTTP response receives a generated `X-Request-ID` and a `Server-Timing: backend;dur=...` value. Both are exposed through CORS, together with the cache/revision headers. The timing measures backend time until response headers are emitted, including rendering and gzip where those occur before headers. It does not measure the browser's total load time or network latency.

The middleware retains only the latest 500 request records per API process. Each record contains the generated ID, method, public route template, status, header duration, total backend duration, and encoded response-body bytes. Query strings, path parameter values, headers, tokens, request bodies, uploaded data, and control values are excluded. Unexpected failures before a response starts receive the generic 500 response with diagnostics headers, then re-raise for normal server exception logging. Errors after a response has started cannot replace headers or status already sent.

Structured `f1.request` JSON lines are emitted by default. To use the complete logging configuration, start the server from this directory with:

```powershell
& ..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-config logging.json
```

The supplied configuration suppresses Uvicorn's separate access lines, which would otherwise include raw URL query strings. Diagnostic privacy applies to the structured request records. Normal server exception tracebacks remain available to administrators.

`GET /api/enhancements/metrics` uses the same access policy as research jobs: the trusted local launcher allows direct use on this computer; hosted mode requires the `X-F1-Admin-Token` header. Set `F1_ADMIN_TOKEN` in the hosted server environment. In hosted mode, an unset token returns 503; a missing or wrong token returns 403. The token is compared in constant time and is never included in diagnostic records.

## Flags and limits

| Environment variable | Default | Effect |
| --- | --- | --- |
| `F1_ENHANCEMENTS` | `1` | `0` at worker startup restores the original view route and removes status/metrics/jobs, revision dependencies, and diagnostic middleware. Quality-aware gzip and the global body limit remain in place. |
| `F1_VIEW_RESPONSE_CACHE` | `1` | `0` bypasses server response reuse. Revision checks and diagnostics remain active. |
| `F1_REQUEST_LOGS` | `1` | `0` at worker startup stops structured request log emission. Diagnostic headers and bounded in-memory records remain active. |
| `F1_BUILD_REVISION` | `local-working-tree` | Optional deployment commit/build identity in status and exported context. |
| `F1_ADMIN_TOKEN` | unset | Enables authenticated access to diagnostics and research-job endpoints. |
| `F1_TRUSTED_LOCAL` | `0` | Explicit opt-in to token-free research/diagnostics for direct trusted local requests. The local launcher sets it only for its own process tree; Docker keeps it off. |
| `F1_LOCAL_ORIGINS` | Localhost/127.0.0.1/[::1] HTTP addresses on ports 5173, 5174 and 8000 | Optional comma-separated exact local origins. Entries must remain localhost addresses, with no credentials, paths, queries or fragments. Include the API origin as well as the frontend origin when changing ports. |
| `F1_MAX_REQUEST_BYTES` | `1048576` | Positive aggregate body-byte limit before JSON parsing; align with Nginx client_max_body_size. |

Caches, locks, diagnostics and jobs are process-local. Use one API worker when using this local research queue. Multiple workers have distinct queues; reliable shared scheduling/routing requires an external queue/result store. Every worker independently notices artifact changes on its next request. No new external cache, queue service or Python dependency is required.

## B4: research jobs and local access

From the repository root in PowerShell, run `.\fastapi_react\start-local.ps1` to use research tools on this computer without managing a token. The script enables `F1_TRUSTED_LOCAL=1` only for the launched API process, binds to `127.0.0.1:8000`, uses the structured logging configuration and passes `--no-proxy-headers`. It restores the calling shell's previous local-mode flag when the API exits.

Every protected request is checked independently: its socket peer must be loopback, its Host must match a configured local address, and any Origin must match a configured local app origin. Duplicate Host/Origin values, forwarded headers and cross-site browser requests cannot obtain local access. Local mode also restricts CORS to the configured local origins. Direct local command-line requests without browser headers are supported. This mode is intended for a single-user computer without a public reverse proxy. `GET /api/enhancements/research-access` returns only the applicable access mode and whether a token is required; it grants no authority by itself and returns no secret.

The React form checks that endpoint and omits the password field in local mode. Loading or failed access checks disable submission and offer retry; stale checks are cancelled. Original research buttons focus the task selector locally and the token field in hosted mode. An explicit **Queue calculation** submits `POST /api/enhancements/jobs` with task `leakage-audit` or `bin-comparison` and task-only values. Submit, status, result, cancel and diagnostics all use the same server-side policy. Normal page requests do not start a research worker.

For hosted use, keep `F1_TRUSTED_LOCAL=0`, set `F1_ADMIN_TOKEN` in the server environment, restart the API and enter the token in the Administrator research jobs form. The Docker configuration explicitly keeps local mode off. The token stays in component memory and is sent only when required. Missing hosted configuration returns 503; missing/incorrect credentials return 403. A valid configured administrator token also authorizes requests that are ineligible for local access. Never put a token in a URL or saved view.

Jobs are lazily created. A one-thread coordinator feeds one ProcessPoolExecutor worker using Windows-compatible spawn, with eight retained jobs, inputs <=64 KiB, each uncompressed and compressed result <=32 MiB, and ten-minute expiry after completion. Pending and completed results occupy slots; expiry is pruned on access. Queue capacity returns 429 with Retry-After:10. Retained compressed results are bounded to at most eight times 32 MiB; worker computation and serialization use additional temporary memory.

States are queued, running, succeeded, failed or cancelled. GET /jobs/{identity} reports state and timestamps. GET /jobs/{identity}/result returns successful presentation JSON; unfinished/failed/cancelled states return409, absent/expired IDs404. DELETE /jobs/{identity} cancels a waiting task and returns both cancelled:boolean and the current job state. Running work finishes normally, including during graceful API shutdown. Completed snapshots expire and all state is lost at restart. In this implementation running jobs have no forcibly enforced execution deadline.

Task inputs are tightly bounded: leakage audit accepts only `Rows to read (0 = all)` with integer1–100000, default1000. The admin queue deliberately rejects unbounded0. Bin comparison accepts only `Select q values (number of bins)` with one to nine distinct integers2–10, default[2]. Extra controls, uploads and ledgers are rejected. Workers use the existing reference calculations, clear their own source caches, and recheck pinned source revision before and after calculation. Bin experiments train temporary comparison estimators without replacing production model files. Reference-calculation notices remain in the returned presentation; succeeded means the worker returned a valid snapshot, so inspect its findings/notices.

The active React shell intercepts the original research buttons to open the queue without submitting. The synchronous /api/views route rejects those two actions with409. Other hosted-mode training controls remain disabled. UI polling aborts stale requests, reports failures with a status retry, continues across section navigation, and filters page controls from the read-only result snapshot. Neither browser acceptance nor process tests execute real model training: fixtures prove state, cancellation, process PID isolation, expiry and bounds.

## B5: aggregate request size

BodyLimit is installed globally before routing/JSON decoding, even with enhancements disabled. Its default 1 MiB includes the complete encoded JSON body. Public betting upload workflows are disabled. Oversized declared Content-Length receives413 without reading the body. Actual streamed bytes are measured too; malformed/duplicate/mismatched length headers receive400 and disconnects stop replay. Empty chunks cannot create an unbounded retained message list. Accepted bodies are buffered and replayed once; JSON decoding and concurrent requests require memory beyond this payload limit.

CORS wraps BodyLimit and RequestMetrics wraps both, so API rejections keep applicable CORS, generated IDs and timing headers. Nginx's 1m limit rejects oversized requests earlier at the hosting boundary; those proxy-generated responses do not contain backend-generated IDs or timing. Align both limits when changing the request policy.

## Verification

From the backend directory:

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest --basetemp ..\..\.test-tmp-parity\backend-enhancements
& ..\..\.venv\Scripts\python.exe -m ruff check .
& ..\..\.venv\Scripts\python.exe -m mypy app
$backendPythonFiles = @(rg --files app -g '*.py')
& ..\..\.venv\Scripts\python.exe -m py_compile @backendPythonFiles
```

The full suite passes 125 tests with 90.50% coverage. It includes the existing real-data API/presentation tests, 20 cache/revision/diagnostic contracts, 14 research/body-limit cases and 31 local/hosted access checks (including parameterized inputs). The access module has 100% statement coverage. Process fixtures prove actual PID isolation without training models. All 22 application Python sources compile, and Ruff/strict mypy pass. The original 80% coverage gate is retained.
