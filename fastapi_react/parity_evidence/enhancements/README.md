# Main application enhancement acceptance

This directory verifies the installed React/FastAPI application against all 17 requested D1–D4, F1–F6, B1–B5 and O1–O2 enhancements. The proposal-preview checks are historical evidence; these checks target the main application.

Run after the frontend and API are started with the enhancement routes and response cache available:

```powershell
node fastapi_react/parity_evidence/enhancements/verify.mjs
node fastapi_react/parity_evidence/enhancements/verify-driver-comparison.mjs
```

The defaults are React at `http://127.0.0.1:5174` and FastAPI at `http://127.0.0.1:8000`. Override `REACT_BASE_URL` and `F1_API_BASE_URL` as needed. The script resolves Playwright and Papa Parse from the installed frontend dependencies, so it does not need a dependency install of its own. `HEADLESS=0` opens a visible test browser. The local launcher permits checking protected diagnostic records without a token; hosted verification can supply an optional existing `F1_ADMIN_TOKEN`. The token is never written into evidence.

For the local research form, run `node fastapi_react/parity_evidence/enhancements/verify-local-access.mjs` after starting the API with `fastapi_react/start-local.ps1`. This checks real token-free local access and rejection of unrelated origins, wrong Host, forwarded headers and cross-site requests. Raw HTTP probes use an unsupported task so they cannot launch calculations, even if authorization regresses. The job lifecycle uses browser fixtures; the access endpoint and ordinary analysis views use the real running API. `verify-queued.mjs` separately fixtures hosted access mode to keep the password form and credential privacy covered.

The script does not start or stop servers, alter datasets/model files, or execute training/research actions. It uses real application responses for normal flows and controlled HTTP fixtures for slow loading, stale cancellation, and a 503/retry case. Explicit injected errors and expected aborted requests are recorded separately from unexpected errors.

## Completion checklist

| ID | Required behavior | Main-app browser/API evidence | Additional fixture/unit evidence needed |
| --- | --- | --- | --- |
| D1 | Legible captions/active navigation in both themes, visible keyboard focus, usable target sizes; retain numeric/word fonts and plain years | Computed contrast ≥4.5:1, opacity 1, keyboard outline ≥3px, table buttons ≥44×44, light/dark/mobile screenshots, semantic numeric/year formatting | Component tests for default/readability controls and contrast CSS; full WCAG certification is outside this claim |
| D2 | Compact branding/header, sticky navigation, practical mobile spacing | Logo ≤280 desktop /210 mobile, padding 40/56px, sticky nav, mobile heading ≤30px, no root horizontal overflow, mobile keyboard navigation and sidebar | Browser measurements at 1280×900 and 390×844 are representative, rather than every device |
| D3 | Always visible table tools that preserve the original grid features | Computed opacity/pointer availability, search, field visibility toggle, full CSV row/column export | Original grid sort/pin/copy/resize regression coverage from existing browser/component checks |
| D4 | Understandable loading/errors, stale request cancellation, safe Plotly lifecycle | Polite loading outside busy main, elapsed visual time, observed canceled browser request, stale content rejected, readable 503, successful Retry | Plotly import/newPlot/resize failures, unmount cleanup; request timeout and independent subscriber cancellation contracts |
| F1 | Save, reload, restore, and delete named local views | Unicode named save survives reload; restores section and Constructor; restored filters `false` survive reload; delete removes option | Storage limits, corrupt/quota-unavailable storage, duplicate names |
| F2 | Unicode share links restore safe settings, including fresh browser/reload | Decodes copied URL; restores Unicode filters on new context; upload bodies, betting probability, token absent | Unsupported versions, malformed and oversized links, safe-key whitelist and nesting bounds |
| F3 | Keyboard-searchable native command palette | Ctrl+K /Cmd+K, native `:modal`, autofocus, filtering, Enter navigation, Escape/Close | Arrow/Home/End selection and focus restoration component coverage |
| F4 | Semantic table alternative without losing the original grid | Actual column/row header scopes, 50-row paging, hidden-field search matches real data, every field selectable, grid switch and CSV | Empty data/search, last column preserved, search resets page, formatting contracts |
| F5 | Compare ≤4 drivers using useful descriptive source metrics | Fifth checkbox disabled, becomes enabled after removing one; record counts and means/DNF rate independently recomputed from real API table. verify-driver-comparison: race tire metrics match the live source, chart follows the same three drivers, clear/close restores all drivers, annual Races counts retained, race/year scope explicit, mobile page does not overflow | Unknown/nonfinite/missing values, filtered-out selection, direct summary values, suppress comparison when no useful metrics exist, original chart data and unrounded values preserved |
| F6 | Export safe analysis context and recorded model provenance | Downloaded JSON contains UTC time, section, safe settings, dataset/build/revision and manifest metadata; print invoked | Disabled export while metadata unavailable/loading/stale and explicit matched analysis revision |
| B1 | Bounded short-lived client/server reuse and deduplicated requests | Identical real API response HIT and unchanged body; settings MISS; private/raw bypass; `gzip;q=0`; client revisits avoid POST but check revision | TTL, entry/byte eviction, action invalidation, concurrent consumer deduplication, abort behavior |
| B2 | Invalidate every cache when eligible artifacts change | Real status and `X-F1-Revision` agree; context export includes current revision; client checks status before hits | Mutate a temporary artifact only; prove presentation/source loaders and response caches clear; metadata race/fallback/manifest scenarios |
| B3 | Request timing/IDs and bounded, private diagnostic records | Success/failure headers, unique 32-digit IDs, route response timing, unauthorized metrics rejected; optional token proves actual record schema | 500-record eviction, query/body/token redaction, byte count including compressed bodies, log format |
| B4 | Explicit authenticated research in a separate process | verify-queued: real unauthorized rejection/409 action guard; fixture UI submission, queued cancellation, running/success snapshots, continued browsing, memory-only token | test_research_jobs: actual spawned fixture worker PID, input snapshot, queue/result bounds, expiry/failure/cancellation, strict allowlist, authenticated routes, source revision checks |
| B5 | Aggregate body limit before parsing | verify-queued: real oversized declared length returns413 without a large upload, with API diagnostic headers | Tests check chunked overflow, exact boundary replay, malformed/duplicate/mismatched length, disconnect, CORS, non-HTTP scopes |
| O1 | Efficient responsive footer | verify-queued: real 1x/2x requests, no original PNG requested, 60px height and original proportions, screenshots and byte counts | Sharp generation preserves transparency using lossless WebP; every build regenerates both variants |
| O2 | Enforced budget and production caching | verify-queued: production browser; check-hosting: actual Nginx HTML/assets/API/gzip/map policies; recorded build-budget.json | Budget fixture checks static dependency accounting, oversized default entry and source-map rejection; CI invokes build and container header checks |

## Generated evidence

- `results.json`: structured flow results, exact measured styles, API/cache observations, expected events and unexpected errors.
- `RESULTS.md`: readable executed-flow report and screenshot links.
- `screenshots/`: real screenshots for desktop light/dark, mobile, semantic table, comparison, saved views, command palette, loading, and failure/retry.
- `exported-context.json`: actual exported model/dataset analysis context; safe analysis settings only.
- `verify-queued.mjs`, `queued-results.json`, `QUEUED_RESULTS.md`: six additional flows for B4/B5/O1/O2. Set optional F1_PRODUCTION_BASE_URL to test a production server too.
- `verify-local-access.mjs`, `local-access-results.json`, `LOCAL_ACCESS_RESULTS.md`: direct local access and token-free browser workflow, with `screenshots/trusted-local-research.png`. Backend tests additionally cover hosted defaults, every protected route, IPv6, duplicate headers, custom ports and invalid origin configuration.
- `verify-driver-comparison.mjs`, `driver-comparison-results.json`, `DRIVER_COMPARISON_RESULTS.md`: four checks using the live 2025 Canadian Grand Prix tire data, with desktop/mobile selected-driver screenshots. The summary table shows real metric values without a one-row count; the selected chart preserves the original unrounded source values. The seasonal comparison retains its recorded race count. No research calculation is started.
- `hosting-results.json`: actual headers from temporary localhost Nginx using the production config (adapted port/root/upstream only).
- `build-budget.json`: exact measured bytes from the production checker. Regenerate after a production build.

Passing this script is necessary evidence for these browser/API flows. Completion also requires the appropriate backend/frontend unit contracts, original parity regression checks, Python compilation, frontend production build, lint and type checks. A narrow browser flow does not prove cache bounds or real artifact invalidation by itself.
