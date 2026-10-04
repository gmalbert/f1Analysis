# Raw-data backend optimization

Measured October 1, 2026 (America/New_York). The final browser run completed at
2026-10-02T02:17:20.791Z. All numbers below compare the original performance
report with three fresh-browser repetitions against the updated backend.

The raw-table wait for response headers fell **75.7%**, from **8.15 seconds to
1.98 seconds**. The complete view finished **70.6% faster**, in **2.53 seconds**.
All 4,629 rows, 561 columns, values and table metadata remain identical.

| Measurement, median | Before | After |
|---|---:|---:|
| Wait for response headers | 8.148 s | 1.982 s |
| Raw table finished loading | 8.615 s | 2.533 s |
| Download, including response headers | 5.766 MB | 6.070 MB |
| Raw table rows | 4,629 | 4,629 |
| Raw table columns | 561 | 561 |

After-change response-header waits ranged from 1.955 to 2.001 seconds; complete
view times ranged from 2.516 to 2.533 seconds. MB means 1,000,000 bytes.

## What changed and why

Profiling identified three avoidable costs in the backend:

1. Table construction normalized each numeric cell separately. Numeric missing
   values are now normalized one column at a time, using a private copy of the
   table. Mixed, text and date columns retain the existing conversion. Numeric
   precision, integer and boolean values, missing values and column order are
   preserved. Standalone presentation construction fell from a median 3.76 s
   to 0.91 s in the diagnostic runs.
2. FastAPI recursively converted an already JSON-compatible presentation again.
   The view endpoint now returns a JSON response directly. This removes the
   original approximately 1.53 s conversion pass over millions of cells. The
   JSON serialization step itself still runs, and the API response schema is
   retained.
3. The default gzip level 9 was expensive for this 20.55 MB JSON body. The API
   now uses gzip level 5. On the same pre-change body, isolated compression
   medians were 3.56 s at level 9 and 0.63 s at level 5. This increases the
   measured download by **0.303 MB, or 5.3%**, in exchange for much less server
   processing. Compression applies to other eligible API responses too.

The standalone diagnostics explain the costs; they are separate runs and their
times should not be added to reconstruct the browser measurement. The
post-change diagnostic deliberately still measures the framework conversion
and gzip level 9 for comparison; those steps are no longer used by the live view
endpoint. Its `gzip9_and_equivalence_ms` field includes validation work and is
not an isolated compression measurement.

## Data and feature verification

- The final complete raw-table checksum matches the pre-change checksum:
  `0389ff31e162ebc06710cbbfc77c9ea8d3028ecce30dd502f4de5f044598415e`.
  The comparison includes values, column definitions, display settings and
  other table metadata.
- Browser verification confirms the full CSV contains 4,629 data rows and all
  561 original column keys. Search and column visibility controls work.
- All **60 backend tests pass**, with **87.45% coverage**. The new regression
  test compares the old and new conversions across large integers, precise
  floats, infinities, missing values, booleans, dates, categories, nested
  objects, duplicate columns and empty tables.
- The final modified Python files pass `py_compile`, Ruff and strict mypy.
- All **11 Playwright interaction groups pass**, including every model and its
  seven panels, charts and exports, filtering, raw data, simulation, CSV
  uploads, paper replay, calibration, theme and mobile navigation. Both browser
  verification runs record **zero application errors**.

## Conditions and evidence

Both measurements use a local React production preview proxying the same
FastAPI service, Chromium at 1280×900, warmed server data/model caches and fresh
browser contexts. Filter Results is enabled and the minimum Year is advanced
to 2017. One warmup is excluded from the three final samples. These are local
response measurements without network throttling or an isolated operating
system. They do not measure server startup, WAN performance or concurrent
heavy requests. The change adds no response cache or reduced-data mode.

Baseline: [original report](REPORT.md), [summary](summary.json).
Profiling: [before](raw-profile-before.json), [after](raw-profile-after.json),
[profiler script](profile_raw_data.py),
[before call profile](raw-render-profile-before.txt),
[after call profile](raw-render-profile-after.txt).
Final checks: [browser timings and CSV](raw-optimization-browser.json),
[browser verification script](verify_raw_optimization.mjs),
[complete table checksum](raw-checksum-final.json),
[checksum verification script](verify_raw_checksum.py),
[interaction results](raw-optimization-interactions.json),
[validation results](raw-optimization-validation.json).
