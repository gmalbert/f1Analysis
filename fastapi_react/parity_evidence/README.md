# Parity evidence

These scripts compare the live local applications and the original Streamlit
testing tree. Results describe the current working tree.

## Data and formatting comparisons

With the repository Python environment activated, from the repository root:

    python fastapi_react/parity_evidence/reference_oracle.py
    python fastapi_react/parity_evidence/reference_filter_oracle.py

Streamlit's AppTest executes the actual original application, not the exported
React view code. The first script compares table values, labels, displayed
column selection/order and index visibility, then all six model selections.
The second applies actual boolean, category, numeric-year and date widgets and
compares resulting row values, formats and labels.

These tools require development Streamlit and the original model dependencies.
They are not production runtime dependencies. They do not activate training
controls or replace artifacts.

## Browser workflows and exports

With Streamlit at :8502, FastAPI at :8000 and React at :5174:

    node fastapi_react/parity_evidence/verify_interactions.mjs
    node fastapi_react/parity_evidence/verify_experiments.mjs
    node fastapi_react/parity_evidence/inspect_downloads.mjs

For the Next Race export comparison in PowerShell:

    $env:DOWNLOAD_SECTION = 'Next Race'
    node fastapi_react/parity_evidence/inspect_downloads.mjs
    Remove-Item Env:DOWNLOAD_SECTION

CSV comparison uses the reference's browser-download fallback by disabling the
OS save-picker API in the test context. It preserves actual downloaded contents.
Runtime reports capture page exceptions, console errors and HTTP failures.
The experiment script invokes a selected bin-count experiment and a bounded
administrative audit.

## Screenshot comparison

From the repository root, in PowerShell:

    $env:PARITY_SCREENSHOT_DIR = Join-Path $PWD 'fastapi_react/parity_evidence/visual/run-name'
    node fastapi_react/parity_evidence/capture_streamlit.mjs
    node fastapi_react/parity_evidence/capture_react.mjs
    node fastapi_react/parity_evidence/diff_screenshots.mjs

REACT_BASE_URL and STREAMLIT_BASE_URL can override the default :5174 and :8502
URLs. Capture eight states at desktop 1280×800, tablet 768×1024 and mobile
390×844. The reference is allowed to complete its rerun and both apps load their
actual fonts. Animations and fonts are not replaced; no screenshot areas are
masked. The identical bundled footer image avoids external-host availability
affecting the reference capture.

Both applications enable Filter Results before Analytics. Raw Data is checked
on desktop/tablet and unchecked on mobile; this is not a mobile full-table visual
comparison. Captures start at the top of the page. Missing expected screenshot
pairs fail the comparison instead of silently reducing the sample.

Unchanged acceptance thresholds are 2% desktop and 3% tablet/mobile. These are
visual tolerances, not pixel identity. Latest successful local evidence is in
visual/parity-2026-10-01/. Earlier local-validation folders are historical.

## Current results

- 77 reference table/model comparisons and four filter scenarios pass.
- Data Explorer and Next Race CSV contents match.
- All 24 screenshot comparisons pass.
- Interaction, experiment and viewport capture reports contain zero browser errors.
- The separate axe report retains reference color-contrast findings. See
  ../PARITY_REPORT.md for details and limits.
