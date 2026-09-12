# Visual diff evidence (PARITY_CHECKLIST \u00A713)

This directory holds the visual-diff capture scripts and their outputs.

## Layout

```
parity_evidence/
  capture_react.mjs       # drive the React/Vite app, screenshot every page
  capture_streamlit.mjs   # drive the Streamlit app, screenshot every section
  diff_screenshots.mjs    # compare paired PNGs, write diff/ + summary.json
  screenshots/
    react/                # produced by capture_react.mjs
    streamlit/            # produced by capture_streamlit.mjs
  diff/
    summary.json          # per-page diff ratio + within_tolerance flag
    *.png                 # per-page pixel-diff image
  README.md
```

## Running the full capture

In three terminals against the same repository checkout:

```bash
# 1) FastAPI backend
cd fastapi_react
docker compose up backend

# 2) React dev server
cd fastapi_react/frontend
npm run dev          # serves on http://127.0.0.1:5173

# 3) Streamlit reference
streamlit run raceAnalysis.py --server.port 8501 --server.headless true

# 4) Capture + diff
cd fastapi_react/frontend
npm run capture:react
npm run capture:streamlit
npm run capture:diff
```

The capture scripts are also runnable directly:

```bash
node fastapi_react/parity_evidence/capture_react.mjs
node fastapi_react/parity_evidence/capture_streamlit.mjs
node fastapi_react/parity_evidence/diff_screenshots.mjs
```

## Acceptance rule (\u00A713)

For each page at each viewport:

- `diff_ratio <= 0.02` for desktop (1280\u00D7800)
- `diff_ratio <= 0.03` for tablet (768\u00D71024)

Pages above tolerance are either fixed or recorded as intentional
UI-only differences in `PARITY_REPORT.md`.

## Current status

The capture and diff scripts are in place but have **not yet been
exercised** end-to-end against running servers in this environment.
The Chromium dependency (Playwright) is not yet installed; install
with `npm install --save-dev playwright` and `npx playwright install
chromium`. Once installed, the three commands above produce the
`screenshots/` and `diff/` artifacts. Results will be summarized in
`PARITY_REPORT.md` once the first full run is complete.
