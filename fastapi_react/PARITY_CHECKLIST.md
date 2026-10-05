# FastAPI + React Parity Checklist

raceAnalysis.py is the behavioral and visual reference. Current local results
are in PARITY_REPORT.md; historical PR evidence is separate.

## Application parity

- [x] Seven sections, nested panels, labels and source descriptions
- [x] Source fonts, branding, headings, sidebar, spacing and responsive tabs
- [x] Boolean, category, numeric and date filters; cross-section state
- [x] Explicit displayed column selection/order, duplicates and hidden fields
- [x] Full 4,629-row raw dataset with 530 configured visible columns
- [x] Numeric precision, dates, localized times, indices and conditional styles
- [x] Schedule enrichment/highlighting, Next Race tables and forecasts
- [x] Analytics encodings, tire selectors, regressions and diagnostic tables
- [x] All six models and seven nested model panels
- [x] Artifact selection, custom ensembles and feature-order validation
- [x] Betting calculator, simulation, paper replay, calibration and CSV uploads
- [x] CSV downloads, PNG exports, fullscreen and chart data views
- [x] Grid search, selection/copy, sorting, resizing, visibility and pinning
- [x] Explicit bin-count comparison and shared temporal audit
- [x] Training controls match the disabled research configuration
- [x] Production application has no Streamlit server/runtime dependency

## Evidence

- [x] Actual Streamlit table/model oracle: 77 comparisons, zero failures
- [x] Filtered-data oracle: four comparisons, zero failures
- [x] Data Explorer and Next Race CSV contents match the live reference
- [x] All 24 screenshot pairs pass unchanged 2% desktop / 3% tablet/mobile tolerances
- [x] Missing expected screenshot files fail verification
- [x] Actual fonts, normal animations and no region masking
- [x] Capture/interaction/experiment reports contain zero browser errors
- [x] Uploads, downloads, themes, mobile sidebar and explicit experiments exercised

Mobile Raw Data uses the unchecked state in both captures. Visual comparison
establishes parity within the stated tolerances, not pixel identity.

## Quality gates

- [x] Python compilation, including exports, chart helpers and shared audit
- [x] Ruff and strict mypy
- [x] Backend: 59 passing tests; 87.36% coverage (80% required)
- [x] ESLint and TypeScript
- [x] Frontend: 57 passing tests; 73.74% line coverage; thresholds unchanged
- [x] Production React build
- [x] Production npm audit: zero vulnerabilities
- [x] Reproducible dependency install and narrow Glide patch

## Separate release considerations

- [ ] WCAG AA contrast: preserved reference styling still produces axe findings
- [ ] Deployment/rollback verification in the intended production environment
- [ ] Equivalent concurrent-load benchmark if comparative claims are needed
- [ ] Integrate local work into the intended PR and run CI on that exact head

These release considerations are not claims about work already performed.
No deployment or GitHub mutation was requested or performed in this completion.
