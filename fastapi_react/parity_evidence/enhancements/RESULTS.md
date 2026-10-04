# Main-app enhancement acceptance

Generated: 2026-10-04T11:42:57.712Z

All automated acceptance flows passed.

| Requirement | Flow | Result |
| --- | --- | --- |
| B1-B3 | Real API response reuse, revision, timing, and cache exclusions | pass |
| D1-D2 | Light contrast, focus, compact brand/header, and sticky navigation | pass |
| D3-F4 | Semantic all-field search, selectable columns, paging, years, and fonts | pass |
| F5 | Four-driver maximum and mathematically correct descriptive comparison | pass |
| D3-F4-regression | Original canvas grid search, column visibility, and complete CSV export | pass |
| F1 | Named views survive reload and restore both page and filter settings | pass |
| F2 | Unicode sharing restores settings in a fresh browser and excludes private values | pass |
| F3 | Native command palette opens with keyboard, filters, navigates with Enter, and closes with Escape | pass |
| F6 | Context export records safe settings, UTC time, revision and model provenance | pass |
| B1-client | Repeated navigation checks revision and reuses the bounded client response | pass |
| D1-D2-dark | Dark contrast and theme controls remain usable | pass |
| D1-D2-mobile | Compact mobile layout, visible navigation, and sidebar controls | pass |
| D4-loading | Loading feedback is visible, polite, outside the busy region, and clears on completion | pass |
| D4-update | Updating a control retains the existing analysis while its replacement is loading | pass |
| D4-stale | Navigating away aborts a stale request and late content cannot replace current results | pass |
| D4-failure | A failure has understandable text and Retry recovers | pass |

## Screenshots

- [desktop-light](screenshots/desktop-light.png)
- [accessible-table](screenshots/accessible-table.png)
- [driver-comparison](screenshots/driver-comparison.png)
- [saved-views](screenshots/saved-views.png)
- [command-palette](screenshots/command-palette.png)
- [desktop-dark](screenshots/desktop-dark.png)
- [mobile-light](screenshots/mobile-light.png)
- [loading](screenshots/loading.png)
- [failure-retry](screenshots/failure-retry.png)

The explicit loading, cancellation and failure fixtures are listed separately from unexpected browser/network errors in [results.json](results.json). Cache invalidation and bounded retention also require the fixture/unit checks listed there; production artifacts are never modified by this script.
