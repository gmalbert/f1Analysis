# Streamlit versus React performance report

**Follow-up:** The raw-data backend has since been optimized. See [Raw-data backend optimization](RAW_DATA_OPTIMIZATION.md) for the updated measurements, changes, verification and bandwidth tradeoff. The original measurements below are retained as the baseline.

Measured October 1, 2026 (America/New_York), using the current local working tree.
The run completed at 2026-10-02T01:42:34.900Z. Chromium 153.0.8010.12, desktop 1280×900.
MB means 1,000,000 bytes; KB means 1,000 bytes. Times below are medians.

## Findings

The local React production build finished its first home load **24.2× faster** and downloaded **94.8% less data** than the running Streamlit application. The actual React development site also loaded much faster, but its development modules are larger than the production bundle.

Streamlit is faster on most later tab switches because it has already calculated and sent the hidden sections. React loads each selected section through the API. React's large raw-data view is slower to finish, although it transfers **74.2% less data** and retains much less browser JavaScript memory.

These are measurements of the current implementations and their current compression/cache settings, not inherent limits of either framework.

## Home-page load, bandwidth and browser work

| Measurement | Streamlit | React dev | React production |
|---|---:|---:|---:|
| Fresh browser: home finished loading | 5.83 s | 0.39 s | 0.24 s |
| Cached browser: home reload | 5.59 s | 0.35 s | 0.21 s |
| Fresh browser: first content painted | 0.28 s | 0.23 s | 0.15 s |
| Fresh browser: largest viewport paint observed | 0.47 s | 0.25 s | 0.16 s |
| First-load download | 31.32 MB | 4.24 MB | 1.64 MB |
| Cached reload download | 25584.3 KB | 4.6 KB | 1.6 KB |
| First-load HTTP requests, including cache lookups | 202 | 29 | 7 |
| First-load API view requests | 0 | 2 | 1 |
| Home retained JavaScript heap | 66.68 MB | 4.79 MB | 2.97 MB |
| Home browser main-thread task time | 4.43 s | 0.21 s | 0.13 s |
| Two simultaneous fresh home loads: median per user | 12.34 s | 0.60 s | 0.33 s |

Finished loading means the application's script/request has completed, visible navigation is present, local requests have settled, fonts are ready and two animation frames have passed. It is different from first paint: Streamlit paints its initial page much earlier than it finishes preparing all sections. Largest-contentful-paint here is a local lab observation for the initial viewport, not a real-user Core Web Vitals assessment.

Fresh home completion ranges across the three runs: Streamlit :8502 5.80 s–6.47 s; React dev :5174 0.38 s–0.40 s; React production preview 0.23 s–0.24 s.

Streamlit's fresh home download comprises 6.05 MB of HTTP transfer and 25.27 MB of WebSocket data. Its cached reload still receives essentially the same WebSocket application payload. Its WebSocket handshake negotiated **no compression extension**. The React API sends gzip-compressed JSON. The production preview additionally gzip-compresses JavaScript, CSS and HTML; Streamlit and React dev are measured as actually configured at their existing URLs.

At 1,000 independent fresh visits, these measured payloads correspond to approximately 31.3 GB for Streamlit, 4.2 GB for React dev, and 1.64 GB for the React production preview. This is payload arithmetic, not a hosting bill or a WAN load-time prediction.

## Filtering and navigation

Each workflow starts in a fresh context, enables Filter Results and increases the minimum Year by one. It then visits the same six remaining root tabs, enables the full Raw Data checkbox, and returns to Analytics. Existing model artifacts are used. No explicit training, simulation or administrative experiment buttons are clicked; normal page-load calculations still run as implemented in each app, including React's offline-exported diagnostics.

| Measurement | Streamlit | React dev | React production |
|---|---:|---:|---:|
| Enable filters | 6.04 s | 1.00 s | 0.98 s |
| Change minimum Year by one | 5.92 s | 1.02 s | 0.98 s |
| Open Analytics | 2.30 s | 3.66 s | 3.14 s |
| Open Schedule | 0.43 s | 1.01 s | 0.96 s |
| Open Next Race | 0.43 s | 2.03 s | 1.99 s |
| Open Predictive Models | 0.47 s | 3.12 s | 3.00 s |
| Open Data & Debug | 0.37 s | 1.01 s | 0.97 s |
| Show full Raw Data | 6.99 s | 8.27 s | 8.62 s |
| Open Betting Research | 0.34 s | 1.00 s | 0.96 s |
| Return to Analytics | 2.31 s | 3.02 s | 3.18 s |

For the complete 11-action workflow, the sum of measured action completion times is 32.38 s for Streamlit, 25.46 s for React dev and 24.58 s for React production. The corresponding downloads are 99.41 MB, 14.07 MB and 9.85 MB. These totals exclude human think time, the instrument's 500 ms collection windows and explicit garbage collection between actions.

Changing Year downloads 21.32 MB in Streamlit versus 120.8 KB in React production. Most Streamlit tab switches transfer no additional application data; React requests a fresh presentation on each selected tab, including a return to Analytics.

## Full raw-data view

| Measurement | Streamlit | React dev | React production |
|---|---:|---:|---:|
| Time to finish | 6.99 s | 8.27 s | 8.62 s |
| Additional download | 22.39 MB | 5.77 MB | 5.77 MB |
| Retained browser JavaScript heap | 141.19 MB | 42.85 MB | 35.77 MB |
| Browser main-thread task time during this action | 6.78 s | 0.34 s | 0.26 s |

React production's raw-table API took a median **8.15 s before response headers arrived**, accounting for most of its 8.62 s completion time. The full JSON body is about 20.55 MB before gzip. Browser main-thread work is much lower than Streamlit's. The clearest raw-table bottleneck is therefore in backend presentation construction, JSON serialization/compression and any request queuing, rather than drawing the visible canvas grid. Timing alone does not separate those backend components; no CPU profiler was run.

## Frontend size

| Size on disk | Streamlit packaged static client | React production build |
|---|---:|---:|
| Runtime files, excluding source maps | 450 | 15 |
| All runtime assets, including lazy chunks, fonts and images | 23.23 MB | 8.09 MB |
| All JavaScript before compression | 19.87 MB | 6.35 MB |
| All JavaScript gzip size, calculated offline | 5.99 MB | 2.00 MB |
| Directory including source maps | 23.23 MB | 25.89 MB |

Whole-directory size is different from first-load bandwidth: lazy Plotly/Vega chunks are included in disk totals but are not all fetched at startup. Source maps are also included in the final directory row but are not part of the ordinary browser transfers measured above. The models and data files are shared repository artifacts; no separate container image or clean server installation was built to compare total deployment size.

The React main JavaScript chunk is 682.1 KB uncompressed. The bundled footer logo alone is 1.13 MB, a substantial share of the production home download. It is displayed at a much smaller size than its stored image.

## Server memory and concurrency

| Python server process measurement | Streamlit | React FastAPI backend |
|---|---:|---:|
| Highest sampled working set during the entire run | 1015 MiB | 1201 MiB |
| Highest sampled private resident memory | 931 MiB | 1100 MiB |
| Working set at the end | 715 MiB | 1201 MiB |

Memory was sampled approximately every three seconds across the listening Python process and its launch/reload helper processes. The highest value is a sampled peak, not a guaranteed maximum. Summed working sets include shared pages and can overstate unique RAM, so private resident memory is also shown. The Vite development server separately peaked at 233 MiB of summed working set. The temporary production static-serving harness, browser processes, GPU/off-heap browser memory and operating-system caches are excluded from the Python table. Retained JavaScript heap above was recorded after explicit browser garbage collection and is not total browser RAM or peak heap usage.

All six two-user home loads per application completed successfully, with zero recorded browser exceptions or console errors across the complete 138-sample run. Median per-user times appear in the first table. This is a small two-user smoke test of fresh home loads, not a capacity, throughput or high-concurrency certification. The API's presentation rendering lock can serialize expensive requests; the home-load test does not establish heavy-page scaling.

## Priorities suggested by these measurements

1. **Reduce raw-table backend work.** Profile the full presentation/serialization path, consider a compact columnar response or chunked data loading, and preserve access to every original row and column. The measured wait is predominantly before the raw-table response begins.
2. **Cache recently visited React presentations where inputs have not changed.** Currently a tab change or return fetches and reconstructs the selected section. A cache keyed by filters, model choice and artifact freshness could improve warm navigation, with invalidation for actions/uploads and updated data.
3. **Serve an appropriately sized footer image.** Its 1.13 MB source image dominates the small production home payload. Verify visual quality after any asset optimization.
4. **Use a production build with compression for hosting.** Development mode sends larger modules and React Strict Mode issues two initial view requests in this app; the production build issues one. These measured differences are a deployment-mode effect, not a user feature difference.
5. **Measure a deployed target before making WAN or capacity promises.** Add real network latency, representative concurrency and expensive page requests. Streamlit compression and production cache settings could materially change its transfer results.

## Method and evidence limits

- Both existing servers were kept running. Their data/model caches were warmed first; these are fresh-browser measurements, **not cold Python-server startup measurements**.
- Three fresh loads, three cached reloads and three complete workflows were measured for each app; concurrent home loads were measured in three pairs. Warmups are excluded from reported medians.
- A 100 ms readiness floor applies to all apps, so tiny differences between fast tab switches should not be interpreted as sub-100 ms precision. A separate 500 ms collection window captures late network activity and is excluded from completion times.
- Transfer counts include HTTP response headers plus incoming WebSocket payloads. WebSocket framing, TCP/TLS overhead and HTTP request/upload bytes are excluded. The direct Streamlit handshake confirms its payload is not per-message compressed in this setup. Cached resource reads are tracked as requests but contribute zero encoded transfer where reported by Chromium.
- The production preview serves the current built React assets on a temporary loopback port with gzip level 6, immutable hashed-asset caching, and a byte-preserving proxy to the same FastAPI backend. It is a controlled local preview, **not an already-deployed React production service**.
- Local disk and loopback network were used, without WAN throttling, CPU throttling or an isolated idle operating system. Background activity can affect short timings. Run order was Streamlit, React dev, then React production; warm application/model caches and first-time lazy-code loading affect different actions.
- Browser DOM-node telemetry in the raw JSON can include detached nodes and is not used as a retained-DOM comparison here. Browser heap, paint and task-time observations are lab diagnostics, not field INP/Core Web Vitals statistics.
- Embedded data URLs retain their media type, original character count and SHA-256; repeated inline image payloads are omitted from the evidence files. Request counts, timings, transfer sizes and all other measurements are unchanged.
- Initial benchmark setup retries corrected Streamlit locator/readiness checks; the retained final data contains three successful repetitions per reported group and no benchmark/runtime errors. The small pilot JSON files are separate and are not the report's source.

Evidence: [measurements.json](measurements.json), [summary.json](summary.json), [compare.mjs](compare.mjs), [server_memory.py](server_memory.py), [summarize.mjs](summarize.mjs).
