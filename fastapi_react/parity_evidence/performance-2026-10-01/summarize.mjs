import {readFile,writeFile} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const here=dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(await readFile(resolve(here,'measurements.json'),'utf8'));
const median=values=>{const a=values.filter(Number.isFinite).sort((a,b)=>a-b);return a.length?a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2:null;};
const stats=values=>({median:median(values),min:Math.min(...values),max:Math.max(...values),n:values.length});
const fields=['ready_ms','download_bytes','http_bytes','ws_received_bytes','http_requests','api_requests','retained_js_heap_bytes','task_ms','script_ms','layout_ms'];
const groups={};
for(const row of data.samples){
  if(row.label==='warmup')continue;
  const label=row.label.startsWith('concurrent_2_')?'concurrent_2':row.label.replace(/_\d+$/,'');
  groups[row.app]??={};groups[row.app][label]??=[];groups[row.app][label].push(row);
}
const summary={generated_at:data.generated_at,completed_at:data.completed_at,browser_version:data.browser_version,sample_count:data.samples.length,errors:data.errors,applications:{},disk:{},memory:{}};
for(const [app,labels] of Object.entries(groups)){
  const target=summary.applications[app]={};
  for(const [label,rows] of Object.entries(labels)){
    target[label]=Object.fromEntries(fields.map(field=>[field,stats(rows.map(r=>r[field]))]));
    for(const field of ['fcp_ms','lcp_ms','ttfb_ms','dom_content_ms'])target[label][field]=stats(rows.map(r=>r.timing[field]));
    target[label].api_response_ms=stats(rows.flatMap(r=>r.requests.filter(p=>p.url.includes('/api/views')).map(p=>p.response_ms)));
  }
  const totals=[];
  for(let run=1;run<=3;run++){
    const rows=data.samples.filter(r=>r.app===app&&r.label.endsWith('_'+run)&&/^(workflow_home|enable_filters|filter_year|navigate_|show_raw_data|return_analytics)/.test(r.label));
    totals.push({run,actions:rows.length,ready_ms:rows.reduce((a,r)=>a+r.ready_ms,0),download_bytes:rows.reduce((a,r)=>a+r.download_bytes,0)});
  }
  target.workflow={runs:totals,ready_ms:stats(totals.map(t=>t.ready_ms)),download_bytes:stats(totals.map(t=>t.download_bytes))};
}
for(const [name,files] of Object.entries(data.disk_inventory)){
  const runtime=files.filter(f=>!f.path.endsWith('.map')),js=runtime.filter(f=>f.path.endsWith('.js'));
  summary.disk[name]={runtime_files:runtime.length,runtime_bytes:runtime.reduce((a,f)=>a+f.bytes,0),total_with_maps_bytes:files.reduce((a,f)=>a+f.bytes,0),javascript_bytes:js.reduce((a,f)=>a+f.bytes,0),javascript_gzip_bytes:js.reduce((a,f)=>a+f.gzip_bytes,0),largest_runtime_files:[...runtime].sort((a,b)=>b.bytes-a.bytes).slice(0,5)};
}
for(const app of ['streamlit','fastapi','vite_dev']){
  const rows=data.memory_samples.map(s=>s[app]).filter(Boolean);
  summary.memory[app]={sampled_peak_rss_mb:Math.max(...rows.map(r=>r.rss_mb)),sampled_peak_private_mb:Math.max(...rows.map(r=>r.private_mb)),end_rss_mb:rows.at(-1)?.rss_mb,end_private_mb:rows.at(-1)?.private_mb,samples:rows.length};
}
await writeFile(resolve(here,'summary.json'),JSON.stringify(summary,null,2)+'\n');
const names=['streamlit','react_dev','react_production'],title={streamlit:'Streamlit :8502',react_dev:'React dev :5174',react_production:'React production preview'};
const group=(app,label)=>summary.applications[app][label];
const get=(app,label,field='ready_ms')=>group(app,label)[field].median;
const seconds=n=>(n/1000).toFixed(2)+' s',mb=n=>(n/1000000).toFixed(2)+' MB',kb=n=>(n/1000).toFixed(1)+' KB';
const cells=(label,field,format)=>names.map(app=>format(get(app,label,field))).join(' | ');
const tableHeader='| Measurement | Streamlit | React dev | React production |\n|---|---:|---:|---:|';
const loadRows=[['Fresh browser: home finished loading','fresh','ready_ms',seconds],['Cached browser: home reload','cached_reload','ready_ms',seconds],['Fresh browser: first content painted','fresh','fcp_ms',seconds],['Fresh browser: largest viewport paint observed','fresh','lcp_ms',seconds],['First-load download','fresh','download_bytes',mb],['Cached reload download','cached_reload','download_bytes',kb],['First-load HTTP requests, including cache lookups','fresh','http_requests',n=>String(n)],['First-load API view requests','fresh','api_requests',n=>String(n)],['Home retained JavaScript heap','fresh','retained_js_heap_bytes',mb],['Home browser main-thread task time','fresh','task_ms',seconds],['Two simultaneous fresh home loads: median per user','concurrent_2','ready_ms',seconds]];
const actionRows=[['Enable filters','enable_filters'],['Change minimum Year by one','filter_year'],['Open Analytics','navigate_Analytics & Visualizations'],['Open Schedule','navigate_Schedule'],['Open Next Race','navigate_Next Race'],['Open Predictive Models','navigate_Predictive Models'],['Open Data & Debug','navigate_Data & Debug'],['Show full Raw Data','show_raw_data'],['Open Betting Research','navigate_Betting Research'],['Return to Analytics','return_analytics']];
const speed=get('streamlit','fresh')/get('react_production','fresh');
const reduction=100*(1-get('react_production','fresh','download_bytes')/get('streamlit','fresh','download_bytes'));
const rawReduction=100*(1-get('react_production','show_raw_data','download_bytes')/get('streamlit','show_raw_data','download_bytes'));
const rawWait=get('react_production','show_raw_data','api_response_ms');
const workflowTime=(app)=>summary.applications[app].workflow.ready_ms.median;
const workflowBytes=(app)=>summary.applications[app].workflow.download_bytes.median;
const source=summary.disk.streamlit_static,react=summary.disk.react_dist;
const followup=await readFile(resolve(here,'RAW_DATA_OPTIMIZATION.md'),'utf8').catch(()=>null);
const report=`# Streamlit versus React performance report

${followup?'**Follow-up:** The raw-data backend has since been optimized. See [Raw-data backend optimization](RAW_DATA_OPTIMIZATION.md) for the updated measurements, changes, verification and bandwidth tradeoff. The original measurements below are retained as the baseline.\n\n':''}Measured October 1, 2026 (America/New_York), using the current local working tree.
The run completed at ${data.completed_at}. Chromium ${data.browser_version}, desktop 1280×900.
MB means 1,000,000 bytes; KB means 1,000 bytes. Times below are medians.

## Findings

The local React production build finished its first home load **${speed.toFixed(1)}× faster** and downloaded **${reduction.toFixed(1)}% less data** than the running Streamlit application. The actual React development site also loaded much faster, but its development modules are larger than the production bundle.

Streamlit is faster on most later tab switches because it has already calculated and sent the hidden sections. React loads each selected section through the API. React's large raw-data view is slower to finish, although it transfers **${rawReduction.toFixed(1)}% less data** and retains much less browser JavaScript memory.

These are measurements of the current implementations and their current compression/cache settings, not inherent limits of either framework.

## Home-page load, bandwidth and browser work

${tableHeader}
${loadRows.map(([name,label,field,format])=>`| ${name} | ${cells(label,field,format)} |`).join('\n')}

Finished loading means the application's script/request has completed, visible navigation is present, local requests have settled, fonts are ready and two animation frames have passed. It is different from first paint: Streamlit paints its initial page much earlier than it finishes preparing all sections. Largest-contentful-paint here is a local lab observation for the initial viewport, not a real-user Core Web Vitals assessment.

Fresh home completion ranges across the three runs: ${names.map(app=>`${title[app]} ${seconds(group(app,'fresh').ready_ms.min)}–${seconds(group(app,'fresh').ready_ms.max)}`).join('; ')}.

Streamlit's fresh home download comprises ${mb(get('streamlit','fresh','http_bytes'))} of HTTP transfer and ${mb(get('streamlit','fresh','ws_received_bytes'))} of WebSocket data. Its cached reload still receives essentially the same WebSocket application payload. Its WebSocket handshake negotiated **no compression extension**. The React API sends gzip-compressed JSON. The production preview additionally gzip-compresses JavaScript, CSS and HTML; Streamlit and React dev are measured as actually configured at their existing URLs.

At 1,000 independent fresh visits, these measured payloads correspond to approximately ${(get('streamlit','fresh','download_bytes')/1000000).toFixed(1)} GB for Streamlit, ${(get('react_dev','fresh','download_bytes')/1000000).toFixed(1)} GB for React dev, and ${(get('react_production','fresh','download_bytes')/1000000).toFixed(2)} GB for the React production preview. This is payload arithmetic, not a hosting bill or a WAN load-time prediction.

## Filtering and navigation

Each workflow starts in a fresh context, enables Filter Results and increases the minimum Year by one. It then visits the same six remaining root tabs, enables the full Raw Data checkbox, and returns to Analytics. Existing model artifacts are used. No explicit training, simulation or administrative experiment buttons are clicked; normal page-load calculations still run as implemented in each app, including React's offline-exported diagnostics.

${tableHeader}
${actionRows.map(([name,label])=>`| ${name} | ${cells(label,'ready_ms',seconds)} |`).join('\n')}

For the complete 11-action workflow, the sum of measured action completion times is ${seconds(workflowTime('streamlit'))} for Streamlit, ${seconds(workflowTime('react_dev'))} for React dev and ${seconds(workflowTime('react_production'))} for React production. The corresponding downloads are ${mb(workflowBytes('streamlit'))}, ${mb(workflowBytes('react_dev'))} and ${mb(workflowBytes('react_production'))}. These totals exclude human think time, the instrument's 500 ms collection windows and explicit garbage collection between actions.

Changing Year downloads ${mb(get('streamlit','filter_year','download_bytes'))} in Streamlit versus ${kb(get('react_production','filter_year','download_bytes'))} in React production. Most Streamlit tab switches transfer no additional application data; React requests a fresh presentation on each selected tab, including a return to Analytics.

## Full raw-data view

${tableHeader}
| Time to finish | ${cells('show_raw_data','ready_ms',seconds)} |
| Additional download | ${cells('show_raw_data','download_bytes',mb)} |
| Retained browser JavaScript heap | ${cells('show_raw_data','retained_js_heap_bytes',mb)} |
| Browser main-thread task time during this action | ${cells('show_raw_data','task_ms',seconds)} |

React production's raw-table API took a median **${seconds(rawWait)} before response headers arrived**, accounting for most of its ${seconds(get('react_production','show_raw_data'))} completion time. The full JSON body is about ${mb(median(data.samples.filter(r=>r.app==='react_production'&&r.label.startsWith('show_raw_data_')).flatMap(r=>r.requests.map(p=>p.decoded))))} before gzip. Browser main-thread work is much lower than Streamlit's. The clearest raw-table bottleneck is therefore in backend presentation construction, JSON serialization/compression and any request queuing, rather than drawing the visible canvas grid. Timing alone does not separate those backend components; no CPU profiler was run.

## Frontend size

| Size on disk | Streamlit packaged static client | React production build |
|---|---:|---:|
| Runtime files, excluding source maps | ${source.runtime_files} | ${react.runtime_files} |
| All runtime assets, including lazy chunks, fonts and images | ${mb(source.runtime_bytes)} | ${mb(react.runtime_bytes)} |
| All JavaScript before compression | ${mb(source.javascript_bytes)} | ${mb(react.javascript_bytes)} |
| All JavaScript gzip size, calculated offline | ${mb(source.javascript_gzip_bytes)} | ${mb(react.javascript_gzip_bytes)} |
| Directory including source maps | ${mb(source.total_with_maps_bytes)} | ${mb(react.total_with_maps_bytes)} |

Whole-directory size is different from first-load bandwidth: lazy Plotly/Vega chunks are included in disk totals but are not all fetched at startup. Source maps are also included in the final directory row but are not part of the ordinary browser transfers measured above. The models and data files are shared repository artifacts; no separate container image or clean server installation was built to compare total deployment size.

The React main JavaScript chunk is ${kb(react.largest_runtime_files.find(f=>/assets\/index-.*\.js$/.test(f.path))?.bytes||0)} uncompressed. The bundled footer logo alone is ${mb(data.disk_inventory.react_dist.find(f=>f.path==='betting-oracle-logo.png').bytes)}, a substantial share of the production home download. It is displayed at a much smaller size than its stored image.

## Server memory and concurrency

| Python server process measurement | Streamlit | React FastAPI backend |
|---|---:|---:|
| Highest sampled working set during the entire run | ${summary.memory.streamlit.sampled_peak_rss_mb.toFixed(0)} MiB | ${summary.memory.fastapi.sampled_peak_rss_mb.toFixed(0)} MiB |
| Highest sampled private resident memory | ${summary.memory.streamlit.sampled_peak_private_mb.toFixed(0)} MiB | ${summary.memory.fastapi.sampled_peak_private_mb.toFixed(0)} MiB |
| Working set at the end | ${summary.memory.streamlit.end_rss_mb.toFixed(0)} MiB | ${summary.memory.fastapi.end_rss_mb.toFixed(0)} MiB |

Memory was sampled approximately every three seconds across the listening Python process and its launch/reload helper processes. The highest value is a sampled peak, not a guaranteed maximum. Summed working sets include shared pages and can overstate unique RAM, so private resident memory is also shown. The Vite development server separately peaked at ${summary.memory.vite_dev.sampled_peak_rss_mb.toFixed(0)} MiB of summed working set. The temporary production static-serving harness, browser processes, GPU/off-heap browser memory and operating-system caches are excluded from the Python table. Retained JavaScript heap above was recorded after explicit browser garbage collection and is not total browser RAM or peak heap usage.

All six two-user home loads per application completed successfully, with zero recorded browser exceptions or console errors across the complete ${data.samples.length}-sample run. Median per-user times appear in the first table. This is a small two-user smoke test of fresh home loads, not a capacity, throughput or high-concurrency certification. The API's presentation rendering lock can serialize expensive requests; the home-load test does not establish heavy-page scaling.

## Priorities suggested by these measurements

1. **Reduce raw-table backend work.** Profile the full presentation/serialization path, consider a compact columnar response or chunked data loading, and preserve access to every original row and column. The measured wait is predominantly before the raw-table response begins.
2. **Cache recently visited React presentations where inputs have not changed.** Currently a tab change or return fetches and reconstructs the selected section. A cache keyed by filters, model choice and artifact freshness could improve warm navigation, with invalidation for actions/uploads and updated data.
3. **Serve an appropriately sized footer image.** Its ${mb(data.disk_inventory.react_dist.find(f=>f.path==='betting-oracle-logo.png').bytes)} source image dominates the small production home payload. Verify visual quality after any asset optimization.
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
`;
await writeFile(resolve(here,'REPORT.md'),report);
console.log(JSON.stringify({sample_count:summary.sample_count,errors:summary.errors,home:names.map(app=>({app,ready_s:get(app,'fresh')/1000,download_mb:get(app,'fresh','download_bytes')/1000000,fcp_s:get(app,'fresh','fcp_ms')/1000,heap_mb:get(app,'fresh','retained_js_heap_bytes')/1000000})),year:names.map(app=>({app,seconds:get(app,'filter_year')/1000,mb:get(app,'filter_year','download_bytes')/1000000})),raw:names.map(app=>({app,seconds:get(app,'show_raw_data')/1000,mb:get(app,'show_raw_data','download_bytes')/1000000,heap_mb:get(app,'show_raw_data','retained_js_heap_bytes')/1000000})),workflow:names.map(app=>({app,seconds:workflowTime(app)/1000,mb:workflowBytes(app)/1000000})),memory:summary.memory,disk:summary.disk},null,2));
