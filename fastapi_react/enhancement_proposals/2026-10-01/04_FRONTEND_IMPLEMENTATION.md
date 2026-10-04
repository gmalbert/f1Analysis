# Complete frontend implementation

These are complete source files, not pseudocode. They were integrated in the isolated preview and passed the build, ESLint, type checks, the 63-test frontend suite, and the browser checks. Use a review branch and compare against [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) before replacing files in a checkout that has moved on.

## Copy map

Paths on the right are relative to `fastapi_react/frontend/`.

| Supplied source | Destination |
| --- | --- |
| `code/frontend/App.jsx` | `src/App.jsx` |
| `code/frontend/App.test.jsx` | `src/App.test.jsx` |
| `code/frontend/main.jsx` | `src/main.jsx` |
| `code/frontend/Presentation.jsx` | `src/components/Presentation.jsx` |
| `preferences.js`, `viewClient.js`, `FeatureBar.jsx`, `EnhancedTable.jsx`, `SafePlotlyChart.jsx`, `ResearchJobs.jsx`, `enhancements.css`, `enhancements-env.d.ts`, `Enhancements.test.jsx` | Corresponding files under `src/enhancements/` |
| `code/deployment/optimize-assets.mjs` | `scripts/optimize-assets.mjs` |
| `code/deployment/check-budgets.mjs` | `scripts/check-budgets.mjs` |
| `code/deployment/vite.config.js` | `vite.config.js` |
| `code/deployment/package.json` | `package.json` |

The supplied package file retains the existing dependency list and adds asset optimization before build and budget enforcement after build. Keep the current lockfile; these changes do not introduce a new dependency. The existing Glide patch postinstall script remains.

## Integration behavior

1. `main.jsx` imports the enhancement CSS after the current parity CSS.
2. `App.jsx` mounts optional tools, restores safe shared settings, uses cancellable requests, adds the loading status, and uses the responsive footer image with the original PNG fallback.
3. `Presentation.jsx` routes table nodes through `EnhancedTable` and Plotly nodes through `SafePlotlyChart`. The original table grid remains available.
4. `FeatureBar.jsx` owns saved-view/link/context/command controls and persisted nonprivate options.
5. `viewClient.js` defaults to uncached requests. Its reuse mode requires the backend status route from the backend integration.
6. `App.test.jsx` adjusts the existing test mock to the new request boundary. `Enhancements.test.jsx` adds new contracts; it does not remove the existing suite.

## Activation

From a normal frontend checkout with dependencies installed:

```powershell
$env:VITE_F1_ENHANCEMENTS = '1'
npm run lint
npm run typecheck
npm test
npm run build
```

Vite embeds this flag at build time; changing the server environment after building will not toggle the compiled bundle. **Improve readability** and **Reuse recent views** are off by default. The backend enhancement status route must be installed before enabling client cache/context export or local jobs. For production, serve the built `dist` directory using the configuration in the deployment chapter.

To disable optional controls, rebuild with `VITE_F1_ENHANCEMENTS=0`. The candidate's request cleanup, safer Plotly lifecycle, and optimized footer loading are present regardless of this UI flag. Restore the prior tracked files to revert those implementation changes as well.

## Full source

Every source file below also exists separately in [code/frontend](code/frontend). Tests are included so installation retains useful checks. Deployment script/configuration source is in the [deployment chapter](06_DEPLOYMENT_AND_VALIDATION.md).

## code/frontend/App.jsx

[Separate source file](code/frontend/App.jsx)

```jsx
import { useEffect, useRef, useState } from 'react';
import { viewClient } from './enhancements/viewClient';
import { FeatureBar, LoadingFeedback, readOptions } from './enhancements/FeatureBar';
import { ResearchJobs } from './enhancements/ResearchJobs';
import { readSharedView, safeValues } from './enhancements/preferences';
import { ViewNodes } from './components/Presentation';
import { TabScroll } from './components/TabScroll';

const labels = ['📊 Data Explorer', '📈 Analytics & Visualizations', '🏎️ Schedule', '🏁 Next Race', '🤖 Predictive Models', '💾 Data & Debug', '📐 Betting Research'];
const routes = ['Data Explorer', 'Analytics', 'Current Season', 'Next Race', 'Predictive Models', 'Raw Data', 'Betting Research'];
const FEATURES_ENABLED = import.meta.env.VITE_F1_ENHANCEMENTS === '1';
const BASE_TITLE = 'Gridlocked - Formula 1 Betting & Analytics';

function sharedView() {
  try {return FEATURES_ENABLED ? readSharedView() : null;} catch {return null;}
}

function readPage() {
  const shared = sharedView();
  if (shared) return shared.page;
  let route;
  try {route = decodeURIComponent(location.hash.replace('#/', '').split('?')[0]);} catch {return 1;}
  const index = routes.indexOf(route);
  return index < 0 ? 1 : index + 1;
}

function readValues() {
  const shared = sharedView();
  if (shared) return shared.values;
  try {
    const values = JSON.parse(sessionStorage.getItem('f1analysis.view-values') || '{}');
    const oldFilters = JSON.parse(sessionStorage.getItem('f1analysis.filters') || 'null');
    if (oldFilters?.applied) values.filter_results_main = true;
    return values;
  } catch { return {}; }
}

export default function App() {
  const [options, setOptions] = useState(readOptions);
  const [page, setPage] = useState(readPage);
  const [values, setValues] = useState(readValues);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(true);
  const [request, setRequest] = useState(null);
  const [sidebarClosed, setSidebarClosed] = useState(false);
  const [settings, setSettings] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('f1analysis.theme') || 'light');
  const generation = useRef(0);
  const navigation = useRef(null);

  useEffect(() => {
    document.title = BASE_TITLE;
    const update = () => {
      const shared = sharedView();
      if (shared) {setValues(shared.values);setRequest(null);}
      setPage(readPage());
    };
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {localStorage.setItem('f1analysis.theme', theme);} catch { /* Optional storage. */ }
  }, [theme]);

  useEffect(() => {
    document.documentElement.dataset.enhancements = FEATURES_ENABLED && options.design ? 'on' : 'off';
  }, [options.design]);

  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    setBusy(true); setError(null);
    viewClient.load({page, values, action: request?.key}, {signal: controller.signal, enabled: FEATURES_ENABLED && options.cache})
      .then(result => {if (current === generation.current) setData({...result, page});})
      .catch(err => {if (err.name !== 'AbortError' && current === generation.current) setError(err.message);})
      .finally(() => {if (current === generation.current) setBusy(false);});
    return () => controller.abort();
  }, [page, values, request, options.cache]);

  function change(key, value) {
    const next = {...values, [key]: value};
    setValues(next); setRequest(null);
    try {
      sessionStorage.setItem('f1analysis.view-values', JSON.stringify(FEATURES_ENABLED ? safeValues(next) : next));
      sessionStorage.setItem('f1analysis.filters', JSON.stringify({applied: Boolean(next.filter_results_main), values: FEATURES_ENABLED ? safeValues(next) : next}));
    } catch { /* Uploaded CSVs may exceed the browser storage quota. */ }
  }

  function restore(view) {
    setValues(view.values); setPage(view.page); setRequest(null);
    try {sessionStorage.setItem('f1analysis.view-values', JSON.stringify(view.values));} catch { /* Optional storage. */ }
    location.hash = '/' + encodeURIComponent(routes[view.page-1]);
  }

  function navigate(index) {
    setPage(index + 1); setRequest(null);
    location.hash = `/${encodeURIComponent(routes[index])}`;
    window.scrollTo({top: 0});
  }

  useEffect(() => {
    const active = navigation.current?.querySelector('[aria-selected="true"]');
    if (active) {
      const parent = navigation.current;
      if (active.offsetLeft < parent.scrollLeft) parent.scrollLeft = active.offsetLeft;
      else if (active.offsetLeft + active.offsetWidth > parent.scrollLeft + parent.clientWidth) parent.scrollLeft = active.offsetLeft + active.offsetWidth - parent.clientWidth;
    }
  }, [page]);

  const sidebar = Boolean(values.filter_results_main) && !sidebarClosed;
  const shell = (data?.shell || []).filter(node => ['heading', 'caption'].includes(node.type));
  const act = key => setRequest({key, id: Date.now()});

  return <div className={`app-shell parity-app ${sidebar ? 'with-sidebar' : ''}`}>
    <a className="skip-link" href="#main-content">Skip to main content</a>
    <div className="app-toolbar">
      {values.filter_results_main && <button aria-label={sidebarClosed ? 'Open sidebar' : 'Close sidebar'} className="sidebar-toggle" style={{left: sidebarClosed ? 16 : 252}} onClick={() => setSidebarClosed(s => !s)}>{sidebarClosed ? '»' : '«'}</button>}
      <button className="settings-toggle" aria-label="Settings" aria-expanded={settings} onClick={() => setSettings(s => !s)}>⋮</button>
      {settings && <div className="settings-menu"><label><input aria-label="Use light theme" type="checkbox" checked={theme === 'light'} onChange={e => setTheme(e.target.checked ? 'light' : 'dark')} />Light theme</label></div>}
    </div>
    {sidebar && <aside className="filter-sidebar" aria-label="Data filters"><div className="view-flow"><ViewNodes nodes={data?.sidebar} values={values} change={change} action={act} /></div></aside>}
    <div className="main-shell">
      {FEATURES_ENABLED && <details><summary>Analysis tools</summary><FeatureBar page={page} values={values} options={options} setOptions={setOptions} restore={restore} navigate={navigate}/></details>}
      <header className="parity-header">
        <img src="/api/brand/logo" alt="Gridlocked" width="450" height="264" />
        {shell.length ? <div className="view-flow shell-copy"><ViewNodes nodes={shell} /></div> : <h1 className="shell-title">F1 Races from 2016 to {new Date().getFullYear()}</h1>}
      </header>
      <nav className="parity-nav" aria-label="Sections"><div role="tablist" aria-label="Analysis sections" ref={navigation}>
        {(data?.tabs?.length ? data.tabs : labels).map((label, i) => <button role="tab" id={`section-tab-${i}`} aria-selected={page === i + 1} aria-controls={`section-panel-${i}`} tabIndex={page === i + 1 ? 0 : -1} key={label} onClick={() => navigate(i)} onKeyDown={event => {if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {event.preventDefault(); const index = event.key === 'Home' ? 0 : event.key === 'End' ? labels.length - 1 : (i + (event.key === 'ArrowRight' ? 1 : -1) + labels.length) % labels.length; navigate(index); navigation.current?.querySelectorAll('button')[index]?.focus();}}}>{label}</button>)}
      </div><TabScroll target={navigation} /></nav>
      {FEATURES_ENABLED && <LoadingFeedback busy={busy} hasResults={data?.page === page}/>}
      <main id="main-content" tabIndex={-1} aria-busy={busy}>
        {error && <div className="view-notice error" role="alert">{error}<button className="view-button" onClick={() => setRequest({key: null, id: Date.now()})}>Retry</button></div>}
        {labels.map((_, i) => <div key={i} role="tabpanel" id={`section-panel-${i}`} aria-labelledby={`section-tab-${i}`} hidden={page !== i + 1} className="view-flow">{data?.page === i + 1 && page === i + 1 && <ViewNodes nodes={data.nodes} values={values} change={change} action={act} />}</div>)}
        {FEATURES_ENABLED && <ResearchJobs values={values}/>}
        {busy && !FEATURES_ENABLED && <span className="sr-only" role="status">Loading analysis…</span>}
        <footer className="parity-footer"><p>Powered by <a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer">Betting Oracle</a></p><p>Sports Prediction Analytics</p><a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer">{FEATURES_ENABLED ? <picture><source type="image/webp" srcSet="/betting-oracle-logo-60.webp 1x, /betting-oracle-logo-120.webp 2x"/><img src="/betting-oracle-logo.png" alt="Betting Oracle Logo" loading="lazy" decoding="async"/></picture> : <img src="/betting-oracle-logo.png" alt="Betting Oracle Logo"/>}</a></footer>
      </main>
    </div>
  </div>;
}
```

## code/frontend/App.test.jsx

[Separate source file](code/frontend/App.test.jsx)

```jsx
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {beforeEach,describe,expect,it,vi} from 'vitest';
import App from './App';
import {viewClient} from './enhancements/viewClient';
vi.mock('./enhancements/viewClient',()=>({viewClient:{load:vi.fn(),clear:vi.fn()}}));
vi.mock('./components/ViewTable',()=>({ViewTable:()=>null}));
const nodes=[{type:'heading',level:2,text:'Data Explorer'},{type:'checkbox',key:'filter_results_main',label:'Filter Results',value:false}];
describe('reference application shell',()=>{
  beforeEach(()=>{
    window.history.replaceState({},'','/');sessionStorage.clear();localStorage.clear();
    Object.defineProperty(window,'scrollTo',{configurable:true,value:vi.fn()});
    viewClient.load.mockImplementation(async(payload)=>({shell:[{type:'heading',level:1,text:'F1 Races from 2016 to 2026'}],nodes:payload.page===1?nodes:[{type:'heading',level:2,text:`Page ${payload.page}`}],sidebar:[{type:'heading',level:2,text:'Select filters to apply:'}]}));
  });
  it('renders the reference heading, brand and seven accessible tabs',async()=>{
    render(<App/>);
    expect(await screen.findByRole('heading',{name:'Data Explorer'})).toBeInTheDocument();
    expect(screen.getByRole('img',{name:'Gridlocked'})).toHaveAttribute('src','/api/brand/logo');
    expect(screen.getAllByRole('tab')).toHaveLength(7);
    expect(document.title).toBe('Gridlocked - Formula 1 Betting & Analytics');
  });
  it('navigates and carries filter state into the next page',async()=>{
    render(<App/>);fireEvent.click(await screen.findByRole('checkbox',{name:'Filter Results'}));
    expect(await screen.findByRole('complementary',{name:'Data filters'})).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab',{name:/Analytics & Visualizations/}));
    expect(await screen.findByRole('heading',{name:'Page 2'})).toBeInTheDocument();
    expect(viewClient.load).toHaveBeenLastCalledWith(expect.objectContaining({page:2,values:{filter_results_main:true}}),expect.objectContaining({enabled:false}));
    expect(window.location.hash).toBe('#/Analytics');
    expect(JSON.parse(sessionStorage.getItem('f1analysis.view-values'))).toEqual({filter_results_main:true});
    fireEvent.click(screen.getByRole('button',{name:'Close sidebar'}));
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Open sidebar'}));
    expect(screen.getByRole('complementary')).toBeInTheDocument();
  });
  it('supports keyboard tab navigation and persistent theme selection',async()=>{
    render(<App/>);await screen.findByRole('heading',{name:'Data Explorer'});
    fireEvent.keyDown(screen.getByRole('tab',{name:/Data Explorer/}),{key:'ArrowRight'});
    expect(await screen.findByRole('heading',{name:'Page 2'})).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Settings'}));
    fireEvent.click(screen.getByRole('checkbox',{name:'Use light theme'}));
    await waitFor(()=>expect(document.documentElement.dataset.theme).toBe('dark'));
    expect(localStorage.getItem('f1analysis.theme')).toBe('dark');
  });
  it('shows a failed request and successfully retries it',async()=>{
    viewClient.load.mockRejectedValueOnce(new Error('Unable to load analysis'));
    render(<App/>);expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load analysis');
    fireEvent.click(screen.getByRole('button',{name:'Retry'}));
    expect(await screen.findByRole('heading',{name:'Data Explorer'})).toBeInTheDocument();
  });
});
```

## code/frontend/EnhancedTable.jsx

[Separate source file](code/frontend/EnhancedTable.jsx)

```jsx
import {useMemo, useState} from 'react';
import {ViewTable} from '../components/ViewTable';
import {displayCell} from '../components/Presentation';

const driverKeys = ['resultsDriverName','driverName','Driver'];

export function EnhancedTable({node}) {
  const [mode, setMode] = useState('grid'), [compare, setCompare] = useState(false);
  if (import.meta.env.VITE_F1_ENHANCEMENTS !== '1') return <ViewTable node={node}/>;
  const hasDrivers = node.columns.some(c => driverKeys.includes(c.key));
  return <section aria-label="Table display">
    <div className="enhancement-bar">
      <button aria-pressed={mode === 'grid'} onClick={() => setMode('grid')}>Interactive grid</button>
      <button aria-pressed={mode === 'accessible'} onClick={() => setMode('accessible')}>Accessible table</button>
      {hasDrivers && <button aria-expanded={compare} onClick={() => setCompare(s => !s)}>Compare drivers</button>}
    </div>
    {mode === 'grid' ? <ViewTable node={node}/> : <AccessibleTable node={node}/>}
    {compare && <DriverComparison node={node}/>}
  </section>;
}

function AccessibleTable({node}) {
  const columns = node.columns, rows = node.rows;
  const [selected, setSelected] = useState(() => columns.slice(0,8).map((_,i) => i));
  const [query, setQuery] = useState(''), [page, setPage] = useState(0);
  const visible = selected.filter(i => columns[i]);
  const matches = useMemo(() => rows.map((_,i) => i).filter(i =>
    !query || rows[i].some(value => String(value ?? 'None').toLowerCase().includes(query.toLowerCase()))
  ), [rows,query]);
  const pageCount = Math.max(1, Math.ceil(matches.length/50)), current = Math.min(page,pageCount-1);
  function toggle(index) {setSelected(old => old.includes(index) ? old.filter(i => i !== index) : [...old,index].sort((a,b) => a-b));}
  return <div className="accessible-table">
    <label>Search all fields <input value={query} onChange={e => {setQuery(e.target.value);setPage(0);}}/></label>
    <details><summary>Choose fields ({visible.length} of {columns.length})</summary>
      <div className="columns-list">{columns.map((column,index) => <label key={index}><input type="checkbox" checked={visible.includes(index)} onChange={() => toggle(index)}/>{column.label} ({column.key})</label>)}</div>
    </details>
    <div className="table-viewport"><table>
      <caption>{matches.length.toLocaleString()} matching rows · showing rows {matches.length ? current*50+1 : 0}–{Math.min((current+1)*50,matches.length)}. All fields are available in Choose fields.</caption>
      <thead><tr>{!node.hide_index && <th scope="col">{node.index_name || 'Row'}</th>}{visible.map(index => <th scope="col" key={index}>{columns[index].label}</th>)}</tr></thead>
      <tbody>{matches.slice(current*50,(current+1)*50).map(row => <tr key={row}>
        {!node.hide_index && <th scope="row">{String(node.index?.[row] ?? row)}</th>}
        {visible.map(index => <td key={index}>{displayCell(rows[row][index],columns[index],node.display?.[row]?.[index])}</td>)}
      </tr>)}</tbody>
    </table></div>
    <nav aria-label="Table row pages"><button disabled={!current} onClick={() => setPage(current-1)}>Previous rows</button> Page {current+1} of {pageCount} <button disabled={current+1 >= pageCount} onClick={() => setPage(current+1)}>Next rows</button></nav>
    <p>Use Interactive grid for the original sorting, selection, copying and full CSV export.</p>
  </div>;
}

function DriverComparison({node}) {
  const driverIndex = node.columns.findIndex(c => driverKeys.includes(c.key));
  const [drivers, setDrivers] = useState([]);
  const choices = useMemo(() => [...new Set(node.rows.map(row => row[driverIndex]).filter(Boolean))].sort(), [node.rows,driverIndex]);
  const fields = useMemo(() => ['resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained','DNF']
    .map(key => ({key,index:node.columns.findIndex(c => c.key === key)})).filter(f => f.index >= 0), [node.columns]);
  const summaries = useMemo(() => drivers.map(driver => {
    const sample = node.rows.filter(row => row[driverIndex] === driver);
    return {driver,rows:sample.length,values:fields.map(field => {
      const values = sample.map(row => row[field.index]);
      if (field.key === 'DNF') {
        const known = values.filter(v => v != null);
        return known.length ? (100*known.filter(v => v === true || v === 1 || String(v).toLowerCase() === 'true').length/known.length).toFixed(1)+'%' : 'No data';
      }
      const known = values.filter(v => typeof v === 'number' && Number.isFinite(v));
      return known.length ? (known.reduce((a,b) => a+b,0)/known.length).toFixed(2) : 'No data';
    })};
  }), [drivers,node.rows,driverIndex,fields]);
  return <div className="accessible-table">
    <h3>Driver comparison within this table</h3>
    <p>Historical descriptive averages over the currently filtered rows. Sample rows can differ from unique races. These are not forecasts or calibrated probabilities.</p>
    <div className="columns-list">{choices.map(driver => <label key={driver}><input type="checkbox" checked={drivers.includes(driver)} disabled={!drivers.includes(driver) && drivers.length >= 4} onChange={() => setDrivers(old => old.includes(driver) ? old.filter(d => d !== driver) : [...old,driver])}/>{driver}</label>)}</div>
    <table><caption>Compare up to four drivers</caption><thead><tr><th scope="col">Driver</th><th scope="col">Sample rows</th>{fields.map(f => <th scope="col" key={f.key}>{f.key === 'DNF' ? 'DNF rate among known rows' : 'Mean '+node.columns[f.index].label}</th>)}</tr></thead>
      <tbody>{summaries.map(summary => <tr key={summary.driver}><th scope="row">{summary.driver}</th><td>{summary.rows}</td>{summary.values.map((value,i) => <td key={i}>{value}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
```

## code/frontend/enhancements-env.d.ts

[Separate source file](code/frontend/enhancements-env.d.ts)

```typescript
/// <reference types="vite/client" />
```

## code/frontend/enhancements.css

[Separate source file](code/frontend/enhancements.css)

```css
/* Opt-in presentation profile. Load after parity.css. */
:root[data-enhancements='on']:not([data-theme='dark']) {
  --accent:#b4232d; --muted:#596273; --border:#cbd2dc;
}
:root[data-enhancements='on'][data-theme='dark'] {
  --accent:#ffb4ab; --muted:#c5cbd7; --border:#626b7c;
}
:root[data-enhancements='on'] .view-caption {opacity:1;color:var(--muted)}
:root[data-enhancements='on'] .main-shell {padding-top:40px;padding-bottom:64px}
:root[data-enhancements='on'] .parity-header>img {width:280px;height:auto}
:root[data-enhancements='on'] .view-help {color:var(--muted);border-color:currentColor}
:root[data-enhancements='on'] .view-notice.info {color:#17436a}
:root[data-enhancements='on'] .slider-values {font-weight:600}
:root[data-enhancements='on'] .table-toolbar {
  position:relative;top:auto;right:auto;opacity:1;pointer-events:auto;
  height:auto;min-height:44px;width:max-content;max-width:100%;margin-left:auto;z-index:5;
}
:root[data-enhancements='on'] .table-toolbar button {min-width:44px;min-height:44px;color:var(--text)}
:root[data-enhancements='on'] .canvas-table .column-picker {top:44px}
:root[data-enhancements='on'] button:focus-visible,
:root[data-enhancements='on'] a:focus-visible,
:root[data-enhancements='on'] input:focus-visible,
:root[data-enhancements='on'] select:focus-visible {outline:3px solid var(--accent);outline-offset:3px}
:root[data-enhancements='on'] .parity-nav {position:sticky;top:0;background:var(--page-bg);z-index:10}
:root[data-enhancements='on'] .parity-nav button {min-height:44px}
:root[data-enhancements='on'] .parity-footer {font-family:'Source Sans',sans-serif;color:var(--muted)}
:root[data-enhancements='on'] .parity-footer p+p {color:var(--muted)}
.enhancement-bar {display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:12px 0;font-family:'Source Sans',sans-serif}
.enhancement-bar button,.enhancement-bar select,.enhancement-bar input,.accessible-table button,.accessible-table select {min-height:44px}
.enhancement-error {color:#b4232d}
.load-feedback {position:fixed;bottom:12px;right:12px;max-width:calc(100vw - 24px);background:var(--page-bg);border:1px solid var(--border);padding:12px 16px;border-radius:8px;z-index:11}
.command-dialog {background:var(--page-bg);color:var(--text);border:1px solid var(--border);border-radius:12px;width:min(520px,calc(100vw - 32px));max-height:80vh}
.command-dialog::backdrop {background:#0008}
.command-dialog input {width:100%;min-height:44px}
.command-dialog ul {padding:0;list-style:none}
.command-dialog li button {width:100%;min-height:44px;text-align:left}
.accessible-table {max-width:100%;margin:16px 0}
.accessible-table .table-viewport {overflow:auto;max-height:500px}
.accessible-table table {border-collapse:collapse;width:100%;font-family:'Source Sans',sans-serif}
.accessible-table th,.accessible-table td {padding:8px;border:1px solid var(--border);text-align:left}
.accessible-table th {position:sticky;top:0;background:var(--page-bg)}
.accessible-table .columns-list {display:flex;gap:12px;flex-wrap:wrap;max-height:200px;overflow:auto}
.provenance {font-size:14px;color:var(--muted);padding:8px 0}
.research-job {border:1px solid var(--border);border-radius:8px;padding:16px;margin:16px 0}
@media(max-width:640px) {
  :root[data-enhancements='on'] .main-shell {padding-top:56px}
  :root[data-enhancements='on'] .parity-header>img {width:210px}
  :root[data-enhancements='on'] h1.view-heading,
  :root[data-enhancements='on'] .shell-title {font-size:30px}
  :root[data-enhancements='on'] .filter-sidebar {width:min(300px,calc(100vw - 56px));padding-bottom:80px}
  .enhancement-bar>* {max-width:100%}
}
@media(prefers-reduced-motion:reduce) {
  :root[data-enhancements='on'] * {scroll-behavior:auto!important;animation:none!important;transition:none!important}
}
@media print {
  :root[data-enhancements='on'] .app-toolbar,
  :root[data-enhancements='on'] .filter-sidebar,
  :root[data-enhancements='on'] .parity-nav,
  .enhancement-bar,.table-toolbar,.load-feedback {display:none!important}
  :root[data-enhancements='on'] .main-shell {margin:0!important;width:100%!important;padding:0!important}
  .accessible-table .table-viewport {max-height:none;overflow:visible}
}
```

## code/frontend/Enhancements.test.jsx

[Separate source file](code/frontend/Enhancements.test.jsx)

```jsx
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import {FeatureBar,readOptions} from './FeatureBar';
import {EnhancedTable} from './EnhancedTable';
import {createViewClient} from './viewClient.js';
import {hasUpload,readPresets,readSharedView,safeValues,savePreset,shareUrl} from './preferences.js';

vi.mock('../components/ViewTable',()=>({ViewTable:()=> <p>Original interactive grid</p>}));
beforeEach(() => {localStorage.clear();vi.stubEnv('VITE_F1_ENHANCEMENTS','1');});
afterEach(() => {vi.unstubAllEnvs();vi.unstubAllGlobals();});

it('saves and shares Unicode filters while excluding private uploaded and financial values',() => {
  const values = {filter_results_main:true,filter_driver:'José',range_filter_grandPrixYear:[2017,2026],f1bet_field_upload:{content:'private'},bankroll:5000};
  savePreset('Recent',2,values);
  expect(readPresets()[0].values).toEqual(safeValues(values));
  expect(readSharedView(new URL(shareUrl(2,values,'http://localhost/')).hash).values).toEqual(safeValues(values));
  expect(hasUpload(values)).toBe(true);
  expect(readOptions()).toEqual({design:false,cache:false});
  localStorage.setItem('f1analysis.enhancement-options','invalid');
  expect(readOptions().design).toBe(false);
});

it('deduplicates requests, invalidates changed revisions and bypasses action caching',async() => {
  let revision = 'r1', posts = 0;
  const fetcher = async url => {
    if(url.endsWith('/status'))return {ok:true,json:async() => ({revision})};
    posts++;await new Promise(resolve => setTimeout(resolve,5));
    return {ok:true,json:async() => ({nodes:[],posts})};
  };
  const client = createViewClient({fetcher}), payload = {page:1,values:{}};
  await Promise.all([client.load(payload,{enabled:true}),client.load(payload,{enabled:true})]);
  expect(posts).toBe(1);
  await client.load(payload,{enabled:true});expect(posts).toBe(1);
  revision='r2';await client.load(payload,{enabled:true});expect(posts).toBe(2);
  await client.load({...payload,action:'explicit'},{enabled:true});expect(posts).toBe(3);
  await client.load(payload,{enabled:true});expect(posts).toBe(4);
});

it('shows all-field semantic table paging and historical comparison without grouping years',() => {
  const node = {hide_index:true,columns:[
    {key:'grandPrixYear',label:'Year',kind:'NumberColumn'},
    {key:'resultsDriverName',label:'Driver',kind:'TextColumn'},
    {key:'resultsFinalPositionNumber',label:'Finish',kind:'NumberColumn'},
    {key:'DNF',label:'DNF',kind:'CheckboxColumn'}
  ],rows:Array.from({length:70},(_,i) => [2026,i%2 ? 'Driver A' : 'Driver B',i%2 ? 2 : 4,false])};
  render(<EnhancedTable node={node}/>);
  fireEvent.click(screen.getByRole('button',{name:'Accessible table'}));
  expect(screen.getAllByRole('cell',{name:'2026'})).toHaveLength(50);
  fireEvent.click(screen.getByRole('button',{name:'Next rows'}));
  expect(screen.getAllByRole('cell',{name:'2026'})).toHaveLength(20);
  fireEvent.click(screen.getByRole('button',{name:'Compare drivers'}));
  fireEvent.click(screen.getByRole('checkbox',{name:'Driver A'}));
  expect(screen.getByRole('cell',{name:'2.00'})).toBeInTheDocument();
});

it('keeps original grid rendering when enhancements are disabled',() => {
  vi.stubEnv('VITE_F1_ENHANCEMENTS','0');
  render(<EnhancedTable node={{columns:[],rows:[]}}/>);
  expect(screen.getByText('Original interactive grid')).toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Accessible table'})).not.toBeInTheDocument();
});

it('renders the saved-view tools and records a named preset',async() => {
  vi.stubGlobal('fetch',vi.fn(async() => ({ok:true,json:async() => ({revision:'abcdefghijklmno',build_revision:'test',dataset:{name:'data.parquet',modified_at:'today'},models:[]})})));
  render(<FeatureBar page={1} values={{filter_results_main:true}} options={{design:false,cache:false}} setOptions={vi.fn()} restore={vi.fn()} navigate={vi.fn()}/>);
  fireEvent.change(screen.getByRole('textbox',{name:'View name'}),{target:{value:'History'}});
  fireEvent.click(screen.getByRole('button',{name:'Save view'}));
  await waitFor(() => expect(readPresets()[0].name).toBe('History'));
  expect(await screen.findByRole('status')).toHaveTextContent('View saved');
});
```

## code/frontend/FeatureBar.jsx

[Separate source file](code/frontend/FeatureBar.jsx)

```jsx
import {useEffect, useId, useRef, useState} from 'react';
import {deletePreset, readPresets, routes, safeValues, savePreset, shareUrl} from './preferences.js';

export function readOptions() {
  try {return {...{design: false, cache: false}, ...JSON.parse(localStorage.getItem('f1analysis.enhancement-options') || '{}')};}
  catch {return {design: false, cache: false};}
}

function downloadJSON(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function FeatureBar({page, values, options, setOptions, restore, navigate}) {
  const [presets, setPresets] = useState(() => readPresets());
  const [name, setName] = useState(''), [chosen, setChosen] = useState('');
  const [message, setMessage] = useState(''), [error, setError] = useState('');
  const [provenance, setProvenance] = useState(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/enhancements/status', {signal: controller.signal, cache: 'no-store'})
      .then(response => {if (!response.ok) throw new Error('Unavailable'); return response.json();})
      .then(setProvenance).catch(() => {});
    return () => controller.abort();
  }, [page, values]);
  function run(operation) {
    setError(''); setMessage('');
    Promise.resolve().then(operation).catch(err => setError(err.message));
  }
  function option(key, checked) {
    const next = {...options, [key]: checked}; setOptions(next);
    run(() => localStorage.setItem('f1analysis.enhancement-options', JSON.stringify(next)));
  }
  return <section aria-label="Analysis tools">
    <div className="enhancement-bar">
      <label><input type="checkbox" checked={options.design} onChange={e => option('design', e.target.checked)}/> Improve readability</label>
      <label><input type="checkbox" checked={options.cache} onChange={e => option('cache', e.target.checked)}/> Reuse recent views</label>
      <label>View name <input value={name} maxLength={80} onChange={e => setName(e.target.value)}/></label>
      <button onClick={() => run(() => {setPresets(savePreset(name, page, values));setMessage('View saved on this device.');})}>Save view</button>
      <label>Saved views <select value={chosen} onChange={e => setChosen(e.target.value)}><option value="">Choose a view</option>{presets.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label>
      <button disabled={!chosen} onClick={() => {const item = presets.find(p => p.name === chosen);if(item)restore(item);}}>Load view</button>
      <button disabled={!chosen} onClick={() => run(() => {setPresets(deletePreset(chosen));setChosen('');})}>Delete view</button>
      <button onClick={() => run(async () => {await navigator.clipboard.writeText(shareUrl(page, values));setMessage('View link copied. Uploads and betting inputs are excluded.');})}>Copy view link</button>
      <button onClick={() => downloadJSON({
        schema: 'f1-analysis-context-v1', exported_at: new Date().toISOString(),
        page: routes[page-1], values: safeValues(values), provenance
      }, 'analysis-context-' + new Date().toISOString().slice(0,10) + '.json')}>Download analysis context</button>
      <button onClick={() => window.print()}>Print current view</button>
      <CommandPalette navigate={navigate}/>
    </div>
    {provenance && <details className="provenance"><summary>Data revision {provenance.revision.slice(0,12)} · artifact details</summary>
      <p>Dataset: {provenance.dataset.name} · file updated {provenance.dataset.modified_at || 'unknown'}</p>
      <p>Build: {provenance.build_revision}. File revision tracks changes; model data hashes below come from existing manifests.</p>
      {provenance.models.map((model, i) => <p key={i}>{model.estimator || model.model_name} · {model.model_version || 'unversioned'} · trained {model.trained_at || 'unknown'} · training ends at {model.training_end_event || 'unknown'} · calibration {model.calibration_method || 'not recorded'}. {(model.notes || []).join(' ')}</p>)}
    </details>}
    {message && <p role="status">{message}</p>}
    {error && <p role="alert" className="enhancement-error">{error}</p>}
  </section>;
}

function CommandPalette({navigate}) {
  const dialog = useRef(null), label = useId();
  const [open, setOpen] = useState(false), [query, setQuery] = useState('');
  const matches = routes.map((name,index) => ({name,index})).filter(item => item.name.toLowerCase().includes(query.toLowerCase()));
  useEffect(() => {
    function key(event) {if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {event.preventDefault();setOpen(s => !s);}}
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key);
  }, []);
  useEffect(() => {
    if (open && !dialog.current.open) {dialog.current.showModal();dialog.current.querySelector('input')?.focus();}
    else if (!open && dialog.current.open) dialog.current.close();
  }, [open]);
  const choose = index => {navigate(index);setOpen(false);setQuery('');};
  return <>
    <button onClick={() => setOpen(true)}>Find section (Ctrl/⌘ K)</button>
    <dialog ref={dialog} className="command-dialog" aria-labelledby={label} onCancel={() => setOpen(false)} onClose={() => setOpen(false)}>
      <h2 id={label}>Find a section</h2>
      <label>Search sections <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => {if(e.key === 'Enter' && matches[0]) {e.preventDefault();choose(matches[0].index);}}}/></label>
      <ul>{matches.map(item => <li key={item.name}><button onClick={() => choose(item.index)}>{item.name}</button></li>)}</ul>
      {!matches.length && <p>No matching sections.</p>}
      <button onClick={() => setOpen(false)}>Close</button>
    </dialog>
  </>;
}

export function LoadingFeedback({busy, hasResults}) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    setSeconds(0);
    if (!busy) return;
    const start = Date.now(), timer = setInterval(() => setSeconds(Math.floor((Date.now()-start)/1000)), 1000);
    return () => clearInterval(timer);
  }, [busy]);
  if (!busy) return null;
  return <div className="load-feedback" role="status" aria-live="polite">
    {hasResults ? 'Updating analysis; existing results remain visible.' : 'Loading analysis.'}
    {seconds >= 2 && <span aria-hidden="true"> {seconds}s elapsed.</span>}
    {seconds >= 10 && <span> This calculation is taking longer than usual.</span>}
  </div>;
}
```

## code/frontend/main.jsx

[Separate source file](code/frontend/main.jsx)

```jsx
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./parity.css";
import "./enhancements/enhancements.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>
);
```

## code/frontend/preferences.js

[Separate source file](code/frontend/preferences.js)

```javascript
export const routes = ['Data Explorer', 'Analytics', 'Current Season', 'Next Race', 'Predictive Models', 'Raw Data', 'Betting Research'];
const storageKey = 'f1analysis.saved-views.v1';
const allowed = /^(filter_results_main|(?:range_filter_|checkbox_filter_|filter_).+|_tabs:.+|Select Model Type|tire_year_select|tire_race_select)$/;

export function safeValues(values = {}) {
  return Object.fromEntries(Object.entries(values).filter(([key, value]) =>
    allowed.test(key) && (
      value == null || ['string', 'number', 'boolean'].includes(typeof value) ||
      Array.isArray(value) && value.length <= 20 && value.every(item =>
        item == null || ['string', 'number', 'boolean'].includes(typeof item))
    )
  ));
}

export function validateView(view) {
  if (!view || view.version !== 1 || !Number.isInteger(view.page) ||
      view.page < 1 || view.page > routes.length) throw new Error('Unsupported saved view.');
  return {version: 1, page: view.page, values: safeValues(view.values)};
}

export function readPresets(storage = localStorage) {
  try {
    return JSON.parse(storage.getItem(storageKey) || '[]').slice(0, 20)
      .map(item => ({...validateView(item), name: String(item.name || 'Saved view').slice(0, 80)}));
  } catch { return []; }
}

export function savePreset(name, page, values, storage = localStorage) {
  const label = name.trim().slice(0, 80);
  if (!label) throw new Error('Enter a name for this view.');
  const view = {...validateView({version: 1, page, values}), name: label};
  const next = [view, ...readPresets(storage).filter(item => item.name !== label)].slice(0, 20);
  storage.setItem(storageKey, JSON.stringify(next));
  return next;
}

export function deletePreset(name, storage = localStorage) {
  const next = readPresets(storage).filter(item => item.name !== name);
  storage.setItem(storageKey, JSON.stringify(next));
  return next;
}

export function shareUrl(page, values, base = location.href) {
  const view = validateView({version: 1, page, values});
  const bytes = new TextEncoder().encode(JSON.stringify(view));
  const token = btoa(Array.from(bytes, byte => String.fromCharCode(byte)).join(''))
    .replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
  if (token.length > 6000) throw new Error('This view is too large for a link. Save it locally instead.');
  const url = new URL(base);
  url.hash = '/' + encodeURIComponent(routes[page - 1]) + '?view=' + token;
  return url.toString();
}

export function readSharedView(hash = location.hash) {
  const token = new URLSearchParams(hash.split('?')[1] || '').get('view');
  if (!token) return null;
  if (token.length > 6000) throw new Error('The shared link is too large.');
  const normalized = token.replaceAll('-', '+').replaceAll('_', '/');
  const decoded = atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '='));
  return validateView(JSON.parse(new TextDecoder().decode(Uint8Array.from(decoded, c => c.charCodeAt(0)))));
}

export function stableKey(value) {
  if (Array.isArray(value)) return '[' + value.map(stableKey).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort()
    .map(key => JSON.stringify(key) + ':' + stableKey(value[key])).join(',') + '}';
  return JSON.stringify(value);
}

export function hasUpload(values) {
  return Object.entries(values).some(([key,value]) =>
    /upload|csv|ledger/i.test(key) ||
    value && typeof value === 'object' && !Array.isArray(value) ||
    Array.isArray(value) && value.some(item => item && typeof item === 'object'));
}
```

## code/frontend/Presentation.jsx

[Separate source file](code/frontend/Presentation.jsx)

```jsx
import { useEffect, useId, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import {EnhancedTable} from '../enhancements/EnhancedTable';
import {SafePlotlyChart} from '../enhancements/SafePlotlyChart';
import {TabScroll} from './TabScroll';
import {useTheme} from './useTheme';

const numberFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 });

function isYearField(column) {
  return [column.key, column.label, column.field, column.title].some(name =>
    typeof name === 'string' && /\byear\b/i.test(name.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[_-]/g, ' '))
  );
}

function formatYearEncodings(spec) {
  if (!spec || typeof spec !== 'object') return;
  if (spec.encoding) {
    for (const [channel, definition] of Object.entries(spec.encoding)) {
      for (const field of Array.isArray(definition) ? definition : [definition]) {
        if (!field || !isYearField(field) || field.type === 'temporal') continue;
        if (channel === 'tooltip' || channel === 'text') field.format = 'd';
        else if ((channel === 'x' || channel === 'y') && field.axis !== null) field.axis = {...field.axis, format: 'd'};
      }
    }
  }
  for (const key of ['layer', 'hconcat', 'vconcat', 'concat']) {
    for (const child of spec[key] || []) formatYearEncodings(child);
  }
  if (spec.spec) formatYearEncodings(spec.spec);
}

export function displayCell(value, column, styled) {
  if (value == null) return 'None';
  if (column.kind === 'CheckboxColumn') return value ? '☑' : '☐';
  if (column.kind === 'DateColumn' || column.kind === 'DatetimeColumn') return String(value).slice(0, column.kind === 'DateColumn' ? 10 : 19).replace('T', ' ');
  if (column.kind === 'TimeColumn') {
    const time=String(value).slice(0,8);
    if(column.format==='localized') {
      const [hours,minutes,seconds]=time.split(':').map(Number);
      const date=new Date(); date.setUTCHours(hours,minutes,seconds||0,0);
      return date.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',second:'2-digit'});
    }
    return time;
  }
  if (typeof value === 'number') {
    if (isYearField(column)) return String(Math.trunc(value));
    const format = column.format;
    if (format === '%d') return String(Math.trunc(value));
    const precision = /^%\.(\d+)f$/.exec(format || '');
    if (precision) return value.toFixed(Number(precision[1]));
    if (format === '%.0f%%') return `${value.toFixed(0)}%`;
    if (format === 'percent') return `${(value * 100).toFixed(2)}%`;
    if (styled != null) return String(styled);
    return numberFormat.format(value);
  }
  return styled ?? String(value);
}

function VegaChart({ node }) {
  const ref = useRef(null);
  const outer=useRef(null),viewRef=useRef(null);
  const [showData,setShowData]=useState(false);
  const [error, setError] = useState(null);
  const theme=useTheme();
  useEffect(() => {
    let view, observer, disposed = false;
    const el = ref.current;
    import('vega-embed').then(async ({default: embed}) => {
      const spec = structuredClone(node.spec);
      formatYearEncodings(spec);
      const dark = theme === 'dark';
      const text = dark ? '#fafafa' : '#31333f';
      spec.width = Math.max(120, el.clientWidth);
      if (typeof spec.height==='number' && spec.height<=0) delete spec.height;
      spec.padding={...(typeof spec.padding==='object'?spec.padding:{}),bottom:20};
      spec.background = 'transparent';
      const gridColor=dark?'#333640':'#e6e7eb';
      const defaults={font:'Source Sans',background:'transparent',fieldTitle:'verbal',autosize:{type:'fit',contains:'padding'},view:{columns:1,strokeWidth:0,stroke:'transparent',continuousHeight:350,continuousWidth:400},axis:{labelFontSize:12,labelFontWeight:400,labelColor:text,labelFontStyle:'normal',titleFontWeight:400,titleFontSize:14,titleColor:text,titleFontStyle:'normal',ticks:false,gridColor,domain:false,domainWidth:1,domainColor:gridColor,labelFlush:true,labelFlushOffset:1,labelBound:false,labelLimit:100,titlePadding:16,labelPadding:16,labelSeparation:2,labelOverlap:true},legend:{labelFontSize:14,labelFontWeight:400,labelColor:text,titleFontSize:14,titleFontWeight:400,titleColor:text,titlePadding:2,labelPadding:16,columnPadding:8,rowPadding:2,padding:8,symbolStrokeWidth:2},range:{category:['#0068c9','#83c9ff','#ff2b2b','#ffabab','#29b09d','#7defa1','#ff8700','#ffd16a','#6d3fc0','#d5dae5']},concat:{columns:1},facet:{columns:1},mark:{tooltip:{content:'encoding'},color:'#0068c9'},bar:{binSpacing:2,discreteBandSize:{band:.85}},axisDiscrete:{grid:false},axisXPoint:{grid:false},axisTemporal:{grid:false},axisXBand:{grid:false}};
      spec.config=Object.fromEntries(Object.keys({...defaults,...spec.config}).map(key=>[key,typeof defaults[key]==='object' && !Array.isArray(defaults[key])?{...defaults[key],...spec.config?.[key]}:spec.config?.[key]??defaults[key]]));
      if (disposed) return;
      const result = await embed(el, spec, {renderer: 'canvas', actions: false, defaultStyle: false});
      view = result.view;
      viewRef.current=view;
      if (disposed) {view.finalize(); return;}
      observer = new ResizeObserver(() => {view.width(Math.max(120, el.clientWidth)).runAsync().catch(() => {});}); observer.observe(el);
    }).catch(e => {if (!disposed) setError(e.message);});
    return () => {disposed = true; observer?.disconnect(); view?.finalize();};
  }, [node.spec,theme]);
  const records=node.spec.data?.values || Object.values(node.spec.datasets || {})[0] || [];
  const keys=records.length?Object.keys(records[0]):[];
  const table={rows:records.map(row=>keys.map(key=>row[key])),columns:keys.map(key=>({key,label:key,kind:typeof records[0]?.[key]==='number'?'NumberColumn':'TextColumn'})),hide_index:true,height:350};
  async function download(){const url=await viewRef.current?.toImageURL('png',Math.max(2,window.devicePixelRatio || 1));if(url){const link=document.createElement('a');link.href=url;link.download=`${new Date().toISOString().slice(0,16).replaceAll(':','-')}_chart.png`;link.click();}}
  return <div className="chart-shell" ref={outer} role="group" aria-label={node.label || 'Interactive analysis chart'}><div className="table-toolbar"><button aria-label={showData?'Show chart':'Show data'} title={showData?'Show chart':'Show data'} onClick={()=>setShowData(s=>!s)}>▥</button><button aria-label="Download chart as PNG" title="Download as PNG" onClick={download}>⇩</button><button aria-label="Copy Vega-Lite spec" title="Copy Vega-Lite spec" onClick={()=>navigator.clipboard?.writeText(JSON.stringify(node.spec,null,2)).catch(()=>{})}>⧉</button><button aria-label="Fullscreen chart" title="Fullscreen" onClick={()=>document.fullscreenElement?document.exitFullscreen():outer.current?.requestFullscreen?.()}>⛶</button></div><div className="view-chart" ref={ref} style={{display:showData?'none':undefined}}>{error && <div role="alert">{error}</div>}</div>{showData && <EnhancedTable node={table}/>}</div>;
}


function Slider({ node, change }) {
  const dates = typeof node.min === 'string';
  const numeric = value => dates ? Date.parse(value) / 86400000 : Number(value);
  const output = value => dates ? new Date(value * 86400000).toISOString().slice(0, 10) : value;
  const range = Array.isArray(node.value);
  const [value, setValue] = useState(node.value);
  useEffect(() => setValue(node.value), [node.value]);
  const min = numeric(node.min), max = numeric(node.max);
  const lower = range ? numeric(value[0]) : min, upper = range ? numeric(value[1]) : numeric(value);
  function update(next, index) {
    const result = range ? [...value] : output(next);
    if (range) result[index] = output(index === 0 ? Math.min(next, upper) : Math.max(next, lower));
    setValue(result);
  }
  function finish() {change(node.key, value);}
  return <div className="view-slider">
    <label>{node.label}</label>
    <div className="slider-values"><span>{range ? value[0] : value}</span>{range && <span>{value[1]}</span>}</div>
    <div className="range-track" style={/** @type {import('react').CSSProperties} */ ({'--start': `${max === min ? 0 : (lower - min) / (max - min) * 100}%`, '--end': `${max === min ? 100 : (upper - min) / (max - min) * 100}%`})}>
      {range && <input type="range" aria-label={`${node.label} minimum`} min={min} max={max} step={node.step} value={lower} onChange={e => update(Number(e.target.value), 0)} onPointerUp={finish} onKeyUp={finish} />}
      <input type="range" aria-label={range ? `${node.label} maximum` : node.label} min={min} max={max} step={node.step} value={upper} onChange={e => update(Number(e.target.value), 1)} onPointerUp={finish} onKeyUp={finish} />
    </div>
    <div className="slider-bounds"><span>{node.min}</span><span>{node.max}</span></div>
  </div>;
}

function NumberInput({ node, change }) {
  const format = next => {const precision=/^%\.(\d+)f$/.exec(node.format || ''); return precision ? Number(next).toFixed(Number(precision[1])) : String(next);};
  const [value, setValue] = useState(() => format(node.value));
  useEffect(() => {const precision=/^%\.(\d+)f$/.exec(node.format || ''); setValue(precision ? Number(node.value).toFixed(Number(precision[1])) : String(node.value));}, [node.value,node.format]);
  function save(next) {if (next === '' || !Number.isFinite(Number(next))) return; const n = Math.min(node.max ?? Infinity, Math.max(node.min ?? -Infinity, Number(next))); setValue(format(n)); if (n !== node.value) change(node.key, n);}
  const id = useId();
  return <div className="view-field"><label htmlFor={id}>{node.label}</label><div className="number-input"><input id={id} type="number" min={node.min} max={node.max} step={node.step} value={value} onChange={e => setValue(e.target.value)} onBlur={() => save(value)} onKeyDown={e => {if (e.key === 'Enter') save(value);}} /><button aria-label={`Decrease ${node.label}`} disabled={Number(value) <= node.min} onClick={() => save(Number(value) - node.step)}>−</button><button aria-label={`Increase ${node.label}`} disabled={Number(value) >= node.max} onClick={() => save(Number(value) + node.step)}>+</button></div></div>;
}

function ViewTabs({ node, values, change, action }) {
  const strip=useRef(null);
  const key = `_tabs:${node.labels[0]}`;
  const active = Number(values[key] || 0);
  const id = useId();
  return <div className="view-tabs"><div className="tab-strip"><div className="view-tablist" role="tablist" ref={strip}>{node.labels.map((label, i) => <button role="tab" aria-selected={active === i} aria-controls={`${id}-panel-${i}`} id={`${id}-tab-${i}`} tabIndex={active === i ? 0 : -1} key={label} onClick={() => change(key, i)} onKeyDown={event => {if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {event.preventDefault(); const next = event.key === 'Home' ? 0 : event.key === 'End' ? node.labels.length - 1 : (i + (event.key === 'ArrowRight' ? 1 : -1) + node.labels.length) % node.labels.length; change(key, next); event.currentTarget.parentElement?.querySelectorAll('button')[next]?.focus();}}}>{label}</button>)}</div><TabScroll target={strip}/></div>{node.children.map((child, i) => <div key={i} role="tabpanel" id={`${id}-panel-${i}`} aria-labelledby={`${id}-tab-${i}`} hidden={active !== i} className="view-flow tab-content">{active === i && <ViewNodes nodes={child.children} values={values} change={change} action={action} />}</div>)}</div>;
}

function Upload({ node, change }) {
  const id = useId();
  const [error,setError]=useState(null);
  async function load(file){if(!file)return;if(!file.name.toLowerCase().endsWith('.csv') || file.size>200*1024*1024){setError('Choose a CSV file smaller than 200MB.');return;}setError(null);change(node.key,{name:file.name,content:await file.text()});}
  return <div className="view-field view-upload"><label htmlFor={id}>{node.label}</label><label className="upload-zone" htmlFor={id} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();load(e.dataTransfer.files?.[0]);}}><span>⇧</span><div>Drag and drop file here<small>Limit 200MB per file • CSV</small></div><span className="upload-browse">Browse files</span><input id={id} type="file" accept=".csv,text/csv" onChange={e=>load(e.target.files?.[0])} /></label>{node.filename && <div className="upload-file">{node.filename}<button aria-label={`Remove ${node.filename}`} onClick={()=>change(node.key,null)}>×</button></div>}{error && <span role="alert">{error}</span>}</div>;
}

function Expander({node,children}) {
  const [open,setOpen]=useState(Boolean(node.expanded));
  return <details className="view-expander" open={open} onToggle={e=>setOpen(e.currentTarget.open)}><summary>{node.label}</summary><div className="view-flow">{children}</div></details>;
}

function MultiSelect({node,values,change}) {
  const [open,setOpen]=useState(false),[search,setSearch]=useState('');
  const selected=values[node.key] ?? node.value;
  const id=useId();
  return <div className="view-field multiselect-field"><label htmlFor={id}>{node.label}</label><div className="multiselect-box">{selected.map(value=><span className="select-tag" key={value}>{value}<button aria-label={`Remove ${value}`} onClick={()=>change(node.key,selected.filter(v=>v!==value))}>×</button></span>)}<input id={id} role="combobox" aria-expanded={open} aria-controls={`${id}-options`} aria-autocomplete="list" value={search} onFocus={()=>setOpen(true)} onChange={e=>{setSearch(e.target.value);setOpen(true);}} onKeyDown={e=>{if(e.key==='Escape')setOpen(false);if(e.key==='Backspace' && !search && selected.length)change(node.key,selected.slice(0,-1));if(e.key==='Enter'){const option=node.options.find(v=>!selected.includes(v) && String(v).includes(search));if(option!==undefined){change(node.key,[...selected,option]);setSearch('');}e.preventDefault();}}}/><button aria-label={`Clear ${node.label}`} onClick={()=>change(node.key,[])}>×</button><button aria-label={`Toggle ${node.label} options`} onClick={()=>setOpen(s=>!s)}>⌄</button></div><div id={`${id}-options`} role="listbox" aria-label={node.label} hidden={!open} className="multiselect-options">{node.options.filter(v=>!selected.includes(v) && String(v).toLowerCase().includes(search.toLowerCase())).map(v=><button role="option" aria-selected="false" key={v} onClick={()=>{change(node.key,[...selected,v]);setSearch('');}}>{v}</button>)}</div></div>;
}

export function ViewNodes({ nodes = [], values = {}, change = (_key, _value) => {}, action = (_key) => {} }) {
  return nodes.map((node, index) => {
    const key = `${index}-${node.type}-${node.label || ''}`;
    const children = () => <ViewNodes nodes={node.children} values={values} change={change} action={action} />;
    switch (node.type) {
      case 'heading': {const Heading = /** @type {keyof import('react').JSX.IntrinsicElements} */ (`h${node.level}`); return <Heading key={key} className="view-heading">{node.text}</Heading>;}
      case 'markdown': return <div className="view-markdown" key={key}><Markdown>{node.text}</Markdown></div>;
      case 'caption': return <div className="view-caption" key={key}><Markdown>{node.text}</Markdown></div>;
      case 'html': return <div key={key} className="view-html" dangerouslySetInnerHTML={{__html: node.text}} />;
      case 'text': case 'code': return <pre key={key} className="view-code">{node.text}</pre>;
      case 'json': return <pre key={key} className="view-json">{JSON.stringify(node.value, null, 2)}</pre>;
      case 'notice':
        if (node.text === 'Research controls are disabled in hosted mode. Enable F1_RESEARCH_MODE=1 only for a trusted local/admin session; precomputed analyses remain available below.') return null;
        return <div key={key} className={`view-notice ${node.severity}`} role={node.severity === 'error' ? 'alert' : 'status'}>{node.icon && <span>{node.icon}</span>}<Markdown>{node.text}</Markdown></div>;
      case 'metric': return <div key={key} className="view-metric"><span>{node.label}</span><strong>{node.value}</strong>{node.delta != null && <small>{node.delta}</small>}</div>;
      case 'divider': return <hr key={key} className="view-divider" />;
      case 'image': return <img key={key} alt={node.alt || 'Analysis visualization'} src={node.src} className="view-image" style={{width: node.width === 'stretch' ? '100%' : node.width, maxWidth: '100%'}} />;
      case 'table': return <EnhancedTable key={key} node={node} />;
      case 'vega': return <VegaChart key={key} node={node} />;
      case 'plotly': return <SafePlotlyChart key={key} node={node} />;
      case 'columns': return <div key={key} className="view-columns" style={{gridTemplateColumns: node.widths.map(w => `minmax(0, ${w}fr)`).join(' ')}}>{node.children.map((col, i) => <div className="view-flow" key={i}><ViewNodes nodes={col.children} values={values} change={change} action={action} /></div>)}</div>;
      case 'tabs': return <ViewTabs key={key} node={node} values={values} change={change} action={action} />;
      case 'expander': return <Expander key={key} node={node}>{children()}</Expander>;
      case 'checkbox': return <label className="view-checkbox" key={key}><input aria-label={node.label} type="checkbox" checked={Boolean(values[node.key] ?? node.value)} disabled={node.disabled} onChange={e => change(node.key, e.target.checked)} /><span>{node.label}</span></label>;
      case 'select': return <label key={key} className="view-field"><span>{node.label}{node.help && <span className="view-help" title={node.help}>?</span>}</span><select aria-label={node.label} title={node.help} value={JSON.stringify(node.options.includes(values[node.key])?values[node.key]:node.value)} onChange={e => change(node.key, JSON.parse(e.target.value))}>{node.options.map((option, i) => <option key={i} value={JSON.stringify(option)}>{String(option)}</option>)}</select></label>;
      case 'multiselect': return <MultiSelect key={key} node={node} values={values} change={change} />;
      case 'slider': return <Slider key={key} node={node} change={change} />;
      case 'number': return <NumberInput key={key} node={node} change={change} />;
      case 'button': return <button key={key} className="view-button" disabled={node.disabled} title={node.help} onClick={() => action(node.key)}>{node.label}</button>;
      case 'upload': return <Upload key={key} node={node} change={change} />;
      case 'download': return <a key={key} className="view-button view-download" download={node.filename} href={`data:${node.mime};base64,${node.data}`}>{node.label}</a>;
      default: return null;
    }
  });
}
```

## code/frontend/ResearchJobs.jsx

[Separate source file](code/frontend/ResearchJobs.jsx)

```jsx
import {useCallback,useEffect,useState} from 'react';
import {ViewNodes} from '../components/Presentation';

export function ResearchJobs({values}) {
  const [token,setToken] = useState(''), [task,setTask] = useState('leakage-audit');
  const [job,setJob] = useState(null), [result,setResult] = useState(null), [error,setError] = useState('');
  const call = useCallback(async (path, options = {}, signal) => {
    const response = await fetch('/api/enhancements/jobs'+path, {...options,signal,
      headers:{'Content-Type':'application/json','X-F1-Admin-Token':token}});
    const body = await response.json();
    if(!response.ok)throw new Error(typeof body.detail === 'string' ? body.detail : 'Job request failed.');
    return body;
  }, [token]);
  async function submit() {
    setError('');setResult(null);
    try {setJob(await call('',{method:'POST',body:JSON.stringify({task,values})}));}
    catch(err){setError(err.message);}
  }
  useEffect(() => {
    if(!job || !['queued','running'].includes(job.state))return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const state = await call('/'+job.id,{},controller.signal);
        if(state.state === 'succeeded')setResult(await call('/'+job.id+'/result',{},controller.signal));
        if(state.state === 'failed')setError(state.error);
        setJob(state);
      }catch(err){if(err.name !== 'AbortError'){setError(err.message);setJob(old => ({...old,state:'unavailable'}));}}
    },1000);
    return () => {clearTimeout(timer);controller.abort();};
  },[job,call]);
  async function cancel() {
    try {
      const cancelled = await call('/'+job.id,{method:'DELETE'});
      if(cancelled.cancelled)setJob(old => ({...old,state:'cancelled'}));
      else setError('This calculation has started and cannot be cancelled safely.');
    }catch(err){setError(err.message);}
  }
  return <details className="research-job">
    <summary>Local administrator research jobs</summary>
    <p>Uses a separate calculation process. Queued jobs can be cancelled; running calculations finish normally. Results expire after ten minutes and are lost on restart.</p>
    <label>Administrator token <input type="password" autoComplete="off" value={token} onChange={e => setToken(e.target.value)}/></label>
    <label>Task <select value={task} onChange={e => setTask(e.target.value)}><option value="leakage-audit">Temporal leakage audit</option><option value="bin-comparison">Bin-count comparison</option></select></label>
    <p>Uses current settings. Audit defaults to 1,000 rows; bin comparison defaults to q=2. Configure those values in their existing panels before submitting.</p>
    <button disabled={!token || ['queued','running'].includes(job?.state)} onClick={submit}>Queue calculation</button>
    {job && <p role="status">Job {job.id}: {job.state}</p>}
    {job?.state === 'queued' && <button onClick={cancel}>Cancel queued job</button>}
    {error && <p role="alert">{error}</p>}
    {result && <ViewNodes nodes={result.nodes}/>}
  </details>;
}
```

## code/frontend/SafePlotlyChart.jsx

[Separate source file](code/frontend/SafePlotlyChart.jsx)

```jsx
import {useEffect,useRef,useState} from 'react';

export function SafePlotlyChart({node}) {
  const ref = useRef(null);
  const [error,setError] = useState(null);
  useEffect(() => {
    const element = ref.current;
    let chart, observer, disposed = false;
    setError(null);
    import('plotly.js-dist-min').then(async ({default:plotly}) => {
      if (disposed) return;
      chart = plotly;
      await chart.newPlot(element,node.spec.data,{...node.spec.layout,autosize:true},{responsive:true});
      if (disposed) {chart.purge(element);return;}
      observer = new ResizeObserver(() => chart.Plots.resize(element).catch(() => {}));
      observer.observe(element);
    }).catch(err => {if (!disposed) setError(err.message);});
    return () => {disposed=true;observer?.disconnect();if(chart)chart.purge(element);};
  }, [node.spec]);
  return <div className="view-chart" ref={ref} role="img" aria-label={node.label || 'Interactive Plotly chart'}>
    {error && <div role="alert">Chart unavailable: {error}. Other analysis remains available.</div>}
  </div>;
}
```

## code/frontend/viewClient.js

[Separate source file](code/frontend/viewClient.js)

```javascript
import {hasUpload, stableKey} from './preferences.js';

// Keep actions and uploaded data out of shared requests and retained responses.
export function createViewClient({fetcher = fetch, now = Date.now, ttl = 15000, maxEntries = 6, maxBytes = 12000000, normalTimeout = 120000, actionTimeout = 600000} = {}) {
  const cache = new Map(), pending = new Map();
  let revision = '', retainedBytes = 0;
  const clear = () => {cache.clear(); retainedBytes = 0;};
  async function json(url, options) {
    const response = await fetcher(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(typeof body.detail === 'string' ? body.detail : 'Request failed (' + response.status + ').'), {status: response.status});
    return body;
  }
  /** @param {object} payload @param {{signal?: AbortSignal, enabled?: boolean}} [options] */
  async function load(payload, {signal, enabled = false} = {}) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    const reusable = enabled && payload.page <= 5 && !payload.action && !hasUpload(payload.values || {});
    if (payload.action || hasUpload(payload.values || {})) clear();
    // Probe on every reusable navigation: never serve a client hit under an old revision.
    if (reusable) {
      const state = await json('/api/enhancements/status', {signal, cache: 'no-store'});
      if (revision !== state.revision) {clear(); revision = state.revision;}
    }
    const key = stableKey({revision, ...payload});
    const hit = reusable && cache.get(key);
    if (hit && hit.until > now()) {
      cache.delete(key); cache.set(key, hit);
      return hit.value;
    }
    if (hit) {cache.delete(key); retainedBytes -= hit.bytes;}
    let task = reusable && pending.get(key);
    if (!task) {
      const controller = new AbortController();
      task = {controller, consumers: 0, promise: null};
      let timedOut = false;
      const timeout = setTimeout(() => {timedOut = true;controller.abort();}, payload.action ? actionTimeout : normalTimeout);
      task.promise = json('/api/views', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload), signal: controller.signal
      }).then(value => {
        if (reusable) {
          // This budgets serialized data; actual JS heap must also be measured.
          const bytes = new TextEncoder().encode(JSON.stringify(value)).byteLength;
          if (bytes <= maxBytes) {
            const replaced = cache.get(key);
            if (replaced) retainedBytes -= replaced.bytes;
            cache.set(key, {value, bytes, until: now() + ttl}); retainedBytes += bytes;
            while (cache.size > maxEntries || retainedBytes > maxBytes) {
              const oldest = cache.keys().next().value;
              retainedBytes -= cache.get(oldest).bytes; cache.delete(oldest);
            }
          }
        }
        return value;
      }).catch(error => {
        if(timedOut)throw new Error('The analysis request timed out. Retry or reduce the selected workload.');
        throw error;
      }).finally(() => {clearTimeout(timeout); if (pending.get(key) === task) pending.delete(key);});
      if (reusable) pending.set(key, task);
    }
    task.consumers++;
    return new Promise((resolve, reject) => {
      let finished = false;
      function release() {
        if (finished) return;
        finished = true; signal?.removeEventListener('abort', abort); task.consumers--;
        // Strict Mode can subscribe again before this timer; allow it to share the request.
        setTimeout(() => {if (!task.consumers) task.controller.abort();}, 100);
      }
      function abort() {release(); reject(new DOMException('Cancelled', 'AbortError'));}
      signal?.addEventListener('abort', abort, {once: true});
      if (signal?.aborted) {abort(); return;}
      task.promise.then(value => {if (!finished) {release(); resolve(value);}},
        error => {if (!finished) {release(); reject(error);}});
    });
  }
  return {load, clear, retainedBytes: () => retainedBytes};
}

export const viewClient = createViewClient();
```
