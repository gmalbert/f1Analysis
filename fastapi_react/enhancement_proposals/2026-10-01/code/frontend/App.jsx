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
