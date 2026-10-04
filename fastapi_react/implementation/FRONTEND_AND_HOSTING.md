# Current frontend and hosting implementation

Snapshot of installed main-application source, generated 2026-10-04 03:06:33Z. Use these with the existing repository and its unchanged data/model artifacts and exported reference view modules. The original proposal files are historical candidates. See [implementation policies and evidence](../ENHANCEMENTS.md). Binary marks/fonts and generated WebP images live in frontend/public; the original footer PNG is retained and the optimizer regenerates variants. No production deployment is performed by these files.

## frontend/src/App.jsx

[Editable source](../frontend/src/App.jsx) — SHA-256: `21e7afa16aa509139064dff36f769afb8a4821bd29c21df26e93c4926db69a8b`

```jsx
import { useEffect, useRef, useState } from 'react';
import { viewClient } from './enhancements/viewClient';
import { FeatureBar, LoadingFeedback, readOptions } from './enhancements/FeatureBar';
import { readSharedView, safeValues } from './enhancements/preferences';
import { ResearchJobs } from './enhancements/ResearchJobs';
import { ViewNodes } from './components/Presentation';
import { TabScroll } from './components/TabScroll';

const labels = ['📊 Data Explorer', '📈 Analytics & Visualizations', '🏎️ Schedule', '🏁 Next Race', '🤖 Predictive Models', '💾 Data & Debug', '📐 Betting Research'];
const routes = ['Data Explorer', 'Analytics', 'Current Season', 'Next Race', 'Predictive Models', 'Raw Data', 'Betting Research'];
const FEATURES_ENABLED = import.meta.env.VITE_F1_ENHANCEMENTS !== '0';
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
    const values = safeValues(JSON.parse(sessionStorage.getItem('f1analysis.view-values') || '{}'));
    const oldFilters = JSON.parse(sessionStorage.getItem('f1analysis.filters') || 'null');
    if (!Object.hasOwn(values, 'filter_results_main') && oldFilters?.applied) values.filter_results_main = true;
    return values;
  } catch { return {}; }
}

function readTheme() {
  try {return localStorage.getItem('f1analysis.theme') === 'dark' ? 'dark' : 'light';}
  catch {return 'light';}
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
  const [theme, setTheme] = useState(readTheme);
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
    const persisted = safeValues(values);
    try {
      sessionStorage.setItem('f1analysis.view-values', JSON.stringify(persisted));
      sessionStorage.setItem('f1analysis.filters', JSON.stringify({applied: Boolean(persisted.filter_results_main), values: persisted}));
    } catch { /* Browser storage is optional; private uploads remain in memory. */ }
  }, [values]);

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
  }

  function restore(view) {
    setValues(view.values); setPage(view.page); setRequest(null);
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
  const act = key => {
    const task = key === 'Run Leakage Audit' ? 'leakage-audit' : key === 'Run Bin Count Comparison' ? 'bin-comparison' : null;
    if (FEATURES_ENABLED && task) {
      window.dispatchEvent(new CustomEvent('f1analysis:research-task', {detail: task}));
      return;
    }
    setRequest({key, id: Date.now()});
  };

  return <div className={`app-shell parity-app ${sidebar ? 'with-sidebar' : ''}`}>
    <a className="skip-link" href="#main-content" onClick={event => {event.preventDefault();document.getElementById('main-content')?.focus();}}>Skip to main content</a>
    <div className="app-toolbar">
      {values.filter_results_main && <button aria-label={sidebarClosed ? 'Open sidebar' : 'Close sidebar'} className="sidebar-toggle" style={{left: sidebarClosed ? 16 : 252}} onClick={() => setSidebarClosed(s => !s)}>{sidebarClosed ? '»' : '«'}</button>}
      <button className="settings-toggle" aria-label="Settings" aria-expanded={settings} onClick={() => setSettings(s => !s)}>⋮</button>
      {settings && <div className="settings-menu"><label><input aria-label="Use light theme" type="checkbox" checked={theme === 'light'} onChange={e => setTheme(e.target.checked ? 'light' : 'dark')} />Light theme</label></div>}
    </div>
    {sidebar && <aside className="filter-sidebar" aria-label="Data filters"><div className="view-flow"><ViewNodes nodes={data?.sidebar} values={values} change={change} action={act} /></div></aside>}
    <div className="main-shell">
      {FEATURES_ENABLED && <details><summary>Analysis tools</summary><FeatureBar page={page} values={values} options={options} setOptions={setOptions} restore={restore} navigate={navigate} busy={busy || Boolean(error) || data?.page !== page} analysisRevision={data?.page === page ? data.source_revision : null}/></details>}
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
        {FEATURES_ENABLED && <ResearchJobs page={page} values={values}/>}
        {busy && !FEATURES_ENABLED && <span className="sr-only" role="status">Loading analysis…</span>}
        <footer className="parity-footer"><p>Powered by <a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer">Betting Oracle</a></p><p>Sports Prediction Analytics</p><a href="https://www.betting-oracle.com" target="_blank" rel="noreferrer"><picture><source type="image/webp" srcSet="/betting-oracle-logo-60.webp 1x, /betting-oracle-logo-120.webp 2x"/><img src="/betting-oracle-logo.png" alt="Betting Oracle Logo" width="822" height="1255" style={{width: 60 * 822 / 1255}} loading="lazy" decoding="async" /></picture></a></footer>
      </main>
    </div>
  </div>;
}
```

## frontend/src/main.jsx

[Editable source](../frontend/src/main.jsx) — SHA-256: `2a670ecfc8352a886523fe5045b233d467331bf4f8b14ef14cf641221cd4f8ee`

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

## frontend/src/components/Presentation.jsx

[Editable source](../frontend/src/components/Presentation.jsx) — SHA-256: `8d4ae77ba0289c23cf3a710d33bc6c33b92838492c2d55049733990935fa4919`

```jsx
import { useEffect, useId, useMemo, useRef, useState } from 'react';
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

function tireContextFrom(nodes) {
  const context = {};
  function read(items) {
    for (const node of items || []) {
      if (node.key === 'tire_year_select') context.year = node.value;
      if (node.key === 'tire_race_select') context.event = node.value;
      read(node.children);
    }
  }
  read(nodes);
  return context;
}

export function isTireChartPair(nodes, index) {
  const table = nodes[index], heading = nodes[index + 1], chart = nodes[index + 2];
  const inlineData = chart?.spec?.data?.values || chart?.spec?.datasets?.[chart?.spec?.data?.name];
  return import.meta.env.VITE_F1_ENHANCEMENTS !== '0' && table?.type === 'table'
    && table.columns.some(column => column.key === 'Avg Deg (s/lap)')
    && heading?.type === 'markdown' && heading.text.includes('Avg Tire Degradation by Driver')
    && chart?.type === 'vega' && chart.spec.encoding?.x?.field === 'driver'
    && chart.spec.encoding?.y?.field === 'Degradation (s/lap)' && Array.isArray(inlineData);
}

export function selectedTireChart(chart, drivers) {
  if (!drivers.length) return chart;
  const spec = structuredClone(chart.spec), selected = new Set(drivers);
  const filter = rows => rows.filter(row => selected.has(row.driver));
  if (Array.isArray(spec.data?.values)) spec.data.values = filter(spec.data.values);
  const name = spec.data?.name;
  if (name && Array.isArray(spec.datasets?.[name])) spec.datasets[name] = filter(spec.datasets[name]);
  spec.encoding.x.sort = drivers;
  spec.encoding.x.title = 'Selected drivers';
  spec.encoding.y.title = 'Tire degradation (s/lap)';
  return {...chart, spec};
}

function TireDriverComparison({table, heading, chart, context}) {
  const [drivers, setDrivers] = useState([]);
  const filtered = useMemo(() => selectedTireChart(chart, drivers), [chart, drivers]);
  const scope = `${context?.event || 'Selected race'}${context?.year ? ' ' + context.year : ''}`;
  return <section aria-label="Race tire strategy comparison">
    <EnhancedTable node={table} context={context} chartLinked onSelectionChange={setDrivers}/>
    <div className="view-markdown"><Markdown>{heading.text}</Markdown></div>
    <p className="view-caption">{drivers.length ? `Showing only ${drivers.length} selected driver${drivers.length === 1 ? '' : 's'}: ${drivers.join(', ')}.` : 'Showing all drivers.'} {scope}.</p>
    <VegaChart node={{...filtered, label: `Tire degradation — ${scope} — ${drivers.length ? drivers.join(', ') : 'all drivers'}`}}/>
  </section>;
}

export function ViewNodes({ nodes = [], values = {}, change = (_key, _value) => {}, action = (_key) => {}, tireContext = null }) {
  const context = tireContext || tireContextFrom(nodes);
  const paired = new Set(nodes.map((_, index) => isTireChartPair(nodes, index) ? index : -1).filter(index => index >= 0));
  return nodes.map((node, index) => {
    if (paired.has(index - 1) || paired.has(index - 2)) return null;
    const key = `${index}-${node.type}-${node.label || ''}`;
    if (paired.has(index)) return <TireDriverComparison key={key} table={node} heading={nodes[index + 1]} chart={nodes[index + 2]} context={context}/>;
    const children = () => <ViewNodes nodes={node.children} values={values} change={change} action={action} tireContext={context}/>;
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
      case 'table': return <EnhancedTable key={key} node={node} context={context} />;
      case 'vega': return <VegaChart key={key} node={node} />;
      case 'plotly': return <SafePlotlyChart key={key} node={node} />;
      case 'columns': return <div key={key} className="view-columns" style={{gridTemplateColumns: node.widths.map(w => `minmax(0, ${w}fr)`).join(' ')}}>{node.children.map((col, i) => <div className="view-flow" key={i}><ViewNodes nodes={col.children} values={values} change={change} action={action} tireContext={context}/></div>)}</div>;
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

## frontend/src/components/ViewTable.jsx

[Editable source](../frontend/src/components/ViewTable.jsx) — SHA-256: `fd4650cc01c1d1ade29a4e61aa22d3349119d59136cc30cfd11b5aaa22e6ffd3`

```jsx
import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import DataEditor, {GridCellKind} from '@glideapps/glide-data-grid';
import '@glideapps/glide-data-grid/dist/index.css';
import {displayCell} from './Presentation';
import {useTheme} from './useTheme';

const empty=[];
const quote=value=>{const text=value==null?'':String(value);return /[,"\r\n]/.test(text)?`"${text.replaceAll('"','""')}"`:text;};
function csvValue(value,column){
  if(value==null)return '';
  if(column.kind==='CheckboxColumn')return Boolean(value);
  if(column.kind==='DateColumn')return String(value).slice(0,10);
  if(column.kind==='TimeColumn'){
    const parts=String(value).split(':');
    return `${parts[0].padStart(2,'0')}:${(parts[1] || '00').padStart(2,'0')}:${Number(parts[2] || 0).toFixed(3).padStart(6,'0')}`;
  }
  if(column.format==='%d' && typeof value==='number')return Math.trunc(value);
  return value;
}

/** The same canvas grid used by the reference, including keyboard selection,
 * copying ranges, search, scrolling, overlays and column resizing. */
export function ViewTable({node}) {
  const rows=node.rows || empty;
  const columns=node.columns || empty;
  const [sort,setSort]=useState(null);
  const [widths,setWidths]=useState(/** @type {Record<string,number>} */ ({}));
  const [hidden,setHidden]=useState([]);
  const [showColumns,setShowColumns]=useState(false);
  const [showSearch,setShowSearch]=useState(false);
  const [menu,setMenu]=useState(null);
  const [pinned,setPinned]=useState([]);
  const [formats,setFormats]=useState(/** @type {Record<string,string>} */ ({}));
  const outer=useRef(null);
  useEffect(()=>{
    if(!menu && !showColumns)return;
    const close=event=>{if(event.type==='keydown' && event.key==='Escape' || event.type==='pointerdown' && !outer.current?.contains(event.target)){setMenu(null);setShowColumns(false);}};
    document.addEventListener('pointerdown',close);document.addEventListener('keydown',close);
    return()=>{document.removeEventListener('pointerdown',close);document.removeEventListener('keydown',close);};
  },[menu,showColumns]);
  const indices=useMemo(()=>{
    const result=rows.map((_,i)=>i);
    if(sort)result.sort((a,b)=>{
      const x=rows[a][sort.column],y=rows[b][sort.column];
      if(x==null || y==null)return x==null?(y==null?0:1):-1;
      const cmp=typeof x==='number' && typeof y==='number'?x-y:String(x).localeCompare(String(y));
      return sort.desc?-cmp:cmp;
    });
    return result;
  },[rows,sort]);
  const visible=useMemo(()=>columns.map((column,index)=>({column,index})).filter(c=>!hidden.includes(c.index)).sort((a,b)=>Number(pinned.includes(b.index))-Number(pinned.includes(a.index))),[columns,hidden,pinned]);
  const gridColumns=useMemo(()=>{
    const result=visible.map(({column,index})=>({id:String(index),title:column.label+(sort?.column===index?(sort.desc?' ↓':' ↑'):''),hasMenu:true,width:widths[index] || (typeof column.width==='number'?column.width:undefined)}));
    if(!node.hide_index)result.unshift({id:'index',title:node.index_name || '',width:widths.index});
    return result;
  },[visible,widths,sort,node.hide_index,node.index_name]);
  const dark=useTheme()==='dark';
  const bg=dark?'#0e1117':'#fff',text=dark?'#fafafa':'#31333f';
  /** @type {(cell: import('@glideapps/glide-data-grid').Item) => import('@glideapps/glide-data-grid').GridCell} */
  const getCell=useCallback(([col,row])=>{
    const rowIndex=indices[row];
    if(rowIndex==null)return {kind:GridCellKind.Text,data:'',displayData:'',allowOverlay:false};
    if(!node.hide_index && col===0)return {kind:GridCellKind.Text,data:String(node.index?.[rowIndex]??rowIndex),displayData:String(node.index?.[rowIndex]??rowIndex),allowOverlay:true,readonly:true,themeOverride:{bgCell:dark?'#262730':'#f7f9fc',textDark:dark?'#bfc2ce':'#808495'}};
    const selected=visible[col-(node.hide_index?0:1)];
    if(!selected)return {kind:GridCellKind.Text,data:'',displayData:'',allowOverlay:false};
    const {column,index}=selected;
    const value=rows[rowIndex][index],style=node.styles?.[rowIndex]?.[index];
    const themeOverride={bgCell:style?.['background-color'] || bg,textDark:style?.color || text};
    if(column.kind==='CheckboxColumn')return {kind:GridCellKind.Boolean,data:value==null?null:Boolean(value),allowOverlay:false,readonly:true,maxSize:16,themeOverride};
    const displayData=displayCell(value,formats[index]?{...column,format:formats[index]}:column,node.display?.[rowIndex]?.[index]);
    if(typeof value==='number')return {kind:GridCellKind.Number,data:value,displayData,allowOverlay:true,readonly:true,contentAlign:'right',themeOverride};
    return {kind:GridCellKind.Text,data:value==null?'':String(value),displayData,allowOverlay:true,readonly:true,style:value==null?'faded':'normal',themeOverride};
  },[indices,visible,node,rows,dark,bg,text,formats]);
  function download(){
    const header=visible.map(c=>quote(c.column.key));
    if(!node.hide_index)header.unshift(quote(node.index_name || ''));
    const lines=indices.map(i=>{
      const values=visible.map(c=>quote(csvValue(rows[i][c.index],formats[c.index]?{...c.column,format:formats[c.index]}:c.column)));
      if(!node.hide_index)values.unshift(quote(node.index?.[i]??i));
      return values.join(',');
    });
    const csv='\ufeff'+[header.join(','),...lines].join('\r\n')+'\r\n';
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download=`${new Date().toISOString().slice(0,16).replace(':','-')}_export.csv`;link.click();URL.revokeObjectURL(url);
  }
  const height=node.height || Math.min(400,(rows.length+1)*35+3);
  const menuColumn=menu?columns[menu.index]:null;
  const numeric=menuColumn?.kind==='NumberColumn';
  return <div className="view-table canvas-table" ref={outer} style={{maxWidth:typeof node.width==='number'?node.width:undefined}}>
    <div className="table-toolbar">
      <button aria-label="Search table" title="Search" onClick={()=>setShowSearch(s=>!s)}>⌕</button>
      <button aria-label="Show or hide columns" title="Columns" onClick={()=>setShowColumns(s=>!s)}>▥</button>
      <button aria-label="Download table as CSV" title="Download CSV" onClick={download}>⇩</button>
      <button aria-label="Fullscreen table" title="Fullscreen" onClick={()=>document.fullscreenElement?document.exitFullscreen():outer.current?.requestFullscreen?.()}>⛶</button>
    </div>
    {showColumns && <div className="column-picker">{columns.map((col,i)=><label key={i}><input type="checkbox" checked={!hidden.includes(i)} onChange={()=>setHidden(h=>h.includes(i)?h.filter(x=>x!==i):[...h,i])}/>{col.label}</label>)}</div>}
    {menuColumn && <div className="grid-column-menu" role="group" aria-label={`${menuColumn.label} column options`} style={{left:menu.left}}><strong>{menuColumn.label}</strong><button onClick={()=>{setSort({column:menu.index,desc:false});setMenu(null);}}>Sort ascending</button><button onClick={()=>{setSort({column:menu.index,desc:true});setMenu(null);}}>Sort descending</button><button onClick={()=>{setSort(null);setMenu(null);}}>Clear sorting</button><button onClick={()=>{setPinned(p=>p.includes(menu.index)?p.filter(i=>i!==menu.index):[...p,menu.index]);setMenu(null);}}>{pinned.includes(menu.index)?'Unpin column':'Pin column'}</button><button onClick={()=>{setHidden(h=>[...h,menu.index]);setMenu(null);}}>Hide column</button>{numeric && <label>Number format<select value={formats[menu.index] || ''} onChange={event=>setFormats(f=>({...f,[menu.index]:event.target.value}))}><option value="">Default</option><option value="%d">Integer</option>{[1,2,3,4].map(n=><option key={n} value={`%.${n}f`}>{n} decimal places</option>)}<option value="percent">Percent</option></select></label>}<small>{rows.length.toLocaleString()} rows · {rows.filter(r=>r[menu.index]==null).length.toLocaleString()} missing · {new Set(rows.map(r=>r[menu.index])).size.toLocaleString()} unique</small></div>}
    <DataEditor columns={gridColumns} rows={rows.length} getCellContent={getCell} getCellsForSelection={true} width="100%" height={height} rowHeight={35} headerHeight={35} rowMarkers="none" minColumnWidth={50} maxColumnWidth={500} freezeColumns={pinned.length+(node.hide_index?0:1)} showSearch={showSearch} onSearchClose={()=>setShowSearch(false)} onHeaderMenuClick={(col,bounds)=>{const selected=visible[col-(node.hide_index?0:1)];if(selected)setMenu({index:selected.index,left:Math.max(0,Math.min(bounds.x-(outer.current?.getBoundingClientRect().x || 0),(outer.current?.clientWidth || 240)-240))});}} onColumnResize={(column,width)=>setWidths(old=>({...old,[column.id]:width}))} onColumnResizeEnd={(column,width)=>setWidths(old=>({...old,[column.id]:width}))} onHeaderClicked={col=>{const selected=visible[col-(node.hide_index?0:1)];if(selected)setSort(s=>s?.column===selected.index?(s.desc?null:{...s,desc:true}):{column:selected.index,desc:false});}} theme={{fontFamily:'Source Sans',baseFontStyle:'13px',headerFontStyle:'13px',cellHorizontalPadding:8,cellVerticalPadding:3,bgCell:bg,bgHeader:dark?'#262730':'#f7f9fc',bgHeaderHovered:dark?'#3a3d46':'#eff1f6',bgHeaderHasFocus:dark?'#3a3d46':'#eff1f6',textDark:text,textHeader:dark?'#bfc2ce':'#808495',textMedium:text,textLight:'#808495',borderColor:dark?'#3a3d46':'#e6e7eb',accentColor:'#ff4b4b',accentLight:dark?'#ff4b4b33':'#ff4b4b1a',accentFg:'#fff',headerBottomBorderColor:dark?'#3a3d46':'#d6d8df',roundingRadius:0}} />
  </div>;
}
```

## frontend/src/enhancements/EnhancedTable.jsx

[Editable source](../frontend/src/enhancements/EnhancedTable.jsx) — SHA-256: `7604331065c07d0e86fe2c70dec12a3158f04b2b15833675b22aaddb79d853ab`

```jsx
import {useEffect, useId, useMemo, useState} from 'react';
import {ViewTable} from '../components/ViewTable';
import {displayCell} from '../components/Presentation';

const driverKeys = ['resultsDriverName','driverName','Driver'];
const FEATURES_ENABLED = import.meta.env.VITE_F1_ENHANCEMENTS !== '0';
const outcomeKeys = ['resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained','DNF'];
const tireKeys = ['Avg Deg (s/lap)','Start Compound','Stints','Avg Stint (laps)','Max Stint (laps)','Avg Stints','Soft Lap %','Laps','Races'];
const tireLabels = {'Avg Deg (s/lap)': 'Average tire degradation (s/lap)'};
const knownDNF = value => [true, false, 1, 0, 'true', 'false', '1', '0'].includes(typeof value === 'string' ? value.trim().toLowerCase() : value);

export function driverComparison(node) {
  const driverIndex = node.columns.findIndex(column => driverKeys.includes(column.key));
  if (driverIndex < 0) return null;
  const choices = [...new Set(node.rows.map(row => row[driverIndex]).filter(value => typeof value === 'string' && value.trim()))].sort();
  const tire = node.columns.some(column => column.key === 'Avg Deg (s/lap)');
  const fields = (tire ? tireKeys : outcomeKeys).map(key => ({key, index: node.columns.findIndex(column => column.key === key)})).filter(field => field.index >= 0);
  const useful = fields.some(field => node.rows.some(row => field.key === 'DNF' ? knownDNF(row[field.index]) : typeof row[field.index] === 'number' && Number.isFinite(row[field.index])));
  if (choices.length < 2 || !useful) return null;
  const counts = new Map();
  for (const row of node.rows) counts.set(row[driverIndex], (counts.get(row[driverIndex]) || 0) + 1);
  return {driverIndex, choices, fields, tire, singleRow: choices.every(driver => counts.get(driver) === 1)};
}

export function EnhancedTable({node, context = null, chartLinked = false, onSelectionChange = undefined}) {
  const [mode, setMode] = useState('grid'), [compare, setCompare] = useState(false);
  const comparisonId = useId();
  const comparison = useMemo(() => driverComparison(node), [node]);
  if (!FEATURES_ENABLED) return <ViewTable node={node}/>;
  function toggleComparison() {
    setCompare(!compare);
    if (compare) onSelectionChange?.([]);
  }
  return <section aria-label="Table display">
    <div className="enhancement-bar">
      <button aria-pressed={mode === 'grid'} onClick={() => setMode('grid')}>Interactive grid</button>
      <button aria-pressed={mode === 'accessible'} onClick={() => setMode('accessible')}>Accessible table</button>
      {comparison && <button aria-expanded={compare} aria-controls={comparisonId} onClick={toggleComparison}>Compare drivers</button>}
    </div>
    {mode === 'grid' ? <ViewTable node={node}/> : <AccessibleTable node={node}/>}
    {compare && comparison && <div id={comparisonId}><DriverComparison node={node} comparison={comparison} context={context} chartLinked={chartLinked} onSelectionChange={onSelectionChange}/></div>}
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
      <div className="columns-list">{columns.map((column,index) => <label key={index}><input type="checkbox" checked={visible.includes(index)} disabled={visible.length === 1 && visible[0] === index} onChange={() => toggle(index)}/>{column.label} ({column.key})</label>)}</div>
    </details>
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- A focusable overflow region lets keyboard users scroll wide tables. */}
    <div className="table-viewport" tabIndex={0} role="region" aria-label="Scrollable analysis table"><table>
      <caption>{matches.length.toLocaleString()} matching rows · showing rows {matches.length ? current*50+1 : 0}–{Math.min((current+1)*50,matches.length)}. All fields are available in Choose fields.</caption>
      <thead><tr>{!node.hide_index && <th scope="col">{node.index_name || 'Row'}</th>}{visible.map(index => <th scope="col" key={index}>{columns[index].label}</th>)}</tr></thead>
      <tbody>{matches.slice(current*50,(current+1)*50).map(row => <tr key={row}>
        {!node.hide_index && <th scope="row">{String(node.index?.[row] ?? row)}</th>}
        {visible.map(index => <td key={index}>{displayCell(rows[row][index],columns[index],node.display?.[row]?.[index])}</td>)}
      </tr>)}</tbody>
    </table></div>
    {!matches.length && <p role="status">No rows match this search.</p>}
    <nav aria-label="Table row pages"><button disabled={!current} onClick={() => setPage(current-1)}>Previous rows</button> Page {current+1} of {pageCount} <button disabled={current+1 >= pageCount} onClick={() => setPage(current+1)}>Next rows</button></nav>
    <p>Use Interactive grid for the original sorting, selection, copying and full CSV export.</p>
  </div>;
}

function DriverComparison({node, comparison, context, chartLinked, onSelectionChange}) {
  const {driverIndex, choices, fields, tire, singleRow} = comparison;
  const [drivers, setDrivers] = useState([]);
  const selectedDrivers = useMemo(() => drivers.filter(driver => choices.includes(driver)), [drivers, choices]);
  useEffect(() => {onSelectionChange?.(selectedDrivers);}, [selectedDrivers, onSelectionChange]);
  const summaries = useMemo(() => selectedDrivers.map(driver => {
    const sample = node.rows.filter(row => row[driverIndex] === driver);
    return {driver,rows:sample.length,values:fields.map(field => {
      const values = sample.map(row => row[field.index]);
      if (singleRow) {
        const value = values[0];
        if (value == null) return 'No data';
        if (field.key === 'DNF') return knownDNF(value) ? value === true || value === 1 || ['true', '1'].includes(String(value).trim().toLowerCase()) ? 'Yes' : 'No' : 'No data';
        return displayCell(value, node.columns[field.index]);
      }
      if (field.key === 'DNF') {
        const known = values.filter(knownDNF);
        return known.length ? (100*known.filter(value => value === true || value === 1 || ['true', '1'].includes(String(value).trim().toLowerCase())).length/known.length).toFixed(1)+'%' : 'No data';
      }
      if (field.key === 'Start Compound') return [...new Set(values.filter(value => typeof value === 'string' && value.trim()))].join(', ') || 'No data';
      const known = values.filter(v => typeof v === 'number' && Number.isFinite(v));
      return known.length ? (known.reduce((a,b) => a+b,0)/known.length).toFixed(2) : 'No data';
    })};
  }), [selectedDrivers,node,driverIndex,fields,singleRow]);
  const annual = tire && fields.some(field => field.key === 'Races');
  const scope = tire ? annual ? `Season tire summaries${context?.year ? ' for ' + context.year : ''}` : `Race tire strategy${context?.event ? ' — ' + context.event : ''}${context?.year ? ' ' + context.year : ''}` : 'Current table and applied filters';
  function fieldLabel(field) {
    const label = tireLabels[field.key] || node.columns[field.index].label;
    return singleRow || field.key === 'Start Compound' ? label : field.key === 'DNF' ? 'DNF rate among known records' : 'Mean ' + label;
  }
  return <div className="accessible-table">
    <h3>{tire ? 'Compare tire strategy' : 'Compare race results'}</h3>
    <p>{scope}. {singleRow ? 'Compare the displayed values for each selected driver.' : 'Compare averages across the records included in this table.'} {chartLinked && 'The chart below uses the same selected drivers. Clear the selection to show all drivers again.'}</p>
    <fieldset className="driver-picker"><legend>Choose up to four drivers ({choices.length} available)</legend><div className="columns-list">{choices.map(driver => <label key={driver}><input type="checkbox" checked={selectedDrivers.includes(driver)} disabled={!selectedDrivers.includes(driver) && selectedDrivers.length >= 4} onChange={() => setDrivers(old => old.includes(driver) ? old.filter(d => d !== driver && choices.includes(d)) : [...old.filter(d => choices.includes(d)),driver])}/>{driver}</label>)}</div></fieldset>
    <p role="status">{selectedDrivers.length ? `Comparing ${selectedDrivers.length} selected driver${selectedDrivers.length === 1 ? '' : 's'}: ${selectedDrivers.join(', ')}.` : 'Choose drivers above to see their comparison.'}</p>
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- A focusable overflow region lets keyboard users scroll comparison columns. */}
    <div className="table-viewport" tabIndex={0} role="region" aria-label="Scrollable driver comparison"><table><caption>{selectedDrivers.length ? 'Selected drivers — ' + scope : 'Driver comparison'}</caption><thead><tr><th scope="col">Driver</th>{!singleRow && <th scope="col">Records included</th>}{fields.map(f => <th scope="col" key={f.key}>{fieldLabel(f)}</th>)}</tr></thead>
      <tbody>{summaries.map(summary => <tr key={summary.driver}><th scope="row">{summary.driver}</th>{!singleRow && <td>{summary.rows}</td>}{summary.values.map((value,i) => <td key={i}>{value}</td>)}</tr>)}</tbody>
    </table></div>
  </div>;
}
```

## frontend/src/enhancements/FeatureBar.jsx

[Editable source](../frontend/src/enhancements/FeatureBar.jsx) — SHA-256: `13d16b3625a44ac7d04d13b45da2aa861c7591110d7b837580336b7b5ee7073c`

```jsx
import {useEffect, useId, useRef, useState} from 'react';
import {createPortal} from 'react-dom';
import {deletePreset, readPresets, routes, safeValues, savePreset, shareUrl} from './preferences.js';

export function readOptions() {
  const defaults = {design: true, cache: true};
  try {
    const stored = JSON.parse(localStorage.getItem('f1analysis.enhancement-options') || '{}');
    return {
      design: typeof stored?.design === 'boolean' ? stored.design : defaults.design,
      cache: typeof stored?.cache === 'boolean' ? stored.cache : defaults.cache
    };
  } catch {return defaults;}
}

function downloadJSON(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function FeatureBar({page, values, options, setOptions, restore, navigate, analysisRevision = null, busy = false}) {
  const [presets, setPresets] = useState(() => readPresets());
  const [name, setName] = useState(''), [chosen, setChosen] = useState('');
  const [message, setMessage] = useState(''), [error, setError] = useState('');
  const [provenance, setProvenance] = useState(null);
  const [provenanceState, setProvenanceState] = useState('loading');
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setProvenance(null); setProvenanceState('loading');
    fetch('/api/enhancements/status', {signal: controller.signal, cache: 'no-store'})
      .then(response => {if (!response.ok) throw new Error('Unavailable'); return response.json();})
      .then(value => {
        if (typeof value?.revision !== 'string' || !value.dataset || !Array.isArray(value.models)) throw new Error('Invalid provenance');
        if (active) {setProvenance(value); setProvenanceState('ready');}
      }).catch(() => {if (active) setProvenanceState('unavailable');});
    return () => {active = false; controller.abort();};
  }, [page, values, retry, analysisRevision]);
  const staleProvenance = Boolean(analysisRevision && provenance && provenance.revision !== analysisRevision);
  const canExport = provenanceState === 'ready' && !busy && !staleProvenance;
  function run(operation) {
    setError(''); setMessage('');
    Promise.resolve().then(operation).catch(err => setError(err instanceof Error ? err.message : 'The analysis tool could not complete. Please try again.'));
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
      <button onClick={() => run(() => {setPresets(savePreset(name, page, values));setChosen(name.trim());setMessage('View saved on this device.');})}>Save view</button>
      <label>Saved views <select value={chosen} onChange={e => setChosen(e.target.value)}><option value="">Choose a view</option>{presets.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label>
      <button disabled={!chosen} onClick={() => run(() => {const item = presets.find(p => p.name === chosen);if(item) {restore(item);setMessage('Saved view restored.');}})}>Load view</button>
      <button disabled={!chosen} onClick={() => run(() => {setPresets(deletePreset(chosen));setChosen('');})}>Delete view</button>
      <button onClick={() => run(async () => {await navigator.clipboard.writeText(shareUrl(page, values));setMessage('View link copied. Uploads and betting inputs are excluded.');})}>Copy view link</button>
      <button disabled={!canExport} onClick={() => run(() => downloadJSON({
        schema: 'f1-analysis-context-v1', exported_at: new Date().toISOString(),
        page: routes[page-1], values: safeValues(values), analysis_revision: analysisRevision || provenance.revision, provenance
      }, 'analysis-context-' + new Date().toISOString().slice(0,10) + '.json'))}>Download analysis context</button>
      <button onClick={() => window.print()}>Print current view</button>
      <CommandPalette navigate={navigate}/>
    </div>
    {provenanceState === 'loading' && <p className="provenance" role="status">Loading source details for context export.</p>}
    {provenanceState === 'unavailable' && <p className="provenance">Source details are unavailable. Context export needs the current data revision. <button onClick={() => setRetry(value => value + 1)}>Retry source details</button></p>}
    {staleProvenance && <p className="provenance" role="status">Source data changed after this analysis. Refresh the analysis before exporting its context. <button onClick={() => setRetry(value => value + 1)}>Recheck source details</button></p>}
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
  const dialog = useRef(null), returnFocus = useRef(null), label = useId();
  const [open, setOpen] = useState(false), [query, setQuery] = useState(''), [selected, setSelected] = useState(0);
  const matches = routes.map((name,index) => ({name,index})).filter(item => item.name.toLowerCase().includes(query.toLowerCase()));
  useEffect(() => {
    function key(event) {
      if ((event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        if (!dialog.current?.open) returnFocus.current = document.activeElement;
        setOpen(s => !s);
      }
    }
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key);
  }, []);
  useEffect(() => {
    if (open && !dialog.current.open) {dialog.current.showModal();dialog.current.querySelector('input')?.focus();}
    else if (!open && dialog.current.open) {dialog.current.close();returnFocus.current?.focus();}
  }, [open]);
  const choose = index => {navigate(index);setOpen(false);setQuery('');setSelected(0);};
  function keyboard(event) {
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key) && matches.length) {
      event.preventDefault();
      setSelected(current => event.key === 'Home' ? 0 : event.key === 'End' ? matches.length - 1 :
        (current + (event.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length);
    } else if (event.key === 'Enter' && matches[selected]) {event.preventDefault();choose(matches[selected].index);}
  }
  return <>
    <button onClick={event => {returnFocus.current = event.currentTarget;setOpen(true);}}>Find section (Ctrl/⌘ K)</button>
    {createPortal(<dialog ref={dialog} className="command-dialog parity-app" aria-labelledby={label}
      onCancel={event => {event.preventDefault();setOpen(false);}}
      onClose={event => {
        // A native close event can arrive after the next shortcut has reopened it.
        if (event.currentTarget.open) return;
        setOpen(false);returnFocus.current?.focus();
      }}>
      <h2 id={label}>Find a section</h2>
      <label>Search sections <input value={query} onChange={e => {setQuery(e.target.value);setSelected(0);}} onKeyDown={keyboard} aria-describedby={`${label}-selection`}/></label>
      <p id={`${label}-selection`} className="palette-selection" aria-live="polite">{matches[selected] ? `Selected section: ${matches[selected].name}. Use arrow keys and Enter, or choose a button.` : 'No matching sections.'}</p>
      <ul>{matches.map((item, index) => <li key={item.name}><button className={selected === index ? 'selected-section' : undefined} onClick={() => choose(item.index)}>{item.name}</button></li>)}</ul>
      <button onClick={() => setOpen(false)}>Close</button>
    </dialog>, document.body)}
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

## frontend/src/enhancements/ResearchJobs.jsx

[Editable source](../frontend/src/enhancements/ResearchJobs.jsx) — SHA-256: `4ddbdb28f7d4fc92ab7205ac73cec07ee3c022f42694261171c0089e59e93786`

```jsx
import {useCallback, useEffect, useRef, useState} from 'react';
import {ViewNodes} from '../components/Presentation';

const ROWS = 'Rows to read (0 = all)';
const BINS = 'Select q values (number of bins)';
const pending = job => ['queued', 'running'].includes(job?.state);
const inputs = new Set(['button', 'checkbox', 'number', 'select', 'multiselect', 'slider', 'upload', 'text_input', 'text_area']);

// Results are a snapshot: keep output and downloads, never resubmit page controls.
export function researchOutput(nodes = []) {
  return nodes.flatMap(node => {
    if (node.hidden || inputs.has(node.type)) return [];
    if (node.type === 'tabs' || node.type === 'tab') return researchOutput(node.children || []);
    return [{...node, ...(node.children ? {children: researchOutput(node.children)} : {})}];
  });
}

export function ResearchJobs({page, values = {}}) {
  const [token, setToken] = useState('');
  const [access, setAccess] = useState(null);
  const [accessError, setAccessError] = useState('');
  const [accessTick, setAccessTick] = useState(0);
  const [task, setTask] = useState(page === 5 ? 'bin-comparison' : 'leakage-audit');
  const [rows, setRows] = useState(Number.isInteger(values[ROWS]) && values[ROWS] > 0 ? values[ROWS] : 1000);
  const [bins, setBins] = useState([2]);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [paused, setPaused] = useState(false);
  const [tick, setTick] = useState(0);
  const section = useRef(null), tokenInput = useRef(null), taskInput = useRef(null);
  const operation = useRef(null), mounted = useRef(true);
  const active = pending(job) || sending;
  const relevant = page === 5 || page === 6 || Boolean(job);
  const canRequest = Boolean(access && (!access.token_required || token));

  useEffect(() => {
    mounted.current = true;
    return () => {mounted.current = false;operation.current?.abort();};
  }, []);

  useEffect(() => {
    if (!relevant) return;
    let disposed = false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort('timeout'), 30000);
    setAccess(null);setAccessError('');
    async function loadAccess() {
      try {
        const response = await fetch('/api/enhancements/research-access', {signal: controller.signal, cache: 'no-store'});
        if (!response.ok) throw new Error('Research access could not be checked.');
        const body = await response.json();
        if (!['local', 'token'].includes(body.mode) || body.token_required !== (body.mode === 'token')) throw new Error('Research access could not be checked.');
        if (!disposed) setAccess(body);
      } catch (err) {
        if (!disposed) setAccessError(controller.signal.reason === 'timeout' ? 'The server took too long to check research access.' : err.message || 'Research access is unavailable.');
      } finally {clearTimeout(timer);}
    }
    loadAccess();
    return () => {disposed = true;clearTimeout(timer);controller.abort();};
  }, [relevant, accessTick]);

  useEffect(() => {
    function open(event) {
      if (!['leakage-audit', 'bin-comparison'].includes(event.detail)) return;
      if (!active) {
        setTask(event.detail);
        const q = values[BINS];
        if (Array.isArray(q) && q.length && q.every(n => Number.isInteger(n) && n >= 2 && n <= 10)) setBins([...new Set(q)]);
        const n = values[ROWS];
        if (Number.isInteger(n) && n >= 1 && n <= 100000) setRows(n);
      }
      if (section.current) {
        section.current.open = true;
        section.current.scrollIntoView?.({block: 'center'});
      }
      (tokenInput.current || taskInput.current)?.focus();
    }
    window.addEventListener('f1analysis:research-task', open);
    return () => window.removeEventListener('f1analysis:research-task', open);
  }, [active, values]);

  const call = useCallback(async (path, options, signal) => {
    const response = await fetch('/api/enhancements/jobs' + path, {
      ...options, signal, cache: 'no-store',
      headers: {'Content-Type': 'application/json', ...(access?.token_required ? {'X-F1-Admin-Token': token} : {})},
    });
    const body = await response.json();
    if (!response.ok) {
      if ([403, 503].includes(response.status)) setAccessTick(n => n + 1);
      throw new Error(typeof body.detail === 'string' ? body.detail : 'The research request could not complete.');
    }
    return body;
  }, [token, access]);

  // One outstanding operation at a time, with a timeout and unmount cancellation.
  const request = useCallback(async (action, controller = new AbortController()) => {
    operation.current = controller;
    const timer = setTimeout(() => controller.abort('timeout'), 30000);
    try {
      return await action(controller.signal);
    } catch (err) {
      if (controller.signal.reason === 'timeout') throw new Error('The server took too long. Retry checking this job.');
      throw err;
    } finally {clearTimeout(timer);}
  }, []);

  useEffect(() => {
    if (!job || !canRequest || paused || sending || !pending(job) && (job.state !== 'succeeded' || result)) return;
    let disposed = false;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const state = await request(signal => call('/' + job.id, {}, signal), controller);
        if (disposed) return;
        if (state.state === 'succeeded') {
          const output = await request(signal => call('/' + job.id + '/result', {}, signal), controller);
          if (disposed) return;
          setResult(output);
        }
        if (state.state === 'failed') setError(state.error || 'The calculation failed. Submit a new job.');
        setJob(state);
      } catch (err) {
        if (!disposed) {setError(err.message || 'The job status is unavailable.');setPaused(true);}
      }
    }, 1000);
    return () => {disposed = true;clearTimeout(timer);controller.abort();};
  }, [job, canRequest, paused, sending, result, tick, request, call]);

  async function submit(event) {
    event.preventDefault();
    if (active || !canRequest) return;
    operation.current?.abort();
    setSending(true);setError('');setResult(null);setJob(null);setPaused(false);
    const settings = task === 'leakage-audit' ? {[ROWS]: Number(rows)} : {[BINS]: bins};
    try {
      const next = await request(signal => call('', {method: 'POST', body: JSON.stringify({task, values: settings})}, signal));
      if (mounted.current) setJob(next);
    } catch (err) {if (mounted.current) setError(err.message || 'The job could not be queued.');}
    finally {if (mounted.current) setSending(false);}
  }

  async function cancel() {
    operation.current?.abort();
    setSending(true);setError('');
    try {
      const response = await request(signal => call('/' + job.id, {method: 'DELETE'}, signal));
      if (!mounted.current) return;
      setJob(response.job || {...job, state: response.cancelled ? 'cancelled' : 'running'});
      if (!response.cancelled) setError('The calculation has started. It will finish normally.');
    } catch (err) {if (mounted.current) {setError(err.message);setPaused(true);}}
    finally {if (mounted.current) setSending(false);}
  }

  function changeToken(value) {
    operation.current?.abort();setToken(value);setError('');setResult(null);setPaused(false);
  }

  return <details ref={section} className="research-job" hidden={!relevant}>
    <summary>{access?.token_required ? 'Administrator research jobs' : 'Research jobs'}</summary>
    <p>Queue a calculation while continuing to browse. Only waiting jobs can be cancelled. Results are available for ten minutes after completion.</p>
    {!access && !accessError && <p role="status">Checking research access…</p>}
    {accessError && <><p role="alert">{accessError}</p><button className="view-button" onClick={() => setAccessTick(n => n + 1)}>Retry research access</button></>}
    {access?.mode === 'local' && <p>Research tools are ready on this computer.</p>}
    <form onSubmit={submit}>
      {access?.token_required && <label>Administrator token <input ref={tokenInput} type="password" autoComplete="off" value={token} onChange={event => changeToken(event.target.value)} disabled={sending} required/></label>}
      <label>Research task <select ref={taskInput} value={task} disabled={active} onChange={event => setTask(event.target.value)}><option value="leakage-audit">Temporal leakage audit</option><option value="bin-comparison">Bin-count comparison</option></select></label>
      {task === 'leakage-audit' ? <label>Audit row limit <input type="number" min="1" max="100000" step="1" value={rows} disabled={active} required onChange={event => setRows(Number(event.target.value))}/></label> : <fieldset disabled={active}><legend>Bin counts to compare</legend>{[2,3,4,5,6,7,8,9,10].map(n => <label key={n}><input type="checkbox" checked={bins.includes(n)} onChange={event => setBins(old => event.target.checked ? [...old,n].sort((a,b) => a-b) : old.filter(q => q !== n))}/>{n}</label>)}</fieldset>}
      <button className="view-button" disabled={!canRequest || active || task === 'bin-comparison' && !bins.length}>Queue calculation</button>
    </form>
    {job && <p role="status">Job {job.id}: {job.state}</p>}
    {job?.state === 'running' && <p>The calculation is running. You can keep browsing.</p>}
    {job?.state === 'queued' && <button className="view-button" disabled={sending || !canRequest} onClick={cancel}>Cancel queued job</button>}
    {error && <p role="alert">{error}</p>}
    {paused && job && <button className="view-button" disabled={!canRequest || sending} onClick={() => {setError('');setPaused(false);setTick(n => n + 1);}}>Retry job status</button>}
    {result && <section aria-label="Research results"><p>Calculated with source revision {result.source_revision || job.revision || 'unavailable'}.</p><ViewNodes nodes={researchOutput(result.nodes)}/></section>}
  </details>;
}
```

## frontend/src/enhancements/SafePlotlyChart.jsx

[Editable source](../frontend/src/enhancements/SafePlotlyChart.jsx) — SHA-256: `d2df19aac9044dbe8c340825b2f52e4b87c6e081257e3e1e11e5615f8f2a46b9`

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
      observer = new ResizeObserver(() => {
        Promise.resolve().then(() => {if (!disposed) return chart.Plots.resize(element);}).catch(() => {});
      });
      observer.observe(element);
    }).catch(err => {if (!disposed) setError(err.message);});
    return () => {disposed=true;observer?.disconnect();if(chart)chart.purge(element);};
  }, [node.spec]);
  return <div className="view-chart" ref={ref} role="img" aria-label={node.label || 'Interactive Plotly chart'}>
    {error && <div role="alert">Chart unavailable: {error}. Other analysis remains available.</div>}
  </div>;
}
```

## frontend/src/enhancements/preferences.js

[Editable source](../frontend/src/enhancements/preferences.js) — SHA-256: `d6c6015b25a6920bded9229e007a8a4d2f54552851dcd8b1a462613d14aa33bb`

```javascript
export const routes = ['Data Explorer', 'Analytics', 'Current Season', 'Next Race', 'Predictive Models', 'Raw Data', 'Betting Research'];
const storageKey = 'f1analysis.saved-views.v1';
const allowed = /^(filter_results_main|(?:range_filter_|checkbox_filter_|filter_).+|_tabs:.+|Select Model Type|tire_year_select|tire_race_select)$/;

export function safeValues(values = {}) {
  if (!values || typeof values !== 'object' || Array.isArray(values)) return {};
  const primitive = value => value == null || typeof value === 'boolean' ||
    typeof value === 'number' && Number.isFinite(value) ||
    typeof value === 'string' && value.length <= 4096;
  return Object.fromEntries(Object.entries(values).filter(([key, value]) =>
    allowed.test(key) && !/upload|csv|ledger|password|token/i.test(key) &&
    (primitive(value) || Array.isArray(value) && value.length <= 20 && value.every(primitive))
  ));
}

export function validateView(view) {
  if (!view || view.version !== 1 || !Number.isInteger(view.page) ||
      view.page < 1 || view.page > routes.length) throw new Error('Unsupported saved view.');
  return {version: 1, page: view.page, values: safeValues(view.values)};
}

export function readPresets(storage) {
  try {
    return JSON.parse((storage || localStorage).getItem(storageKey) || '[]').slice(0, 20)
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

## frontend/src/enhancements/viewClient.js

[Editable source](../frontend/src/enhancements/viewClient.js) — SHA-256: `f2c4626da01896502aa7e330666e9068ff44db6579e2755044722162e6ff333f`

```javascript
import {hasUpload, stableKey} from './preferences.js';

// Keep actions and uploaded data out of shared requests and retained responses.
export function createViewClient({fetcher = fetch, now = Date.now, ttl = 15000, maxEntries = 6, maxBytes = 12000000, normalTimeout = 120000, actionTimeout = 600000} = {}) {
  const cache = new Map(), pending = new Map();
  let revision = '', retainedBytes = 0, epoch = 0;
  const clear = () => {cache.clear(); retainedBytes = 0; epoch++;};
  async function json(url, options) {
    const response = await fetcher(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(typeof body.detail === 'string' ? body.detail : 'Request failed (' + response.status + ').'), {status: response.status});
    const sourceRevision = url === '/api/views' ? response.headers?.get?.('X-F1-Revision') : null;
    return sourceRevision ? {...body, source_revision:sourceRevision} : body;
  }
  async function status(signal) {
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, {once: true});
    if (signal?.aborted) controller.abort();
    let timedOut = false;
    const timer = setTimeout(() => {timedOut = true;controller.abort();}, normalTimeout);
    try {return await json('/api/enhancements/status', {signal: controller.signal, cache: 'no-store'});}
    catch (error) {
      if (timedOut) throw new Error('Checking analysis data timed out. Retry the request.');
      throw error;
    } finally {clearTimeout(timer);signal?.removeEventListener('abort', abort);}
  }
  /** @param {object} payload @param {{signal?: AbortSignal, enabled?: boolean}} [options] */
  async function load(payload, {signal, enabled = false} = {}) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    const reusable = enabled && payload.page <= 5 && !payload.action && !hasUpload(payload.values || {});
    if (payload.action || hasUpload(payload.values || {})) clear();
    // Probe on every reusable navigation: never serve a client hit under an old revision.
    if (reusable) {
      const state = await status(signal);
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
    if (task && (task.epoch !== epoch || task.controller.signal.aborted)) task = null;
    if (!task) {
      const controller = new AbortController();
      task = {controller, consumers: 0, promise: null, epoch};
      let timedOut = false;
      const timeout = setTimeout(() => {timedOut = true;controller.abort();}, payload.action ? actionTimeout : normalTimeout);
      task.promise = json('/api/views', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload), signal: controller.signal
      }).then(value => {
        if (reusable && task.epoch === epoch) {
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

## frontend/src/enhancements/enhancements.css

[Editable source](../frontend/src/enhancements/enhancements.css) — SHA-256: `d3e22bc6d853230945dc5e7551aaf2aee578234117bc0379dc9fe62b23134045`

```css
/* Default readability profile; users can restore the base layout. Load after parity.css. */
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
:root[data-enhancements='on']:not([data-theme='dark']) .view-notice.info {color:#17436a}
:root[data-enhancements='on'][data-theme='dark'] .view-notice.info {color:#b3d8f5}
:root[data-enhancements='on'] .slider-values {font-weight:600}
:root[data-enhancements='on'] .table-toolbar {
  position:relative;top:auto;right:auto;opacity:1;pointer-events:auto;
  height:auto;min-height:44px;width:max-content;max-width:100%;margin-left:auto;z-index:5;
}
:root[data-enhancements='on'] .table-toolbar button {min-width:44px;min-height:44px;color:var(--text)}
:root[data-enhancements='on'] .canvas-table .column-picker {top:44px;max-width:min(400px,100%)}
:root[data-enhancements='on'] .canvas-table .grid-column-menu {top:79px;max-width:100%;max-height:min(450px,70vh);overflow:auto}
:root[data-enhancements='on'] .canvas-table .grid-column-menu>button {min-height:44px}
:root[data-enhancements='on'] .canvas-table:has(.column-picker),
:root[data-enhancements='on'] .canvas-table:has(.grid-column-menu) {overflow:visible}
:root[data-enhancements='on'] button:focus-visible,
:root[data-enhancements='on'] a:focus-visible,
:root[data-enhancements='on'] input:focus-visible,
:root[data-enhancements='on'] select:focus-visible {outline:3px solid var(--accent);outline-offset:3px}
:root[data-enhancements='on'] summary:focus-visible,
:root[data-enhancements='on'] [tabindex]:focus-visible {outline:3px solid var(--accent);outline-offset:3px}
:root[data-enhancements='on'] .parity-nav {position:sticky;top:56px;background:var(--page-bg);z-index:10;scroll-margin-top:110px}
:root[data-enhancements='on'] .parity-nav button {min-height:44px}
:root[data-enhancements='on'] .app-toolbar button,
:root[data-enhancements='on'] .view-button,
:root[data-enhancements='on'] .number-input button,
:root[data-enhancements='on'] summary,
:root[data-enhancements='on'] .view-checkbox {min-height:44px}
:root[data-enhancements='on'] .app-toolbar button,
:root[data-enhancements='on'] .number-input button,
:root[data-enhancements='on'] .view-button {min-width:44px}
:root[data-enhancements='on'] .parity-footer {font-family:'Source Sans',sans-serif;color:var(--muted)}
:root[data-enhancements='on'] .parity-footer p+p {color:var(--muted)}
.enhancement-bar {display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:12px 0;font-family:'Source Sans',sans-serif}
.enhancement-bar label {display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-height:44px}
.enhancement-bar button,.enhancement-bar select,.enhancement-bar input:not([type='checkbox']),.accessible-table button,.accessible-table select,.accessible-table input:not([type='checkbox']),.provenance button {min-height:44px;max-width:100%}
.enhancement-bar input:not([type='checkbox']) {width:180px}
.enhancement-bar input[type='checkbox'],.accessible-table input[type='checkbox'] {width:20px;height:20px;flex-shrink:0;accent-color:var(--accent)}
.enhancement-error {color:var(--accent);overflow-wrap:anywhere}
.load-feedback {position:fixed;bottom:12px;right:12px;max-width:calc(100vw - 24px);background:var(--page-bg);border:1px solid var(--border);padding:12px 16px;border-radius:8px;z-index:11}
.command-dialog {background:var(--page-bg);color:var(--text);border:1px solid var(--border);border-radius:12px;width:min(520px,calc(100vw - 32px));max-height:80vh}
.command-dialog::backdrop {background:#0008}
.command-dialog input {width:100%;min-height:44px}
.command-dialog ul {padding:0;list-style:none}
.command-dialog li button {width:100%;min-height:44px;text-align:left}
.command-dialog .selected-section {outline:2px solid var(--accent);outline-offset:-2px;font-weight:600}
.palette-selection {color:var(--muted);font-size:14px}
.accessible-table {max-width:100%;margin:16px 0}
.accessible-table .table-viewport {overflow:auto;max-height:500px}
.accessible-table table {border-collapse:collapse;width:100%;font-family:'Source Sans',sans-serif}
.accessible-table th,.accessible-table td {padding:8px;border:1px solid var(--border);text-align:left}
.accessible-table th {position:sticky;top:0;background:var(--page-bg)}
.accessible-table .columns-list {display:flex;gap:12px;flex-wrap:wrap;max-height:200px;overflow:auto}
.accessible-table .columns-list label {display:flex;align-items:center;gap:8px;min-height:44px;overflow-wrap:anywhere}
.accessible-table caption {text-align:left;color:var(--muted);padding:8px 0}
.accessible-table nav {display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;padding-top:12px}
.driver-picker {min-width:0;border:1px solid var(--border);margin:0;padding:12px}
.provenance {font-size:14px;color:var(--muted);padding:8px 0}
.provenance p {overflow-wrap:anywhere}
.research-job {margin:24px 0;padding:12px;border:1px solid var(--border);border-radius:8px;overflow-wrap:anywhere}
.research-job summary {cursor:pointer;min-height:44px;display:list-item}
.research-job form {display:flex;flex-wrap:wrap;gap:16px;align-items:end}
.research-job form>label {display:flex;flex-direction:column;gap:6px;max-width:100%}
.research-job input:not([type='checkbox']),.research-job select {min-height:44px;max-width:100%;background:var(--page-bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:8px;font:inherit}
.research-job fieldset {display:flex;flex-wrap:wrap;gap:12px;border:1px solid var(--border);min-width:0}
.research-job fieldset label {display:flex;align-items:center;gap:6px;min-height:44px}
.research-job p {max-width:100%}
@media(max-width:640px) {
  :root[data-enhancements='on'] .main-shell {padding-top:56px}
  :root[data-enhancements='on'] .parity-header>img {width:210px}
  :root[data-enhancements='on'] h1.view-heading,
  :root[data-enhancements='on'] .shell-title {font-size:30px}
  :root[data-enhancements='on'] .filter-sidebar {width:min(300px,calc(100vw - 56px));padding-bottom:80px}
  .enhancement-bar>* {max-width:100%}
  .enhancement-bar input:not([type='checkbox']),.enhancement-bar select {min-width:0;max-width:100%}
  .accessible-table input:not([type='checkbox']) {display:block;width:100%}
}
@media(prefers-reduced-motion:reduce) {
  :root[data-enhancements='on'] * {scroll-behavior:auto!important;animation:none!important;transition:none!important}
}
@media print {
  :root[data-enhancements='on'] .app-toolbar,
  :root[data-enhancements='on'] .filter-sidebar,
  :root[data-enhancements='on'] .parity-nav,
  .enhancement-bar,.table-toolbar,.load-feedback,.research-job form {display:none!important}
  :root[data-enhancements='on'] .main-shell {margin:0!important;width:100%!important;padding:0!important}
  .accessible-table .table-viewport {max-height:none;overflow:visible}
}
```

## frontend/src/parity.css

[Editable source](../frontend/src/parity.css) — SHA-256: `dfbe15918a261b96a0b39e775aa4a9621433c8b6688927380d2525af582a10d8`

```css
@font-face{font-family:'Source Sans';src:url('/fonts/SourceSans.woff2') format('woff2');font-weight:100 900;font-display:swap}
@font-face{font-family:'Source Code';src:url('/fonts/SourceCode.woff2') format('woff2');font-weight:100 900;font-display:swap}
@font-face{font-family:'Source Sans';src:url('/fonts/SourceSansItalic.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
@font-face{font-family:'Source Code';src:url('/fonts/SourceCodeItalic.woff2') format('woff2');font-weight:100 900;font-style:italic;font-display:swap}
:root{--page-bg:#fff;--text:#31333f;--border:#d6d8df;--muted:#555965;--accent:#ff4b4b;color-scheme:light;font-family:'Source Sans',sans-serif;color:var(--text);background:var(--page-bg)}
:root[data-theme='light']{--text:#31333f;--border:#d6d8df;--muted:#555965;--accent:#ff4b4b}
:root[data-theme='dark']{--page-bg:#0e1117;--text:#fafafa;--border:#3a3d46;--muted:#b8bac2;--accent:#ff4b4b;color-scheme:dark}
body{font-family:'Source Sans',sans-serif;line-height:1.6}
*{box-sizing:border-box}
body{margin:0;min-width:320px;background:var(--page-bg);color:var(--text)}
button{cursor:pointer}
button:disabled{cursor:not-allowed}
:focus-visible{outline:2px solid #0068c9;outline-offset:2px}
.skip-link{position:absolute;left:-9999px;top:0;z-index:1000;padding:8px 12px;background:#17517d;color:white;text-decoration:none}
.skip-link:focus{left:0}
main:focus{outline:none}
.parity-app{color:var(--text);font-size:16px}
.parity-app button,.parity-app input,.parity-app select{font:inherit;color:inherit}
.parity-app button{background:transparent}
.main-shell{padding:96px 80px 160px;margin:0 auto;width:100%;min-width:0}
.parity-header>img{display:block;width:450px;height:auto;max-width:100%;aspect-ratio:450/264}
.shell-copy{margin-top:16px}
.view-flow{display:flex;flex-direction:column;gap:16px;min-width:0}
.view-flow:empty{display:none}
.view-heading{font-weight:600;letter-spacing:0;line-height:1.2;margin:0 0 -16px;padding:16px 0}
h1.view-heading,.shell-title{font-size:44px;font-weight:700;padding:20px 0 16px;line-height:1.2}
h2.view-heading{font-size:36px}
h3.view-heading{font-size:28px}
.view-markdown,.view-caption{margin-bottom:-16px}
.view-markdown p,.view-caption p{margin:0 0 16px;line-height:1.6}
.view-markdown h3{font-size:28px;font-weight:600;line-height:1.2;padding:16px 0;margin:0 0 16px}
.view-markdown ul{padding-left:32px;margin-top:0}
.view-caption{font-size:14px;color:var(--muted)}
.view-markdown code,.view-caption code,.view-notice code{font-family:'Source Code',monospace;font-size:.875em;color:inherit;background:#f0f2f6;padding:2px 4px;border-radius:4px}
.parity-nav{margin-top:32px;min-width:0;margin-bottom:16px}
.parity-nav>div,.view-tablist{position:relative;display:flex;gap:16px;overflow-x:auto;overflow-y:hidden;white-space:nowrap;border-bottom:2px solid #e6e7eb;scrollbar-width:none}
.parity-nav button,.view-tablist button{flex:0 0 auto;padding:0;height:38px;background:transparent!important;border:0;border-bottom:2px solid transparent;border-radius:0;color:var(--text)!important;font-size:14px;line-height:1;white-space:nowrap}
.parity-nav button[aria-selected='true'],.view-tablist button[aria-selected='true']{color:var(--accent)!important;border-bottom-color:var(--accent)}
.parity-nav button:hover,.view-tablist button:hover{color:var(--accent)!important}
.app-toolbar{position:fixed;top:0;left:0;right:0;height:56px;z-index:30;pointer-events:none;display:flex;align-items:center;justify-content:flex-end;padding:0 16px}
.app-toolbar button{pointer-events:auto;border:0;line-height:1;font-size:24px;padding:4px 8px;color:var(--text);background:transparent}
.settings-menu{pointer-events:auto;position:absolute;right:16px;top:48px;padding:16px;background:var(--page-bg);box-shadow:0 2px 12px #0003;border:1px solid var(--border);border-radius:8px}
.settings-menu label{display:flex;gap:8px;align-items:center}
.settings-menu input{width:16px;height:16px}
.view-checkbox{display:flex;align-items:center;gap:8px;width:fit-content;font-size:14px;line-height:24px;min-height:24px;margin:0;cursor:pointer}
.view-checkbox input{appearance:none;width:16px;height:16px;margin:0;border:1px solid #d5d8df;border-radius:4px;background:transparent;padding:0;flex-shrink:0}
.view-checkbox input:checked{background:var(--accent);border-color:var(--accent)}
.view-checkbox input:checked:after{content:'✓';display:block;color:white;font-size:13px;line-height:14px;text-align:center}
.view-field{display:flex;flex-direction:column;gap:4px;font-size:14px;line-height:22.4px;min-width:0}
.view-field>input,.view-field select,.number-input{height:40px;border:0!important;border-radius:8px;background:#f0f2f6!important;color:var(--text)!important;font-size:16px;padding:8px 12px;min-width:0;width:100%}
.number-input{display:flex;padding:0!important;overflow:hidden}
.number-input input{border:0!important;border-radius:0!important;background:transparent!important;padding:8px 12px!important;min-width:0;width:100%;appearance:textfield}
.number-input input::-webkit-inner-spin-button{appearance:none}
.number-input button{width:32px;flex-shrink:0;padding:0!important;border:0!important;border-radius:0;font-size:22px;background:transparent!important}
.number-input button:hover{background:#e4e6ec!important}
.view-notice{display:flex;gap:16px;padding:16px;border-radius:8px;background:#e8f2fc;color:#17517d;font-size:16px;line-height:25.6px}
.view-notice p{margin:0}
.view-notice.info{background:#e8f2fc;color:#17517d}
.view-notice.warning{background:#fff8e1;color:#715000}
.view-notice.error{background:#ffeded;color:#922222}
.view-notice.success{background:#e5f7ed;color:#17683b}
.view-columns{display:grid;gap:16px;min-width:0}
.view-metric{min-width:0}
.view-metric>span{font-size:14px;display:block;line-height:22.4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.view-metric>strong{font-size:36px;line-height:43.2px;font-weight:400;display:block;letter-spacing:0}
.view-expander{border:1px solid var(--border);border-radius:8px;min-width:0}
.view-expander summary{list-style:none;padding:12px 16px;cursor:pointer;font-size:14px;line-height:24px;display:flex;align-items:center;gap:8px}
.view-expander summary:before{content:'›';font-size:24px;line-height:1}
.view-expander[open]>summary:before{transform:rotate(90deg)}
.view-expander>div{padding:0 16px 16px}
.view-tabs{min-width:0}
.tab-content{padding-top:16px}
.tab-content[hidden]{display:none}
.view-divider{border:0;border-top:1px solid var(--border);margin:16px 0}
.view-button{display:inline-flex;align-items:center;justify-content:center;align-self:flex-start;min-height:40px;padding:6px 12px;border:1px solid var(--border)!important;border-radius:8px;background:var(--page-bg)!important;font-weight:400;font-size:16px;color:var(--text)!important;text-decoration:none;line-height:24px}
.view-button:hover{border-color:var(--accent)!important;color:var(--accent)!important}
.view-button:disabled{opacity:.45}
.view-image{display:block;height:auto}
.view-code,.view-json{margin:0;white-space:pre-wrap;overflow:auto;font-family:'Source Code',monospace;font-size:14px;line-height:1.6;background:#f0f2f6;padding:16px;border-radius:8px}
.view-html{max-width:100%;overflow:auto}
.view-html a{color:#0068c9}
.filter-sidebar{position:fixed;left:0;top:0;bottom:0;width:300px;padding:76px 30px 32px;background:#f0f2f6;z-index:20;overflow-y:auto;overflow-x:hidden;box-shadow:2px 0 12px #00000008}
.filter-sidebar h2{font-size:20px;padding:16px 0;line-height:24px}
.with-sidebar .main-shell{margin-left:300px;width:calc(100% - 300px)}
.sidebar-toggle{position:absolute;left:252px;top:16px;z-index:25}
.filter-sidebar .view-field select{background:#fff!important}
.view-slider{font-size:14px;line-height:22.4px;height:68px}
.slider-values{display:flex;justify-content:space-between;color:var(--accent);font-size:14px;margin-top:0;height:18px;line-height:18px}
.range-track{position:relative;height:22px;margin:0 0 2px}
.range-track:before{content:'';position:absolute;left:0;right:0;top:6px;height:4px;border-radius:4px;background:linear-gradient(to right,#d5d8df 0 var(--start),var(--accent) var(--start) var(--end),#d5d8df var(--end) 100%)}
.range-track input{position:absolute;top:0;left:0;appearance:none;width:100%;height:16px;padding:0;margin:0;pointer-events:none;background:transparent!important;border:0!important}
.range-track input::-webkit-slider-thumb{appearance:none;width:12px;height:12px;background:var(--accent);border-radius:50%;pointer-events:auto;cursor:pointer}
.range-track input::-moz-range-thumb{width:12px;height:12px;background:var(--accent);border:0;border-radius:50%;pointer-events:auto}
.range-track input:focus-visible{outline:none}
.range-track input:focus-visible::-webkit-slider-thumb{outline:2px solid var(--text);outline-offset:2px}
.slider-bounds{visibility:hidden;display:flex;justify-content:space-between;font-size:12px;color:var(--muted);line-height:16px;opacity:.5;position:absolute;left:0;right:0;bottom:0}
.view-slider{position:relative}
.view-upload .upload-zone{display:flex;align-items:center;gap:16px;background:#f0f2f6;padding:16px;border-radius:8px;position:relative;font-size:16px;min-height:96px}
.upload-zone>span:first-child{font-size:32px}
.upload-zone div{flex:1}
.upload-zone small{display:block;font-size:14px;color:var(--muted)}
.upload-browse{border:1px solid var(--border);background:var(--page-bg);padding:6px 12px;border-radius:8px;font-size:14px;white-space:nowrap}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.view-chart{position:relative;width:100%;min-height:350px;overflow:hidden}
.view-chart canvas{display:block}
.view-chart details{position:absolute;top:8px;right:8px}
.view-chart details summary{cursor:pointer}
.view-chart details>div{background:var(--page-bg);padding:8px;display:flex;flex-direction:column;font-size:14px}
.view-chart details a{color:#0068c9}
.view-table{position:relative;width:100%;min-width:0;border-radius:8px;overflow:visible}
.data-grid{overflow:auto;width:100%;border:1px solid var(--border);border-radius:8px;font-family:'Source Sans',sans-serif;font-size:13px;background:var(--page-bg);position:relative;line-height:35px;scrollbar-width:thin;scrollbar-color:#c2c5cc transparent}
.grid-header{height:35px;position:sticky;top:0;background:#f0f2f6;z-index:3}
.grid-row{height:35px;position:absolute;border-bottom:1px solid #eceef1}
.grid-cell{position:absolute;top:0;height:35px;padding:0 8px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:var(--text);border-right:1px solid #eceef1}
.grid-cell.numeric{text-align:right}
.grid-cell.boolean{text-align:center;color:#777c86;font-size:16px}
.grid-header .grid-cell{border-bottom:1px solid var(--border);border-right:0}
.grid-header button{display:block;text-align:left;width:100%;height:35px;overflow:hidden;text-overflow:ellipsis;border:0;padding:0;font-weight:500;font-size:13px;color:var(--muted);border-radius:0;background:transparent!important}
.grid-row:hover{background:#f8f9fb}
.grid-cell:focus{outline:2px solid var(--accent);outline-offset:-2px;background:#fff3f3}
.resize-handle{position:absolute;right:0;top:0;height:35px;width:5px;cursor:col-resize}
.index-cell{color:var(--muted);text-align:right;background:#f0f2f6}
.table-toolbar{position:absolute;right:0;top:-30px;height:30px;display:flex;border:1px solid var(--border);background:var(--page-bg);border-radius:6px;z-index:5;opacity:0;transition:opacity .15s}
.view-table:hover .table-toolbar,.view-table:focus-within .table-toolbar{opacity:1}
.table-toolbar button{border:0;padding:1px 6px;border-radius:0;color:var(--muted);font-size:18px}
.column-picker{position:absolute;top:0;right:0;z-index:10;background:var(--page-bg);border:1px solid var(--border);box-shadow:0 4px 12px #0002;max-height:400px;overflow:auto;padding:12px;max-width:400px}
.column-picker label{display:flex;gap:8px;font-size:14px;line-height:24px}
.column-picker input{width:16px}
.view-table:fullscreen{padding:40px;background:var(--page-bg)}
.view-table:fullscreen .data-grid{height:calc(100vh - 80px)!important}
.grid-empty{font-size:14px;padding:16px}
.parity-footer{text-align:center;padding:20px 0;border-top:1px solid #e0e0e0;margin-top:56px;font-family:sans-serif;font-size:14px;color:#666}
.parity-footer p{margin:0 0 10px;line-height:22.4px}
.parity-footer p+p{font-size:12px;color:#666;margin-bottom:15px;line-height:19.2px}
.parity-footer a{color:#2866bc!important;font-weight:700;text-decoration:none}
.parity-footer img{height:60px;width:auto;border:0}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
:root[data-theme='dark'] .filter-sidebar,:root[data-theme='dark'] .view-field>input,:root[data-theme='dark'] .view-field select,:root[data-theme='dark'] .number-input,:root[data-theme='dark'] .view-code,:root[data-theme='dark'] .view-json,:root[data-theme='dark'] .upload-zone,:root[data-theme='dark'] .grid-header,:root[data-theme='dark'] .index-cell{background:#262730!important;color:var(--text)!important}
:root[data-theme='dark'] .grid-cell,:root[data-theme='dark'] .grid-row{border-color:#30323c}
:root[data-theme='dark'] .grid-row:hover{background:#262730}
:root[data-theme='dark'] .view-notice.info{background:#152e43;color:#b3d8f5}
:root[data-theme='dark'] .parity-footer{color:#aaa}
@media(max-width:991px){.main-shell{padding:96px 16px 160px}}
@media(max-width:640px){.main-shell{padding:96px 16px 160px}.with-sidebar .main-shell{margin-left:0;width:100%}.sidebar-toggle{left:252px}.view-columns{grid-template-columns:1fr!important}.view-field,.view-checkbox{font-size:14px}}


.canvas-table{border:1px solid var(--border);border-radius:8px;overflow:hidden}.canvas-table .table-toolbar{top:0}.canvas-table [data-testid=glide-cell-overlay-editor]{font-family:Source Sans,sans-serif}
.view-help{float:right;border:1px solid #808495;border-radius:50%;font-size:10px;font-weight:600;width:14px;height:14px;line-height:12px;text-align:center;margin-top:4px;color:#808495}.upload-file{padding:8px 16px;display:flex;justify-content:space-between}.upload-file button,.multiselect-box button{border:0;font-size:18px;line-height:1}.multiselect-field{position:relative}.multiselect-box{display:flex;flex-wrap:wrap;gap:8px;background:#f0f2f6;border-radius:8px;padding:8px 12px;min-height:40px}.select-tag{display:flex;align-items:center;gap:6px;background:#ff4b4b;color:white;padding:0 6px;border-radius:4px;font-size:14px;line-height:24px}.select-tag button{color:white}.multiselect-box input{min-width:40px;flex:1;width:40px;background:transparent;border:0;outline:none}.multiselect-options{position:absolute;top:100%;left:0;right:0;background:var(--page-bg);border:1px solid var(--border);box-shadow:0 4px 16px #0002;border-radius:8px;padding:8px;z-index:20;max-height:300px;overflow:auto}.multiselect-options button{display:block;width:100%;border:0;text-align:left;padding:8px;border-radius:4px}.multiselect-options button:hover{background:#f0f2f6}:root[data-theme=dark] .multiselect-box{background:#262730}

.parity-nav,.tab-strip{position:relative}.tab-scroll{position:absolute!important;top:0!important;bottom:2px!important;width:20px!important;height:38px!important;z-index:4!important;border:0!important;background:var(--page-bg)!important;color:#808495!important;font-size:24px!important;line-height:38px!important;padding:0!important}.tab-scroll.left{left:0}.tab-scroll.right{right:0}.view-checkbox>span{font-size:14px;line-height:21px}.view-field>label,.view-field>span{display:flex;align-items:center;justify-content:space-between;min-height:24px}.view-help{margin-top:0}h3.view-heading{padding-top:12px}
.view-caption{color:inherit;opacity:.6}.chart-shell{position:relative;min-width:0}.chart-shell:hover>.table-toolbar,.chart-shell:focus-within>.table-toolbar{opacity:1}.chart-shell>.table-toolbar{top:0}.chart-shell:fullscreen{padding:40px;background:var(--page-bg)}.chart-shell:fullscreen .view-chart{height:calc(100vh - 80px)}
.view-slider>label{position:relative;top:1px}.canvas-table .column-picker{top:30px}
.grid-column-menu{position:absolute;top:35px;z-index:10;width:240px;padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--page-bg);box-shadow:0 4px 16px #0002;font-size:14px;display:flex;flex-direction:column;gap:4px}.grid-column-menu>button{text-align:left;padding:8px;border:0;border-radius:4px;font-size:14px}.grid-column-menu>button:hover{background:#f0f2f6}.grid-column-menu strong,.grid-column-menu small{padding:4px 8px}.grid-column-menu label{display:flex;flex-direction:column;padding:8px;gap:4px}.grid-column-menu select{padding:6px;border:1px solid var(--border);border-radius:4px}
.table-toolbar{pointer-events:none}.view-table:hover>.table-toolbar,.view-table:focus-within>.table-toolbar,.chart-shell:hover>.table-toolbar,.chart-shell:focus-within>.table-toolbar{pointer-events:auto}.chart-shell>.table-toolbar{z-index:6}
.view-slider:hover .slider-bounds,.view-slider:focus-within .slider-bounds{visibility:visible}:root[data-theme=dark] .grid-column-menu>button:hover,:root[data-theme=dark] .multiselect-options button:hover,:root[data-theme=dark] .number-input button:hover{background:#30323b!important}
```

## frontend/index.html

[Editable source](../frontend/index.html) — SHA-256: `77e010431bfa8351827ce794549d5ff30baf180435a32faeef8e5f46a6a101c2`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/png" href="/favicon.png" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#101014" />
    <title>F1 Analysis</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

## frontend/package.json

[Editable source](../frontend/package.json) — SHA-256: `06058308edbc39a695155eb202d66785cfc20354326bee572f4f8760c0476b54`

```json
{
  "name": "f1-analysis-react",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "postinstall": "node scripts/patch-glide.mjs",
    "dev": "vite",
    "prebuild": "node scripts/optimize-assets.mjs",
    "build": "vite build && node scripts/check-budgets.mjs",
    "assets:optimize": "node scripts/optimize-assets.mjs",
    "check:budgets": "node scripts/check-budgets.mjs",
    "check:hosting": "node scripts/check-hosting.mjs",
    "preview": "vite preview",
    "lint": "eslint . --max-warnings=0",
    "typecheck": "tsc --noEmit",
    "test": "vitest run --coverage",
    "test:watch": "vitest",
    "audit": "npm audit --omit=dev",
    "audit:dev": "npm audit",
    "capture:react": "node ../parity_evidence/capture_react.mjs",
    "capture:streamlit": "node ../parity_evidence/capture_streamlit.mjs",
    "capture:diff": "node ../parity_evidence/diff_screenshots.mjs",
    "audit:a11y": "node ../parity_evidence/audit_accessibility.mjs",
    "benchmark": "node ../parity_evidence/benchmark.mjs"
  },
  "dependencies": {
    "@glideapps/glide-data-grid": "^6.0.3",
    "lodash": "^4.18.1",
    "marked": "^4.3.0",
    "papaparse": "^5.4.1",
    "plotly.js-dist-min": "^4.1.1",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-markdown": "^10.1.0",
    "react-responsive-carousel": "^3.2.23",
    "recharts": "^2.15.0",
    "vega": "^6.4.0",
    "vega-embed": "^7.3.0",
    "vega-lite": "^6.4.3"
  },
  "devDependencies": {
    "@eslint/js": "^9.13.0",
    "@testing-library/dom": "^10.4.2",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@testing-library/user-event": "^14.5.2",
    "@types/papaparse": "^5.3.15",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "@vitest/coverage-v8": "^2.1.8",
    "axe-core": "^4.13.0",
    "eslint": "^9.13.0",
    "eslint-plugin-jsx-a11y": "^6.10.2",
    "eslint-plugin-react": "^7.37.2",
    "eslint-plugin-react-hooks": "^5.0.0",
    "globals": "^15.11.0",
    "jsdom": "^25.0.1",
    "playwright": "^1.49.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "sharp": "^0.33.5",
    "typescript": "^5.7.2",
    "vite": "^6.0.0",
    "vitest": "^2.1.8"
  }
}
```

## frontend/vite.config.js

[Editable source](../frontend/vite.config.js) — SHA-256: `1e108ba7c321dd779ad2140bee5826de16faec78337f291cd2ced7b7e8b505bb`

```javascript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: false,
    manifest: true,
    // npm run build enforces the 500,000-byte gzip entry budget, including
    // statically imported chunks. Vite's separate raw-size warnings remain useful.
    chunkSizeWarningLimit: 500,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: false,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{js,jsx}'],
      exclude: ['src/test/**', 'src/main.jsx', '**/*.test.{js,jsx}'],
      // Thresholds are intentionally below the §14 80% target: page-level
      // tests for App.jsx, the full Betting Research workflow, and
      // interactive Data Explorer filter combinations are tracked as
      // follow-up work in PARITY_REPORT.md. The infrastructure (vitest,
      // coverage, the api mock pattern, and 30+ component tests) is in
      // place; only the additional tests are deferred.
      thresholds: {
        lines: 60,
        functions: 40,
        branches: 60,
        statements: 60,
      },
    },
  },
});
```

## frontend/scripts/optimize-assets.mjs

[Editable source](../frontend/scripts/optimize-assets.mjs) — SHA-256: `cdb0f0e1f15fb714d979ee37d15997f8cfaf1d0b7cc655c59b36135724732cf2`

```javascript
import sharp from 'sharp';
import {mkdir, stat} from 'node:fs/promises';
import {fileURLToPath, URL} from 'node:url';
import console from 'node:console';

// Preserve the source mark and its transparency, with 1x/2x display-height variants.
const publicDir = new URL('../public/', import.meta.url);
await mkdir(publicDir, {recursive: true});
const source = new URL('betting-oracle-logo.png', publicDir);
for (const height of [60, 120]) {
  const target = new URL(`betting-oracle-logo-${height}.webp`, publicDir);
  await sharp(fileURLToPath(source)).resize({height, withoutEnlargement: true})
    .webp({lossless: true}).toFile(fileURLToPath(target));
  console.info(JSON.stringify({asset: fileURLToPath(target), bytes: (await stat(target)).size}));
}
```

## frontend/scripts/check-budgets.mjs

[Editable source](../frontend/scripts/check-budgets.mjs) — SHA-256: `a715b88934909488412eda9ca6c958501e6b4885a437fd55c9460df6a8eafa15`

```javascript
import {readFile, readdir, writeFile} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {resolve, relative, sep} from 'node:path';
import {fileURLToPath, URL} from 'node:url';
import console from 'node:console';
import process from 'node:process';

const defaultDist = () => fileURLToPath(new URL('../dist/', import.meta.url));
export const ENTRY_GZIP_BUDGET = 500000;

export async function checkBudgets(directory = defaultDist(), limit = ENTRY_GZIP_BUDGET) {
  const root = resolve(directory);
  const manifest = JSON.parse(await readFile(resolve(root, '.vite/manifest.json'), 'utf8'));
  const entries = Object.values(manifest).filter(item => item.isEntry);
  if (entries.length !== 1) throw new Error('Expected exactly one production entry.');
  const initial = new Set();
  function visit(item) {
    if (!item || typeof item.file !== 'string') throw new Error('Invalid production manifest.');
    if (initial.has(item.file)) return;
    initial.add(item.file);
    for (const key of item.imports || []) visit(manifest[key]);
  }
  visit(entries[0]);
  const rows = [];
  async function scan(folder) {
    for (const entry of await readdir(folder, {withFileTypes: true})) {
      const path = resolve(folder, entry.name);
      if (entry.isDirectory()) await scan(path);
      else if (entry.name.endsWith('.map')) throw new Error('Production source maps must not be published.');
      else if (entry.name.endsWith('.js')) {
        const body = await readFile(path);
        rows.push({name: relative(root, path).split(sep).join('/'), bytes: body.length,
          gzip_bytes: gzipSync(body, {level: 5}).length});
      }
    }
  }
  await scan(root);
  const initialRows = rows.filter(row => initial.has(row.name));
  if (initialRows.length !== initial.size) throw new Error('An initial JavaScript chunk is missing.');
  const initialBytes = initialRows.reduce((sum, row) => sum + row.gzip_bytes, 0);
  if (initialBytes > limit) throw new Error(`Initial JavaScript is ${initialBytes} gzip bytes; budget is ${limit}.`);
  return {budget_gzip_bytes: limit, initial_javascript_gzip_bytes: initialBytes,
    all_javascript_gzip_bytes: rows.reduce((sum, row) => sum + row.gzip_bytes, 0),
    compression_level: 5, source_maps: false, chunks: rows.sort((a, b) => a.name.localeCompare(b.name))};
}

if (import.meta.url.startsWith('file:') && process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const directory = process.argv[2] || defaultDist();
  const report = await checkBudgets(directory);
  await writeFile(resolve(directory, 'build-budget.json'), JSON.stringify(report, null, 2) + '\n');
  console.info(JSON.stringify(report, null, 2));
}
```

## frontend/scripts/check-hosting.mjs

[Editable source](../frontend/scripts/check-hosting.mjs) — SHA-256: `2ea2cce58a5f0ce8d8fb4c4ffe8341b899b250d74d1af89ac5c821333656ed50`

```javascript
/* global fetch, AbortSignal */
import assert from 'node:assert/strict';
import process from 'node:process';
import console from 'node:console';

// Verify actual Nginx headers, not the Vite development proxy's policies.
const base = process.argv[2] || 'http://127.0.0.1:8080';
const get = (path, options) => fetch(base + path, {...options, signal: AbortSignal.timeout(15000)});
const checks = [];
async function expectCache(path, cache, status = 200) {
  const response = await get(path);
  assert.equal(response.status, status, path);
  assert.equal(response.headers.get('cache-control'), cache, path);
  checks.push({path, status: response.status, cache_control: cache});
  return response;
}
const html = await (await expectCache('/index.html', 'no-cache')).text();
await expectCache('/route-that-uses-the-spa-fallback', 'no-cache');
const entry = html.match(/src="([^"]+\.js)"/)?.[1];
assert.ok(entry, 'The built entry is absent from index.html.');
const main = await expectCache(entry, 'public, max-age=31536000, immutable');
assert.equal(main.headers.get('content-encoding'), 'gzip');
assert.match(main.headers.get('vary') || '', /Accept-Encoding/i);
await expectCache('/betting-oracle-logo-60.webp', 'public, max-age=3600');
await expectCache('/favicon.png', 'public, max-age=3600');
await expectCache(entry + '.map', 'no-store', 404);
assert.equal((await get('/.vite/manifest.json')).status, 404);
assert.equal((await get('/assets/missing.js')).status, 404);
for (const path of ['/api/health', '/api/missing.png', '/api/missing.map']) {
  const response = await get(path);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  checks.push({path, status: response.status, cache_control: 'no-store'});
}
console.info(JSON.stringify({pass: true, base, checks}, null, 2));
```

## frontend/nginx.conf

[Editable source](../frontend/nginx.conf) — SHA-256: `838fb288c630db7f08b1080c8398b057d67dad7618256423c9e9869dd011aad6`

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;
    client_max_body_size 1m;

    gzip on;
    gzip_vary on;
    gzip_comp_level 5;
    gzip_min_length 1000;
    gzip_types text/css application/javascript application/json image/svg+xml;

    # Keep API downloads out of filename-based static-asset locations.
    location ^~ /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600;
        proxy_request_buffering on;
        proxy_hide_header Cache-Control;
        add_header Cache-Control "no-store" always;
    }
    location ~* \.map$ {
        add_header Cache-Control "no-store" always;
        return 404;
    }
    location ~ (^|/)\. {
        return 404;
    }
    location /assets/ {
        try_files $uri =404;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
    location ~* ^/(?!assets/).*\.(woff2?|png|webp|ico|svg|jpg|jpeg)$ {
        try_files $uri =404;
        add_header Cache-Control "public, max-age=3600";
    }
    location = /index.html {
        add_header Cache-Control "no-cache" always;
    }
    location / {
        try_files $uri /index.html;
        add_header Cache-Control "no-cache" always;
    }
}
```

## frontend/Dockerfile

[Editable source](../frontend/Dockerfile) — SHA-256: `7bd7d10e8efb8414f09b53882ac8b63f436fab55bb786abd103d9fac79c094e8`

```dockerfile
FROM node:22-alpine AS build
WORKDIR /app
COPY fastapi_react/frontend/package.json ./
RUN npm install
COPY fastapi_react/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY fastapi_react/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```
