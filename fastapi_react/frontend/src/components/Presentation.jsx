import { useEffect, useId, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import {ViewTable} from './ViewTable';
import {TabScroll} from './TabScroll';
import {useTheme} from './useTheme';

const numberFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 });


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
  return <div className="chart-shell" ref={outer} role="group" aria-label={node.label || 'Interactive analysis chart'}><div className="table-toolbar"><button aria-label={showData?'Show chart':'Show data'} title={showData?'Show chart':'Show data'} onClick={()=>setShowData(s=>!s)}>▥</button><button aria-label="Download chart as PNG" title="Download as PNG" onClick={download}>⇩</button><button aria-label="Copy Vega-Lite spec" title="Copy Vega-Lite spec" onClick={()=>navigator.clipboard?.writeText(JSON.stringify(node.spec,null,2)).catch(()=>{})}>⧉</button><button aria-label="Fullscreen chart" title="Fullscreen" onClick={()=>document.fullscreenElement?document.exitFullscreen():outer.current?.requestFullscreen?.()}>⛶</button></div><div className="view-chart" ref={ref} style={{display:showData?'none':undefined}}>{error && <div role="alert">{error}</div>}</div>{showData && <ViewTable node={table}/>}</div>;
}

function PlotlyChart({ node }) {
  const ref = useRef(null);
  useEffect(() => {
    let chart;
    const el = ref.current;
    import('plotly.js-dist-min').then(({default: plotly}) => {chart = plotly; chart.newPlot(el, node.spec.data, {...node.spec.layout, autosize: true}, {responsive: true});});
    return () => {if (chart && el) chart.purge(el);};
  }, [node.spec]);
  return <div className="view-chart" ref={ref} role="img" aria-label="Interactive analysis chart" />;
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
      case 'notice': return <div key={key} className={`view-notice ${node.severity}`} role={node.severity === 'error' ? 'alert' : 'status'}>{node.icon && <span>{node.icon}</span>}<Markdown>{node.text}</Markdown></div>;
      case 'metric': return <div key={key} className="view-metric"><span>{node.label}</span><strong>{node.value}</strong>{node.delta != null && <small>{node.delta}</small>}</div>;
      case 'divider': return <hr key={key} className="view-divider" />;
      case 'image': return <img key={key} alt={node.alt || 'Analysis visualization'} src={node.src} className="view-image" style={{width: node.width === 'stretch' ? '100%' : node.width, maxWidth: '100%'}} />;
      case 'table': return <ViewTable key={key} node={node} />;
      case 'vega': return <VegaChart key={key} node={node} />;
      case 'plotly': return <PlotlyChart key={key} node={node} />;
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
