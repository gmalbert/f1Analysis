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
