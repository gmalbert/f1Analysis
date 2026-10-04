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
