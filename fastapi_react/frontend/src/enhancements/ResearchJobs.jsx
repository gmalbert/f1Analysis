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
