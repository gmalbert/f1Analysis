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
