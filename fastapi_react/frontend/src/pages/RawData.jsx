import { useEffect, useMemo, useState } from 'react'
import { api, downloadUrl } from "../api";
import { Card, DataTable, JsonBlock, Status, Tabs } from "../components/UI";

const RAW_TABS = ["Raw Data", "Temporal Leakage Audit", "Hyperparameter Tuning", "File Browser"];

export default function RawData() {
  const [tab, setTab] = useState(RAW_TABS[0]);
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [toolResult, setToolResult] = useState(null);
  const [toolBusy, setToolBusy] = useState(false);
  const [showDataset, setShowDataset] = useState(false);
  const [dataset, setDataset] = useState(null);
  const [displaySchema, setDisplaySchema] = useState(null);
  const [displaySchemaLoading, setDisplaySchemaLoading] = useState(true);
  const [datasetPage, setDatasetPage] = useState(0);
  const datasetPageSize = 50;

  useEffect(() => {
    api.get("/api/raw/files").then(r => { setFiles(r.files); setLoading(false); }).catch(e => { setError(e); setLoading(false); });
    api.get("/api/health").then(setHealth).catch(() => {});
    api.get("/api/data-explorer/display-schema")
      .then(setDisplaySchema).catch(setError).finally(() => setDisplaySchemaLoading(false));
  }, []);

  useEffect(() => {
    if (!showDataset) return;
    let cancelled = false;
    setLoading(true);
    api.post("/api/raw/analysis-data", {
      limit: datasetPageSize,
      offset: datasetPage * datasetPageSize,
    }).then(result => {
      if (!cancelled) setDataset(result);
    }).catch(setError).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [showDataset, datasetPage]);

  const shown = useMemo(
    () => files.filter(f => f.path.toLowerCase().includes(query.toLowerCase())).slice(0, 500),
    [files, query]
  );

  async function open(path) {
    setSelected(path); setPreview(null); setError(null);
    try { setPreview(await api.get(`/api/raw/preview?path=${encodeURIComponent(path)}`)); }
    catch (e) { setError(e); }
  }

  async function runTool(tool, args = []) {
    setToolBusy(true); setToolResult(null); setError(null);
    try { setToolResult(await api.post("/api/tools/run", { tool, args })); }
    catch (e) { setError(e); }
    finally { setToolBusy(false); }
  }

  const enabled = !!health?.expensive_tools_enabled;
  const sourceColumns = displaySchema?.columns?.filter(column => dataset?.columns.includes(column)) ?? [];
  const displayColumns = sourceColumns.map(column => displaySchema?.labels?.[column] ?? column);
  const displayRows = dataset?.rows.map(row => Object.fromEntries(
    sourceColumns.map((column, index) => [displayColumns[index], row[column]])
  )) ?? [];

  return (
    <div>
      <p>Tab 6 START</p>
      <header className="page-header"><div><h1>Data & Debug Tools</h1></div></header>
      <Tabs tabs={RAW_TABS} active={tab} onChange={setTab} />

      {tab === "Raw Data" && <section>
        <p>View the complete unfiltered dataset.</p>
        <label className="dataset-toggle">
          <input type="checkbox" aria-label="Show Raw Data" checked={showDataset} onChange={event => { setShowDataset(event.target.checked); setDatasetPage(0); setError(null); }} />
          Show Raw Data
        </label>
        {showDataset && <Status loading={loading || displaySchemaLoading} error={error}>
          {dataset && displaySchema && <>
            <p>Total number of results: {dataset.total.toLocaleString()}</p>
            <DataTable rows={displayRows} columns={displayColumns} maxHeight={600} />
            <div className="button-row">
              <button disabled={datasetPage === 0 || loading} onClick={() => setDatasetPage(page => page - 1)}>Previous</button>
              <button disabled={(datasetPage + 1) * datasetPageSize >= dataset.total || loading} onClick={() => setDatasetPage(page => page + 1)}>Next</button>
            </div>
          </>}
        </Status>}
      </section>}

      {tab === "File Browser" && <div className="raw-grid">
        <Card title={`Files (${files.length})`}>
          <input className="search" aria-label="Filter filenames" placeholder="Filter filenames…" value={query} onChange={e => setQuery(e.target.value)} />
          <div className="file-list">
            {shown.length === 0 ? (
              <div className="empty">No files in data_files/</div>
            ) : shown.map(f => (
              <button key={f.path} className={selected === f.path ? "file active" : "file"} onClick={() => open(f.path)}>
                <span>{f.path}</span><small>{(f.size / 1024).toFixed(1)} KB</small>
              </button>
            ))}
          </div>
        </Card>
        <Card title={selected || "Preview"}>
          <Status loading={loading && !showDataset} error={!showDataset ? error : null}>
            {!selected && <div className="empty">Choose a file to preview.</div>}
            {preview?.kind === "table" && <DataTable rows={preview.rows} columns={preview.columns} />}
            {preview?.kind === "json" && <JsonBlock value={preview.data} />}
            {preview?.kind === "text" && <pre className="json">{preview.data}</pre>}
            {preview?.kind === "binary" && <div className="empty">Binary preview unavailable.</div>}
            {selected && <p><a className="button-link" href={downloadUrl(selected)}>Download original</a></p>}
          </Status>
        </Card>
      </div>}

      {tab === "Temporal Leakage Audit" && <Card title="Temporal Leakage Audit">
        <p className="muted">Runs the repository's heuristics-based temporal leakage audit. This is intentionally disabled by default on a public host because it can read the full analysis dataset.</p>
        {!enabled && <div className="warning">Manual analysis tools are disabled. Set <code>ENABLE_EXPENSIVE_TOOLS=1</code> only on a test deployment.</div>}
        <button className="primary" disabled={!enabled || toolBusy} onClick={() => runTool("temporal_leakage")}>{toolBusy ? "Running…" : "Run Leakage Audit"}</button>
        {error && <div className="status error">{String(error.message || error)}</div>}
        {toolResult && <JsonBlock value={toolResult} />}
      </Card>}

      {tab === "Hyperparameter Tuning" && <Card title="Hyperparameter Tuning">
        <p className="muted">The live site should normally consume precomputed HPO artifacts. These controls provide test-only parity with the manual tuning area without enabling them on production by default.</p>
        {!enabled && <div className="warning">Manual tuning is disabled. Set <code>ENABLE_EXPENSIVE_TOOLS=1</code> on a test host to enable it.</div>}
        <div className="button-row wrap">
          <button className="primary" disabled={!enabled || toolBusy} onClick={() => runTool("hyperparameter_grid")}>Run Grid Search</button>
          <button disabled={!enabled || toolBusy} onClick={() => runTool("hyperparameter_bayesian")}>Run Bayesian Optimization</button>
        </div>
        {error && <div className="status error">{String(error.message || error)}</div>}
        {toolResult && <JsonBlock value={toolResult} />}
      </Card>}
    </div>
  );
}
