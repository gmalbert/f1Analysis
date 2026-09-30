import { useEffect, useState } from "react";
import { api } from "../api";
import { DataTable, JsonBlock, Status, Tabs } from "../components/UI";

const RAW_TABS = ["Raw Data", "Temporal Leakage Audit", "Hyperparameter Tuning"];

export default function RawData() {
  const [tab, setTab] = useState("Raw Data");
  const [health, setHealth] = useState(null);
  const [showDataset, setShowDataset] = useState(false);
  const [dataset, setDataset] = useState(null);
  const [datasetPage, setDatasetPage] = useState(0);
  const [toolResult, setToolResult] = useState(null);
  const [toolBusy, setToolBusy] = useState(false);
  const [error, setError] = useState(null);
  const pageSize = 1000;

  useEffect(() => { api.get("/api/health").then(setHealth).catch(() => {}); }, []);

  useEffect(() => {
    if (!showDataset) return;
    setError(null);
    api.post("/api/data-explorer/query", {
      filters: [], offset: datasetPage * pageSize, limit: pageSize,
    }).then(setDataset).catch(setError);
  }, [showDataset, datasetPage]);

  async function runTool(tool) {
    setToolBusy(true);
    setToolResult(null);
    setError(null);
    try { setToolResult(await api.post("/api/tools/run", { tool, args: [] })); }
    catch (err) { setError(err); }
    finally { setToolBusy(false); }
  }

  const enabled = Boolean(health?.expensive_tools_enabled);

  return (
    <div>
      <header className="page-header">
        <h1>Data & Debug Tools</h1>
      </header>
      <Tabs tabs={RAW_TABS} active={tab} onChange={setTab} />

      {tab === "Raw Data" && <>
        <p>View the complete unfiltered dataset.</p>
        <label className="filter-results-toggle">
          <input type="checkbox" checked={showDataset} onChange={e => { setShowDataset(e.target.checked); setDatasetPage(0); }} />
          Show Raw Data
        </label>
        {showDataset && <Status loading={!dataset && !error} error={error}>
          {dataset && <>
            <p>Total number of results: {dataset.total.toLocaleString()}</p>
            <DataTable rows={dataset.rows} columns={dataset.columns} maxHeight={600} />
            {dataset.total > pageSize && <div className="button-row">
              <button disabled={datasetPage === 0} onClick={() => setDatasetPage(page => page - 1)}>Previous</button>
              <button disabled={(datasetPage + 1) * pageSize >= dataset.total} onClick={() => setDatasetPage(page => page + 1)}>Next</button>
            </div>}
          </>}
        </Status>}
      </>}

      {tab === "Temporal Leakage Audit" && <>
        <p>Run heuristics-based checks for features that may leak future information into models.</p>
        <details>
          <summary>🔍 Run Temporal Leakage Audit (Admin)</summary>
          <p>This audit scans the analysis dataset for features that may leak future or post-event information into training.</p>
          <details>
            <summary>About this Leakage Audit</summary>
            <p>It applies name-pattern checks, very high target-correlation checks, per-driver lagged-correlation checks, and safety-car candidate checks.</p>
            <p>Recommendation: review flagged features and remove or re-engineer any that use post-race or future information before training models.</p>
          </details>
          {!enabled && <div className="warning">Research controls are disabled in hosted mode.</div>}
          <button disabled={!enabled || toolBusy} onClick={() => runTool("temporal_leakage")}>{toolBusy ? "Running leakage audit…" : "Run Leakage Audit"}</button>
          {toolResult && <JsonBlock value={toolResult} />}
        </details>
        {error && <Status error={error} />}
      </>}

      {tab === "Hyperparameter Tuning" && <>
        <p>Run basic hyperparameter tuning (GridSearch) on the full dataset.</p>
        {!enabled && <div className="warning">Research controls are disabled in hosted mode.</div>}
        <button disabled={!enabled || toolBusy} onClick={() => runTool("hyperparameter_grid")}>{toolBusy ? "Running…" : "Run Hyperparameter Tuning (subtab)"}</button>
        {toolResult && <JsonBlock value={toolResult} />}
        {error && <Status error={error} />}
      </>}
    </div>
  );
}
