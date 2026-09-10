import { useEffect, useState } from 'react'
import { api } from "../api";
import { Card, DataTable, JsonBlock, Status, Tabs } from "../components/UI";

const advancedTabs = [
  "Performance", "Feature Importance", "Feature Selection", "Position Analysis",
  "Hyperparameters", "Historical Validation", "Debug"
];

const artifactByTab = {
  "Feature Importance": ["shap", "permutation"],
  "Feature Selection": ["monte_carlo", "monte_carlo_log", "rfe", "boruta"],
  "Position Analysis": ["position_mae"],
  "Hyperparameters": ["hyperparam_bayesian", "hyperparam_grid"],
  "Historical Validation": ["historical_validation"],
};

function Artifact({ name, data }) {
  const payload = data?.data;
  if (payload == null) return <Card title={name}><div className="empty">No precomputed artifact found.</div></Card>;
  const firstArray = Object.entries(payload).find(([, value]) => Array.isArray(value) && value.length && typeof value[0] === "object");
  return (
    <Card title={name.replaceAll("_", " ")}>
      {payload.metadata && <JsonBlock value={payload.metadata} />}
      {firstArray ? <DataTable rows={firstArray[1]} /> : <JsonBlock value={payload} />}
    </Card>
  );
}

export default function Models() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("XGBoost");
  const [tab, setTab] = useState("Performance");
  const [artifacts, setArtifacts] = useState({});
  const [health, setHealth] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [toolOutput, setToolOutput] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.get("/api/models"), api.get("/api/health")])
      .then(([m, h]) => { setModels(m.models); setHealth(h); })
      .catch(setError);
  }, []);

  useEffect(() => {
    api.get(`/api/models/manifest?model_type=${encodeURIComponent(selectedModel)}`)
      .then(r => setManifest(r.manifest)).catch(() => setManifest(null));
  }, [selectedModel]);

  useEffect(() => {
    const names = artifactByTab[tab] || [];
    Promise.all(names.map(name => api.get(`/api/models/precomputed/${name}`).then(data => [name, data])))
      .then(entries => setArtifacts(Object.fromEntries(entries))).catch(setError);
  }, [tab]);

  async function runTool(name) {
    setToolOutput({ running: true, name });
    try {
      const out = await api.post("/api/tools/run", { tool: name, args: [] });
      setToolOutput({ name, ...out });
    } catch (e) { setToolOutput({ name, error: e.message }); }
  }

  return (
    <div>
      <header className="page-header">
        <div><h1>Predictive Models & Advanced Options</h1><p>Pretrained model selection, performance diagnostics, feature selection, HPO and historical validation.</p></div>
      </header>
      <Status loading={!health && !error} error={error}>
        <Card title="Model Selection">
          <label className="field-label" htmlFor="model-type-select">Model type</label>
          {models.length === 0 ? (
            <div className="empty">No trained model for the selected type. See <a href="https://github.com/anomalyco/opencode/blob/main/scripts/precompute/README.md" target="_blank" rel="noreferrer">the precompute docs</a> for how to generate one.</div>
          ) : (
            <select id="model-type-select" value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
              {models.map(model => <option key={model}>{model}</option>)}
            </select>
          )}
          <p className="muted">
            Selected: <strong>{selectedModel}</strong>. The React migration keeps production inference artifact-first; it does not train models on page load.
          </p>
        </Card>

        <Tabs tabs={advancedTabs} active={tab} onChange={setTab} />

        {tab === "Performance" && (
          <Card title="Model Performance">
            <div className="metrics">
              <div className="metric"><span>MAE</span><strong>{manifest?.metrics?.mae?.toFixed?.(3) ?? "—"}</strong></div>
              <div className="metric"><span>MSE</span><strong>{manifest?.metrics?.mse?.toFixed?.(3) ?? "—"}</strong></div>
              <div className="metric"><span>R²</span><strong>{manifest?.metrics?.r2?.toFixed?.(3) ?? "—"}</strong></div>
              <div className="metric"><span>Features</span><strong>{manifest?.feature_names?.length ?? "—"}</strong></div>
            </div>
            <JsonBlock value={manifest || { selected_model: selectedModel, artifact_policy: "precomputed-only" }} />
          </Card>
        )}

        {tab === "Feature Importance" && manifest?.feature_names?.length > 0 && (
          <Card title="Model Feature Contract">
            <p className="muted">Ordered feature contract recorded in the model manifest.</p>
            <DataTable rows={manifest.feature_names.map((feature, i) => ({ rank: i + 1, feature }))} />
          </Card>
        )}

        {tab === "Debug" && (
          <>
            <Card title="Runtime">
              <JsonBlock value={health} />
            </Card>
            <Card title="Manual / Expensive Analysis Tools">
              <p className="muted">These controls mirror the Streamlit research tools but are disabled by default. Set <code>ENABLE_EXPENSIVE_TOOLS=1</code> only on a test host.</p>
              <div className="button-row wrap">
                {["monte_carlo", "rfe", "boruta", "shap", "permutation"].map(name => (
                  <button key={name} disabled={!health?.expensive_tools_enabled} onClick={() => runTool(name)}>
                    Run {name.replaceAll("_", " ")}
                  </button>
                ))}
              </div>
              {toolOutput && <JsonBlock value={toolOutput} />}
            </Card>
          </>
        )}

        {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
      </Status>
    </div>
  );
}
