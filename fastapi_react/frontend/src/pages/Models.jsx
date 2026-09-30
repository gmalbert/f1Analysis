import { useEffect, useState } from "react";
import { api } from "../api";
import { Card, DataTable, JsonBlock, Metric, Status, Tabs } from "../components/UI";

const advancedTabs = [
  "📊 Model Performance",
  "🔍 Feature Analysis",
  "🎯 Feature Selection",
  "🏎️ Position-Specific Analysis",
  "⚙️ Hyperparameters",
  "📈 Historical Validation",
  "🛠️ Debug & Experiments",
];

const artifactByTab = {
  "🔍 Feature Analysis": ["shap", "permutation"],
  "🎯 Feature Selection": ["monte_carlo", "monte_carlo_log", "rfe", "boruta"],
  "🏎️ Position-Specific Analysis": ["position_mae"],
  "⚙️ Hyperparameters": ["hyperparam_bayesian", "hyperparam_grid"],
  "📈 Historical Validation": ["historical_validation"],
};

function firstObjectArray(payload) {
  if (!payload || typeof payload !== "object") return null;
  for (const [name, value] of Object.entries(payload)) {
    if (Array.isArray(value) && value.length && typeof value[0] === "object") return [name, value];
  }
  return null;
}

function Artifact({ name, data }) {
  const payload = data?.data;
  if (payload == null) return <div className="empty">No precomputed artifact found for {name.replaceAll("_", " ")}.</div>;
  const table = firstObjectArray(payload);
  return (
    <Card title={name.replaceAll("_", " ")}>
      {payload.metadata && <p className="caption">Precomputed artifact metadata is shown below.</p>}
      {table ? <DataTable rows={table[1]} maxHeight={650} /> : <JsonBlock value={payload} />}
    </Card>
  );
}

export default function Models() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("XGBoost");
  const [tab, setTab] = useState(advancedTabs[0]);
  const [artifacts, setArtifacts] = useState({});
  const [health, setHealth] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [toolOutput, setToolOutput] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.get("/api/models"), api.get("/api/health")])
      .then(([modelResponse, healthResponse]) => {
        setModels(modelResponse.models || []);
        setHealth(healthResponse);
      })
      .catch(setError);
  }, []);

  useEffect(() => {
    api.get(`/api/models/manifest?model_type=${encodeURIComponent(selectedModel)}`)
      .then(response => setManifest(response.manifest))
      .catch(() => setManifest(null));
  }, [selectedModel]);

  useEffect(() => {
    const names = artifactByTab[tab] || [];
    Promise.all(names.map(name => api.get(`/api/models/precomputed/${name}`).then(data => [name, data])))
      .then(entries => setArtifacts(Object.fromEntries(entries)))
      .catch(setError);
  }, [tab]);

  async function runTool(name) {
    setToolOutput({ running: true, name });
    try {
      setToolOutput({ name, ...(await api.post("/api/tools/run", { tool: name, args: [] })) });
    } catch (err) {
      setToolOutput({ name, error: err.message });
    }
  }

  const metrics = manifest?.metrics || {};
  const featureNames = manifest?.feature_names || [];

  return (
    <div>
      <header className="page-header">
        <h1>Predictive Models & Advanced Options</h1>
        <p>Advanced machine learning models, hyperparameter tuning, and feature selection tools.</p>
      </header>

      <label className="field-label" htmlFor="model-type-select">Select Model Type</label>
      <select id="model-type-select" value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
        {models.map(model => <option key={model}>{model}</option>)}
      </select>

      <details>
        <summary>ℹ️ Model Information & Recommendations</summary>
        <h3>Model Descriptions & Use Cases</h3>
        <p><strong>🏆 XGBoost (Recommended Default)</strong></p>
        <ul><li>Excellent performance, handles missing data, and provides built-in feature importance.</li><li>Best for general-purpose predictions and reliable interpretability.</li><li>Training speed: Fast · Memory usage: Moderate.</li></ul>
        <p><strong>🚀 LightGBM</strong></p>
        <ul><li>Very fast training and efficient memory use.</li><li>Best when training speed is critical or datasets are large.</li></ul>
        <p><strong>🐱 CatBoost</strong></p>
        <ul><li>Strong categorical-data handling and robustness to overfitting.</li><li>Training speed: Moderate · Memory usage: Moderate.</li></ul>
        <p><strong>🎯 Ensemble (XGBoost + LightGBM + CatBoost)</strong></p>
        <ul><li>Combines all three base models for maximum prediction accuracy.</li><li>Training speed: Slowest · Memory usage: High.</li></ul>
        <p><strong>🏎️ Position Group</strong></p>
        <ul><li>Separate sub-models for podium, points, and outside-points segments.</li></ul>
        <p><strong>🗺️ Track-Weighted Ensemble</strong></p>
        <ul><li>Circuit-type-specific blend weights across XGBoost, LightGBM, and CatBoost.</li></ul>
      </details>

      <p className="caption">Models are pre-trained by GitHub Actions; training controls are kept out of the live app to protect responsiveness.</p>
      {!health?.expensive_tools_enabled && (
        <div className="warning">Research controls are disabled in hosted mode. Enable F1_RESEARCH_MODE=1 only for a trusted local/admin session; precomputed analyses remain available below.</div>
      )}

      <Status loading={!health && !error} error={error}>
        {!manifest ? <div className="empty">No trained model artifact is available for {selectedModel}.</div> : (
          <details open>
            <summary>🔧 Advanced Options</summary>
            <Tabs tabs={advancedTabs} active={tab} onChange={setTab} />

            {tab === "📊 Model Performance" && <>
              <h2>Predictive Data Model Metrics</h2>
              <div className="metrics">
                <Metric label="Mean Squared Error" value={metrics.mse != null ? Number(metrics.mse).toFixed(3) : "—"} />
                <Metric label="R² Score" value={metrics.r2 != null ? Number(metrics.r2).toFixed(3) : "—"} />
                <Metric label="Mean Absolute Error" value={metrics.mae != null ? Number(metrics.mae).toFixed(2) : "—"} />
                <Metric label="Mean Error" value={metrics.mean_error != null ? Number(metrics.mean_error).toFixed(2) : "—"} />
              </div>
              {manifest.best_iteration != null && <p>Boosting rounds used: {Number(manifest.best_iteration) + 1}</p>}
              {manifest.position_mae && <DataTable rows={Object.entries(manifest.position_mae).map(([group, mae]) => ({ "Position Group": group, MAE: mae }))} />}
              <h2>Predictive Results with Features</h2>
              <p className="caption">The deployed React site consumes the same workflow-generated model contract and artifacts as Streamlit.</p>
            </>}

            {tab === "🔍 Feature Analysis" && <>
              <h2>Feature Importances</h2>
              {featureNames.length > 0 && <DataTable rows={featureNames.map((feature, index) => ({ Feature: feature, Rank: index + 1 })).slice(0, 100)} maxHeight={650} />}
              {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
            </>}

            {tab === "🎯 Feature Selection" && <>
              <h2>Feature Selection Tools</h2>
              {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
            </>}

            {tab === "🏎️ Position-Specific Analysis" && <>
              <h2>Position Group Analysis</h2>
              {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
            </>}

            {tab === "⚙️ Hyperparameters" && <>
              <h2>Hyperparameter Tuning</h2>
              {manifest.hyperparameters && <JsonBlock value={manifest.hyperparameters} />}
              {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
            </>}

            {tab === "📈 Historical Validation" && <>
              <h2>Historical Validation</h2>
              {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
            </>}

            {tab === "🛠️ Debug & Experiments" && <>
              <h2>Debug & Experiments</h2>
              <h3>Compare Different Bin Counts (q)</h3>
              <p className="caption">Manual experiments are disabled by default on the hosted site, matching the Streamlit research-mode gate.</p>
              <div className="button-row wrap">
                {["monte_carlo", "rfe", "boruta", "shap", "permutation"].map(name => (
                  <button key={name} disabled={!health?.expensive_tools_enabled} onClick={() => runTool(name)}>Run {name.replaceAll("_", " ")}</button>
                ))}
              </div>
              {toolOutput && <JsonBlock value={toolOutput} />}
            </>}
          </details>
        )}
      </Status>
    </div>
  );
}
