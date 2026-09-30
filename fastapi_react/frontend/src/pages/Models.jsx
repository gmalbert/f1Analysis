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
  "📊 Model Performance": ["position_mae", "historical_validation"],
  "🔍 Feature Analysis": ["shap", "permutation"],
  "🎯 Feature Selection": ["monte_carlo", "monte_carlo_log", "rfe", "boruta"],
  "🏎️ Position-Specific Analysis": ["position_mae", "historical_validation"],
  "⚙️ Hyperparameters": ["hyperparam_bayesian", "hyperparam_grid"],
  "📈 Historical Validation": ["historical_validation"],
};

function firstObjectArray(payload) {
  if (!payload || typeof payload !== "object") return null;
  for (const value of Object.values(payload)) {
    if (Array.isArray(value) && value.length && typeof value[0] === "object") return value;
  }
  return null;
}

function objectRows(value, keyLabel = "Name") {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value).map(([name, details]) => ({
    [keyLabel]: name,
    ...(details && typeof details === "object" && !Array.isArray(details) ? details : { value: details }),
  }));
}

function positionRowsFromHistorical(payload) {
  const rows = payload?.holdout?.rows || [];
  const buckets = new Map();
  for (const row of rows) {
    const actual = Number(row.ActualFinalPosition);
    const predicted = Number(row.PredictedFinalPosition);
    if (!Number.isFinite(actual) || !Number.isFinite(predicted)) continue;
    const key = String(actual);
    const bucket = buckets.get(key) || [];
    bucket.push(Math.abs(actual - predicted));
    buckets.set(key, bucket);
  }
  return [...buckets.entries()]
    .map(([Position, errors]) => ({
      Position: Number(Position),
      MAE: errors.reduce((sum, value) => sum + value, 0) / errors.length,
      Count: errors.length,
    }))
    .sort((a, b) => a.Position - b.Position);
}

function groupRowsFromHistorical(payload) {
  const rows = payload?.holdout?.rows || [];
  const groups = [
    ["Winners", value => value === 1],
    ["Podium", value => value >= 1 && value <= 3],
    ["Top 5", value => value >= 1 && value <= 5],
    ["Points", value => value >= 1 && value <= 10],
    ["Midfield", value => value >= 11 && value <= 15],
    ["Backmarkers", value => value >= 16],
  ];
  return groups.flatMap(([name, predicate]) => {
    const subset = rows.filter(row => predicate(Number(row.ActualFinalPosition)));
    if (!subset.length) return [];
    const errors = subset.map(row => Math.abs(Number(row.ActualFinalPosition) - Number(row.PredictedFinalPosition)));
    return [{
      "Position Group": name,
      MAE: errors.reduce((sum, value) => sum + value, 0) / errors.length,
      Count: subset.length,
    }];
  });
}

function Artifact({ name, data }) {
  const payload = data?.data;
  if (payload == null) return <div className="empty">No precomputed artifact found for {name.replaceAll("_", " ")}.</div>;
  const table = firstObjectArray(payload);
  return (
    <Card title={name.replaceAll("_", " ")}>
      {payload.metadata && <p className="caption">Precomputed artifact metadata is shown below.</p>}
      {table ? <DataTable rows={table} maxHeight={650} /> : <JsonBlock value={payload} />}
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

            {tab === "📊 Model Performance" && (() => {
              const position = artifacts.position_mae?.data || {};
              const historical = artifacts.historical_validation?.data || {};
              const driverRows = objectRows(position.by_driver, "Driver").sort((a, b) => Number(b.mae || 0) - Number(a.mae || 0));
              const positionRows = positionRowsFromHistorical(historical);
              const groupRows = objectRows(position.position_groups, "Position Group");
              const validationRows = historical?.holdout?.rows || [];
              return <>
                <h2>Predictive Data Model Metrics</h2>
                <div className="metrics">
                  <Metric label="Mean Squared Error" value={metrics.mse != null ? Number(metrics.mse).toFixed(3) : "—"} />
                  <Metric label="R² Score" value={metrics.r2 != null ? Number(metrics.r2).toFixed(3) : "—"} />
                  <Metric label="Mean Absolute Error" value={metrics.mae != null ? Number(metrics.mae).toFixed(2) : "—"} />
                  <Metric label="Mean Error" value={metrics.mean_error != null ? Number(metrics.mean_error).toFixed(2) : "—"} />
                </div>
                {manifest.best_iteration != null && <p>Boosting rounds used: {Number(manifest.best_iteration) + 1}</p>}

                <h2>Mean Error (ME) and Mean Absolute Error (MAE) per Driver</h2>
                <p>Total number of drivers: {driverRows.length}</p>
                <p>Total number of results: {validationRows.length}</p>
                <DataTable rows={driverRows} maxHeight={600} />

                <h2>Error Metrics per Driver</h2>
                <DataTable rows={driverRows} maxHeight={600} />

                <h2>Predictive Results with Features</h2>
                <DataTable rows={validationRows.slice(0, 100)} maxHeight={600} />

                <h2>Feature Importances</h2>
                {featureNames.length > 0 ? <DataTable rows={featureNames.map((feature, index) => ({ Feature: feature, Rank: index + 1 })).slice(0, 100)} maxHeight={600} /> : <div className="empty">Feature importances are not present in this model manifest.</div>}

                <h2>MAE by Position Groups</h2>
                <DataTable rows={groupRows.length ? groupRows : groupRowsFromHistorical(historical)} />

                <h2>MAE by Individual Positions</h2>
                <DataTable rows={positionRows} maxHeight={750} />

                <h2>Position Group Summary</h2>
                <DataTable rows={groupRows.length ? groupRows : groupRowsFromHistorical(historical)} />

                <h2>Prediction Error Distribution by Position Groups</h2>
                <DataTable rows={groupRows.length ? groupRows : groupRowsFromHistorical(historical)} />
              </>;
            })()}

            {tab === "🔍 Feature Analysis" && (() => {
              const permutation = artifacts.permutation?.data || {};
              const shap = artifacts.shap?.data || {};
              const permutationRows = permutation.feature_importance || [];
              const shapRows = shap.feature_importance || [];
              return <>
                <h2>Feature Analysis</h2>
                <h3>Permutation Importance (Feature Impact on Model Error)</h3>
                <p className="caption">Precomputed by GitHub Actions; higher values indicate a larger increase in model error when the feature is permuted.</p>
                <DataTable rows={permutationRows.slice(0, 100)} maxHeight={600} />
                {permutationRows.length > 0 && <>
                  <p>Features with lowest permutation importance (least helpful):</p>
                  <DataTable rows={[...permutationRows].sort((a, b) => Number(a.importance) - Number(b.importance)).slice(0, 10)} />
                  <p>Features with highest permutation importance (most helpful):</p>
                  <DataTable rows={[...permutationRows].sort((a, b) => Number(b.importance) - Number(a.importance)).slice(0, 10)} />
                </>}

                <h3>High-Cardinality Features (Potential Overfitting Risk)</h3>
                <p>Features with high cardinality (many unique values) are more likely to cause overfitting, especially if they are IDs or post-event info.</p>

                <h3>Safety Car Feature Importance</h3>
                <DataTable rows={shapRows.slice(0, 30)} />

                <h3>Correlation Matrix</h3>
                <p className="caption">Correlation diagnostics remain artifact/data-backed; no model is trained during this request.</p>

                <h2>Feature Importances</h2>
                {featureNames.length > 0 && <DataTable rows={featureNames.map((feature, index) => ({ Feature: feature, Rank: index + 1 })).slice(0, 100)} maxHeight={650} />}
              </>;
            })()}

            {tab === "🎯 Feature Selection" && (() => {
              const monte = artifacts.monte_carlo?.data || {};
              const rfe = artifacts.rfe?.data || {};
              const boruta = artifacts.boruta?.data || {};
              const shap = artifacts.shap?.data || {};
              const top20 = monte.top_20_results || [];
              const freq = objectRows(monte.feature_frequency_top_20, "Feature").map(row => ({ Feature: row.Feature, Count: row.value }));
              return <>
                <h2>Feature Selection Tools</h2>
                <div className="status">📦 Precomputed feature selection results available from GitHub Actions!</div>
                <details open>
                  <summary>📊 View Precomputed Results</summary>
                  <h3>Monte Carlo Results (Precomputed)</h3>
                  {monte.metadata && <JsonBlock value={monte.metadata} />}
                  {monte.best_result && <JsonBlock value={monte.best_result} />}
                  <h3>SHAP Analysis (Precomputed)</h3>
                  <DataTable rows={(shap.top_20 || shap.feature_importance || []).slice(0, 20)} />
                  <h3>RFE Results (Precomputed)</h3>
                  <DataTable rows={(rfe.feature_ranking || []).slice(0, 100)} maxHeight={600} />
                  <h3>Boruta Results (Precomputed)</h3>
                  <DataTable rows={(boruta.feature_ranking || []).slice(0, 100)} maxHeight={600} />
                  <h3>Permutation Importance (Precomputed)</h3>
                  <Artifact name="permutation" data={artifacts.permutation} />
                </details>

                <h3>Monte Carlo Feature Subset Search</h3>
                <h2>Top 20 Feature Subsets</h2>
                <DataTable rows={top20} maxHeight={650} />
                <h2>Feature Appearance in Top 20 Subsets</h2>
                <DataTable rows={freq} />

                <h3>Recursive Feature Elimination (RFE)</h3>
                <h3>Boruta Feature Selection</h3>
                <h3>RFE to Minimize MAE</h3>
                <h3>External Feature Selection Script</h3>

                <h2>Feature selection summary</h2>
                <p>{rfe.selected_features?.length || 0} RFE features and {boruta.selected_features?.length || 0} Boruta features are present in the latest committed artifacts.</p>

                <h2>Boruta Selected Features</h2>
                <DataTable rows={(boruta.selected_features || []).map(feature => ({ Feature: feature }))} />

                <h2>SHAP Ranking (top 20)</h2>
                <DataTable rows={(shap.top_20 || []).slice(0, 20)} />

                <h2>Highly Correlated Pairs (&gt;0.95)</h2>
                <p className="caption">Correlation-pair exports are surfaced when present under Data &amp; Debug; this tab does not recompute them on page load.</p>

                <h3>Exported Summaries</h3>
                <p className="caption">Feature-selection exports remain downloadable from the committed data files.</p>
              </>;
            })()}

            {tab === "🏎️ Position-Specific Analysis" && (() => {
              const position = artifacts.position_mae?.data || {};
              const historical = artifacts.historical_validation?.data || {};
              const groupRows = objectRows(position.position_groups, "Position Group");
              const individualRows = positionRowsFromHistorical(historical);
              return <>
                <h2>Position Group Analysis</h2>
                <p>Based on current UI test set (same as Model Performance tab)</p>
                <h2>📊 Position Group MAE Summary</h2>
                <Metric label="Overall Model MAE" value={position.metadata?.model_mae != null ? Number(position.metadata.model_mae).toFixed(3) : metrics.mae ?? "—"} />
                <DataTable rows={groupRows} />
                <p className="caption">Lower MAE indicates better prediction accuracy for that position group. These values match the Model Performance tab.</p>

                <details>
                  <summary>🔍 Example Predictions for Winners (P1)</summary>
                  <DataTable rows={(historical?.holdout?.rows || []).filter(row => Number(row.ActualFinalPosition) === 1).slice(0, 10)} />
                </details>

                <h2>MAE by Season</h2>
                <DataTable rows={(position.metadata?.seasons || []).map(season => ({ Season: season, MAE: position.metadata?.model_mae }))} />

                <h2>MAE by Individual Positions</h2>
                <DataTable rows={individualRows} maxHeight={750} />
              </>;
            })()}

            {tab === "⚙️ Hyperparameters" && <>
              <h2>Hyperparameter Tuning</h2>
              {manifest.hyperparameters && <JsonBlock value={manifest.hyperparameters} />}
              {(artifactByTab[tab] || []).map(name => <Artifact key={name} name={name} data={artifacts[name]} />)}
            </>}

            {tab === "📈 Historical Validation" && (() => {
              const historical = artifacts.historical_validation?.data || {};
              const holdout = historical.holdout || {};
              return <>
                <h2>Historical Validation</h2>
                {historical.metadata && <p className="caption">Precomputed by GitHub Actions: {historical.metadata.generated_at || "Unknown"} · validation model: {historical.metadata.model_type || "XGBoost"}</p>}
                <h3>Model Evaluation Metrics (Cross-Validation)</h3>
                <JsonBlock value={{ position_cv: historical.position_cv, dnf_validation: historical.dnf_validation, safety_car_validation: historical.safety_car_validation }} />
                <h3>Model Accuracy Across All Races</h3>
                <JsonBlock value={holdout.metrics || {}} />
                <DataTable rows={holdout.rows || []} maxHeight={650} />
                <h2>Actual vs Predicted Final Position (All Races)</h2>
                <DataTable rows={(holdout.rows || []).slice(0, 100)} maxHeight={650} />
              </>;
            })()}

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
