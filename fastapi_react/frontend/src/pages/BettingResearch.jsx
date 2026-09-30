import { useEffect, useState } from "react";
import Papa from "papaparse";
import { api } from "../api";
import { DataTable, JsonBlock, Metric, Tabs } from "../components/UI";
import { LinePanel } from "../components/Charts";

const tabs = ["Value & stake", "Field simulation", "Paper replay", "Calibration"];
const defaultEntries = [
  { driver_id: "driver-a", constructor_id: "team-1", pace_score: 1.0, dnf_probability: 0.05, uncertainty: 0.8, race_sensitivity: 0.8 },
  { driver_id: "driver-b", constructor_id: "team-1", pace_score: 1.4, dnf_probability: 0.06, uncertainty: 0.9, race_sensitivity: 1.0 },
  { driver_id: "driver-c", constructor_id: "team-2", pace_score: 2.2, dnf_probability: 0.08, uncertainty: 1.0, race_sensitivity: 1.2 },
];

function csvDataUrl(rows, columns) {
  if (!rows?.length) return "#";
  const cols = columns?.length ? columns : Object.keys(rows[0]);
  const csv = Papa.unparse({ fields: cols, data: rows.map(row => cols.map(column => row[column] ?? "")) });
  return `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`;
}

function CsvInput({ onRows, label }) {
  return <label className="upload-label">{label}<input aria-label={label} type="file" accept=".csv,text/csv" onChange={event => {
    const file = event.target.files?.[0];
    if (!file) return;
    Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: result => onRows(result.data, result.errors) });
  }} /></label>;
}

export default function BettingResearch() {
  const [tab, setTab] = useState(tabs[0]);
  const [calc, setCalc] = useState({
    model_probability: .25, decimal_odds: 2.10, opposing_odds: 1.80,
    uncertainty: .02, devig_method: "multiplicative", bankroll: 10000,
  });
  const [calcOut, setCalcOut] = useState(null);
  const [simEntries, setSimEntries] = useState(defaultEntries);
  const [simulations, setSimulations] = useState(10000);
  const [simOut, setSimOut] = useState(null);
  const [replayRows, setReplayRows] = useState([]);
  const [replayOut, setReplayOut] = useState(null);
  const [calRows, setCalRows] = useState([]);
  const [calOut, setCalOut] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState("");

  useEffect(() => {
    let cancelled = false;
    setBusy("value");
    api.post("/api/betting/value", calc)
      .then(result => { if (!cancelled) setCalcOut(result); })
      .catch(err => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setBusy(""); });
    return () => { cancelled = true; };
  }, [calc]);

  useEffect(() => {
    if (!calRows.length) { setCalOut(null); return; }
    let cancelled = false;
    setBusy("calibration");
    api.post("/api/betting/calibration", { rows: calRows })
      .then(result => { if (!cancelled) setCalOut(result); })
      .catch(err => { if (!cancelled) setError(`Calibration input is invalid: ${err.message}`); })
      .finally(() => { if (!cancelled) setBusy(""); });
    return () => { cancelled = true; };
  }, [calRows]);

  async function runSimulation() {
    setBusy("simulation");
    setError(null);
    try { setSimOut(await api.post("/api/betting/simulate", { entries: simEntries, simulations, seed: 42 })); }
    catch (err) { setError(`Simulation input is invalid: ${err.message}`); }
    finally { setBusy(""); }
  }

  async function runReplay() {
    setBusy("replay");
    setError(null);
    try { setReplayOut(await api.post("/api/betting/backtest", { rows: replayRows })); }
    catch (err) { setError(`Backtest rejected: ${err.message}`); }
    finally { setBusy(""); }
  }

  return (
    <div>
      <header className="page-header"><h1>Probability & Betting Research</h1></header>
      <Tabs tabs={tabs} active={tab} onChange={setTab} />
      {error && <div className="status error" role="alert">{error}</div>}

      {tab === "Value & stake" && <>
        <div className="form-grid betting-three">
          <label>Model probability<input type="number" min=".001" max=".999" step=".005" value={calc.model_probability} onChange={e => setCalc({ ...calc, model_probability: Number(e.target.value) })} /></label>
          <label>Selection decimal odds<input type="number" min="1.01" max="1000" step=".05" value={calc.decimal_odds} onChange={e => setCalc({ ...calc, decimal_odds: Number(e.target.value) })} /></label>
          <label>Probability uncertainty<input type="number" min="0" max=".5" step=".005" value={calc.uncertainty} onChange={e => setCalc({ ...calc, uncertainty: Number(e.target.value) })} /></label>
        </div>
        <label className="field-label">Opposing decimal odds (complete two-way market)
          <input type="number" min="1.01" max="1000" step=".05" value={calc.opposing_odds} onChange={e => setCalc({ ...calc, opposing_odds: Number(e.target.value) })} />
        </label>
        <label className="field-label">De-vig method
          <select value={calc.devig_method} onChange={e => setCalc({ ...calc, devig_method: e.target.value })}>
            <option>multiplicative</option><option>additive</option><option>power</option>
          </select>
        </label>
        {calcOut && <div className="metrics">
          <Metric label="De-vigged market probability" value={`${(calcOut.market_probability * 100).toFixed(2)}%`} />
          <Metric label="Raw EV / unit" value={`${calcOut.raw_ev >= 0 ? "+" : ""}${(calcOut.raw_ev * 100).toFixed(2)}%`} />
          <Metric label="Conservative probability" value={`${(calcOut.adjusted_probability * 100).toFixed(2)}%`} />
          <Metric label="Paper stake on $10k" value={`$${calcOut.stake.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} />
        </div>}
        {calcOut && <p className="caption">Decision: {calcOut.reason_code}.</p>}
      </>}

      {tab === "Field simulation" && <>
        <p>Upload one row per driver. Pace is an arbitrary lower-is-faster score; drivers sharing a constructor receive correlated shocks and all simulations produce unique finishing positions.</p>
        <p><a className="button-link" href={csvDataUrl(defaultEntries, Object.keys(defaultEntries[0]))} download="f1_field_simulation_template.csv">Download input template</a></p>
        <CsvInput label="Field CSV" onRows={rows => setSimEntries(rows)} />
        <label className="field-label">Simulations
          <input type="range" min="1000" max="50000" step="1000" value={simulations} onChange={e => setSimulations(Number(e.target.value))} />
          <span>{simulations.toLocaleString()}</span>
        </label>
        <p><button onClick={runSimulation} disabled={busy === "simulation"}>{busy === "simulation" ? "Running…" : "Run coherent field simulation"}</button></p>
        {simOut && <>
          <DataTable rows={simOut.rows} columns={simOut.columns} />
          <p><a className="button-link" href={csvDataUrl(simOut.rows, simOut.columns)} download="f1_market_probabilities.csv">Download probabilities</a></p>
        </>}
      </>}

      {tab === "Paper replay" && <>
        <p>Replay requires timestamps, real pre-event prices, de-vigged market probability, and settled outcomes. Records using a forecast or quote after event start are rejected.</p>
        <CsvInput label="Backtest ledger CSV" onRows={rows => { setReplayRows(rows); setReplayOut(null); }} />
        {!replayRows.length ? <div className="status">No odds ledger is bundled, so profitability is intentionally not estimated.</div> : (
          <p><button onClick={runReplay} disabled={busy === "replay"}>{busy === "replay" ? "Running…" : "Run paper backtest"}</button></p>
        )}
        {replayOut && <>
          <JsonBlock value={replayOut.summary} />
          <h2>Placed paper bets</h2><DataTable rows={replayOut.ledger} />
          <h2>All decisions and abstentions</h2><DataTable rows={replayOut.decisions} />
          <h2>Required staking sensitivity</h2><DataTable rows={replayOut.sensitivity} />
        </>}
      </>}

      {tab === "Calibration" && <>
        <p>Upload frozen probabilities and binary outcomes. Diagnostics include Brier score, log loss, adaptive reliability bins, ECE, calibration slope/intercept, and ROC AUC.</p>
        <CsvInput label="Calibration CSV" onRows={rows => setCalRows(rows)} />
        {!calRows.length && <div className="status">Required columns: probability and outcome. Optional columns: market and stage.</div>}
        {busy === "calibration" && <div className="status">Analyzing calibration…</div>}
        {calOut && <>
          <DataTable rows={calOut.metrics} />
          <h2>Adaptive reliability table</h2>
          <DataTable rows={calOut.reliability} />
          <LinePanel title="" rows={calOut.reliability || []} x="mean_probability" y="observed_rate" />
        </>}
      </>}
    </div>
  );
}
