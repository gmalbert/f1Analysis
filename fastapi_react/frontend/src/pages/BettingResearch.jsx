import { useEffect, useState } from 'react'
import Papa from "papaparse";
import { api } from "../api";
import { Card, DataTable, JsonBlock, Metric, Tabs } from "../components/UI";
import { LinePanel } from "../components/Charts";

const tabs = ["Value & stake", "Field simulation", "Paper replay", "Calibration"];

const defaultEntries = [
  { driver_id: "driver-a", constructor_id: "team-1", pace_score: 1.0, dnf_probability: 0.05, uncertainty: 0.8, race_sensitivity: 0.8 },
  { driver_id: "driver-b", constructor_id: "team-1", pace_score: 1.4, dnf_probability: 0.06, uncertainty: 0.9, race_sensitivity: 1.0 },
  { driver_id: "driver-c", constructor_id: "team-2", pace_score: 2.2, dnf_probability: 0.08, uncertainty: 1.0, race_sensitivity: 1.2 }
];

function csvDataUrl(rows, columns) {
  if (!rows?.length) return "#";
  const cols = columns?.length ? columns : Object.keys(rows[0]);
  const csv = Papa.unparse({ fields: cols, data: rows.map(row => cols.map(column => row[column] ?? "")) });
  return `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`;
}

function ReliabilityChart({ rows }) {
  // The reliability table from f1bet has a 'reliability'/'reliability_observed'
  // bin-mean field. Plot observed rate vs predicted mean with the y=x reference.
  const mapped = rows
    .map(r => ({
      predicted: Number(r.bin ?? r.predicted ?? r.reliability ?? r.center),
      observed: Number(r.observed ?? r.observed_rate ?? r.reliability_observed),
    }))
    .filter(p => Number.isFinite(p.predicted) && Number.isFinite(p.observed));
  if (mapped.length < 2) return null;
  const chartRows = mapped.map(p => ({ ...p, perfect: p.predicted }));
  return (
    <LinePanel
      title="Reliability curve"
      rows={chartRows}
      x="predicted"
      y="observed"
    />
  );
}

function CsvInput({ onRows, label }) {
  function load(file) {
    if (!file) return;
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: result => onRows(result.data, result.errors),
    });
  }
  return <label className="upload-label">{label}<input aria-label={label} type="file" accept=".csv,text/csv" onChange={e => load(e.target.files?.[0])} /></label>;
}

export default function BettingResearch() {
  const [tab, setTab] = useState(tabs[0]);
  const [calc, setCalc] = useState({ model_probability: .25, decimal_odds: 2.1, opposing_odds: 1.8, uncertainty: .02, devig_method: "multiplicative", bankroll: 10000 });
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
    setError(null);
    api.post("/api/betting/value", calc)
      .then(result => { if (!cancelled) setCalcOut(result); })
      .catch(e => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setBusy(""); });
    return () => { cancelled = true; };
  }, [calc]);

  async function runSimulation() {
    setBusy("simulation");
    try { setError(null); setSimOut(await api.post("/api/betting/simulate", { entries: simEntries, simulations, seed: 42 })); } catch (e) { setError(e.message); }
    finally { setBusy(""); }
  }
  async function runReplay() {
    setBusy("replay");
    try { setError(null); setReplayOut(await api.post("/api/betting/backtest", { rows: replayRows })); } catch (e) { setError(e.message); }
    finally { setBusy(""); }
  }
  async function runCalibration() {
    setBusy("calibration");
    try { setError(null); setCalOut(await api.post("/api/betting/calibration", { rows: calRows })); } catch (e) { setError(e.message); }
    finally { setBusy(""); }
  }
  return (
    <div>
      <header className="page-header"><div><h1>Probability & Betting Research</h1><p>Value, coherent race simulation, replay and calibration.</p></div></header>
      <Tabs tabs={tabs} active={tab} onChange={setTab} />
      {error && <div className="status error" role="alert" aria-live="assertive">{error}</div>}

      {tab === "Value & stake" && <Card title="Value & Stake Calculator">
        <div className="form-grid">
          {[
            ["Model probability", "model_probability", .001],
            ["Selection decimal odds", "decimal_odds", .01],
            ["Opposing decimal odds", "opposing_odds", .01],
            ["Probability uncertainty", "uncertainty", .005],
            ["Bankroll", "bankroll", 100]
          ].map(([label, key, step]) => <label key={key}>{label}<input type="number" step={step} value={calc[key]} onChange={e => setCalc({ ...calc, [key]: Number(e.target.value) })} /></label>)}
          <label>De-vig method<select value={calc.devig_method} onChange={e => setCalc({ ...calc, devig_method: e.target.value })}>
            <option>multiplicative</option><option>additive</option><option>power</option>
          </select></label>
        </div>
        {busy === "value" && <div className="loading-state" role="status" aria-busy="true">Calculating value…<span className="skeleton-line short" aria-hidden="true" /></div>}
        {calcOut && <div className="metrics">
          <Metric label="De-vigged market probability" value={`${(calcOut.market_probability * 100).toFixed(2)}%`} />
          <Metric label="Raw EV / unit" value={`${(calcOut.raw_ev * 100).toFixed(2)}%`} />
          <Metric label="Conservative probability" value={`${(calcOut.adjusted_probability * 100).toFixed(2)}%`} />
          <Metric label={`Paper stake on $${calc.bankroll.toLocaleString()}`} value={`$${calcOut.stake.toFixed(2)}`} />
        </div>}
        {calcOut && <p className="muted">Decision: {calcOut.reason_code}</p>}
      </Card>}

      {tab === "Field simulation" && <Card title="Correlated Field Simulation">
        <p>Upload one row per driver, or use the default three-driver template.</p>
        <a className="button-link" href={csvDataUrl(defaultEntries, Object.keys(defaultEntries[0]))} download="f1_field_simulation_template.csv">Download input template</a>
        <CsvInput label="Field CSV" onRows={rows => setSimEntries(rows)} />
        <DataTable rows={simEntries} />
        <label className="field-label">Simulations<input aria-label="Simulation count" type="range" min="1000" max="50000" step="1000" value={simulations} onChange={event => setSimulations(Number(event.target.value))} /><span>{simulations.toLocaleString()}</span></label>
        <button className="primary" disabled={busy === "simulation"} onClick={runSimulation}>{busy === "simulation" ? "Simulating…" : "Run coherent field simulation"}</button>
        {simOut && <>
          <DataTable rows={simOut.rows} columns={simOut.columns} />
          <p><a className="button-link" href={csvDataUrl(simOut.rows, simOut.columns)} download="f1_market_probabilities.csv">Download probabilities</a></p>
        </>}
      </Card>}

      {tab === "Paper replay" && <Card title="Paper Backtest">
        <p>Upload the timestamped ledger used by the existing f1bet backtest engine.</p>
        <CsvInput label="Backtest ledger CSV" onRows={rows => setReplayRows(rows)} />
        <button className="primary" disabled={!replayRows.length || busy === "replay"} onClick={runReplay}>{busy === "replay" ? "Running backtest…" : "Run paper backtest"}</button>
        {replayOut && <>
          <JsonBlock value={replayOut.summary} />
          <h4>Placed paper bets</h4><DataTable rows={replayOut.ledger} />
          <h4>All decisions and abstentions</h4><DataTable rows={replayOut.decisions} />
          <h4>Staking sensitivity</h4><DataTable rows={replayOut.sensitivity} />
        </>}
      </Card>}

      {tab === "Calibration" && <Card title="Calibration Diagnostics">
        <p>Required columns: <code>probability</code> and <code>outcome</code>. Optional: market and stage.</p>
        <CsvInput label="Calibration CSV" onRows={rows => setCalRows(rows)} />
        <button className="primary" disabled={!calRows.length || busy === "calibration"} onClick={runCalibration}>{busy === "calibration" ? "Analyzing…" : "Analyze calibration"}</button>
        {calOut && <>
          <DataTable rows={calOut.metrics} />
          <h4>Adaptive reliability</h4>
          <DataTable rows={calOut.reliability} />
          {calOut.reliability?.length > 1 && (
            <ReliabilityChart rows={calOut.reliability} />
          )}
        </>}
      </Card>}

    </div>
  );
}
