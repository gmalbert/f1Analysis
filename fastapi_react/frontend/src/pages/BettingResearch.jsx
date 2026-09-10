import { useState } from 'react'
import Papa from "papaparse";
import { api } from "../api";
import { Card, DataTable, JsonBlock, Metric, Tabs } from "../components/UI";

const tabs = ["Value & stake", "Field simulation", "Paper replay", "Calibration", "Release gates"];

const defaultEntries = [
  { driver_id: "driver-a", constructor_id: "team-1", pace_score: 1.0, dnf_probability: 0.05, uncertainty: 0.8, race_sensitivity: 0.8 },
  { driver_id: "driver-b", constructor_id: "team-1", pace_score: 1.4, dnf_probability: 0.06, uncertainty: 0.9, race_sensitivity: 1.0 },
  { driver_id: "driver-c", constructor_id: "team-2", pace_score: 2.2, dnf_probability: 0.08, uncertainty: 1.0, race_sensitivity: 1.2 }
];

function CsvInput({ onRows }) {
  function load(file) {
    if (!file) return;
    Papa.parse(file, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: result => onRows(result.data) });
  }
  return <input type="file" accept=".csv,text/csv" onChange={e => load(e.target.files?.[0])} />;
}

export default function BettingResearch() {
  const [tab, setTab] = useState(tabs[0]);
  const [calc, setCalc] = useState({ model_probability: .25, decimal_odds: 2.1, opposing_odds: 1.8, uncertainty: .02, devig_method: "multiplicative", bankroll: 10000 });
  const [calcOut, setCalcOut] = useState(null);
  const [simEntries, setSimEntries] = useState(defaultEntries);
  const [simOut, setSimOut] = useState(null);
  const [replayRows, setReplayRows] = useState([]);
  const [replayOut, setReplayOut] = useState(null);
  const [calRows, setCalRows] = useState([]);
  const [calOut, setCalOut] = useState(null);
  const [gov, setGov] = useState(null);
  const [error, setError] = useState(null);

  async function calculate() {
    try { setError(null); setCalcOut(await api.post("/api/betting/value", calc)); } catch (e) { setError(e.message); }
  }
  async function runSimulation() {
    try { setError(null); setSimOut(await api.post("/api/betting/simulate", { entries: simEntries, simulations: 10000, seed: 42 })); } catch (e) { setError(e.message); }
  }
  async function runReplay() {
    try { setError(null); setReplayOut(await api.post("/api/betting/backtest", { rows: replayRows })); } catch (e) { setError(e.message); }
  }
  async function runCalibration() {
    try { setError(null); setCalOut(await api.post("/api/betting/calibration", { rows: calRows })); } catch (e) { setError(e.message); }
  }
  async function loadGovernance() {
    try { setError(null); setGov(await api.get("/api/betting/governance")); } catch (e) { setError(e.message); }
  }

  return (
    <div>
      <header className="page-header"><div><h1>Probability & Betting Research</h1><p>Paper-research only: value, coherent race simulation, replay, calibration and release governance.</p></div></header>
      <div className="warning">A finishing-position MAE is not evidence of a betting edge. Release requires frozen real odds, calibration, closing-line value and walk-forward replay.</div>
      <Tabs tabs={tabs} active={tab} onChange={setTab} />
      {error && <div className="status error">{error}</div>}

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
        <button className="primary" onClick={calculate}>Calculate</button>
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
        <CsvInput onRows={setSimEntries} />
        <DataTable rows={simEntries} />
        <button className="primary" onClick={runSimulation}>Run coherent field simulation</button>
        {simOut && <DataTable rows={simOut.rows} columns={simOut.columns} />}
      </Card>}

      {tab === "Paper replay" && <Card title="Paper Backtest">
        <p>Upload the timestamped ledger used by the existing f1bet backtest engine.</p>
        <CsvInput onRows={setReplayRows} />
        <button className="primary" disabled={!replayRows.length} onClick={runReplay}>Run paper backtest</button>
        {replayOut && <>
          <JsonBlock value={replayOut.summary} />
          <h4>Placed paper bets</h4><DataTable rows={replayOut.ledger} />
          <h4>All decisions and abstentions</h4><DataTable rows={replayOut.decisions} />
          <h4>Staking sensitivity</h4><DataTable rows={replayOut.sensitivity} />
        </>}
      </Card>}

      {tab === "Calibration" && <Card title="Calibration Diagnostics">
        <p>Required columns: <code>probability</code> and <code>outcome</code>. Optional: market and stage.</p>
        <CsvInput onRows={setCalRows} />
        <button className="primary" disabled={!calRows.length} onClick={runCalibration}>Analyze calibration</button>
        {calOut && <><DataTable rows={calOut.metrics} /><h4>Adaptive reliability</h4><DataTable rows={calOut.reliability} /></>}
      </Card>}

      {tab === "Release gates" && <Card title="Release Governance">
        <button className="primary" onClick={loadGovernance}>Load current release evidence</button>
        {gov && <>
          <h4>Feature availability registry</h4><DataTable rows={gov.registry} />
          <h4>Current wide-table contract audit</h4><JsonBlock value={gov.contract_audit} />
          <h4>Automated release evidence</h4><JsonBlock value={gov.release_evidence} />
        </>}
      </Card>}
    </div>
  );
}
