import { useEffect, useMemo, useState } from 'react'
import { api } from "../api";
import { Card, DataTable, Metric, Status } from "../components/UI";

function Section({ title, rows }) {
  return <Card title={title}><DataTable rows={rows || []} /></Card>;
}

export default function NextRace() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [modelKey, setModelKey] = useState("xgboost");
  useEffect(() => { api.get("/api/next-race").then(setData).catch(setError); }, []);
  const modelKeys = useMemo(() => Object.keys(data?.predictions?.predictions_by_model || {}), [data]);
  const modelBlock = data?.predictions?.predictions_by_model?.[modelKey] || data?.predictions?.predictions_by_model?.[modelKeys[0]];

  return (
    <div>
      <header className="page-header"><div><h1>Next Race</h1><p>Upcoming race details, predictions, historical performance, flags, pit-stop context and weather.</p></div></header>
      <Status loading={!data && !error} error={error}>
        {data && !data.next_race && <Card><div className="empty">No upcoming race found.</div></Card>}
        {data?.next_race && <>
          <div className="metrics">
            <Metric label="Grand Prix" value={data.race_name} />
            <Metric label="Year" value={data.year} />
            <Metric label="Race ID" value={data.race_id} />
            <Metric label="Past results" value={data.past_results?.length || 0} />
          </div>
          <Card title="Race Details"><DataTable rows={[data.next_race]} /></Card>
          {data.predictions?.format === "json" && <Card title={`Predictions — ${data.predictions.file}`}>
            <div className="button-row">
              <select value={modelKeys.includes(modelKey) ? modelKey : (modelKeys[0] || "")} onChange={e => setModelKey(e.target.value)}>
                {modelKeys.map(key => <option key={key} value={key}>{key}</option>)}
              </select>
            </div>
            {modelBlock?.model_mae != null && <p className="muted">Model MAE: {Number(modelBlock.model_mae).toFixed(3)}</p>}
            <DataTable rows={modelBlock?.predictions || []} />
          </Card>}
          {data.predictions?.format === "csv" && <Card title={`Predictions — ${data.predictions.file}`}><DataTable rows={data.predictions.rows} columns={data.predictions.columns} /></Card>}
          {!data.predictions && <Card title="Predictions"><div className="empty">No precomputed prediction artifact matched the upcoming race.</div></Card>}
          <Section title="Past Results" rows={data.past_results} />
          <Section title="Driver Performance at this Grand Prix" rows={data.driver_performance} />
          <Section title="Constructor Performance at this Grand Prix" rows={data.constructor_performance} />
          <Section title="Flags & Safety Cars" rows={data.race_messages} />
          <Section title="Weather" rows={data.weather} />
        </>}
      </Status>
    </div>
  );
}
