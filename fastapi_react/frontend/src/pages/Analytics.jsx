import { useEffect, useState } from 'react'
import { api } from "../api";
import { Card, DataTable, Metric, Status } from "../components/UI";
import { BarPanel, LinePanel, RegressionPanel, ScatterPanel } from "../components/Charts";

export default function Analytics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tireData, setTireData] = useState(null);
  const [tireError, setTireError] = useState(null);
  const [tireLoading, setTireLoading] = useState(true);
  const [filterState] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null"); }
    catch { return null; }
  });
  const filterApplied = Boolean(filterState?.applied);
  const analyticsFilters = filterState?.filters;
  useEffect(() => {
    api.get("/api/analytics/tire-strategy")
      .then(setTireData).catch(setTireError).finally(() => setTireLoading(false));
    if (!filterApplied) {
      setLoading(false);
      return;
    }
    api.post("/api/analytics", { filters: analyticsFilters || [], max_rows: 5000 })
      .then(setData).catch(setError).finally(() => setLoading(false));
  }, [analyticsFilters, filterApplied]);

  async function selectTire(year, eventName) {
    setTireLoading(true);
    setTireError(null);
    const params = new URLSearchParams();
    if (year != null) params.set("year", year);
    if (eventName) params.set("event_name", eventName);
    try {
      setTireData(await api.get(`/api/analytics/tire-strategy?${params.toString()}`));
    } catch (err) {
      setTireError(err);
    } finally {
      setTireLoading(false);
    }
  }
  return (
    <div>
      <header className="page-header"><div><h1>Analytics & Visualizations</h1><p>Charts, regressions, correlations, driver trends and constructor trends.</p></div></header>
      {filterApplied ? <Status loading={loading} error={error}>
        {data && data.rows_considered === 0 ? (
          <Card><div className="empty">No data for the selected years / drivers.</div></Card>
        ) : data && <>
          <div className="metrics"><Metric label="Rows considered" value={data.rows_considered?.toLocaleString()} /></div>
          <div className="chart-grid">
            <ScatterPanel title="Active Years vs Final Position" rows={data.charts?.active_years_vs_final} x="resultsFinalPositionNumber" y="yearsActive" />
            <LinePanel title="Positions Gained Over Time" rows={data.charts?.positions_gained_over_time} x="short_date" y="positionsGained" />
            <ScatterPanel title="Positions Gained" rows={data.charts?.positions_gained_over_time} x="short_date" y="positionsGained" xLabel="Date" />
            <ScatterPanel title="Last Practice vs Final Position" rows={data.charts?.practice_vs_final} x="lastFPPositionNumber" y="resultsFinalPositionNumber" />
            <ScatterPanel title="Starting Grid vs Final Position" rows={data.charts?.grid_vs_final} x="resultsStartingGridPositionNumber" y="resultsFinalPositionNumber" />
            <ScatterPanel title="Average Practice vs Final Position" rows={data.charts?.avg_practice_vs_final} x="averagePracticePosition" y="resultsFinalPositionNumber" />
            <ScatterPanel title="Pit Stop Time vs Final Position" rows={data.charts?.pit_stop_vs_final} x="averageStopTime" y="resultsFinalPositionNumber" />
            {data.regression_series?.map(regression => (
              <RegressionPanel
                key={regression.x}
                title={regression.title}
                points={regression.points}
                fit={regression.fit}
                x={regression.x}
                y={regression.y}
                xLabel={regression.x_label}
                yLabel="Final Position"
              />
            ))}
          </div>
          <Card title="Regression Statistics"><DataTable rows={data.regressions || []} /></Card>
          {data.correlation && <Card title="Correlation Matrix"><DataTable rows={data.correlation.rows} /></Card>}
          <Card title="Driver Performance Over Time"><DataTable rows={data.driver_performance || []} /></Card>
          <Card title="Constructor Performance Over Time"><DataTable rows={data.constructor_performance || []} /></Card>
          <BarPanel title="Reasons for DNFs" rows={data.dnf_reasons || []} x="resultsReasonRetired" y="count" />
          <Card title="DNF by Driver"><DataTable rows={data.dnf_by_driver || []} /></Card>
        </>}
      </Status> : <div className="status" role="status">Please filter results in the Data Explorer tab first to view analytics.</div>}

      <section aria-labelledby="tire-strategy-title" className="tire-section">
        <h2 id="tire-strategy-title">Tire Strategy Analysis</h2>
        <p className="muted">Compound usage, stint data, and tire degradation per driver/race (FastF1: 2018–present).</p>
        <div className="form-grid tire-controls">
          <label>Year<select aria-label="Tire strategy year" disabled={!tireData?.years?.length} value={tireData?.selected_year ?? ""} onChange={event => selectTire(Number(event.target.value), tireData?.selected_event)}>
            {(tireData?.years || []).map(year => <option key={year} value={year}>{year}</option>)}
          </select></label>
          <label>Grand Prix<select aria-label="Tire strategy Grand Prix" disabled={!tireData?.events?.length} value={tireData?.selected_event ?? ""} onChange={event => selectTire(tireData?.selected_year, event.target.value)}>
            {(tireData?.events || []).map(name => <option key={name}>{name}</option>)}
          </select></label>
        </div>
        <Status loading={tireLoading} error={tireError}>
          {tireData && !tireData.race_rows?.length ? <p className="empty">No tire strategy data is available for the selected race.</p> : tireData && <>
            <Card title="Compound Usage per Driver"><DataTable rows={tireData.race_rows} /></Card>
            <BarPanel title="Avg Tire Degradation by Driver (s/lap)" rows={tireData.degradation_rows || []} x="driver" y="degradation" xLabel="Driver" yLabel="Degradation (s/lap)" />
            <details className="tire-history">
              <summary>Historical Tire Management by Driver (all races in selected year)</summary>
              <DataTable rows={tireData.historical_rows || []} />
            </details>
          </>}
        </Status>
      </section>
    </div>
  );
}
