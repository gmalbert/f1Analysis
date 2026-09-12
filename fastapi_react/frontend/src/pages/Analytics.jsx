import { useEffect, useState } from 'react'
import { api } from "../api";
import { Card, DataTable, Metric, Status } from "../components/UI";
import { BarPanel, LinePanel, ScatterPanel } from "../components/Charts";

export default function Analytics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.post("/api/analytics", { filters: [], max_rows: 5000 }).then(setData).catch(setError);
  }, []);
  return (
    <div>
      <header className="page-header"><div><h1>Analytics & Visualizations</h1><p>Charts, regressions, correlations, driver trends and constructor trends.</p></div></header>
      <Status loading={!data && !error} error={error}>
        {data && data.rows_considered === 0 ? (
          <Card><div className="empty">No data for the selected years / drivers.</div></Card>
        ) : data && <>
          <div className="metrics"><Metric label="Rows considered" value={data.rows_considered?.toLocaleString()} /></div>
          <div className="chart-grid">
            <ScatterPanel title="Active Years vs Final Position" rows={data.charts?.active_years_vs_final} x="resultsFinalPositionNumber" y="yearsActive" />
            <LinePanel title="Positions Gained Over Time" rows={data.charts?.positions_gained_over_time} x="short_date" y="positionsGained" />
            <ScatterPanel title="Last Practice vs Final Position" rows={data.charts?.practice_vs_final} x="lastFPPositionNumber" y="resultsFinalPositionNumber" />
            <ScatterPanel title="Starting Grid vs Final Position" rows={data.charts?.grid_vs_final} x="resultsStartingGridPositionNumber" y="resultsFinalPositionNumber" />
            <ScatterPanel title="Average Practice vs Final Position" rows={data.charts?.avg_practice_vs_final} x="averagePracticePosition" y="resultsFinalPositionNumber" />
            <ScatterPanel title="Pit Stop Time vs Final Position" rows={data.charts?.pit_stop_vs_final} x="averageStopTime" y="resultsFinalPositionNumber" />
          </div>
          <Card title="Regression Statistics"><DataTable rows={data.regressions || []} /></Card>
          {data.correlation && <Card title="Correlation Matrix"><DataTable rows={data.correlation.rows} /></Card>}
          <Card title="Driver Performance Over Time"><DataTable rows={data.driver_performance || []} /></Card>
          <Card title="Constructor Performance Over Time"><DataTable rows={data.constructor_performance || []} /></Card>
          <BarPanel title="Reasons for DNFs" rows={data.dnf_reasons || []} x="resultsReasonRetired" y="count" />
        </>}
      </Status>
    </div>
  );
}
