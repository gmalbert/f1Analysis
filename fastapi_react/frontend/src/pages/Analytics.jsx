import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { BarPanel, LinePanel, MultiLinePanel, PiePanel, RegressionPanel, ScatterPanel } from "../components/Charts";
import { Card, DataTable, Metric, Status } from "../components/UI";

function firstObjectArray(value) {
  if (!value || typeof value !== "object") return [];
  for (const candidate of Object.values(value)) {
    if (Array.isArray(candidate) && candidate.length && typeof candidate[0] === "object") return candidate;
  }
  return [];
}

export default function Analytics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tireData, setTireData] = useState(null);
  const [tireError, setTireError] = useState(null);
  const [tireLoading, setTireLoading] = useState(true);
  const filterState = useMemo(() => {
    try { return JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null"); }
    catch { return null; }
  }, []);
  const filterApplied = Boolean(filterState?.applied);

  useEffect(() => {
    api.get("/api/analytics/tire-strategy")
      .then(setTireData).catch(setTireError).finally(() => setTireLoading(false));
    if (!filterApplied) {
      setLoading(false);
      return;
    }
    api.post("/api/analytics", { filters: filterState?.filters || [], max_rows: 10000 })
      .then(setData).catch(setError).finally(() => setLoading(false));
  }, [filterApplied, filterState]);

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

  const manifestMetrics = data?.model_summary?.metrics || {};
  const importanceRows = firstObjectArray(data?.feature_importance);

  return (
    <div>
      <header className="page-header">
        <h1>Analytics & Visualizations</h1>
        <p>Comprehensive charts, regressions, and analysis of filtered data.</p>
      </header>

      {!filterApplied ? (
        <div className="status">Please filter results in the Data Explorer tab first to view analytics.</div>
      ) : (
        <Status loading={loading} error={error}>
          {data?.rows_considered === 0 ? <div className="empty">No data available after filtering. Please adjust your filters.</div> : data && <>
            <ScatterPanel title="Active Years v. Final Position" rows={data.charts?.active_years_vs_final} x="resultsFinalPositionNumber" y="yearsActive" xLabel="Final Position" yLabel="Years Active" />

            <LinePanel title="Positions Gained" rows={data.charts?.positions_gained_over_time} x="short_date" y="positionsGained" xLabel="Date" yLabel="Positions Gained" />
            <ScatterPanel title="" rows={data.charts?.positions_gained_over_time} x="short_date" y="positionsGained" xLabel="Date" yLabel="Positions Gained" />

            <ScatterPanel title="Practice Position vs Final Position" rows={data.charts?.practice_vs_final} x="lastFPPositionNumber" y="resultsFinalPositionNumber" xLabel="Last FP Position" yLabel="Final Position" />
            <ScatterPanel title="Starting Position vs Final Position" rows={data.charts?.grid_vs_final} x="resultsStartingGridPositionNumber" y="resultsFinalPositionNumber" xLabel="Starting Position" yLabel="Final Position" />
            <ScatterPanel title="Average Practice Position vs Final Position" rows={data.charts?.avg_practice_vs_final} x="averagePracticePosition" y="resultsFinalPositionNumber" xLabel="Average Practice Position" yLabel="Final Position" />

            {(data.regression_series || []).map(regression => (
              <div key={regression.x}>
                <RegressionPanel title={regression.title} points={regression.points} fit={regression.fit} x={regression.x} y={regression.y} xLabel={regression.x_label} yLabel="Final Position" />
                <p><strong>Regression Equation:</strong> y = {Number(data.regressions?.find(item => item.x === regression.x)?.slope ?? 0).toFixed(2)}x + {Number(data.regressions?.find(item => item.x === regression.x)?.intercept ?? 0).toFixed(2)}</p>
                <p><strong>Regression Statistics:</strong></p>
                <p>R-squared: {Number(data.regressions?.find(item => item.x === regression.x)?.r_squared ?? 0).toFixed(2)}</p>
              </div>
            ))}

            <Card title="Correlation Matrix">
              <p className="caption">Correlation values range from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.</p>
              <h2>Feature Correlations with Final Position and Podium</h2>
              <DataTable rows={data.correlation?.rows || []} maxHeight={600} />
            </Card>

            <MultiLinePanel title="Driver Performance Over Time" rows={data.driver_performance || []} x="grandPrixYear" y="average_final_position" series="resultsDriverName" xLabel="Year" yLabel="Average Final Position" />
            <BarPanel title="Constructor Dominance Over the Years" rows={data.constructor_performance || []} x="grandPrixYear" y="total_wins" xLabel="Year" yLabel="Wins and Podiums" />

            <ScatterPanel title="Impact of Starting Grid Position on Final Position" rows={data.charts?.grid_vs_final} x="resultsStartingGridPositionNumber" y="resultsFinalPositionNumber" xLabel="Starting Pos." yLabel="Final Pos." />
            <ScatterPanel title="Pit Stop Analysis" rows={data.charts?.pit_stop_vs_final} x="averageStopTime" y="resultsFinalPositionNumber" xLabel="Avg. Stop Time" yLabel="Final Pos." />

            <Card title="Driver vs Constructor Performance"><DataTable rows={data.driver_vs_constructor || []} maxHeight={600} /></Card>
            <BarPanel title="Reasons for DNFs" rows={data.dnf_reasons || []} x="resultsReasonRetired" y="count" xLabel="Reason" yLabel="Count" />
            <Card title="DNF by Driver"><DataTable rows={data.dnf_by_driver || []} maxHeight={600} /></Card>
            <Card title="DNF by Race"><DataTable rows={data.dnf_by_race || []} maxHeight={600} /></Card>
            <Card title="DNF by Constructor"><DataTable rows={data.dnf_by_constructor || []} maxHeight={600} /></Card>
            <PiePanel title="DNF Reasons" rows={data.dnf_reasons || []} nameKey="resultsReasonRetired" valueKey="count" />

            <ScatterPanel title="Track Characteristics and Performance" rows={data.charts?.track_turns_vs_final} x="turns" y="resultsFinalPositionNumber" xLabel="Turns" yLabel="Final Position" />
            <Card title={`${data.season_year || ""} Season Summary`}><DataTable rows={data.season_summary || []} /></Card>

            <BarPanel title="Driver Consistency" rows={data.driver_consistency || []} x="resultsDriverName" y="finishing_position_std" xLabel="Driver" yLabel="Standard Deviation - Finishing" />
            <p className="caption">(Lower is Better)</p>
            <p className="caption">Lower standard deviation indicates more consistent finishing positions.</p>
            <DataTable rows={data.driver_consistency || []} maxHeight={600} />

            <Card title="Predictive Data Model">
              <div className="metrics">
                <Metric label="Mean Squared Error" value={manifestMetrics.mse != null ? Number(manifestMetrics.mse).toFixed(3) : "—"} />
                <Metric label="R² Score" value={manifestMetrics.r2 != null ? Number(manifestMetrics.r2).toFixed(3) : "—"} />
                <Metric label="Mean Absolute Error" value={manifestMetrics.mae != null ? Number(manifestMetrics.mae).toFixed(2) : "—"} />
                <Metric label="Mean Error" value={manifestMetrics.mean_error != null ? Number(manifestMetrics.mean_error).toFixed(2) : "—"} />
              </div>
              {importanceRows.length > 0 && <>
                <h2>Feature Importance</h2>
                <DataTable rows={importanceRows.slice(0, 50)} maxHeight={600} />
              </>}
            </Card>
          </>}
        </Status>
      )}

      <section className="tire-section" aria-labelledby="tire-strategy-title">
        <h2 id="tire-strategy-title">🏎️ Tire Strategy Analysis</h2>
        <p className="caption">Compound usage, stint data, and tire degradation per driver/race (FastF1:2018–present).</p>
        <div className="form-grid tire-controls">
          <label>Year
            <select value={tireData?.selected_year ?? ""} disabled={!tireData?.years?.length} onChange={e => selectTire(Number(e.target.value), tireData?.selected_event)}>
              {(tireData?.years || []).map(year => <option key={year} value={year}>{year}</option>)}
            </select>
          </label>
          <label>Grand Prix
            <select value={tireData?.selected_event ?? ""} disabled={!tireData?.events?.length} onChange={e => selectTire(tireData?.selected_year, e.target.value)}>
              {(tireData?.events || []).map(name => <option key={name}>{name}</option>)}
            </select>
          </label>
        </div>
        <Status loading={tireLoading} error={tireError}>
          {tireData && !tireData.race_rows?.length ? <p className="empty">No tire strategy data available for this race.</p> : tireData && <>
            <DataTable rows={tireData.race_rows || []} />
            <BarPanel title="Avg Tire Degradation by Driver (s/lap)" rows={tireData.degradation_rows || []} x="driver" y="degradation" xLabel="Driver" yLabel="Degradation (s/lap)" />
            <details>
              <summary>Historical Tire Management by Driver (all races in selected year)</summary>
              <DataTable rows={tireData.historical_rows || []} />
            </details>
          </>}
        </Status>
      </section>
    </div>
  );
}
