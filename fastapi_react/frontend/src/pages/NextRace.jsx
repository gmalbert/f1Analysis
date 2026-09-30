import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { DataTable, Status } from "../components/UI";

const NEXT_RACE_COLUMNS = ["date", "time", "fullName", "courseLength", "turns", "laps"];
const NEXT_RACE_LABELS = {
  date: "Date", time: "Time", fullName: "Grand Prix", courseLength: "Course Length",
  turns: "Turns", laps: "Laps",
};

const PIT_LABELS = {
  year: "Year", round: "Round", constructorName: "Constructor", lap: "Lap",
  pitStopSeconds: "Pit Stop (s)", pit_time_stationary: "Pit Time Stationary (s)",
};

export default function NextRace() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [show, setShow] = useState(true);
  useEffect(() => { api.get("/api/next-race").then(setData).catch(setError); }, []);

  const predictionRows = useMemo(() => {
    const byModel = data?.predictions?.predictions_by_model || {};
    const block = byModel.xgboost || byModel[Object.keys(byModel)[0]];
    return (block?.predictions || []).map(row => ({
      Rank: row.predicted_rank,
      resultsDriverName: row.driverName,
      constructorName: row.constructor,
      PredictedFinalPosition: row.predicted_position,
      ModelMAE: row.mae ?? block?.model_mae,
    })).sort((a, b) => Number(a.Rank) - Number(b.Rank));
  }, [data]);

  return (
    <div>
      <header className="page-header">
        <h1>Next Race</h1>
        <p>Details, predictions, and analysis for the upcoming race.</p>
      </header>

      <label className="filter-results-toggle">
        <input type="checkbox" checked={show} onChange={e => setShow(e.target.checked)} />
        Show Next Race
      </label>

      <Status loading={!data && !error} error={error}>
        {data && !data.next_race ? <div className="warning">No upcoming race found in the schedule.</div> : data?.next_race && <>
          {show && <>
            <h2>Next Race:</h2>
            <DataTable rows={[data.next_race]} columns={NEXT_RACE_COLUMNS} headerMap={NEXT_RACE_LABELS} maxHeight={180} />
          </>}

          <h2>Past Results:</h2>
          <p>Total number of results: {data.past_results?.length || 0}</p>
          <DataTable rows={data.past_results || []} maxHeight={600} />

          <h2>Predictive Results for Active Drivers</h2>
          {predictionRows.length ? (
            <DataTable
              rows={predictionRows}
              columns={["Rank", "constructorName", "resultsDriverName", "PredictedFinalPosition", "ModelMAE"]}
              headerMap={{
                Rank: "Rank", constructorName: "Constructor", resultsDriverName: "Driver",
                PredictedFinalPosition: "Predicted Final Position", ModelMAE: "MAE",
              }}
            />
          ) : <div className="empty">No precomputed active-driver prediction artifact is available for this race.</div>}

          <h2>Predictive DNF</h2>
          {data.legacy_predictions?.length ? (
            <DataTable
              rows={data.legacy_predictions}
              columns={["constructorName", "resultsDriverName", "driverDNFCount", "driverDNFPercentage", "PredictedDNFProbabilityPercentage", "PredictedDNFProbabilityStd"]}
              headerMap={{
                constructorName: "Constructor", resultsDriverName: "Driver", driverDNFCount: "Driver DNF Count",
                driverDNFPercentage: "Driver DNF Percentage", PredictedDNFProbabilityPercentage: "Predicted DNF Probability (%)",
                PredictedDNFProbabilityStd: "Predicted DNF Probability Std.",
              }}
            />
          ) : <div className="empty">No committed DNF prediction rows are available for the upcoming race.</div>}

          <h2>Predicted Safety Car</h2>
          {data.safety_car_predictions?.length ? (
            <DataTable rows={data.safety_car_predictions} />
          ) : (
            <div className="empty">No committed safety-car prediction artifact is available for the upcoming race.</div>
          )}

          <h2>Flags and Safety Cars from {data.race_name}:</h2>
          <p className="caption">Race messages, including flags, are only available going back to 2018.</p>
          <p>Total number of results: {data.race_messages?.length || 0}</p>
          <DataTable rows={data.race_messages || []} columns={["Year", "Round", "SafetyCarStatus", "redFlag", "yellowFlag", "doubleYellowFlag", "dnf_count"]} />

          <h2>Driver Performance in {data.race_name}:</h2>
          <p>Total number of results: {data.driver_performance?.length || 0}</p>
          <DataTable rows={data.driver_performance || []} maxHeight={600} />

          <h2>Constructor Performance in {data.race_name}:</h2>
          <DataTable rows={data.constructor_performance || []} />

          <h2>Fastest Individual Pit Stop per Constructor</h2>
          <p>Total number of fastest pit stops: {data.fastest_pit_stops?.total ?? 0}</p>
          <p>Pit Time Constant: {data.fastest_pit_stops?.pit_lane_time_constant ?? "N/A"}</p>
          {data.fastest_pit_stops?.rows?.length ? (
            <DataTable
              rows={data.fastest_pit_stops.rows}
              columns={["year", "round", "constructorName", "lap", "pitStopSeconds", "pit_time_stationary"]}
              headerMap={PIT_LABELS}
            />
          ) : <div className="empty">No individual pit stop data available for prior races at this Grand Prix.</div>}

          <h2>Weather Data for {data.race_name}:</h2>
          <p>Total number of weather records: {data.weather?.length || 0}</p>
          <DataTable rows={data.weather || []} />
        </>}
      </Status>
    </div>
  );
}
