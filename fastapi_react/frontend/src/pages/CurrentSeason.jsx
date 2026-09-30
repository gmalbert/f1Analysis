import { useEffect, useState } from "react";
import { api } from "../api";
import { Status } from "../components/UI";

const COLUMNS = ["round", "fullName", "date", "time", "circuitType", "courseLength", "laps", "turns", "distance", "totalRacesHeld"];
const LABELS = {
  round: "Round",
  fullName: "Grand Prix",
  date: "Date",
  time: "Time",
  circuitType: "Circuit Type",
  courseLength: "Course Length",
  laps: "Laps",
  turns: "Turns",
  distance: "Distance",
  totalRacesHeld: "Total Races Held",
};

export default function CurrentSeason() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.get("/api/current-season").then(setData).catch(setError); }, []);

  return (
    <div>
      <header className="page-header">
        <h1>{data?.year || "Current"} Season</h1>
        <p>Complete schedule and information for the {data?.year || "current"} Formula 1 season.</p>
      </header>
      <Status loading={!data && !error} error={error}>
        {data && <>
          <p>Total number of races: {data.rows?.length || 0}</p>
          {data.rows?.length ? (
            <div className="table-wrap" role="region" aria-label={`${data.year} Formula 1 schedule`} style={{ maxHeight: 900 }}>
              <table>
                <thead><tr>{COLUMNS.map(column => <th key={column} scope="col">{LABELS[column]}</th>)}</tr></thead>
                <tbody>
                  {data.rows.map((row, index) => (
                    <tr key={index} className={row.seasonStatus === "Next Race" ? "next-race-row" : ""}>
                      {COLUMNS.map(column => <td key={column}>{row[column] == null ? "" : String(row[column])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div className="empty">No race data for the current year.</div>}
        </>}
      </Status>
    </div>
  );
}
