import { useEffect, useState } from 'react'
import { api } from "../api";
import { Card, Metric, Status } from "../components/UI";

export default function CurrentSeason() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.get("/api/current-season").then(setData).catch(setError); }, []);
  return (
    <div>
      <header className="page-header"><div><h1>{data?.year || "Current"} Season</h1><p>Complete schedule and circuit information for the current Formula 1 season.</p></div></header>
      <Status loading={!data && !error} error={error}>
        {data && <>
          <div className="metrics"><Metric label="Races" value={data.rows?.length || 0} /></div>
          <Card title={`${data.year} Schedule`}>
            <div className="table-wrap" style={{maxHeight: 780}}><table><thead><tr>{data.columns.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>
              {data.rows.map((row, i) => <tr key={i} className={row.seasonStatus === "Next Race" ? "next-race-row" : ""}>{data.columns.map(c => <td key={c}>{row[c] == null ? "" : String(row[c])}</td>)}</tr>)}
            </tbody></table></div>
          </Card>
        </>}
      </Status>
    </div>
  );
}
