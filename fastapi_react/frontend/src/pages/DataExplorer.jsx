import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { DataTable, Status, Tabs } from "../components/UI";

const STREAMLIT_COLUMNS = [
  "grandPrixYear", "grandPrixName", "streetRace", "trackRace", "constructorName", "resultsDriverName",
  "resultsPodium", "resultsTop5", "resultsTop10", "resultsStartingGridPositionNumber",
  "resultsFinalPositionNumber", "positionsGained", "DNF", "resultsQualificationPositionNumber",
  "q1End", "q2End", "q3Top10", "averagePracticePosition", "lastFPPositionNumber", "numberOfStops",
  "averageStopTime", "totalStopTime", "driverBestStartingGridPosition", "driverBestRaceResult",
  "driverTotalChampionshipWins", "driverTotalPolePositions", "resultsReasonRetired",
  "driverTotalRaceEntries", "driverTotalRaceStarts", "driverTotalRaceWins", "driverTotalRaceLaps",
  "driverTotalPodiums", "avgLapTime", "finishingTime"
];

const CHECKBOX_COLUMNS = ["streetRace", "trackRace", "resultsPodium", "resultsTop5", "resultsTop10", "DNF", "q1End", "q2End", "q3Top10"];

export default function DataExplorer() {
  const [schema, setSchema] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [values, setValues] = useState({});
  const [result, setResult] = useState({ rows: [], columns: [], total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [innerTab, setInnerTab] = useState("Data");

  useEffect(() => {
    api.get("/api/data-explorer/schema").then(r => {
      setSchema(r.filters);
      try {
        const saved = JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null");
        if (saved?.applied) {
          setShowFilters(true);
          setValues(saved.values || {});
          runQuery(saved.filters || []);
        }
      } catch { /* ignore invalid session state */ }
    }).catch(setError);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const byName = useMemo(() => Object.fromEntries(schema.map(s => [s.column, s])), [schema]);
  const headerMap = useMemo(() => Object.fromEntries(schema.map(s => [s.column, s.label])), [schema]);

  function activeFilters(nextValues = values) {
    return Object.entries(nextValues).flatMap(([column, value]) => {
      if (value == null || value === "" || (Array.isArray(value) && value.some(v => v === ""))) return [];
      const spec = byName[column];
      if (!spec) return [];
      let normalized = value;
      if (spec.kind === "range") normalized = value.map(Number);
      return [{ column, kind: spec.kind, value: normalized }];
    });
  }

  async function runQuery(filters = activeFilters()) {
    setLoading(true);
    setError(null);
    try {
      const body = {
        filters,
        columns: STREAMLIT_COLUMNS,
        sort: ["grandPrixYear", "resultsFinalPositionNumber"],
        ascending: [false, true],
        descending: true,
        offset: 0,
        limit: 5000,
      };
      setResult(await api.post("/api/data-explorer/query", body));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }


  function toggleFilters(enabled) {
    setShowFilters(enabled);
    setError(null);
    if (enabled) {
      const defaults = {};
      const filters = [];
      for (const spec of schema) {
        if (spec.kind === "range" || spec.kind === "date_range") {
          defaults[spec.column] = [spec.min, spec.max];
          filters.push({ column: spec.column, kind: spec.kind, value: [spec.min, spec.max] });
        }
      }
      setValues(defaults);
      sessionStorage.setItem("f1analysis.filters", JSON.stringify({ applied: true, filters, values: defaults }));
      window.dispatchEvent(new CustomEvent("f1analysis:filters-changed"));
      runQuery(filters);
    } else {
      setValues({});
      setResult({ rows: [], columns: [], total: 0 });
      sessionStorage.removeItem("f1analysis.filters");
      window.dispatchEvent(new CustomEvent("f1analysis:filters-changed"));
    }
  }

  return (
    <div>
      <header className="page-header">
        <h1>Data Explorer</h1>
        <p>Filter and explore F1 race data from multiple perspectives.</p>
      </header>

      <label className="filter-results-toggle">
        <input type="checkbox" aria-label="Filter Results" checked={showFilters} disabled={!schema.length} onChange={e => toggleFilters(e.target.checked)} />
        Filter Results
      </label>

      {showFilters && (
        <section>
          <p>Number of filtered results: {result.total.toLocaleString()}</p>
          <Tabs tabs={["Data", "Data & Debug"]} active={innerTab} onChange={setInnerTab} />
          {innerTab === "Data" && (
            <Status loading={loading} error={error}>
              {result.rows?.length ? (
                <DataTable
                  rows={result.rows}
                  columns={result.columns}
                  headerMap={headerMap}
                  checkboxColumns={CHECKBOX_COLUMNS}
                  maxHeight={600}
                  ariaLabel="Filtered Formula 1 results"
                />
              ) : <div className="empty">No rows match the current filters.</div>}
            </Status>
          )}
          {innerTab === "Data & Debug" && <div aria-label="Data and debug placeholder" />}
        </section>
      )}

      {!showFilters && error && <Status error={error} />}
    </div>
  );
}
