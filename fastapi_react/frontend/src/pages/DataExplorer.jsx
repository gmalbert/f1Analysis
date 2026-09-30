import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Card, DataTable, Status } from "../components/UI";

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

function FilterEditor({ spec, value, onChange }) {
  if (spec.kind === "boolean") {
    return (
      <label className="streamlit-checkbox">
        <input
          type="checkbox"
          aria-label={spec.label}
          checked={value === true}
          onChange={e => onChange(e.target.checked ? true : null)}
        />
        <span>{spec.label}</span>
      </label>
    );
  }
  if (spec.kind === "exact" && spec.options) {
    return (
      <label>
        <span>{spec.label}</span>
        <select aria-label={spec.label} value={value ?? ""} onChange={e => onChange(e.target.value || null)}>
          <option value=""> All</option>
          {spec.options.map(v => <option key={v} value={v}>{v}</option>)}
        </select>
      </label>
    );
  }
  if (spec.kind === "range" || spec.kind === "date_range") {
    const current = Array.isArray(value) ? value : [spec.min, spec.max];
    const type = spec.kind === "date_range" ? "date" : "number";
    return (
      <label>
        <span>{spec.label}</span>
        <div className="range-pair">
          <input aria-label={`${spec.label} minimum`} type={type} value={current[0] ?? ""} onChange={e => onChange([e.target.value, current[1]])} />
          <input aria-label={`${spec.label} maximum`} type={type} value={current[1] ?? ""} onChange={e => onChange([current[0], e.target.value])} />
        </div>
      </label>
    );
  }
  return (
    <label>
      <span>{spec.label}</span>
      <input aria-label={spec.label} value={value ?? ""} onChange={e => onChange(e.target.value || null)} />
    </label>
  );
}

export default function DataExplorer() {
  const [schema, setSchema] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [values, setValues] = useState({});
  const [result, setResult] = useState({ rows: [], columns: [], total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
        descending: true,
        offset: 0,
        limit: 1000,
      };
      setResult(await api.post("/api/data-explorer/query", body));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  function apply(nextValues = values) {
    const filters = activeFilters(nextValues);
    sessionStorage.setItem("f1analysis.filters", JSON.stringify({ applied: true, filters, values: nextValues }));
    runQuery(filters);
  }

  function toggleFilters(enabled) {
    setShowFilters(enabled);
    setError(null);
    if (enabled) {
      const defaults = {};
      for (const spec of schema) {
        if (spec.kind === "range" || spec.kind === "date_range") defaults[spec.column] = [spec.min, spec.max];
      }
      setValues(defaults);
      apply(defaults);
    } else {
      setValues({});
      setResult({ rows: [], columns: [], total: 0 });
      sessionStorage.removeItem("f1analysis.filters");
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
        <div className="explorer-grid">
          <aside className="filter-card" aria-label="Select filters to apply">
            <h2>Select filters to apply:</h2>
            <div className="filter-list">
              {schema.map(spec => (
                <div className="filter-row" key={spec.column}>
                  <FilterEditor
                    spec={spec}
                    value={values[spec.column]}
                    onChange={value => {
                      const next = { ...values, [spec.column]: value };
                      setValues(next);
                      apply(next);
                    }}
                  />
                </div>
              ))}
            </div>
          </aside>

          <section>
            <p>Number of filtered results: {result.total.toLocaleString()}</p>
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
          </section>
        </div>
      )}

      {!showFilters && error && <Status error={error} />}
    </div>
  );
}
