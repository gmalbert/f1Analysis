import { useEffect, useMemo, useState } from 'react'
import { api } from "../api";
import { Card, DataTable, Status } from "../components/UI";

const preferredColumns = [
  "grandPrixYear", "grandPrixName", "constructorName", "resultsDriverName",
  "resultsStartingGridPositionNumber", "resultsFinalPositionNumber", "positionsGained",
  "DNF", "resultsQualificationPositionNumber", "averagePracticePosition",
  "lastFPPositionNumber", "numberOfStops", "averageStopTime", "totalStopTime"
];

function FilterEditor({ spec, value, onChange }) {
  if (spec.kind === "boolean") {
    return (
      <select aria-label={spec.label} value={value ?? ""} onChange={e => onChange(e.target.value === "" ? null : e.target.value === "true")}>
        <option value="">All</option><option value="true">True / 1</option><option value="false">False / 0</option>
      </select>
    );
  }
  if (spec.kind === "exact" && spec.options) {
    return (
      <select aria-label={spec.label} value={value ?? ""} onChange={e => onChange(e.target.value || null)}>
        <option value="">All</option>
        {spec.options.map(v => <option key={v} value={v}>{v}</option>)}
      </select>
    );
  }
  if (spec.kind === "range" || spec.kind === "date_range") {
    const current = Array.isArray(value) ? value : [spec.min, spec.max];
    const type = spec.kind === "date_range" ? "date" : "number";
    return (
      <div className="range-pair">
        <input aria-label={`${spec.label} minimum`} type={type} value={current[0] ?? ""} onChange={e => onChange([e.target.value, current[1]])} />
        <input aria-label={`${spec.label} maximum`} type={type} value={current[1] ?? ""} onChange={e => onChange([current[0], e.target.value])} />
      </div>
    );
  }
  return <input aria-label={spec.label} value={value ?? ""} onChange={e => onChange(e.target.value || null)} placeholder="Exact value" />;
}

export default function DataExplorer() {
  const [schema, setSchema] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [selected, setSelected] = useState(["grandPrixYear", "grandPrixName", "resultsDriverName", "constructorName"]);
  const [values, setValues] = useState({});
  const [query, setQuery] = useState("");
  const [result, setResult] = useState({ rows: [], columns: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hasAppliedFilters, setHasAppliedFilters] = useState(false);

  useEffect(() => {
    api.get("/api/data-explorer/schema").then(r => {
      setSchema(r.filters);
      let saved = null;
      try { saved = JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null"); } catch { /* ignore invalid session state */ }
      if (saved?.applied) {
        setValues(saved.values || {});
        setHasAppliedFilters(true);
        setShowFilters(true);
        runQuery(saved.filters, preferredColumns);
      } else {
        setLoading(false);
      }
    }).catch(e => { setError(e); setLoading(false); });
    // runQuery intentionally omitted: it is recreated on every render and we
    // only want this effect to run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const byName = useMemo(() => Object.fromEntries(schema.map(s => [s.column, s])), [schema]);
  const available = useMemo(() => schema.filter(s =>
    `${s.column} ${s.label}`.toLowerCase().includes(query.toLowerCase())
  ), [schema, query]);

  function activeFilters(nextValues = values) {
    return Object.entries(nextValues).flatMap(([column, value]) => {
      if (value == null || value === "") return [];
      const spec = byName[column];
      if (!spec) return [];
      let normalized = value;
      if (spec.kind === "range") normalized = value.map(Number);
      return [{ column, kind: spec.kind, value: normalized }];
    });
  }

  async function runQuery(filters = activeFilters(), columns = preferredColumns) {
    setLoading(true); setError(null);
    try {
      const body = { filters, columns, sort: ["grandPrixYear", "resultsFinalPositionNumber"], descending: true, offset: 0, limit: 500 };
      const r = /** @type {{ total: number, columns: string[], rows: Array<Record<string, any>> }} */ (
        await api.post("/api/data-explorer/query", body)
      );
      setResult(r);
    } catch (e) { setError(e); }
    finally { setLoading(false); }
  }

  function toggleColumn(column) {
    setSelected(prev => prev.includes(column) ? prev.filter(x => x !== column) : [...prev, column]);
  }

  function clear() {
    setValues({});
    setHasAppliedFilters(false);
    sessionStorage.removeItem("f1analysis.filters");
    runQuery([], preferredColumns);
  }

  function apply() {
    const filters = activeFilters();
    sessionStorage.setItem("f1analysis.filters", JSON.stringify({ applied: true, filters, values }));
    setHasAppliedFilters(true);
    runQuery(filters, preferredColumns);
  }

  function toggleFilters(enabled) {
    setShowFilters(enabled);
    if (!enabled) {
      setValues({});
      setHasAppliedFilters(false);
      setResult({ rows: [], columns: [], total: 0 });
      sessionStorage.removeItem("f1analysis.filters");
      setLoading(false);
      return;
    }
    if (schema.length) apply();
  }

  return (
    <div>
      <header className="page-header">
        <div><h1>Data Explorer</h1><p>Filter and explore F1 race data from multiple perspectives.</p></div>
        <div className="count-pill">{result.total.toLocaleString()} rows</div>
      </header>

      <label className="filter-results-toggle">
        <input type="checkbox" aria-label="Filter Results" checked={showFilters} disabled={!schema.length} onChange={event => toggleFilters(event.target.checked)} />
        Filter Results
      </label>
      {!showFilters && error && <Status error={error} />}

      {showFilters && <div className="explorer-grid">
        <Card title="Filters" className="filter-card">
          <input className="search" aria-label="Find a filter field" placeholder="Find one of the dataset fields…" value={query} onChange={e => setQuery(e.target.value)} />
          <div className="filter-list">
            {available.map(spec => (
              <div className="filter-row" key={spec.column}>
                <label><input type="checkbox" checked={selected.includes(spec.column)} onChange={() => toggleColumn(spec.column)} /> {spec.label}</label>
                {selected.includes(spec.column) && (
                  <FilterEditor spec={spec} value={values[spec.column]} onChange={v => setValues(x => ({ ...x, [spec.column]: v }))} />
                )}
              </div>
            ))}
          </div>
          <div className="button-row">
            <button className="primary" onClick={apply}>Apply filters</button>
            <button onClick={clear}>Reset</button>
          </div>
        </Card>

        <Card title="Filtered Results">
          <Status loading={loading} error={error}>
            {result.rows?.length ? (
              <DataTable rows={result.rows} columns={result.columns} />
            ) : (
              <div className="empty">
                <p>No rows match the current filters.</p>
                {hasAppliedFilters && (
                  <button onClick={clear}>Reset filters</button>
                )}
              </div>
            )}
          </Status>
        </Card>
      </div>}
    </div>
  );
}
