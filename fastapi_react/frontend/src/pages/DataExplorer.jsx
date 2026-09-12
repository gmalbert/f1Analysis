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
      <select value={value ?? ""} onChange={e => onChange(e.target.value === "" ? null : e.target.value === "true")}>
        <option value="">All</option><option value="true">True / 1</option><option value="false">False / 0</option>
      </select>
    );
  }
  if (spec.kind === "exact" && spec.options) {
    return (
      <select value={value ?? ""} onChange={e => onChange(e.target.value || null)}>
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
        <input type={type} value={current[0] ?? ""} onChange={e => onChange([e.target.value, current[1]])} />
        <input type={type} value={current[1] ?? ""} onChange={e => onChange([current[0], e.target.value])} />
      </div>
    );
  }
  return <input value={value ?? ""} onChange={e => onChange(e.target.value || null)} placeholder="Exact value" />;
}

export default function DataExplorer() {
  const [schema, setSchema] = useState([]);
  const [selected, setSelected] = useState(["grandPrixYear", "grandPrixName", "resultsDriverName", "constructorName"]);
  const [values, setValues] = useState({});
  const [query, setQuery] = useState("");
  const [result, setResult] = useState({ rows: [], columns: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.get("/api/data-explorer/schema").then(r => {
      setSchema(r.filters);
      setLoading(false);
      runQuery([], preferredColumns);
    }).catch(e => { setError(e); setLoading(false); });
    // runQuery intentionally omitted: it is recreated on every render and we
    // only want this effect to run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const byName = useMemo(() => Object.fromEntries(schema.map(s => [s.column, s])), [schema]);
  const available = useMemo(() => schema.filter(s => s.column.toLowerCase().includes(query.toLowerCase())).slice(0, 120), [schema, query]);

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
      const r = await api.post("/api/data-explorer/query", body);
      setResult(r);
    } catch (e) { setError(e); }
    finally { setLoading(false); }
  }

  function toggleColumn(column) {
    setSelected(prev => prev.includes(column) ? prev.filter(x => x !== column) : [...prev, column]);
  }

  function clear() {
    setValues({});
    runQuery([], preferredColumns);
  }

  return (
    <div>
      <header className="page-header">
        <div><h1>Data Explorer</h1><p>Filter and explore the same wide F1 analysis dataset used by the Streamlit application.</p></div>
        <div className="count-pill">{result.total.toLocaleString()} rows</div>
      </header>

      <div className="explorer-grid">
        <Card title="Filters" className="filter-card">
          <input className="search" placeholder="Find one of the dataset fields…" value={query} onChange={e => setQuery(e.target.value)} />
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
            <button className="primary" onClick={() => runQuery(activeFilters(), preferredColumns)}>Apply filters</button>
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
                {Object.keys(values).length > 0 && (
                  <button onClick={clear}>Reset filters</button>
                )}
              </div>
            )}
          </Status>
        </Card>
      </div>
    </div>
  );
}
