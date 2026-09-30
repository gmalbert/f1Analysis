import { useEffect, useMemo, useState } from "react";
import { api } from "../api";

function FilterControl({ spec, value, onChange }) {
  if (spec.kind === "boolean") {
    return (
      <label className="streamlit-checkbox">
        <input
          type="checkbox"
          aria-label={spec.label}
          checked={value === true}
          onChange={event => onChange(event.target.checked ? true : null)}
        />
        <span>{spec.label}</span>
      </label>
    );
  }

  if (spec.kind === "exact" && spec.options) {
    return (
      <label>
        <span>{spec.label}</span>
        <select aria-label={spec.label} value={value ?? ""} onChange={event => onChange(event.target.value || null)}>
          <option value=""> All</option>
          {spec.options.map(option => <option key={option} value={option}>{option}</option>)}
        </select>
      </label>
    );
  }

  if (spec.kind === "range") {
    const low = Number(Array.isArray(value) ? value[0] : spec.min);
    const high = Number(Array.isArray(value) ? value[1] : spec.max);
    const min = Math.trunc(Number(spec.min));
    const max = Math.trunc(Number(spec.max));
    const currentLow = Math.trunc(Number.isFinite(low) ? low : min);
    const currentHigh = Math.trunc(Number.isFinite(high) ? high : max);
    return (
      <label>
        <span>{spec.label}</span>
        <div className="streamlit-range">
          <div className="streamlit-range-values" aria-hidden="true">
            <span>{currentLow}</span><span>{currentHigh}</span>
          </div>
          <div className="streamlit-range-track" aria-hidden="true" />
          <input
            aria-label={`${spec.label} minimum`}
            type="range"
            min={min}
            max={max}
            step="1"
            value={currentLow}
            onChange={event => onChange([Math.min(Number(event.target.value), currentHigh), currentHigh])}
          />
          <input
            aria-label={`${spec.label} maximum`}
            type="range"
            min={min}
            max={max}
            step="1"
            value={currentHigh}
            onChange={event => onChange([currentLow, Math.max(Number(event.target.value), currentLow)])}
          />
        </div>
      </label>
    );
  }

  if (spec.kind === "date_range") {
    const current = Array.isArray(value) ? value : [spec.min, spec.max];
    return (
      <label>
        <span>{spec.label}</span>
        <div className="range-pair">
          <input aria-label={`${spec.label} minimum`} type="date" value={current[0] ?? ""} onChange={event => onChange([event.target.value, current[1]])} />
          <input aria-label={`${spec.label} maximum`} type="date" value={current[1] ?? ""} onChange={event => onChange([current[0], event.target.value])} />
        </div>
      </label>
    );
  }

  return (
    <label>
      <span>{spec.label}</span>
      <input aria-label={spec.label} value={value ?? ""} onChange={event => onChange(event.target.value || null)} />
    </label>
  );
}

export default function FilterSidebar() {
  const [schema, setSchema] = useState([]);
  const [values, setValues] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem("f1analysis.filters") || "null")?.values || {}; }
    catch { return {}; }
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    api.get("/api/data-explorer/schema")
      .then(response => setSchema(response.filters || []))
      .catch(setError);
  }, []);

  const byName = useMemo(() => Object.fromEntries(schema.map(spec => [spec.column, spec])), [schema]);

  function activeFilters(nextValues) {
    return Object.entries(nextValues).flatMap(([column, value]) => {
      if (value == null || value === "" || (Array.isArray(value) && value.some(item => item === ""))) return [];
      const spec = byName[column];
      if (!spec) return [];
      let normalized = value;
      if (spec.kind === "range") normalized = value.map(Number);
      return [{ column, kind: spec.kind, value: normalized }];
    });
  }

  function apply(column, value) {
    const next = { ...values, [column]: value };
    setValues(next);
    const filters = activeFilters(next);
    sessionStorage.setItem("f1analysis.filters", JSON.stringify({ applied: true, filters, values: next }));
    window.dispatchEvent(new CustomEvent("f1analysis:filters-changed"));
  }

  return (
    <aside className="global-filter-sidebar" aria-label="Select filters to apply">
      <h2>Select filters to apply:</h2>
      {error && <div className="status error">{String(error.message || error)}</div>}
      <div className="filter-list">
        {schema.map(spec => (
          <div className="filter-row" key={spec.column}>
            <FilterControl spec={spec} value={values[spec.column]} onChange={value => apply(spec.column, value)} />
          </div>
        ))}
      </div>
    </aside>
  );
}
