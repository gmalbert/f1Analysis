import React from "react";

export function Card({ title, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {title && <h3>{title}</h3>}
      {children}
    </section>
  );
}

export function Status({ loading, error, children }) {
  if (loading) return <div className="status">Loading…</div>;
  if (error) return <div className="status error">{String(error.message || error)}</div>;
  return children || null;
}

export function DataTable({ rows = [], columns, maxHeight = 560 }) {
  if (!rows?.length) return <div className="empty">No rows available.</div>;
  const cols = columns?.length ? columns : Object.keys(rows[0] || {});
  return (
    <div className="table-wrap" style={{ maxHeight }}>
      <table>
        <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {cols.map(c => <td key={c}>{formatCell(row[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value) {
  if (value == null) return "";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(3).replace(/\.?0+$/, "");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function JsonBlock({ value }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

export function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value ?? "—"}</strong></div>;
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="subtabs">
      {tabs.map(tab => (
        <button key={tab} className={active === tab ? "active" : ""} onClick={() => onChange(tab)}>
          {tab}
        </button>
      ))}
    </div>
  );
}
