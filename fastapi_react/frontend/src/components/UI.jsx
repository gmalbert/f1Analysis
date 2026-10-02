/** @param {{ title?: string, children?: import("react").ReactNode, className?: string }} props */
export function Card(props = {}) {
  const { title = undefined, children = undefined, className = "" } = props;
  return (
    <section className={`card ${className}`}>
      {title && <h2>{title}</h2>}
      {children}
    </section>
  );
}

/** @param {{ loading?: boolean, error?: unknown, children?: import("react").ReactNode, onRetry?: () => void, loadingLabel?: string }} props */
export function Status(props = {}) {
  const { loading = false, error = null, children = null, onRetry = undefined, loadingLabel = "Loading…" } = props;
  if (loading) return (
    <div
      className="loading-state"
      role="status"
      aria-busy="true"
      aria-live="polite"
    >
      <span>{loadingLabel}</span>
      <span className="skeleton-line" aria-hidden="true" />
      <span className="skeleton-line short" aria-hidden="true" />
    </div>
  );
  if (error) return (
    <div
      className="status error"
      role="alert"
      aria-live="assertive"
    >
      {Number(typeof error === "object" && error && "status" in error ? error.status : 0) >= 500 ? "Server error (5xx): " : Number(typeof error === "object" && error && "status" in error ? error.status : 0) >= 400 ? "Request rejected (4xx): " : ""}
      {String(error instanceof Error ? error.message : error)}
      <button type="button" onClick={onRetry || (() => window.location.reload())}>Retry</button>
    </div>
  );
  return children || null;
}

/** @param {{ rows?: Array<Record<string, any>>, columns?: string[], maxHeight?: number, ariaLabel?: string, headerMap?: Record<string,string>, checkboxColumns?: string[] }} props */
/* eslint-disable jsx-a11y/no-noninteractive-tabindex */
export function DataTable(props = {}) {
  const { rows = [], columns = undefined, maxHeight = 560, ariaLabel = undefined, headerMap = {}, checkboxColumns = [] } = props;
  if (!rows?.length) return <div className="empty">No rows available.</div>;
  const cols = columns?.length ? columns : Object.keys(rows[0] || {});
  const landmarkProps = ariaLabel ? { role: "region", "aria-label": ariaLabel } : {};
  return (
    <div className="table-wrap" {...landmarkProps} tabIndex={0} style={{ maxHeight }}>
      <table>
        <thead><tr>{cols.map(c => <th key={c} scope="col">{headerMap[c] || c}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {cols.map(c => <td key={c}>{checkboxColumns.includes(c) ? <input type="checkbox" checked={Boolean(row[c])} readOnly aria-label={`${headerMap[c] || c}: ${Boolean(row[c])}`} /> : formatCell(row[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** @param {unknown} value */
function formatCell(value) {
  if (value == null) return "";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(3).replace(/\.?0+$/, "");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** @param {{ value: unknown }} props */
export function JsonBlock({ value }) {
  return <pre className="json" tabIndex={0} aria-label="JSON data">{JSON.stringify(value, null, 2)}</pre>;
}
/* eslint-enable jsx-a11y/no-noninteractive-tabindex */

/** @param {{ label: string, value?: unknown }} props */
export function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value == null ? "—" : String(value)}</strong></div>;
}

/** @param {{ tabs: string[], active: string, onChange: (tab: string) => void }} props */
export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="subtabs" role="tablist">
      {tabs.map((tab, index) => (
        <button key={tab} type="button" role="tab" className={active === tab ? "active" : ""} aria-selected={active === tab} onClick={() => onChange(tab)} onKeyDown={event => {
          if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
          onChange(tabs[nextIndex]);
          event.currentTarget.parentElement?.querySelectorAll("button")[nextIndex]?.focus();
        }}>
          {tab}
        </button>
      ))}
    </div>
  );
}
