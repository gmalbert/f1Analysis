import {useEffect, useId, useMemo, useState} from 'react';
import {ViewTable} from '../components/ViewTable';
import {displayCell} from '../components/Presentation';

const driverKeys = ['resultsDriverName','driverName','Driver'];
const FEATURES_ENABLED = import.meta.env.VITE_F1_ENHANCEMENTS !== '0';
const outcomeKeys = ['resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained','DNF'];
const tireKeys = ['Avg Deg (s/lap)','Start Compound','Stints','Avg Stint (laps)','Max Stint (laps)','Avg Stints','Soft Lap %','Laps','Races'];
const tireLabels = {'Avg Deg (s/lap)': 'Average tire degradation (s/lap)'};
const knownDNF = value => [true, false, 1, 0, 'true', 'false', '1', '0'].includes(typeof value === 'string' ? value.trim().toLowerCase() : value);

export function driverComparison(node) {
  const driverIndex = node.columns.findIndex(column => driverKeys.includes(column.key));
  if (driverIndex < 0) return null;
  const choices = [...new Set(node.rows.map(row => row[driverIndex]).filter(value => typeof value === 'string' && value.trim()))].sort();
  const tire = node.columns.some(column => column.key === 'Avg Deg (s/lap)');
  const fields = (tire ? tireKeys : outcomeKeys).map(key => ({key, index: node.columns.findIndex(column => column.key === key)})).filter(field => field.index >= 0);
  const useful = fields.some(field => node.rows.some(row => field.key === 'DNF' ? knownDNF(row[field.index]) : typeof row[field.index] === 'number' && Number.isFinite(row[field.index])));
  if (choices.length < 2 || !useful) return null;
  const counts = new Map();
  for (const row of node.rows) counts.set(row[driverIndex], (counts.get(row[driverIndex]) || 0) + 1);
  return {driverIndex, choices, fields, tire, singleRow: choices.every(driver => counts.get(driver) === 1)};
}

export function EnhancedTable({node, context = null, chartLinked = false, onSelectionChange = undefined}) {
  const [mode, setMode] = useState('grid'), [compare, setCompare] = useState(false);
  const comparisonId = useId();
  const comparison = useMemo(() => driverComparison(node), [node]);
  if (!FEATURES_ENABLED) return <ViewTable node={node}/>;
  function toggleComparison() {
    setCompare(!compare);
    if (compare) onSelectionChange?.([]);
  }
  return <section aria-label="Table display">
    <div className="enhancement-bar">
      <button aria-pressed={mode === 'grid'} onClick={() => setMode('grid')}>Interactive grid</button>
      <button aria-pressed={mode === 'accessible'} onClick={() => setMode('accessible')}>Accessible table</button>
      {comparison && <button aria-expanded={compare} aria-controls={comparisonId} onClick={toggleComparison}>Compare drivers</button>}
    </div>
    {mode === 'grid' ? <ViewTable node={node}/> : <AccessibleTable node={node}/>}
    {compare && comparison && <div id={comparisonId}><DriverComparison node={node} comparison={comparison} context={context} chartLinked={chartLinked} onSelectionChange={onSelectionChange}/></div>}
  </section>;
}

function AccessibleTable({node}) {
  const columns = node.columns, rows = node.rows;
  const [selected, setSelected] = useState(() => columns.slice(0,8).map((_,i) => i));
  const [query, setQuery] = useState(''), [page, setPage] = useState(0);
  const visible = selected.filter(i => columns[i]);
  const matches = useMemo(() => rows.map((_,i) => i).filter(i =>
    !query || rows[i].some(value => String(value ?? 'None').toLowerCase().includes(query.toLowerCase()))
  ), [rows,query]);
  const pageCount = Math.max(1, Math.ceil(matches.length/50)), current = Math.min(page,pageCount-1);
  function toggle(index) {setSelected(old => old.includes(index) ? old.filter(i => i !== index) : [...old,index].sort((a,b) => a-b));}
  return <div className="accessible-table">
    <label>Search all fields <input value={query} onChange={e => {setQuery(e.target.value);setPage(0);}}/></label>
    <details><summary>Choose fields ({visible.length} of {columns.length})</summary>
      <div className="columns-list">{columns.map((column,index) => <label key={index}><input type="checkbox" checked={visible.includes(index)} disabled={visible.length === 1 && visible[0] === index} onChange={() => toggle(index)}/>{column.label} ({column.key})</label>)}</div>
    </details>
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- A focusable overflow region lets keyboard users scroll wide tables. */}
    <div className="table-viewport" tabIndex={0} role="region" aria-label="Scrollable analysis table"><table>
      <caption>{matches.length.toLocaleString()} matching rows · showing rows {matches.length ? current*50+1 : 0}–{Math.min((current+1)*50,matches.length)}. All fields are available in Choose fields.</caption>
      <thead><tr>{!node.hide_index && <th scope="col">{node.index_name || 'Row'}</th>}{visible.map(index => <th scope="col" key={index}>{columns[index].label}</th>)}</tr></thead>
      <tbody>{matches.slice(current*50,(current+1)*50).map(row => <tr key={row}>
        {!node.hide_index && <th scope="row">{String(node.index?.[row] ?? row)}</th>}
        {visible.map(index => <td key={index}>{displayCell(rows[row][index],columns[index],node.display?.[row]?.[index])}</td>)}
      </tr>)}</tbody>
    </table></div>
    {!matches.length && <p role="status">No rows match this search.</p>}
    <nav aria-label="Table row pages"><button disabled={!current} onClick={() => setPage(current-1)}>Previous rows</button> Page {current+1} of {pageCount} <button disabled={current+1 >= pageCount} onClick={() => setPage(current+1)}>Next rows</button></nav>
    <p>Use Interactive grid for the original sorting, selection, copying and full CSV export.</p>
  </div>;
}

function DriverComparison({node, comparison, context, chartLinked, onSelectionChange}) {
  const {driverIndex, choices, fields, tire, singleRow} = comparison;
  const [drivers, setDrivers] = useState([]);
  const selectedDrivers = useMemo(() => drivers.filter(driver => choices.includes(driver)), [drivers, choices]);
  useEffect(() => {onSelectionChange?.(selectedDrivers);}, [selectedDrivers, onSelectionChange]);
  const summaries = useMemo(() => selectedDrivers.map(driver => {
    const sample = node.rows.filter(row => row[driverIndex] === driver);
    return {driver,rows:sample.length,values:fields.map(field => {
      const values = sample.map(row => row[field.index]);
      if (singleRow) {
        const value = values[0];
        if (value == null) return 'No data';
        if (field.key === 'DNF') return knownDNF(value) ? value === true || value === 1 || ['true', '1'].includes(String(value).trim().toLowerCase()) ? 'Yes' : 'No' : 'No data';
        return displayCell(value, node.columns[field.index]);
      }
      if (field.key === 'DNF') {
        const known = values.filter(knownDNF);
        return known.length ? (100*known.filter(value => value === true || value === 1 || ['true', '1'].includes(String(value).trim().toLowerCase())).length/known.length).toFixed(1)+'%' : 'No data';
      }
      if (field.key === 'Start Compound') return [...new Set(values.filter(value => typeof value === 'string' && value.trim()))].join(', ') || 'No data';
      const known = values.filter(v => typeof v === 'number' && Number.isFinite(v));
      return known.length ? (known.reduce((a,b) => a+b,0)/known.length).toFixed(2) : 'No data';
    })};
  }), [selectedDrivers,node,driverIndex,fields,singleRow]);
  const annual = tire && fields.some(field => field.key === 'Races');
  const scope = tire ? annual ? `Season tire summaries${context?.year ? ' for ' + context.year : ''}` : `Race tire strategy${context?.event ? ' — ' + context.event : ''}${context?.year ? ' ' + context.year : ''}` : 'Current table and applied filters';
  function fieldLabel(field) {
    const label = tireLabels[field.key] || node.columns[field.index].label;
    return singleRow || field.key === 'Start Compound' ? label : field.key === 'DNF' ? 'DNF rate among known records' : 'Mean ' + label;
  }
  return <div className="accessible-table">
    <h3>{tire ? 'Compare tire strategy' : 'Compare race results'}</h3>
    <p>{scope}. {singleRow ? 'Compare the displayed values for each selected driver.' : 'Compare averages across the records included in this table.'} {chartLinked && 'The chart below uses the same selected drivers. Clear the selection to show all drivers again.'}</p>
    <fieldset className="driver-picker"><legend>Choose up to four drivers ({choices.length} available)</legend><div className="columns-list">{choices.map(driver => <label key={driver}><input type="checkbox" checked={selectedDrivers.includes(driver)} disabled={!selectedDrivers.includes(driver) && selectedDrivers.length >= 4} onChange={() => setDrivers(old => old.includes(driver) ? old.filter(d => d !== driver && choices.includes(d)) : [...old.filter(d => choices.includes(d)),driver])}/>{driver}</label>)}</div></fieldset>
    <p role="status">{selectedDrivers.length ? `Comparing ${selectedDrivers.length} selected driver${selectedDrivers.length === 1 ? '' : 's'}: ${selectedDrivers.join(', ')}.` : 'Choose drivers above to see their comparison.'}</p>
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- A focusable overflow region lets keyboard users scroll comparison columns. */}
    <div className="table-viewport" tabIndex={0} role="region" aria-label="Scrollable driver comparison"><table><caption>{selectedDrivers.length ? 'Selected drivers — ' + scope : 'Driver comparison'}</caption><thead><tr><th scope="col">Driver</th>{!singleRow && <th scope="col">Records included</th>}{fields.map(f => <th scope="col" key={f.key}>{fieldLabel(f)}</th>)}</tr></thead>
      <tbody>{summaries.map(summary => <tr key={summary.driver}><th scope="row">{summary.driver}</th>{!singleRow && <td>{summary.rows}</td>}{summary.values.map((value,i) => <td key={i}>{value}</td>)}</tr>)}</tbody>
    </table></div>
  </div>;
}
