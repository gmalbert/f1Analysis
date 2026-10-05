import {useMemo, useState} from 'react';
import {ViewTable} from '../components/ViewTable';
import {displayCell} from '../components/Presentation';

const driverKeys = ['resultsDriverName','driverName','Driver'];

export function EnhancedTable({node}) {
  const [mode, setMode] = useState('grid'), [compare, setCompare] = useState(false);
  if (import.meta.env.VITE_F1_ENHANCEMENTS !== '1') return <ViewTable node={node}/>;
  const hasDrivers = node.columns.some(c => driverKeys.includes(c.key));
  return <section aria-label="Table display">
    <div className="enhancement-bar">
      <button aria-pressed={mode === 'grid'} onClick={() => setMode('grid')}>Interactive grid</button>
      <button aria-pressed={mode === 'accessible'} onClick={() => setMode('accessible')}>Accessible table</button>
      {hasDrivers && <button aria-expanded={compare} onClick={() => setCompare(s => !s)}>Compare drivers</button>}
    </div>
    {mode === 'grid' ? <ViewTable node={node}/> : <AccessibleTable node={node}/>}
    {compare && <DriverComparison node={node}/>}
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
      <div className="columns-list">{columns.map((column,index) => <label key={index}><input type="checkbox" checked={visible.includes(index)} onChange={() => toggle(index)}/>{column.label} ({column.key})</label>)}</div>
    </details>
    <div className="table-viewport"><table>
      <caption>{matches.length.toLocaleString()} matching rows · showing rows {matches.length ? current*50+1 : 0}–{Math.min((current+1)*50,matches.length)}. All fields are available in Choose fields.</caption>
      <thead><tr>{!node.hide_index && <th scope="col">{node.index_name || 'Row'}</th>}{visible.map(index => <th scope="col" key={index}>{columns[index].label}</th>)}</tr></thead>
      <tbody>{matches.slice(current*50,(current+1)*50).map(row => <tr key={row}>
        {!node.hide_index && <th scope="row">{String(node.index?.[row] ?? row)}</th>}
        {visible.map(index => <td key={index}>{displayCell(rows[row][index],columns[index],node.display?.[row]?.[index])}</td>)}
      </tr>)}</tbody>
    </table></div>
    <nav aria-label="Table row pages"><button disabled={!current} onClick={() => setPage(current-1)}>Previous rows</button> Page {current+1} of {pageCount} <button disabled={current+1 >= pageCount} onClick={() => setPage(current+1)}>Next rows</button></nav>
    <p>Use Interactive grid for the original sorting, selection, copying and full CSV export.</p>
  </div>;
}

function DriverComparison({node}) {
  const driverIndex = node.columns.findIndex(c => driverKeys.includes(c.key));
  const [drivers, setDrivers] = useState([]);
  const choices = useMemo(() => [...new Set(node.rows.map(row => row[driverIndex]).filter(Boolean))].sort(), [node.rows,driverIndex]);
  const fields = useMemo(() => ['resultsStartingGridPositionNumber','resultsFinalPositionNumber','positionsGained','DNF']
    .map(key => ({key,index:node.columns.findIndex(c => c.key === key)})).filter(f => f.index >= 0), [node.columns]);
  const summaries = useMemo(() => drivers.map(driver => {
    const sample = node.rows.filter(row => row[driverIndex] === driver);
    return {driver,rows:sample.length,values:fields.map(field => {
      const values = sample.map(row => row[field.index]);
      if (field.key === 'DNF') {
        const known = values.filter(v => v != null);
        return known.length ? (100*known.filter(v => v === true || v === 1 || String(v).toLowerCase() === 'true').length/known.length).toFixed(1)+'%' : 'No data';
      }
      const known = values.filter(v => typeof v === 'number' && Number.isFinite(v));
      return known.length ? (known.reduce((a,b) => a+b,0)/known.length).toFixed(2) : 'No data';
    })};
  }), [drivers,node.rows,driverIndex,fields]);
  return <div className="accessible-table">
    <h3>Driver comparison within this table</h3>
    <p>Historical descriptive averages over the currently filtered rows. Sample rows can differ from unique races. These are not forecasts or calibrated probabilities.</p>
    <div className="columns-list">{choices.map(driver => <label key={driver}><input type="checkbox" checked={drivers.includes(driver)} disabled={!drivers.includes(driver) && drivers.length >= 4} onChange={() => setDrivers(old => old.includes(driver) ? old.filter(d => d !== driver) : [...old,driver])}/>{driver}</label>)}</div>
    <table><caption>Compare up to four drivers</caption><thead><tr><th scope="col">Driver</th><th scope="col">Sample rows</th>{fields.map(f => <th scope="col" key={f.key}>{f.key === 'DNF' ? 'DNF rate among known rows' : 'Mean '+node.columns[f.index].label}</th>)}</tr></thead>
      <tbody>{summaries.map(summary => <tr key={summary.driver}><th scope="row">{summary.driver}</th><td>{summary.rows}</td>{summary.values.map((value,i) => <td key={i}>{value}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
