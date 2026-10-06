import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import DataEditor, {GridCellKind} from '@glideapps/glide-data-grid';
import '@glideapps/glide-data-grid/dist/index.css';
import {displayCell} from './Presentation';
import {useTheme} from './useTheme';

const empty=[];
const DEFAULT_COLUMN_WIDTH=150;
const estimateColumnWidth=(column,index,rows)=>{
  const longest=rows.reduce((max,row)=>Math.max(max,String(row[index]??'').length),String(column.label??'').length);
  const minimum=column.kind==='NumberColumn'?72:96;
  return Math.min(320,Math.max(minimum,Math.ceil(longest*6+24)));
};
const quote=value=>{const text=value==null?'':String(value);return /[,"\r\n]/.test(text)?`"${text.replaceAll('"','""')}"`:text;};
function csvValue(value,column){
  if(value==null)return '';
  if(column.kind==='CheckboxColumn')return Boolean(value);
  if(column.kind==='DateColumn')return String(value).slice(0,10);
  if(column.kind==='TimeColumn'){
    const parts=String(value).split(':');
    return `${parts[0].padStart(2,'0')}:${(parts[1] || '00').padStart(2,'0')}:${Number(parts[2] || 0).toFixed(3).padStart(6,'0')}`;
  }
  if(column.format==='%d' && typeof value==='number')return Math.trunc(value);
  return value;
}

/** The same canvas grid used by the reference, including keyboard selection,
 * copying ranges, search, scrolling, overlays and column resizing. */
export function ViewTable({node}) {
  const rows=node.rows || empty;
  const columns=node.columns || empty;
  const [sort,setSort]=useState(null);
  const [widths,setWidths]=useState(/** @type {Record<string,number>} */ ({}));
  const [hidden,setHidden]=useState([]);
  const [showColumns,setShowColumns]=useState(false);
  const [showSearch,setShowSearch]=useState(false);
  const [menu,setMenu]=useState(null);
  const [pinned,setPinned]=useState([]);
  const [formats,setFormats]=useState(/** @type {Record<string,string>} */ ({}));
  const outer=useRef(null);
  const [availableWidth,setAvailableWidth]=useState(0);
  useEffect(()=>{
    const parent=outer.current?.parentElement;
    if(!parent)return;
    const measure=()=>setAvailableWidth(parent.clientWidth);
    measure();
    const observer=new ResizeObserver(measure);
    observer.observe(parent);
    return()=>observer.disconnect();
  },[]);
  useEffect(()=>{
    if(!menu && !showColumns)return;
    const close=event=>{if(event.type==='keydown' && event.key==='Escape' || event.type==='pointerdown' && !outer.current?.contains(event.target)){setMenu(null);setShowColumns(false);}};
    document.addEventListener('pointerdown',close);document.addEventListener('keydown',close);
    return()=>{document.removeEventListener('pointerdown',close);document.removeEventListener('keydown',close);};
  },[menu,showColumns]);
  const indices=useMemo(()=>{
    const result=rows.map((_,i)=>i);
    if(sort)result.sort((a,b)=>{
      const x=rows[a][sort.column],y=rows[b][sort.column];
      if(x==null || y==null)return x==null?(y==null?0:1):-1;
      const cmp=typeof x==='number' && typeof y==='number'?x-y:String(x).localeCompare(String(y));
      return sort.desc?-cmp:cmp;
    });
    return result;
  },[rows,sort]);
  const visible=useMemo(()=>columns.map((column,index)=>({column,index})).filter(c=>!hidden.includes(c.index)).sort((a,b)=>Number(pinned.includes(b.index))-Number(pinned.includes(a.index))),[columns,hidden,pinned]);
  const gridColumns=useMemo(()=>{
    const result=visible.map(({column,index})=>({id:String(index),title:column.label+(sort?.column===index?(sort.desc?' ↓':' ↑'):''),hasMenu:true,width:widths[index] || Math.max(typeof column.width==='number'?column.width:0,estimateColumnWidth(column,index,rows))}));
    if(!node.hide_index)result.unshift({id:'index',title:node.index_name || '',width:widths.index || 80});
    return result;
  },[visible,widths,sort,node.hide_index,node.index_name,rows]);
  const contentWidth=gridColumns.reduce((total,column)=>total+(widths[column.id] || column.width || DEFAULT_COLUMN_WIDTH),0);
  const tableWidth=Math.min(contentWidth,availableWidth || Infinity,typeof node.width==='number'?node.width:Infinity);
  const dark=useTheme()==='dark';
  const bg=dark?'#0e1117':'#fff',text=dark?'#fafafa':'#31333f';
  /** @type {(cell: import('@glideapps/glide-data-grid').Item) => import('@glideapps/glide-data-grid').GridCell} */
  const getCell=useCallback(([col,row])=>{
    const rowIndex=indices[row];
    if(rowIndex==null)return {kind:GridCellKind.Text,data:'',displayData:'',allowOverlay:false};
    if(!node.hide_index && col===0)return {kind:GridCellKind.Text,data:String(node.index?.[rowIndex]??rowIndex),displayData:String(node.index?.[rowIndex]??rowIndex),allowOverlay:true,readonly:true,themeOverride:{bgCell:dark?'#262730':'#f7f9fc',textDark:dark?'#bfc2ce':'#808495'}};
    const selected=visible[col-(node.hide_index?0:1)];
    if(!selected)return {kind:GridCellKind.Text,data:'',displayData:'',allowOverlay:false};
    const {column,index}=selected;
    const value=rows[rowIndex][index],style=node.styles?.[rowIndex]?.[index];
    const themeOverride={bgCell:style?.['background-color'] || bg,textDark:style?.color || text};
    if(column.kind==='CheckboxColumn')return {kind:GridCellKind.Boolean,data:value==null?null:Boolean(value),allowOverlay:false,readonly:true,maxSize:16,themeOverride};
    const displayData=displayCell(value,formats[index]?{...column,format:formats[index]}:column,node.display?.[rowIndex]?.[index]);
    if(typeof value==='number')return {kind:GridCellKind.Number,data:value,displayData,allowOverlay:true,readonly:true,contentAlign:'right',themeOverride};
    return {kind:GridCellKind.Text,data:value==null?'':String(value),displayData,allowOverlay:true,readonly:true,style:value==null?'faded':'normal',themeOverride};
  },[indices,visible,node,rows,dark,bg,text,formats]);
  function download(){
    const header=visible.map(c=>quote(c.column.key));
    if(!node.hide_index)header.unshift(quote(node.index_name || ''));
    const lines=indices.map(i=>{
      const values=visible.map(c=>quote(csvValue(rows[i][c.index],formats[c.index]?{...c.column,format:formats[c.index]}:c.column)));
      if(!node.hide_index)values.unshift(quote(node.index?.[i]??i));
      return values.join(',');
    });
    const csv='\ufeff'+[header.join(','),...lines].join('\r\n')+'\r\n';
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download=`${new Date().toISOString().slice(0,16).replace(':','-')}_export.csv`;link.click();URL.revokeObjectURL(url);
  }
  const height=node.height || Math.min(400,(rows.length+1)*35+3);
  const menuColumn=menu?columns[menu.index]:null;
  const numeric=menuColumn?.kind==='NumberColumn';
  return <div className="view-table canvas-table" ref={outer} style={{width:tableWidth,maxWidth:'100%'}}>
    <div className="table-toolbar">
      <button aria-label="Search table" title="Search" onClick={()=>setShowSearch(s=>!s)}>⌕</button>
      <button aria-label="Show or hide columns" title="Columns" onClick={()=>setShowColumns(s=>!s)}>▥</button>
      <button aria-label="Download table as CSV" title="Download CSV" onClick={download}>⇩</button>
      <button aria-label="Fullscreen table" title="Fullscreen" onClick={()=>document.fullscreenElement?document.exitFullscreen():outer.current?.requestFullscreen?.()}>⛶</button>
    </div>
    {showColumns && <div className="column-picker">{columns.map((col,i)=><label key={i}><input type="checkbox" checked={!hidden.includes(i)} onChange={()=>setHidden(h=>h.includes(i)?h.filter(x=>x!==i):[...h,i])}/>{col.label}</label>)}</div>}
    {menuColumn && <div className="grid-column-menu" role="group" aria-label={`${menuColumn.label} column options`} style={{left:menu.left}}><strong>{menuColumn.label}</strong><button onClick={()=>{setSort({column:menu.index,desc:false});setMenu(null);}}>Sort ascending</button><button onClick={()=>{setSort({column:menu.index,desc:true});setMenu(null);}}>Sort descending</button><button onClick={()=>{setSort(null);setMenu(null);}}>Clear sorting</button><button onClick={()=>{setPinned(p=>p.includes(menu.index)?p.filter(i=>i!==menu.index):[...p,menu.index]);setMenu(null);}}>{pinned.includes(menu.index)?'Unpin column':'Pin column'}</button><button onClick={()=>{setHidden(h=>[...h,menu.index]);setMenu(null);}}>Hide column</button>{numeric && <label>Number format<select value={formats[menu.index] || ''} onChange={event=>setFormats(f=>({...f,[menu.index]:event.target.value}))}><option value="">Default</option><option value="%d">Integer</option>{[1,2,3,4].map(n=><option key={n} value={`%.${n}f`}>{n} decimal places</option>)}<option value="percent">Percent</option></select></label>}<small>{rows.length.toLocaleString()} rows · {rows.filter(r=>r[menu.index]==null).length.toLocaleString()} missing · {new Set(rows.map(r=>r[menu.index])).size.toLocaleString()} unique</small></div>}
    <DataEditor columns={gridColumns} rows={rows.length} getCellContent={getCell} getCellsForSelection={true} width={tableWidth} height={height} rowHeight={35} headerHeight={35} rowMarkers="none" minColumnWidth={50} maxColumnWidth={500} freezeColumns={pinned.length+(node.hide_index?0:1)} showSearch={showSearch} onSearchClose={()=>setShowSearch(false)} onHeaderMenuClick={(col,bounds)=>{const selected=visible[col-(node.hide_index?0:1)];if(selected)setMenu({index:selected.index,left:Math.max(0,Math.min(bounds.x-(outer.current?.getBoundingClientRect().x || 0),(outer.current?.clientWidth || 240)-240))});}} onColumnResize={(column,width)=>setWidths(old=>({...old,[column.id]:width}))} onColumnResizeEnd={(column,width)=>setWidths(old=>({...old,[column.id]:width}))} onHeaderClicked={col=>{const selected=visible[col-(node.hide_index?0:1)];if(selected)setSort(s=>s?.column===selected.index?(s.desc?null:{...s,desc:true}):{column:selected.index,desc:false});}} theme={{fontFamily:'Source Sans',baseFontStyle:'13px',headerFontStyle:'13px',cellHorizontalPadding:8,cellVerticalPadding:3,bgCell:bg,bgHeader:dark?'#262730':'#f7f9fc',bgHeaderHovered:dark?'#3a3d46':'#eff1f6',bgHeaderHasFocus:dark?'#3a3d46':'#eff1f6',textDark:text,textHeader:dark?'#bfc2ce':'#555965',textMedium:text,textLight:'#808495',borderColor:dark?'#3a3d46':'#e6e7eb',accentColor:'#ff4b4b',accentLight:dark?'#ff4b4b33':'#ff4b4b1a',accentFg:'#fff',headerBottomBorderColor:dark?'#3a3d46':'#d6d8df',roundingRadius:0}} />
  </div>;
}
