import {act,fireEvent,render,screen,waitFor,within} from '@testing-library/react';
import {afterAll,afterEach,beforeAll,beforeEach,expect,it,vi} from 'vitest';
import {FeatureBar,LoadingFeedback,readOptions} from './FeatureBar';
import {EnhancedTable} from './EnhancedTable';
import {createViewClient} from './viewClient.js';
import {hasUpload,readPresets,readSharedView,safeValues,savePreset,shareUrl} from './preferences.js';

vi.mock('../components/ViewTable',()=>({ViewTable:()=> <p>Original interactive grid</p>}));
const provenance = {revision:'abcdefghijklmno',build_revision:'test',dataset:{name:'data.parquet',modified_at:'today'},models:[]};
const dialogMethods = ['showModal','close'].map(name => [name,Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype,name)]);
beforeAll(() => {
  Object.defineProperty(HTMLDialogElement.prototype,'showModal',{configurable:true,value(){this.setAttribute('open','');}});
  Object.defineProperty(HTMLDialogElement.prototype,'close',{configurable:true,value(){this.removeAttribute('open');this.dispatchEvent(new Event('close'));}});
});
afterAll(() => {for(const [name,descriptor] of dialogMethods) {if(descriptor)Object.defineProperty(HTMLDialogElement.prototype,name,descriptor);else delete HTMLDialogElement.prototype[name];}});
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('fetch',vi.fn(async() => ({ok:true,json:async() => provenance})));
});
afterEach(() => {vi.useRealTimers();vi.restoreAllMocks();vi.unstubAllEnvs();vi.unstubAllGlobals();});

it('saves and shares Unicode filters while excluding private uploaded and financial values',() => {
  const values = {filter_results_main:true,filter_driver:'José',range_filter_grandPrixYear:[2017,2026],f1bet_field_upload:{content:'private'},bankroll:5000};
  savePreset('Recent',2,values);
  expect(readPresets()[0].values).toEqual(safeValues(values));
  expect(readSharedView(new URL(shareUrl(2,values,'http://localhost/')).hash).values).toEqual(safeValues(values));
  expect(hasUpload(values)).toBe(true);
  expect(readOptions()).toEqual({design:true,cache:true});
  localStorage.setItem('f1analysis.enhancement-options','invalid');
  expect(readOptions().design).toBe(true);
  localStorage.setItem('f1analysis.enhancement-options',JSON.stringify({design:false,cache:'false'}));
  expect(readOptions()).toEqual({design:false,cache:true});
});

it('deduplicates requests, invalidates changed revisions and bypasses action caching',async() => {
  let revision = 'r1', posts = 0;
  const fetcher = async url => {
    if(url.endsWith('/status'))return {ok:true,json:async() => ({revision})};
    posts++;await new Promise(resolve => setTimeout(resolve,5));
    return {ok:true,json:async() => ({nodes:[],posts})};
  };
  const client = createViewClient({fetcher}), payload = {page:1,values:{}};
  await Promise.all([client.load(payload,{enabled:true}),client.load(payload,{enabled:true})]);
  expect(posts).toBe(1);
  await client.load(payload,{enabled:true});expect(posts).toBe(1);
  revision='r2';await client.load(payload,{enabled:true});expect(posts).toBe(2);
  await client.load({...payload,action:'explicit'},{enabled:true});expect(posts).toBe(3);
  await client.load(payload,{enabled:true});expect(posts).toBe(4);
});

it('shows all-field semantic table paging and historical comparison without grouping years',() => {
  const node = {hide_index:true,columns:[
    {key:'grandPrixYear',label:'Year',kind:'NumberColumn'},
    {key:'resultsDriverName',label:'Driver',kind:'TextColumn'},
    {key:'resultsFinalPositionNumber',label:'Finish',kind:'NumberColumn'},
    {key:'DNF',label:'DNF',kind:'CheckboxColumn'}
  ],rows:Array.from({length:70},(_,i) => [2026,i%2 ? 'Driver A' : 'Driver B',i%2 ? 2 : 4,false])};
  render(<EnhancedTable node={node}/>);
  fireEvent.click(screen.getByRole('button',{name:'Accessible table'}));
  expect(screen.getAllByRole('cell',{name:'2026'})).toHaveLength(50);
  fireEvent.click(screen.getByRole('button',{name:'Next rows'}));
  expect(screen.getAllByRole('cell',{name:'2026'})).toHaveLength(20);
  fireEvent.click(screen.getByRole('button',{name:'Compare drivers'}));
  fireEvent.click(screen.getByRole('checkbox',{name:'Driver A'}));
  expect(screen.getByRole('cell',{name:'2.00'})).toBeInTheDocument();
});

it('keeps the original interactive grid available with table tools enabled by default',() => {
  render(<EnhancedTable node={{columns:[],rows:[]}}/>);
  expect(screen.getByText('Original interactive grid')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Accessible table'})).toBeInTheDocument();
});

it('compares actual race tire metrics and synchronizes the linked chart selection without a row-count column',() => {
  const onSelectionChange=vi.fn();
  const node={hide_index:true,columns:[
    {key:'Driver',label:'Driver'}, {key:'Avg Deg (s/lap)',label:'Avg Deg (s/lap)',kind:'NumberColumn'},
    {key:'Start Compound',label:'Start Compound'}, {key:'Stints',label:'Stints',kind:'NumberColumn'},
    {key:'Avg Stint (laps)',label:'Avg Stint (laps)',kind:'NumberColumn'},
  ],rows:[['George Russell',-4.223,'MEDIUM',2,26.5],['Kimi Antonelli',-4.178,'SOFT',3,18],['Esteban Ocon',null,'HARD',2,26]]};
  render(<EnhancedTable node={node} context={{year:2025,event:'Canadian Grand Prix'}} chartLinked onSelectionChange={onSelectionChange}/>);
  fireEvent.click(screen.getByRole('button',{name:'Compare drivers'}));
  expect(screen.getByText(/Race tire strategy — Canadian Grand Prix 2025/)).toHaveTextContent('chart below uses the same selected drivers');
  expect(screen.queryByRole('columnheader',{name:'Sample rows'})).not.toBeInTheDocument();
  expect(screen.queryByRole('columnheader',{name:'Records included'})).not.toBeInTheDocument();
  for(const name of ['George Russell','Kimi Antonelli','Esteban Ocon'])fireEvent.click(screen.getByRole('checkbox',{name}));
  expect(onSelectionChange).toHaveBeenLastCalledWith(['George Russell','Kimi Antonelli','Esteban Ocon']);
  expect(screen.getByRole('status')).toHaveTextContent('Comparing 3 selected drivers');
  const russell=screen.getByRole('rowheader',{name:'George Russell'}).closest('tr');
  expect(within(russell).getByRole('cell',{name:'-4.223'})).toBeInTheDocument();
  expect(within(russell).getByRole('cell',{name:'MEDIUM'})).toBeInTheDocument();
  expect(within(screen.getByRole('rowheader',{name:'Esteban Ocon'}).closest('tr')).getByRole('cell',{name:'No data'})).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Compare drivers'}));
  expect(onSelectionChange).toHaveBeenLastCalledWith([]);
});

it('uses existing seasonal race counts and tire averages instead of counting summary rows',() => {
  const node={columns:[{key:'Driver',label:'Driver'},{key:'Avg Deg (s/lap)',label:'Avg Deg (s/lap)'},{key:'Races',label:'Races'}],rows:[['A',-.456,9],['B',-.123,8]]};
  render(<EnhancedTable node={node} context={{year:2025,event:'Canadian Grand Prix'}}/>);
  fireEvent.click(screen.getByRole('button',{name:'Compare drivers'}));
  fireEvent.click(screen.getByRole('checkbox',{name:'A'}));
  expect(screen.getByText(/^Season tire summaries for 2025/)).not.toHaveTextContent('Canadian Grand Prix');
  expect(screen.getByRole('columnheader',{name:'Races'})).toBeInTheDocument();
  expect(screen.getByRole('cell',{name:'9'})).toBeInTheDocument();
  expect(screen.getByRole('cell',{name:'-0.456'})).toBeInTheDocument();
  expect(screen.queryByRole('columnheader',{name:'Records included'})).not.toBeInTheDocument();
});

it('omits driver comparison when a table has no usable comparison metric',() => {
  const {rerender}=render(<EnhancedTable node={{columns:[{key:'Driver',label:'Driver'}],rows:[['A'],['B']]}}/>);
  expect(screen.queryByRole('button',{name:'Compare drivers'})).not.toBeInTheDocument();
  rerender(<EnhancedTable node={{columns:[{key:'Driver',label:'Driver'},{key:'resultsFinalPositionNumber',label:'Finish'}],rows:[['A',null],['B',null]]}}/>);
  expect(screen.queryByRole('button',{name:'Compare drivers'})).not.toBeInTheDocument();
});

it('renders the saved-view tools and records a named preset',async() => {
  render(<FeatureBar page={1} values={{filter_results_main:true}} options={{design:false,cache:false}} setOptions={vi.fn()} restore={vi.fn()} navigate={vi.fn()}/>);
  fireEvent.change(screen.getByRole('textbox',{name:'View name'}),{target:{value:'History'}});
  fireEvent.click(screen.getByRole('button',{name:'Save view'}));
  await waitFor(() => expect(readPresets()[0].name).toBe('History'));
  expect(await screen.findByRole('status')).toHaveTextContent('View saved');
});

it('restores and deletes saved views and reports invalid names',async() => {
  savePreset('Season',3,{range_filter_grandPrixYear:[2020,2026],bankroll:9000});
  const restore = vi.fn();
  render(<FeatureBar page={1} values={{}} options={{design:true,cache:true}} setOptions={vi.fn()} restore={restore} navigate={vi.fn()}/>);
  fireEvent.click(screen.getByRole('button',{name:'Save view'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Enter a name');
  fireEvent.change(screen.getByRole('combobox',{name:'Saved views'}),{target:{value:'Season'}});
  fireEvent.click(screen.getByRole('button',{name:'Load view'}));
  await waitFor(() => expect(restore).toHaveBeenCalledWith({version:1,page:3,name:'Season',values:{range_filter_grandPrixYear:[2020,2026]}}));
  fireEvent.click(screen.getByRole('button',{name:'Delete view'}));
  await waitFor(() => expect(readPresets()).toHaveLength(0));
  expect(screen.getByRole('button',{name:'Load view'})).toBeDisabled();
});

it('supports section search, arrow selection, Enter navigation and focus return',async() => {
  const navigate = vi.fn();
  render(<FeatureBar page={1} values={{}} options={{design:true,cache:true}} setOptions={vi.fn()} restore={vi.fn()} navigate={navigate}/>);
  await waitFor(() => expect(screen.getByRole('button',{name:'Download analysis context'})).toBeEnabled());
  const trigger = screen.getByRole('button',{name:'Find section (Ctrl/⌘ K)'});
  trigger.focus();fireEvent.keyDown(window,{key:'k',ctrlKey:true});
  const search = screen.getByRole('textbox',{name:'Search sections'});
  expect(search).toHaveFocus();
  fireEvent.keyDown(search,{key:'ArrowDown'});
  fireEvent.keyDown(search,{key:'Enter'});
  expect(navigate).toHaveBeenCalledWith(1);
  expect(trigger).toHaveFocus();
  fireEvent.click(trigger);
  fireEvent.change(search,{target:{value:'predictive'}});
  fireEvent.keyDown(search,{key:'Enter'});
  expect(navigate).toHaveBeenLastCalledWith(4);
  fireEvent.click(trigger);
  fireEvent.change(search,{target:{value:'unfindable'}});
  expect(within(screen.getByRole('dialog')).queryAllByRole('button').map(button => button.textContent)).toEqual(['Close']);
  fireEvent.click(screen.getByRole('button',{name:'Close'}));
  expect(trigger).toHaveFocus();
});

it('opens the global section shortcut outside a collapsed tools drawer and restores external focus',async() => {
  const navigate=vi.fn();
  const {container}=render(<><button>Outside tools</button><details><summary>Analysis tools</summary><FeatureBar page={1} values={{}} options={{design:true,cache:true}} setOptions={vi.fn()} restore={vi.fn()} navigate={navigate}/></details></>);
  await waitFor(() => expect(container.querySelector('.provenance')).toHaveTextContent('Data revision'));
  const drawer=container.querySelector('details');
  expect(drawer).not.toHaveAttribute('open');
  const outside=screen.getByRole('button',{name:'Outside tools'});
  outside.focus();fireEvent.keyDown(window,{key:'k',ctrlKey:true});
  const dialog=screen.getByRole('dialog');
  expect(dialog.parentElement).toBe(document.body);
  expect(drawer.contains(dialog)).toBe(false);
  expect(dialog).toHaveAttribute('open');
  expect(dialog).toBeVisible();
  const search=screen.getByRole('textbox',{name:'Search sections'});
  expect(search).toHaveFocus();
  fireEvent.change(search,{target:{value:'models'}});
  fireEvent.keyDown(search,{key:'Enter'});
  expect(navigate).toHaveBeenCalledWith(4);
  expect(outside).toHaveFocus();
  expect(drawer).not.toHaveAttribute('open');
  fireEvent.keyDown(window,{key:'k',metaKey:true});
  expect(search).toHaveFocus();
  fireEvent(dialog,new Event('close'));
  expect(dialog).toHaveAttribute('open');
  expect(search).toHaveFocus();
  fireEvent(dialog,new Event('cancel',{cancelable:true}));
  expect(dialog).not.toHaveAttribute('open');
  expect(outside).toHaveFocus();
});

it('offers retryable context provenance and exports only safe context with recorded model details',async() => {
  const source = {...provenance,models:[{model_name:'Legacy finish model',notes:['Not calibrated for win probabilities.']}]};
  vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce({ok:false}).mockResolvedValue({ok:true,json:async() => source}));
  const originalCreate = URL.createObjectURL, originalRevoke = URL.revokeObjectURL;
  let exported;
  URL.createObjectURL = vi.fn(blob => {exported=blob;return 'blob:test';});
  URL.revokeObjectURL = vi.fn();
  const click = vi.spyOn(HTMLAnchorElement.prototype,'click').mockImplementation(() => {});
  try {
    render(<FeatureBar page={3} values={{filter_driver:'José',bankroll:9000,filter_upload:{content:'secret'}}} options={{design:true,cache:true}} setOptions={vi.fn()} restore={vi.fn()} navigate={vi.fn()}/>);
    const download = screen.getByRole('button',{name:'Download analysis context'});
    expect(download).toBeDisabled();
    fireEvent.click(await screen.findByRole('button',{name:'Retry source details'}));
    await waitFor(() => expect(download).toBeEnabled());
    fireEvent.click(download);
    await waitFor(() => expect(click).toHaveBeenCalled());
    const json = await new Promise(resolve => {const reader=new FileReader();reader.onload=()=>resolve(JSON.parse(String(reader.result)));reader.readAsText(exported);});
    expect(json).toMatchObject({schema:'f1-analysis-context-v1',page:'Current Season',values:{filter_driver:'José'},provenance:source});
    expect(Number.isNaN(Date.parse(json.exported_at))).toBe(false);
  } finally {URL.createObjectURL=originalCreate;URL.revokeObjectURL=originalRevoke;}
});

it('searches hidden fields, preserves styled values and keeps row pages within matching rows',() => {
  const columns=Array.from({length:9},(_,index)=>({key:'field'+index,label:'Field '+index,kind:index===0?'NumberColumn':'TextColumn'}));
  const rows=Array.from({length:51},(_,index)=>[index,...Array(7).fill('visible'),index===50?'needle':'other']);
  const node={hide_index:true,columns,rows,display:rows.map(()=>['styled number'])};
  render(<EnhancedTable node={node}/>);
  fireEvent.click(screen.getByRole('button',{name:'Accessible table'}));
  fireEvent.click(screen.getByRole('button',{name:'Next rows'}));
  fireEvent.change(screen.getByRole('textbox',{name:'Search all fields'}),{target:{value:'needle'}});
  expect(screen.getAllByRole('cell',{name:'styled number'})).toHaveLength(1);
  expect(screen.getByText('Page 1 of 1',{exact:false})).toBeInTheDocument();
  const fields = screen.getByText('Choose fields (8 of 9)');fireEvent.click(fields);
  fireEvent.click(screen.getByRole('checkbox',{name:'Field 8 (field8)'}));
  expect(screen.getByRole('cell',{name:'needle'})).toBeInTheDocument();
  fireEvent.change(screen.getByRole('textbox',{name:'Search all fields'}),{target:{value:'absent'}});
  expect(screen.getByRole('status')).toHaveTextContent('No rows match');
  expect(screen.getByRole('button',{name:'Next rows'})).toBeDisabled();
});

it('blocks context export during requests and when source provenance differs from displayed results',async() => {
  const props = {page:1,values:{},options:{design:true,cache:true},setOptions:vi.fn(),restore:vi.fn(),navigate:vi.fn(),analysisRevision:'old-data'};
  const {rerender}=render(<FeatureBar {...props}/>);
  expect(await screen.findByText('Source data changed after this analysis.',{exact:false})).toBeInTheDocument();
  const download=screen.getByRole('button',{name:'Download analysis context'});
  expect(download).toBeDisabled();
  rerender(<FeatureBar {...props} analysisRevision={provenance.revision} busy/>);
  await waitFor(() => expect(screen.queryByText('Source data changed after this analysis.',{exact:false})).not.toBeInTheDocument());
  expect(download).toBeDisabled();
  rerender(<FeatureBar {...props} analysisRevision={provenance.revision} busy={false}/>);
  await waitFor(() => expect(download).toBeEnabled());
});

it('compares four drivers with known-row DNF rates and drops drivers absent from updated rows',() => {
  const columns=[{key:'driverName',label:'Driver'},{key:'resultsFinalPositionNumber',label:'Finish'},{key:'DNF',label:'DNF'}];
  const rows=[['A',2,true],['A',4,false],['A',null,'unknown'],['A',null,null],['B',1,'FALSE'],['C',3,false],['D',5,false],['E',6,false]];
  const {rerender}=render(<EnhancedTable node={{hide_index:true,columns,rows}}/>);
  fireEvent.click(screen.getByRole('button',{name:'Compare drivers'}));
  for(const name of ['A','B','C','D'])fireEvent.click(screen.getByRole('checkbox',{name}));
  expect(screen.getByRole('checkbox',{name:'E'})).toBeDisabled();
  const row=screen.getByRole('rowheader',{name:'A'}).closest('tr');
  expect(within(row).getByRole('cell',{name:'3.00'})).toBeInTheDocument();
  expect(within(row).getByRole('cell',{name:'50.0%'})).toBeInTheDocument();
  expect(within(row).getByRole('cell',{name:'4'})).toBeInTheDocument();
  rerender(<EnhancedTable node={{hide_index:true,columns,rows:rows.filter(row=>row[0]!=='A')}}/>);
  expect(screen.getByRole('checkbox',{name:'E'})).toBeEnabled();
  expect(screen.queryByRole('rowheader',{name:'A'})).not.toBeInTheDocument();
});

it('keeps existing results and announces a longer calculation without speaking every second',() => {
  vi.useFakeTimers();
  const {rerender}=render(<LoadingFeedback busy hasResults/>);
  expect(screen.getByRole('status')).toHaveTextContent('existing results remain visible');
  act(() => {vi.advanceTimersByTime(11000);});
  rerender(<LoadingFeedback busy hasResults/>);
  expect(screen.getByRole('status')).toHaveTextContent('taking longer than usual');
  expect(screen.getByText('11s elapsed.',{exact:false})).toHaveAttribute('aria-hidden','true');
  rerender(<LoadingFeedback busy={false} hasResults/>);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});
