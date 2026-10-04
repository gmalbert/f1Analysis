import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import {FeatureBar,readOptions} from './FeatureBar';
import {EnhancedTable} from './EnhancedTable';
import {createViewClient} from './viewClient.js';
import {hasUpload,readPresets,readSharedView,safeValues,savePreset,shareUrl} from './preferences.js';

vi.mock('../components/ViewTable',()=>({ViewTable:()=> <p>Original interactive grid</p>}));
beforeEach(() => {localStorage.clear();vi.stubEnv('VITE_F1_ENHANCEMENTS','1');});
afterEach(() => {vi.unstubAllEnvs();vi.unstubAllGlobals();});

it('saves and shares Unicode filters while excluding private uploaded and financial values',() => {
  const values = {filter_results_main:true,filter_driver:'José',range_filter_grandPrixYear:[2017,2026],f1bet_field_upload:{content:'private'},bankroll:5000};
  savePreset('Recent',2,values);
  expect(readPresets()[0].values).toEqual(safeValues(values));
  expect(readSharedView(new URL(shareUrl(2,values,'http://localhost/')).hash).values).toEqual(safeValues(values));
  expect(hasUpload(values)).toBe(true);
  expect(readOptions()).toEqual({design:false,cache:false});
  localStorage.setItem('f1analysis.enhancement-options','invalid');
  expect(readOptions().design).toBe(false);
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

it('keeps original grid rendering when enhancements are disabled',() => {
  vi.stubEnv('VITE_F1_ENHANCEMENTS','0');
  render(<EnhancedTable node={{columns:[],rows:[]}}/>);
  expect(screen.getByText('Original interactive grid')).toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'Accessible table'})).not.toBeInTheDocument();
});

it('renders the saved-view tools and records a named preset',async() => {
  vi.stubGlobal('fetch',vi.fn(async() => ({ok:true,json:async() => ({revision:'abcdefghijklmno',build_revision:'test',dataset:{name:'data.parquet',modified_at:'today'},models:[]})})));
  render(<FeatureBar page={1} values={{filter_results_main:true}} options={{design:false,cache:false}} setOptions={vi.fn()} restore={vi.fn()} navigate={vi.fn()}/>);
  fireEvent.change(screen.getByRole('textbox',{name:'View name'}),{target:{value:'History'}});
  fireEvent.click(screen.getByRole('button',{name:'Save view'}));
  await waitFor(() => expect(readPresets()[0].name).toBe('History'));
  expect(await screen.findByRole('status')).toHaveTextContent('View saved');
});
