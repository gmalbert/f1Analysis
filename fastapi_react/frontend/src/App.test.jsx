import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {afterEach,beforeEach,describe,expect,it,vi} from 'vitest';
import App from './App';
import {viewClient} from './enhancements/viewClient';
vi.mock('./enhancements/viewClient',()=>({viewClient:{load:vi.fn(),clear:vi.fn()}}));
vi.mock('./components/ViewTable',()=>({ViewTable:()=>null}));
const nodes=[{type:'heading',level:2,text:'Data Explorer'},{type:'checkbox',key:'filter_results_main',label:'Filter Results',value:false}];
describe('reference application shell',()=>{
  beforeEach(()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({mode:'token',token_required:true})}));
    window.history.replaceState({},'','/');sessionStorage.clear();localStorage.clear();
    Object.defineProperty(window,'scrollTo',{configurable:true,value:vi.fn()});
    viewClient.load.mockImplementation(async(payload)=>({shell:[{type:'heading',level:1,text:'F1 Races from 2016 to 2026'}],nodes:payload.page===1?nodes:[{type:'heading',level:2,text:`Page ${payload.page}`}],sidebar:[{type:'heading',level:2,text:'Select filters to apply:'}]}));
  });
  afterEach(()=>vi.unstubAllGlobals());
  it('renders the reference heading, brand and seven accessible tabs',async()=>{
    render(<App/>);
    expect(await screen.findByRole('heading',{name:'Data Explorer'})).toBeInTheDocument();
    expect(screen.getByRole('img',{name:'Gridlocked'})).toHaveAttribute('src','/api/brand/logo');
    expect(screen.getAllByRole('tab')).toHaveLength(7);
    expect(document.title).toBe('Gridlocked - Formula 1 Betting & Analytics');
  });
  it('navigates and carries filter state into the next page',async()=>{
    render(<App/>);fireEvent.click(await screen.findByRole('checkbox',{name:'Filter Results'}));
    expect(await screen.findByRole('complementary',{name:'Data filters'})).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab',{name:/Analytics & Visualizations/}));
    expect(await screen.findByRole('heading',{name:'Page 2'})).toBeInTheDocument();
    expect(viewClient.load).toHaveBeenLastCalledWith(expect.objectContaining({page:2,values:{filter_results_main:true}}),expect.objectContaining({enabled:true}));
    expect(window.location.hash).toBe('#/Analytics');
    expect(JSON.parse(sessionStorage.getItem('f1analysis.view-values'))).toEqual({filter_results_main:true});
    fireEvent.click(screen.getByRole('button',{name:'Close sidebar'}));
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Open sidebar'}));
    expect(screen.getByRole('complementary')).toBeInTheDocument();
  });
  it('opens the research queue from the reference action without submitting another view',async()=>{
    window.history.replaceState({},'','/#/Raw%20Data');
    viewClient.load.mockResolvedValue({nodes:[{type:'button',key:'Run Leakage Audit',label:'Run Leakage Audit'}]});
    render(<App/>);
    const button = await screen.findByRole('button',{name:'Run Leakage Audit'});
    await screen.findByLabelText('Administrator token');
    const callsBefore = viewClient.load.mock.calls.length;
    fireEvent.click(button);
    expect(screen.getByLabelText('Administrator token')).toHaveFocus();
    expect(screen.getByRole('button',{name:'Queue calculation'})).toBeDisabled();
    expect(viewClient.load).toHaveBeenCalledTimes(callsBefore);
  });
  it('supports keyboard tab navigation and persistent theme selection',async()=>{
    render(<App/>);await screen.findByRole('heading',{name:'Data Explorer'});
    fireEvent.keyDown(screen.getByRole('tab',{name:/Data Explorer/}),{key:'ArrowRight'});
    expect(await screen.findByRole('heading',{name:'Page 2'})).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Settings'}));
    fireEvent.click(screen.getByRole('checkbox',{name:'Use light theme'}));
    await waitFor(()=>expect(document.documentElement.dataset.theme).toBe('dark'));
    expect(localStorage.getItem('f1analysis.theme')).toBe('dark');
  });
  it('shows a failed request and successfully retries it',async()=>{
    viewClient.load.mockRejectedValueOnce(new Error('Unable to load analysis'));
    render(<App/>);expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load analysis');
    fireEvent.click(screen.getByRole('button',{name:'Retry'}));
    expect(await screen.findByRole('heading',{name:'Data Explorer'})).toBeInTheDocument();
  });
  it('keeps an explicit false filter setting after an older true legacy setting',async()=>{
    sessionStorage.setItem('f1analysis.view-values',JSON.stringify({filter_results_main:false}));
    sessionStorage.setItem('f1analysis.filters',JSON.stringify({applied:true}));
    render(<App/>);
    await screen.findByRole('heading',{name:'Data Explorer'});
    expect(viewClient.load).toHaveBeenLastCalledWith(expect.objectContaining({values:{filter_results_main:false}}),expect.anything());
    expect(JSON.parse(sessionStorage.getItem('f1analysis.filters')).applied).toBe(false);
  });
  it('focuses main content without changing the selected section',async()=>{
    window.history.replaceState({},'','/#/Analytics');
    render(<App/>);
    await screen.findByRole('heading',{name:'Page 2'});
    fireEvent.click(screen.getByRole('link',{name:'Skip to main content'}));
    expect(document.activeElement).toBe(document.getElementById('main-content'));
    expect(location.hash).toBe('#/Analytics');
  });
});
