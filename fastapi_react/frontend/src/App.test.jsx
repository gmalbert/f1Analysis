import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {beforeEach,describe,expect,it,vi} from 'vitest';
import App from './App';
import {api} from './api';
vi.mock('./api',()=>({api:{post:vi.fn()}}));
vi.mock('./components/ViewTable',()=>({ViewTable:()=>null}));
const nodes=[{type:'heading',level:2,text:'Data Explorer'},{type:'checkbox',key:'filter_results_main',label:'Filter Results',value:false}];
describe('reference application shell',()=>{
  beforeEach(()=>{
    window.history.replaceState({},'','/');sessionStorage.clear();localStorage.clear();
    Object.defineProperty(window,'scrollTo',{configurable:true,value:vi.fn()});
    api.post.mockImplementation(async(_,payload)=>({shell:[{type:'heading',level:1,text:'F1 Races from 2016 to 2026'}],nodes:payload.page===1?nodes:[{type:'heading',level:2,text:`Page ${payload.page}`}],sidebar:[{type:'heading',level:2,text:'Select filters to apply:'}]}));
  });
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
    expect(api.post).toHaveBeenLastCalledWith('/api/views',expect.objectContaining({page:2,values:{filter_results_main:true}}));
    expect(window.location.hash).toBe('#/Analytics');
    expect(JSON.parse(sessionStorage.getItem('f1analysis.view-values'))).toEqual({filter_results_main:true});
    fireEvent.click(screen.getByRole('button',{name:'Close sidebar'}));
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Open sidebar'}));
    expect(screen.getByRole('complementary')).toBeInTheDocument();
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
    api.post.mockRejectedValueOnce(new Error('Unable to load analysis'));
    render(<App/>);expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load analysis');
    fireEvent.click(screen.getByRole('button',{name:'Retry'}));
    expect(await screen.findByRole('heading',{name:'Data Explorer'})).toBeInTheDocument();
  });
});
