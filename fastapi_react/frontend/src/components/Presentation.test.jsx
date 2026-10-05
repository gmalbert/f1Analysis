import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {describe,expect,it,vi} from 'vitest';
import {ViewNodes,displayCell,isTireChartPair,selectedTireChart} from './Presentation';

vi.mock('./ViewTable',()=>({ViewTable:()=>null}));

describe('reference presentation controls',()=>{
  it('links only the corresponding tire chart and filters original chart values without mutating the source',()=>{
    const chart={type:'vega',spec:{data:{name:'tire'},encoding:{x:{field:'driver',sort:null},y:{field:'Degradation (s/lap)'}},datasets:{tire:[
      {driver:'George Russell','Degradation (s/lap)':-4.223456},
      {driver:'Kimi Antonelli','Degradation (s/lap)':-4.178},
      {driver:'Esteban Ocon','Degradation (s/lap)':-3.987},
      {driver:'Other driver','Degradation (s/lap)':-3.1},
    ]}}};
    const nodes=[{type:'table',columns:[{key:'Driver'},{key:'Avg Deg (s/lap)'}]},{type:'markdown',text:'**Avg Tire Degradation by Driver (s/lap)**'},chart];
    expect(isTireChartPair(nodes,0)).toBe(true);
    const selected=['George Russell','Kimi Antonelli','Esteban Ocon'];
    const filtered=selectedTireChart(chart,selected);
    expect(filtered.spec.datasets.tire.map(row=>row.driver)).toEqual(selected);
    expect(filtered.spec.datasets.tire[0]['Degradation (s/lap)']).toBe(-4.223456);
    expect(filtered.spec.encoding.x.sort).toEqual(selected);
    expect(chart.spec.datasets.tire).toHaveLength(4);
    expect(chart.spec.encoding.x.sort).toBeNull();
    expect(selectedTireChart(chart,[])).toBe(chart);
    expect(isTireChartPair([{...nodes[0],columns:[{key:'Driver'}]},...nodes.slice(1)],0)).toBe(false);
    expect(isTireChartPair([nodes[0],{type:'markdown',text:'Unrelated chart'},chart],0)).toBe(false);
    const inline={...chart,spec:{...chart.spec,data:{values:chart.spec.datasets.tire},datasets:undefined}};
    expect(selectedTireChart(inline,['Esteban Ocon']).spec.data.values).toEqual([chart.spec.datasets.tire[2]]);
  });
  it('preserves explicit precision, percent, missing and date formatting',()=>{
    expect(displayCell(1.23456,{format:'%.3f'})).toBe('1.235');
    expect(displayCell(.25,{format:'percent'})).toBe('25.00%');
    expect(displayCell(4.8,{format:'%d'})).toBe('4');
    expect(displayCell(null,{})).toBe('None');
    expect(displayCell('2026-10-01T12:30:00',{kind:'DateColumn'})).toBe('2026-10-01');
    expect(displayCell(1.2,{},'1.200')).toBe('1.200');
  });
  it('displays calendar years without grouping while preserving ordinary numeric formatting',()=>{
    expect(displayCell(2026,{key:'year'})).toBe('2026');
    expect(displayCell(2026,{key:'grandPrixYear'},'2,026')).toBe('2026');
    expect(displayCell(2016,{label:'Year',format:'%.2f'})).toBe('2016');
    expect(displayCell(2026,{key:'start_year'})).toBe('2026');
    expect(displayCell(4629,{key:'rows'})).toBe('4,629');
    expect(displayCell(2026,{key:'yearsActive'})).toBe('2,026');
  });
  it('commits numbers on Enter and clamps to the reference limits',()=>{
    const change=vi.fn();
    render(<ViewNodes nodes={[{type:'number',key:'p',label:'Probability',value:.25,min:0,max:1,step:.05,format:'%.2f'}]} change={change}/>);
    const input=screen.getByRole('spinbutton');
    expect(input).toHaveValue(.25);
    fireEvent.change(input,{target:{value:'2'}});fireEvent.keyDown(input,{key:'Enter'});
    expect(change).toHaveBeenCalledWith('p',1);
    expect(screen.getByRole('button',{name:'Increase Probability'})).toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'Decrease Probability'}));
    expect(change).toHaveBeenLastCalledWith('p',.95);
  });
  it('uses the canonical option after a dependent selection changes',()=>{
    const change=vi.fn();
    render(<ViewNodes nodes={[{type:'select',key:'race',label:'Race',value:'Monza',options:['Monza','Spa']}]} values={{race:'Old race'}} change={change}/>);
    expect(screen.getByRole('combobox')).toHaveValue('"Monza"');
    fireEvent.change(screen.getByRole('combobox'),{target:{value:'"Spa"'}});
    expect(change).toHaveBeenCalledWith('race','Spa');
  });
  it('commits date ranges without converting them to local calendar dates',()=>{
    const change=vi.fn();
    render(<ViewNodes nodes={[{type:'slider',key:'date',label:'Race Date',min:'2020-01-01',max:'2026-01-01',step:1,value:['2020-01-01','2026-01-01']}]} change={change}/>);
    const slider=screen.getByRole('slider',{name:'Race Date minimum'});
    fireEvent.change(slider,{target:{value:String(Date.parse('2021-02-03')/86400000)}});fireEvent.keyUp(slider,{key:'ArrowRight'});
    expect(change).toHaveBeenCalledWith('date',['2021-02-03','2026-01-01']);
  });
  it('supports multiselect search, removal and clearing',()=>{
    const change=vi.fn();
    render(<ViewNodes nodes={[{type:'multiselect',key:'q',label:'Bins',options:[2,3,4],value:[2]}]} change={change}/>);
    fireEvent.focus(screen.getByRole('combobox'));fireEvent.change(screen.getByRole('combobox'),{target:{value:'3'}});fireEvent.keyDown(screen.getByRole('combobox'),{key:'Enter'});
    expect(change).toHaveBeenCalledWith('q',[2,3]);
    fireEvent.click(screen.getByRole('button',{name:'Remove 2'}));expect(change).toHaveBeenCalledWith('q',[]);
    fireEvent.click(screen.getByRole('button',{name:'Clear Bins'}));expect(change).toHaveBeenLastCalledWith('q',[]);
  });
  it('uploads the CSV name and contents and supports removing it',async()=>{
    const change=vi.fn();
    render(<ViewNodes nodes={[{type:'upload',key:'csv',label:'Field CSV',filename:'old.csv'}]} change={change}/>);
    const file=new File(['a,b\n1,2\n'],'field.csv',{type:'text/csv'});
    file.text=async()=> 'a,b\n1,2\n';
    fireEvent.change(screen.getByLabelText('Field CSV'),{target:{files:[file]}});
    await waitFor(()=>expect(change).toHaveBeenCalledWith('csv',{name:'field.csv',content:'a,b\n1,2\n'}));
    fireEvent.click(screen.getByRole('button',{name:'Remove old.csv'}));expect(change).toHaveBeenLastCalledWith('csv',null);
  });
  it('keeps disabled actions disabled and preserves download bytes',()=>{
    const action=vi.fn();
    render(<ViewNodes nodes={[{type:'button',key:'run',label:'Run',disabled:true},{type:'download',label:'Export',filename:'data.csv',mime:'text/csv',data:'YWJj'}]} action={action}/>);
    fireEvent.click(screen.getByRole('button',{name:'Run'}));expect(action).not.toHaveBeenCalled();
    expect(screen.getByRole('link',{name:'Export'})).toHaveAttribute('href','data:text/csv;base64,YWJj');
  });
});
