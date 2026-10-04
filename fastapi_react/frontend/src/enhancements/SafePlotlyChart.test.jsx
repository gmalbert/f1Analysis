import {act, render, screen, waitFor} from '@testing-library/react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {SafePlotlyChart} from './SafePlotlyChart';

const chart = vi.hoisted(() => ({newPlot:vi.fn(), purge:vi.fn(), Plots:{resize:vi.fn()}}));
vi.mock('plotly.js-dist-min', () => ({default:chart}));
const node = {label:'Finish distribution', spec:{data:[], layout:{}}};

beforeEach(() => {
  vi.clearAllMocks();
  chart.newPlot.mockResolvedValue(undefined);
});
afterEach(() => vi.unstubAllGlobals());

describe('asynchronous Plotly lifecycle', () => {
  it('renders the chart and releases it when the panel is removed', async () => {
    const view = render(<SafePlotlyChart node={node}/>);
    await waitFor(() => expect(chart.newPlot).toHaveBeenCalledTimes(1));
    expect(screen.getByRole('img',{name:'Finish distribution'})).toBeInTheDocument();
    view.unmount();
    expect(chart.purge).toHaveBeenCalled();
  });

  it('reports a failed chart without throwing a page error', async () => {
    chart.newPlot.mockRejectedValueOnce(new Error('Invalid chart data'));
    render(<SafePlotlyChart node={node}/>);
    expect(await screen.findByRole('alert')).toHaveTextContent('Chart unavailable: Invalid chart data');
  });

  it('disposes a chart that completes after its panel has already unmounted', async () => {
    let finish;
    chart.newPlot.mockImplementationOnce(() => new Promise(resolve => {finish = resolve;}));
    const view = render(<SafePlotlyChart node={node}/>);
    await waitFor(() => expect(finish).toBeTypeOf('function'));
    view.unmount();
    finish();
    await waitFor(() => expect(chart.purge).toHaveBeenCalled());
  });
  it('contains synchronous resize failures and ignores queued callbacks after disposal', async () => {
    let resize;
    const disconnect = vi.fn();
    vi.stubGlobal('ResizeObserver',class {
      constructor(callback) {resize = callback;}
      observe() {}
      disconnect() {disconnect();}
    });
    chart.Plots.resize.mockImplementation(() => {throw new Error('Container was resized');});
    const view = render(<SafePlotlyChart node={node}/>);
    await waitFor(() => expect(resize).toBeTypeOf('function'));
    await act(async () => {resize();});
    expect(chart.Plots.resize).toHaveBeenCalledTimes(1);
    view.unmount();
    expect(disconnect).toHaveBeenCalled();
    await act(async () => {resize();});
    expect(chart.Plots.resize).toHaveBeenCalledTimes(1);
  });
});
