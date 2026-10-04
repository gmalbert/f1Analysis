import {act, fireEvent, render, screen} from '@testing-library/react';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {ResearchJobs, researchOutput} from './ResearchJobs';
vi.mock('../components/Presentation', () => ({ViewNodes: ({nodes}) => <div>{nodes.map(node => node.text).join(' ')}</div>}));
const response = body => ({ok: true, json: async () => body});
let jobFetch;
const open = async () => {
  await act(async () => {});
  fireEvent.click(screen.getByText('Administrator research jobs'));
};
const token = () => fireEvent.change(screen.getByLabelText('Administrator token'), {target: {value: 'memory-only-token'}});
beforeEach(() => {
  vi.useFakeTimers();jobFetch = vi.fn();
  vi.stubGlobal('fetch', vi.fn((path, ...args) => path === '/api/enhancements/research-access'
    ? Promise.resolve(response({mode: 'token', token_required: true})) : jobFetch(path, ...args)));
});
afterEach(() => {vi.useRealTimers();vi.unstubAllGlobals();});

it('opens from an existing research action without running anything or retaining a token', async () => {
  localStorage.clear();sessionStorage.clear();
  render(<ResearchJobs page={5} values={{'Select q values (number of bins)': [3,4], upload_csv: 'private'}}/>);
  await act(async () => {});
  act(() => window.dispatchEvent(new CustomEvent('f1analysis:research-task', {detail: 'bin-comparison'})));
  expect(screen.getByLabelText('Administrator token')).toHaveFocus();
  expect(screen.getByRole('checkbox', {name: '3'})).toBeChecked();
  expect(screen.getByRole('button', {name: 'Queue calculation'})).toBeDisabled();
  token();
  expect(jobFetch).not.toHaveBeenCalled();
  expect(JSON.stringify({...localStorage, ...sessionStorage})).not.toContain('memory-only-token');
});

it('queues only bounded task values, polls completion and displays recorded results', async () => {
  jobFetch.mockResolvedValueOnce(response({id: 'job-one', state: 'queued'}))
    .mockResolvedValueOnce(response({id: 'job-one', state: 'succeeded', revision: 'r1'}))
    .mockResolvedValueOnce(response({source_revision: 'r1', nodes: [{type: 'heading', text: 'Audit finished'}]}));
  render(<ResearchJobs page={6} values={{uploaded_csv: 'private'}}/>);await open();token();
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Queue calculation'})));
  const options = jobFetch.mock.calls[0][1];
  expect(JSON.parse(options.body)).toEqual({task: 'leakage-audit', values: {'Rows to read (0 = all)': 1000}});
  expect(options.headers['X-F1-Admin-Token']).toBe('memory-only-token');
  await act(() => vi.advanceTimersByTimeAsync(1000));
  expect(screen.getByText('Audit finished')).toBeInTheDocument();
  expect(screen.getByText(/Calculated with source revision r1/)).toBeInTheDocument();
});

it('cancels a queued job even while a previous poll is being disposed', async () => {
  jobFetch.mockResolvedValueOnce(response({id: 'job-one', state: 'queued'}))
    .mockImplementationOnce(async (_path, options) => {
      await Promise.resolve();
      expect(options.signal.aborted).toBe(false);
      return response({cancelled: true, job: {id: 'job-one', state: 'cancelled'}});
    });
  render(<ResearchJobs page={6}/>);await open();token();
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Queue calculation'})));
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Cancel queued job'})));
  expect(screen.getByRole('status')).toHaveTextContent('cancelled');
  expect(jobFetch.mock.calls[1][1].method).toBe('DELETE');
});

it('offers status retry after failure and aborts an in-flight poll on unmount', async () => {
  jobFetch.mockResolvedValueOnce(response({id: 'job-one', state: 'running'}))
    .mockRejectedValueOnce(new Error('Connection lost'))
    .mockImplementationOnce(() => new Promise(() => {}));
  const view = render(<ResearchJobs page={6}/>);await open();token();
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Queue calculation'})));
  await act(() => vi.advanceTimersByTimeAsync(1000));
  expect(screen.getByRole('alert')).toHaveTextContent('Connection lost');
  fireEvent.click(screen.getByRole('button', {name: 'Retry job status'}));
  await act(() => vi.advanceTimersByTimeAsync(1000));
  const signal = jobFetch.mock.calls[2][1].signal;
  view.unmount();
  expect(signal.aborted).toBe(true);
});

it('submits, polls results and cancels locally without a token field or credential header', async () => {
  fetch.mockImplementation((path, ...args) => path === '/api/enhancements/research-access'
    ? Promise.resolve(response({mode: 'local', token_required: false})) : jobFetch(path, ...args));
  jobFetch.mockResolvedValueOnce(response({id: 'local-one', state: 'queued'}))
    .mockResolvedValueOnce(response({cancelled: true, job: {id: 'local-one', state: 'cancelled'}}))
    .mockResolvedValueOnce(response({id: 'local-two', state: 'running'}))
    .mockResolvedValueOnce(response({id: 'local-two', state: 'succeeded'}))
    .mockResolvedValueOnce(response({source_revision: 'local-r1', nodes: [{type: 'heading', text: 'Local audit finished'}]}));
  render(<ResearchJobs page={6}/>);
  await act(async () => {});
  act(() => window.dispatchEvent(new CustomEvent('f1analysis:research-task', {detail: 'leakage-audit'})));
  expect(screen.queryByLabelText('Administrator token')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Research task')).toHaveFocus();
  expect(screen.getByRole('button', {name: 'Queue calculation'})).toBeEnabled();
  expect(jobFetch).not.toHaveBeenCalled();
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Queue calculation'})));
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Cancel queued job'})));
  expect(screen.getByRole('status')).toHaveTextContent('cancelled');
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Queue calculation'})));
  await act(() => vi.advanceTimersByTimeAsync(1000));
  expect(screen.getByText('Local audit finished')).toBeInTheDocument();
  expect(jobFetch.mock.calls.every(([, options]) => !('X-F1-Admin-Token' in options.headers))).toBe(true);
});

it('fails closed while checking access and permits retry after a connection failure', async () => {
  fetch.mockRejectedValueOnce(new Error('Connection lost'))
    .mockResolvedValueOnce(response({mode: 'local', token_required: false}));
  render(<ResearchJobs page={6}/>);
  fireEvent.click(screen.getByText('Research jobs'));
  expect(screen.getByRole('button', {name: 'Queue calculation'})).toBeDisabled();
  await act(async () => {});
  expect(screen.getByRole('alert')).toHaveTextContent('Connection lost');
  await act(async () => fireEvent.click(screen.getByRole('button', {name: 'Retry research access'})));
  expect(screen.getByRole('button', {name: 'Queue calculation'})).toBeEnabled();
  expect(jobFetch).not.toHaveBeenCalled();
});

it('aborts a pending access check on unmount', () => {
  fetch.mockImplementation(() => new Promise(() => {}));
  const view = render(<ResearchJobs page={6}/>);
  const signal = fetch.mock.calls[0][1].signal;
  view.unmount();
  expect(signal.aborted).toBe(true);
});

it('does not expose controls from output snapshots', () => {
  expect(researchOutput([{type: 'tabs', children: [
    {type: 'tab', hidden: true, children: [{type: 'heading', text: 'Hidden'}]},
    {type: 'tab', children: [{type: 'button'}, {type: 'table', rows: []}]},
  ]}])).toEqual([{type: 'table', rows: []}]);
});
