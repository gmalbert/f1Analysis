import {afterEach, describe, expect, it, vi} from 'vitest';
import {createViewClient} from './viewClient';

const response = body => ({ok:true, json:async () => body});
const payload = {page:1, values:{filter_results_main:false}};
const enabled = {enabled:true};

afterEach(() => vi.useRealTimers());

describe('bounded analysis requests', () => {
  it('deduplicates subscribers, preserves an active subscriber, and reuses the result', async () => {
    let complete;
    const fetcher = vi.fn(url => url.endsWith('/status') ? Promise.resolve(response({revision:'r1'})) :
      new Promise(resolve => {complete = resolve;}));
    const client = createViewClient({fetcher});
    const controller = new AbortController();
    const first = client.load(payload, {...enabled, signal:controller.signal});
    const rejected = expect(first).rejects.toMatchObject({name:'AbortError'});
    const second = client.load(payload, enabled);
    await vi.waitFor(() => expect(complete).toBeTypeOf('function'));
    controller.abort();
    await rejected;
    complete(response({nodes:['current']}));
    expect(await second).toEqual({nodes:['current']});
    expect(await client.load(payload, enabled)).toEqual({nodes:['current']});
    expect(fetcher.mock.calls.filter(([url]) => url === '/api/views')).toHaveLength(1);
  });

  it('does not refill or reuse an invalidated in-flight response', async () => {
    const completions = [];
    const fetcher = vi.fn(url => url.endsWith('/status') ? Promise.resolve(response({revision:'r1'})) :
      new Promise(resolve => completions.push(resolve)));
    const client = createViewClient({fetcher});
    const old = client.load(payload, enabled);
    await vi.waitFor(() => expect(completions).toHaveLength(1));
    client.clear();
    const current = client.load(payload, enabled);
    await vi.waitFor(() => expect(completions).toHaveLength(2));
    completions[1](response({nodes:['new']}));
    expect(await current).toEqual({nodes:['new']});
    completions[0](response({nodes:['old']}));
    await old;
    expect(await client.load(payload, enabled)).toEqual({nodes:['new']});
  });

  it('checks revision before every hit and honors expiry and memory limits', async () => {
    let revision = 'r1', clock = 1, renders = 0;
    const fetcher = vi.fn(async url => response(url.endsWith('/status') ? {revision} : {render:++renders}));
    const client = createViewClient({fetcher, now:() => clock, ttl:10, maxBytes:40, maxEntries:1});
    expect((await client.load(payload, enabled)).render).toBe(1);
    expect((await client.load(payload, enabled)).render).toBe(1);
    revision = 'r2';
    expect((await client.load(payload, enabled)).render).toBe(2);
    clock += 11;
    expect((await client.load(payload, enabled)).render).toBe(3);
    await client.load({...payload, values:{filter_results_main:true}}, enabled);
    expect((await client.load(payload, enabled)).render).toBe(5);
    expect(client.retainedBytes()).toBeLessThanOrEqual(40);
  });

  it('bypasses private uploads, actions, raw data, and betting responses', async () => {
    let count = 0;
    const fetcher = vi.fn(async url => response(url.endsWith('/status') ? {revision:'r'} : {id:++count}));
    const client = createViewClient({fetcher});
    for (const next of [
      {...payload, action:'Run Leakage Audit'},
      {...payload, values:{csv:{name:'private.csv', content:'secret'}}},
      {...payload, page:6},
      {...payload, page:7},
    ]) {
      const first = await client.load(next, enabled);
      expect(await client.load(next, enabled)).not.toEqual(first);
    }
    expect(fetcher.mock.calls.some(([url]) => url.endsWith('/status'))).toBe(false);
    expect(client.retainedBytes()).toBe(0);
  });

  it('reports status-probe and view-request timeouts', async () => {
    vi.useFakeTimers();
    const hanging = (_url, {signal}) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('Cancelled','AbortError')), {once:true});
    });
    const probe = createViewClient({fetcher:hanging, normalTimeout:5});
    const statusError = expect(probe.load(payload, enabled)).rejects.toThrow('Checking analysis data timed out');
    await vi.advanceTimersByTimeAsync(5);
    await statusError;
    const view = createViewClient({fetcher:hanging, normalTimeout:5});
    const viewError = expect(view.load(payload)).rejects.toThrow('analysis request timed out');
    await vi.advanceTimersByTimeAsync(5);
    await viewError;
  });
});
