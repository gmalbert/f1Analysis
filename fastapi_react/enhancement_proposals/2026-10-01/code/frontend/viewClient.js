import {hasUpload, stableKey} from './preferences.js';

// Keep actions and uploaded data out of shared requests and retained responses.
export function createViewClient({fetcher = fetch, now = Date.now, ttl = 15000, maxEntries = 6, maxBytes = 12000000, normalTimeout = 120000, actionTimeout = 600000} = {}) {
  const cache = new Map(), pending = new Map();
  let revision = '', retainedBytes = 0;
  const clear = () => {cache.clear(); retainedBytes = 0;};
  async function json(url, options) {
    const response = await fetcher(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(typeof body.detail === 'string' ? body.detail : 'Request failed (' + response.status + ').'), {status: response.status});
    return body;
  }
  /** @param {object} payload @param {{signal?: AbortSignal, enabled?: boolean}} [options] */
  async function load(payload, {signal, enabled = false} = {}) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    const reusable = enabled && payload.page <= 5 && !payload.action && !hasUpload(payload.values || {});
    if (payload.action || hasUpload(payload.values || {})) clear();
    // Probe on every reusable navigation: never serve a client hit under an old revision.
    if (reusable) {
      const state = await json('/api/enhancements/status', {signal, cache: 'no-store'});
      if (revision !== state.revision) {clear(); revision = state.revision;}
    }
    const key = stableKey({revision, ...payload});
    const hit = reusable && cache.get(key);
    if (hit && hit.until > now()) {
      cache.delete(key); cache.set(key, hit);
      return hit.value;
    }
    if (hit) {cache.delete(key); retainedBytes -= hit.bytes;}
    let task = reusable && pending.get(key);
    if (!task) {
      const controller = new AbortController();
      task = {controller, consumers: 0, promise: null};
      let timedOut = false;
      const timeout = setTimeout(() => {timedOut = true;controller.abort();}, payload.action ? actionTimeout : normalTimeout);
      task.promise = json('/api/views', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload), signal: controller.signal
      }).then(value => {
        if (reusable) {
          // This budgets serialized data; actual JS heap must also be measured.
          const bytes = new TextEncoder().encode(JSON.stringify(value)).byteLength;
          if (bytes <= maxBytes) {
            const replaced = cache.get(key);
            if (replaced) retainedBytes -= replaced.bytes;
            cache.set(key, {value, bytes, until: now() + ttl}); retainedBytes += bytes;
            while (cache.size > maxEntries || retainedBytes > maxBytes) {
              const oldest = cache.keys().next().value;
              retainedBytes -= cache.get(oldest).bytes; cache.delete(oldest);
            }
          }
        }
        return value;
      }).catch(error => {
        if(timedOut)throw new Error('The analysis request timed out. Retry or reduce the selected workload.');
        throw error;
      }).finally(() => {clearTimeout(timeout); if (pending.get(key) === task) pending.delete(key);});
      if (reusable) pending.set(key, task);
    }
    task.consumers++;
    return new Promise((resolve, reject) => {
      let finished = false;
      function release() {
        if (finished) return;
        finished = true; signal?.removeEventListener('abort', abort); task.consumers--;
        // Strict Mode can subscribe again before this timer; allow it to share the request.
        setTimeout(() => {if (!task.consumers) task.controller.abort();}, 100);
      }
      function abort() {release(); reject(new DOMException('Cancelled', 'AbortError'));}
      signal?.addEventListener('abort', abort, {once: true});
      if (signal?.aborted) {abort(); return;}
      task.promise.then(value => {if (!finished) {release(); resolve(value);}},
        error => {if (!finished) {release(); reject(error);}});
    });
  }
  return {load, clear, retainedBytes: () => retainedBytes};
}

export const viewClient = createViewClient();
