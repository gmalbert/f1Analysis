export const routes = ['Data Explorer', 'Analytics', 'Current Season', 'Next Race', 'Predictive Models', 'Raw Data', 'Betting Research'];
const storageKey = 'f1analysis.saved-views.v1';
const allowed = /^(filter_results_main|(?:range_filter_|checkbox_filter_|filter_).+|_tabs:.+|Select Model Type|tire_year_select|tire_race_select)$/;

export function safeValues(values = {}) {
  return Object.fromEntries(Object.entries(values).filter(([key, value]) =>
    allowed.test(key) && (
      value == null || ['string', 'number', 'boolean'].includes(typeof value) ||
      Array.isArray(value) && value.length <= 20 && value.every(item =>
        item == null || ['string', 'number', 'boolean'].includes(typeof item))
    )
  ));
}

export function validateView(view) {
  if (!view || view.version !== 1 || !Number.isInteger(view.page) ||
      view.page < 1 || view.page > routes.length) throw new Error('Unsupported saved view.');
  return {version: 1, page: view.page, values: safeValues(view.values)};
}

export function readPresets(storage = localStorage) {
  try {
    return JSON.parse(storage.getItem(storageKey) || '[]').slice(0, 20)
      .map(item => ({...validateView(item), name: String(item.name || 'Saved view').slice(0, 80)}));
  } catch { return []; }
}

export function savePreset(name, page, values, storage = localStorage) {
  const label = name.trim().slice(0, 80);
  if (!label) throw new Error('Enter a name for this view.');
  const view = {...validateView({version: 1, page, values}), name: label};
  const next = [view, ...readPresets(storage).filter(item => item.name !== label)].slice(0, 20);
  storage.setItem(storageKey, JSON.stringify(next));
  return next;
}

export function deletePreset(name, storage = localStorage) {
  const next = readPresets(storage).filter(item => item.name !== name);
  storage.setItem(storageKey, JSON.stringify(next));
  return next;
}

export function shareUrl(page, values, base = location.href) {
  const view = validateView({version: 1, page, values});
  const bytes = new TextEncoder().encode(JSON.stringify(view));
  const token = btoa(Array.from(bytes, byte => String.fromCharCode(byte)).join(''))
    .replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
  if (token.length > 6000) throw new Error('This view is too large for a link. Save it locally instead.');
  const url = new URL(base);
  url.hash = '/' + encodeURIComponent(routes[page - 1]) + '?view=' + token;
  return url.toString();
}

export function readSharedView(hash = location.hash) {
  const token = new URLSearchParams(hash.split('?')[1] || '').get('view');
  if (!token) return null;
  if (token.length > 6000) throw new Error('The shared link is too large.');
  const normalized = token.replaceAll('-', '+').replaceAll('_', '/');
  const decoded = atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '='));
  return validateView(JSON.parse(new TextDecoder().decode(Uint8Array.from(decoded, c => c.charCodeAt(0)))));
}

export function stableKey(value) {
  if (Array.isArray(value)) return '[' + value.map(stableKey).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort()
    .map(key => JSON.stringify(key) + ':' + stableKey(value[key])).join(',') + '}';
  return JSON.stringify(value);
}

export function hasUpload(values) {
  return Object.entries(values).some(([key,value]) =>
    /upload|csv|ledger/i.test(key) ||
    value && typeof value === 'object' && !Array.isArray(value) ||
    Array.isArray(value) && value.some(item => item && typeof item === 'object'));
}
