import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api, downloadUrl } from './api.js';

describe('api', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });

  it('parses successful JSON response', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ foo: 1 }),
    });
    const result = await api.get('/api/health');
    expect(result).toEqual({ foo: 1 });
    expect(fetchMock).toHaveBeenCalledWith('/api/health');
  });

  it('throws with detail on 4xx/5xx', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'bad input' }),
    });
    await expect(api.get('/api/x')).rejects.toThrow('bad input');
  });

  it('falls back to status text when body is empty', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Server Error',
      json: async () => {
        throw new Error('no body');
      },
    });
    await expect(api.get('/api/x')).rejects.toThrow('500 Server Error');
  });

  it('serialises JSON in POST requests', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    await api.post('/api/betting/value', { a: 1 });
    expect(fetchMock).toHaveBeenCalledWith('/api/betting/value', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ a: 1 }),
    });
  });
});

describe('downloadUrl', () => {
  it('encodes the path component', () => {
    expect(downloadUrl('foo bar.csv')).toBe('/api/raw/download?path=foo%20bar.csv');
  });

  it('encodes path separators', () => {
    expect(downloadUrl('subdir/file.csv')).toBe('/api/raw/download?path=subdir%2Ffile.csv');
  });
});
