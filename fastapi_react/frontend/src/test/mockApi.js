// Shared helper for page-level smoke tests.
// Each test file vi.mocks the api module and renders a single page
// with a deterministic fixture, then asserts that the page renders
// the expected headings or data without throwing.
import { vi } from 'vitest';

export const mockApi = (responses) => {
  vi.mock('../api.js', () => ({
    api: {
      get: vi.fn((url) => {
        if (responses[url]) return Promise.resolve(responses[url]);
        return Promise.reject(new Error(`unmocked GET ${url}`));
      }),
      post: vi.fn((url, body) => {
        const key = `${url}:${JSON.stringify(body || {})}`;
        if (responses[key]) return Promise.resolve(responses[key]);
        if (responses[url]) return Promise.resolve(responses[url]);
        return Promise.reject(new Error(`unmocked POST ${url}`));
      }),
    },
    downloadUrl: (path) => `/api/raw/download?path=${encodeURIComponent(path)}`,
  }));
};
