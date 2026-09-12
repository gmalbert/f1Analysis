import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import RawData from './RawData.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe('RawData page', () => {
  it('renders the page heading after data loads', async () => {
    apiMock.get.mockImplementation((url) => {
      if (url === '/api/raw/files') {
        return Promise.resolve({
          files: [
            { path: 'active_drivers.csv', size: 100, suffix: '.csv' },
            { path: 'notes.txt', size: 50, suffix: '.txt' },
          ],
        });
      }
      if (url === '/api/health') return Promise.resolve({ status: 'ok' });
      return Promise.resolve({});
    });
    render(<RawData />);
    expect(screen.getByText(/Data & Debug Tools/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/api/raw/files');
    });
  });
});
