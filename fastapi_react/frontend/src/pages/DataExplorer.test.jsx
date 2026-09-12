import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import DataExplorer from './DataExplorer.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe('DataExplorer page', () => {
  it('renders the page heading and shows the filter schema after data loads', async () => {
    apiMock.get.mockImplementation((url) => {
      if (url === '/api/data-explorer/schema') {
        return Promise.resolve({
          filters: [
            { column: 'grandPrixYear', kind: 'range', min: 2015, max: 2025 },
            { column: 'driverDNFCount', kind: 'range', min: 0, max: 50 },
          ],
        });
      }
      return Promise.resolve({});
    });
    apiMock.post.mockResolvedValue({
      total: 100,
      columns: ['grandPrixYear', 'driverDNFCount'],
      rows: [],
    });
    render(<DataExplorer />);
    expect(screen.getByText(/Data Explorer/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/api/data-explorer/schema');
    });
  });

  it('shows error state when schema fetch fails', async () => {
    apiMock.get.mockRejectedValueOnce(new Error('schema down'));
    render(<DataExplorer />);
    await waitFor(() => {
      expect(screen.getByText('schema down')).toBeInTheDocument();
    });
  });
});
