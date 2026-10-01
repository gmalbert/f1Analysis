import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import DataExplorer from './DataExplorer.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
  sessionStorage.clear();
});

describe('DataExplorer page', () => {
  it('matches the reference unchecked filter state until users opt in', async () => {
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
    expect(screen.getByRole('checkbox', { name: 'Filter Results' })).toBeInTheDocument();
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/api/data-explorer/schema');
    });
    expect(screen.queryByText('0 rows')).not.toBeInTheDocument();
    expect(apiMock.post).not.toHaveBeenCalled();
    expect(screen.queryByText('Find one of the dataset fields…')).not.toBeInTheDocument();
  });

  it('shows error state when schema fetch fails', async () => {
    apiMock.get.mockRejectedValueOnce(new Error('schema down'));
    render(<DataExplorer />);
    await waitFor(() => {
      expect(screen.getByText('schema down')).toBeInTheDocument();
    });
  });

  it('applies and shares selected filters with other sections', async () => {
    apiMock.get.mockResolvedValueOnce({ filters: [
      { column: 'grandPrixYear', label: 'Year', kind: 'range', min: 2015, max: 2025 },
    ] });
    apiMock.post.mockResolvedValue({ total: 10, columns: ['grandPrixYear'], rows: [] });
    render(<DataExplorer />);
    fireEvent.click(await screen.findByRole('checkbox', { name: 'Filter Results' }));
    const minimum = await screen.findByRole('spinbutton', { name: 'Year minimum' });
    fireEvent.change(minimum, { target: { value: '2020' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply filters' }));
    await waitFor(() => {
      expect(JSON.parse(sessionStorage.getItem('f1analysis.filters'))).toMatchObject({
        applied: true,
        filters: [{ column: 'grandPrixYear', kind: 'range', value: [2020, 2025] }],
      });
    });
    expect(apiMock.post).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({
      filters: [{ column: 'grandPrixYear', kind: 'range', value: [2020, 2025] }],
    }));
  });

  it('loads the unfiltered result set when Filter Results is enabled', async () => {
    apiMock.get.mockResolvedValueOnce({ filters: [{ column: 'grandPrixYear', label: 'Year', kind: 'range', min: 2015, max: 2025 }] });
    apiMock.post.mockResolvedValue({ total: 12, columns: ['grandPrixYear'], rows: [{ grandPrixYear: 2025 }] });
    render(<DataExplorer />);
    fireEvent.click(await screen.findByRole('checkbox', { name: 'Filter Results' }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({
      filters: [],
      limit: 5000,
    })));
    expect(await screen.findByText('2025')).toBeInTheDocument();
  });
});
