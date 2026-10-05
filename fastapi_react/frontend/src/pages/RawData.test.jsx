import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

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
      if (url === '/api/data-explorer/display-schema') {
        return Promise.resolve({ columns: ['grandPrixYear'], labels: { grandPrixYear: 'Year' } });
      }
      return Promise.resolve({});
    });
    render(<RawData />);
    expect(screen.getByText(/Data & Debug Tools/i)).toBeInTheDocument();
    expect(screen.getByText('View the complete unfiltered dataset.')).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Show Raw Data' })).not.toBeChecked();
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalledWith('/api/raw/files');
    });
  });

  it('previews and downloads a selected source file', async () => {
    apiMock.get.mockImplementation(url => {
      if (url === '/api/raw/files') return Promise.resolve({ files: [{ path: 'active_drivers.csv', size: 100, suffix: '.csv' }] });
      if (url === '/api/health') return Promise.resolve({ status: 'ok' });
      if (url.startsWith('/api/raw/preview')) return Promise.resolve({ kind: 'table', columns: ['driver'], rows: [{ driver: 'Max' }] });
      return Promise.resolve({});
    });
    render(<RawData />);
    fireEvent.click(screen.getByRole('button', { name: 'File Browser' }));
    fireEvent.click(await screen.findByRole('button', { name: /active_drivers.csv/ }));
    expect(await screen.findByText('Max')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Download original' })).toHaveAttribute(
      'href', '/api/raw/download?path=active_drivers.csv',
    );
  });

  it('pages through the full analysis table without requesting all rows at once', async () => {
    apiMock.get.mockImplementation(url => url === '/api/raw/files'
      ? Promise.resolve({ files: [] })
      : url === '/api/data-explorer/display-schema'
        ? Promise.resolve({
          columns: ['grandPrixYear', 'round', 'grandPrixName', 'resultsDriverName'],
          labels: { grandPrixYear: 'Year', grandPrixName: 'Grand Prix', resultsDriverName: 'Driver' },
        })
      : Promise.resolve({ status: 'ok' }));
    apiMock.post.mockResolvedValue({
      total: 51,
      columns: ['grandPrixYear', 'round', 'grandPrixName', 'resultsDriverName', 'driverId'],
      rows: [{ grandPrixYear: 2025, round: 1, grandPrixName: 'Australian Grand Prix', resultsDriverName: 'Max', driverId: 1 }],
    });
    render(<RawData />);
    fireEvent.click(await screen.findByRole('checkbox', { name: 'Show Raw Data' }));
    await screen.findByText('Total number of results: 51');
    expect(screen.getAllByRole('columnheader').map(header => header.textContent)).toEqual([
      'Year', 'round', 'Grand Prix', 'Driver',
    ]);
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    await waitFor(() => expect(apiMock.post).toHaveBeenLastCalledWith('/api/raw/analysis-data', expect.objectContaining({ offset: 50, limit: 50 })));
  });

  it('keeps expensive tools disabled with an actionable environment hint', async () => {
    apiMock.get.mockImplementation(url => url === '/api/raw/files'
      ? Promise.resolve({ files: [] })
      : Promise.resolve({ status: 'ok', expensive_tools_enabled: false }));
    render(<RawData />);
    fireEvent.click(screen.getByRole('button', { name: 'Temporal Leakage Audit' }));
    expect(await screen.findByRole('button', { name: 'Run Leakage Audit' })).toBeDisabled();
    expect(screen.getByText('ENABLE_EXPENSIVE_TOOLS=1')).toBeInTheDocument();
  });
});
