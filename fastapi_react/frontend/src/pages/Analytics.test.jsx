import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

// Make chart containers have size so Recharts renders.
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({ width: 800, height: 400, top: 0, left: 0, right: 800, bottom: 400, x: 0, y: 0 }),
  });
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

import Analytics from './Analytics.jsx';

describe('Analytics page', () => {
  it('renders headings and metric after data loads', async () => {
    apiMock.post.mockResolvedValueOnce({
      rows_considered: 1234,
      charts: {
        active_years_vs_final: [{ resultsFinalPositionNumber: 5, yearsActive: 7 }],
        positions_gained_over_time: [{ short_date: '2024-01-01', positionsGained: 1 }],
      },
      regressions: [],
      driver_performance: [],
      constructor_performance: [],
      dnf_reasons: [],
    });
    render(<Analytics />);
    await waitFor(() => {
      expect(screen.getByText(/Rows considered/)).toBeInTheDocument();
    });
    expect(screen.getByText('Analytics & Visualizations')).toBeInTheDocument();
  });

  it('shows error message on failure', async () => {
    apiMock.post.mockRejectedValueOnce(new Error('boom'));
    render(<Analytics />);
    await waitFor(() => {
      expect(screen.getByText('boom')).toBeInTheDocument();
    });
  });
});
