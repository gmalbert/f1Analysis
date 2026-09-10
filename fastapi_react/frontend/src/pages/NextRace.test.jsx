import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import NextRace from './NextRace.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe('NextRace page', () => {
  it('renders the next-race header when data is present', async () => {
    apiMock.get.mockResolvedValueOnce({
      next_race: { grandPrixId: 'australia', year: 2025, grandPrixName: 'Australia', short_date: '2025-03-23' },
      prediction: { model: 'XGBoost', rows: [] },
    });
    render(<NextRace />);
    await waitFor(() => {
      expect(screen.getByText('Next Race')).toBeInTheDocument();
    });
  });

  it('handles no-next-race response', async () => {
    apiMock.get.mockResolvedValueOnce({ next_race: null });
    render(<NextRace />);
    await waitFor(() => {
      // Page heading still renders
      expect(screen.getByText('Next Race')).toBeInTheDocument();
    });
  });
});
