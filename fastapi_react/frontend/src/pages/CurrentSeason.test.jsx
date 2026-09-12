import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import CurrentSeason from './CurrentSeason.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe('CurrentSeason page', () => {
  it('shows loading then renders the schedule', async () => {
    apiMock.get.mockResolvedValueOnce({
      year: 2025,
      columns: ['round', 'grandPrixName', 'date', 'seasonStatus'],
      rows: [
        { round: 1, grandPrixName: 'Bahrain', date: '2025-03-02', seasonStatus: 'Completed' },
        { round: 2, grandPrixName: 'Saudi Arabia', date: '2025-03-09', seasonStatus: 'Completed' },
        { round: 3, grandPrixName: 'Australia', date: '2025-03-23', seasonStatus: 'Next Race' },
      ],
    });
    render(<CurrentSeason />);
    await waitFor(() => {
      expect(screen.getByText('Bahrain')).toBeInTheDocument();
    });
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(/2025/);
  });

  it('shows error state when the API fails', async () => {
    apiMock.get.mockRejectedValueOnce(new Error('network down'));
    render(<CurrentSeason />);
    await waitFor(() => {
      expect(screen.getByText('network down')).toBeInTheDocument();
    });
  });
});
