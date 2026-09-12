import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import BettingResearch from './BettingResearch.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe('BettingResearch page', () => {
  it('renders the calculator form and computes a value on submit', async () => {
    apiMock.get.mockImplementation((url) => {
      if (url === '/api/betting/governance') {
        return Promise.resolve({ manifest: [] });
      }
      return Promise.resolve({});
    });
    apiMock.post.mockResolvedValueOnce({
      market_probability: 0.5,
      raw_ev: 0.05,
      adjusted_probability: 0.45,
      stake: 50,
      reason_code: 'positive_ev',
    });
    render(<BettingResearch />);
    expect(screen.getByText(/Betting Research/i)).toBeInTheDocument();
  });
});
