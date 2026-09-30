import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

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
    apiMock.post.mockResolvedValue({
      market_probability: 0.5,
      raw_ev: 0.05,
      adjusted_probability: 0.45,
      stake: 50,
      reason_code: 'positive_ev',
    });
    render(<BettingResearch />);
    expect(screen.getByText(/Betting Research/i)).toBeInTheDocument();
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith(
      '/api/betting/value',
      expect.objectContaining({ model_probability: 0.25, decimal_odds: 2.1, bankroll: 10000 }),
    ));
    expect(await screen.findByText('50.00%')).toBeInTheDocument();
  });

  it('offers the field template and submits the selected simulation count', async () => {
    apiMock.post.mockImplementation((url) => Promise.resolve(url === '/api/betting/simulate'
      ? { rows: [{ driver_id: 'driver-a', win_probability: 0.5 }], columns: ['driver_id', 'win_probability'] }
      : { market_probability: 0.5, raw_ev: 0.05, adjusted_probability: 0.45, stake: 50, reason_code: 'positive_ev' }));
    render(<BettingResearch />);
    fireEvent.click(screen.getByRole('button', { name: 'Field simulation' }));
    const template = screen.getByRole('link', { name: 'Download input template' });
    expect(template).toHaveAttribute('download', 'f1_field_simulation_template.csv');
    expect(template.getAttribute('href')).toContain('data:text/csv');
    fireEvent.change(screen.getByRole('slider', { name: 'Simulation count' }), { target: { value: '12000' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run coherent field simulation' }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/api/betting/simulate', expect.objectContaining({ simulations: 12000 })));
    await waitFor(() => expect(screen.getAllByText('driver-a')).toHaveLength(2));
  });
});
