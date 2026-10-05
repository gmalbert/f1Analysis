import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

const apiMock = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../api.js', () => ({
  api: apiMock,
  downloadUrl: (p) => `/api/raw/download?path=${encodeURIComponent(p)}`,
}));

import Models from './Models.jsx';

beforeEach(() => {
  apiMock.get.mockReset();
  apiMock.post.mockReset();
});

describe('Models page', () => {
  it('renders the page heading and model selector after data loads', async () => {
    apiMock.get.mockImplementation((url) => {
      if (url === '/api/health') return Promise.resolve({ status: 'ok' });
      if (url === '/api/models') return Promise.resolve({ models: ['XGBoost', 'LightGBM'] });
      if (url.startsWith('/api/models/manifest')) return Promise.resolve({
        model_type: 'XGBoost',
        manifest: {
          metrics: { mae: 2.252, mse: 12.119, r2: 0.609 },
          feature_names: ['grid', 'wins'],
          estimator: 'XGBoost',
        },
      });
      if (url.endsWith('/monte_carlo')) return Promise.resolve({
        name: 'monte_carlo', data: { metadata: { source: 'offline' }, ranking: [{ feature: 'grid' }] },
      });
      if (url.startsWith('/api/models/precomputed/')) return Promise.resolve({ name: 'x', data: null });
      return Promise.resolve({});
    });
    const { container } = render(<Models />);
    expect(screen.getByText(/Predictive Models/i)).toBeInTheDocument();
    expect(await screen.findByText('2.252')).toBeInTheDocument();
    expect(screen.getByText('12.119')).toBeInTheDocument();
    expect(screen.getByText('0.609')).toBeInTheDocument();
    expect(screen.getByText('2', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.queryByText(/"estimator"/)).not.toBeInTheDocument();
    expect(container.querySelectorAll('pre.json')).toHaveLength(0);

    fireEvent.click(screen.getByRole('button', { name: 'Feature Selection' }));
    expect(await screen.findByText('offline')).toBeInTheDocument();
    expect(container.querySelectorAll('pre.json')).toHaveLength(0);

    fireEvent.click(screen.getByRole('button', { name: 'Debug' }));
    expect(await screen.findByText('ok')).toBeInTheDocument();
    expect(container.querySelectorAll('pre.json')).toHaveLength(0);
  });
});
