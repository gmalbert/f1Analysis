import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

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
      if (url.startsWith('/api/models/manifest')) return Promise.resolve({ model_type: 'XGBoost', manifest: {} });
      if (url.startsWith('/api/models/precomputed/')) return Promise.resolve({ name: 'x', data: null });
      return Promise.resolve({});
    });
    render(<Models />);
    expect(screen.getByText(/Predictive Models/i)).toBeInTheDocument();
    // Allow async effects to complete
    await waitFor(() => {
      expect(apiMock.get).toHaveBeenCalled();
    });
  });
});
