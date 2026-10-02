import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

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

const historical = {
  metadata: { generated_at: '2026-09-30', model_type: 'XGBoost' },
  position_cv: { mae: 1.1 },
  dnf_validation: { auc: 0.7 },
  safety_car_validation: { auc: 0.6 },
  holdout: {
    metrics: { mae: 1.2 },
    rows: [
      { ActualFinalPosition: 1, PredictedFinalPosition: 1.4, Error: 0.4 },
      { ActualFinalPosition: 2, PredictedFinalPosition: 2.5, Error: 0.5 },
      { ActualFinalPosition: 12, PredictedFinalPosition: 10.5, Error: -1.5 },
      { ActualFinalPosition: 17, PredictedFinalPosition: 16.0, Error: -1.0 },
    ],
  },
};

const artifactData = {
  position_mae: {
    metadata: { model_mae: 1.2, seasons: [2024, 2025] },
    by_driver: { Alice: { mae: 1.1 }, Bob: { mae: 1.4 } },
    position_groups: { Podium: { mae: 0.8, count: 3 }, Midfield: { mae: 1.5, count: 10 } },
  },
  historical_validation: historical,
  shap: { metadata: {}, feature_importance: [{ feature: 'grid', importance: 0.4 }], top_20: [{ feature: 'grid', importance: 0.4 }] },
  permutation: { metadata: {}, feature_importance: [{ feature: 'practice', importance: 0.3 }, { feature: 'grid', importance: 0.6 }] },
  monte_carlo: { metadata: {}, best_result: { mae: 1.0 }, top_20_results: [{ features: 'grid,practice', mae: 1.0 }], feature_frequency_top_20: { grid: 20, practice: 18 } },
  monte_carlo_log: { entries: [{ iteration: 1 }] },
  rfe: { feature_ranking: [{ feature: 'grid', rank: 1 }], selected_features: ['grid'] },
  boruta: { feature_ranking: [{ feature: 'practice', rank: 1 }], selected_features: ['practice'] },
  hyperparam_bayesian: { best_params: { depth: 4 } },
  hyperparam_grid: { best_params: { depth: 5 } },
};

function setupApi({ expensive = true } = {}) {
  apiMock.get.mockImplementation((url) => {
    if (url === '/api/health') return Promise.resolve({ status: 'ok', expensive_tools_enabled: expensive });
    if (url === '/api/models') return Promise.resolve({ models: ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble (XGBoost + LightGBM + CatBoost)', 'Position Group', 'Track-Weighted Ensemble'] });
    if (url.startsWith('/api/models/manifest')) return Promise.resolve({
      model_type: 'XGBoost',
      manifest: {
        metrics: { mse: 2.2, r2: 0.7, mae: 1.2, mean_error: -0.1 },
        best_iteration: 19,
        feature_names: ['grid', 'practice'],
        hyperparameters: { max_depth: 4 },
      },
    });
    if (url.startsWith('/api/models/precomputed/')) {
      const name = decodeURIComponent(url.split('/').pop());
      return Promise.resolve({ name, data: artifactData[name] ?? null });
    }
    return Promise.resolve({});
  });
}

describe('Models page', () => {
  it('renders and exercises every advanced model tab', async () => {
    setupApi();
    apiMock.post.mockResolvedValue({ status: 'ok', result: { rows: 3 } });

    render(<Models />);

    await waitFor(() => expect(screen.getByText('Predictive Data Model Metrics')).toBeInTheDocument());
    expect(screen.getByText('Mean Absolute Error')).toBeInTheDocument();
    expect(screen.getByText(/Boosting rounds used: 20/)).toBeInTheDocument();

    const tabs = [
      ['🔍 Feature Analysis', 'Feature Analysis'],
      ['🎯 Feature Selection', 'Feature Selection Tools'],
      ['🏎️ Position-Specific Analysis', 'Position Group Analysis'],
      ['⚙️ Hyperparameters', 'Hyperparameter Tuning'],
      ['📈 Historical Validation', 'Historical Validation'],
      ['🛠️ Debug & Experiments', 'Compare Different Bin Counts (q)'],
    ];

    for (const [tabName, expectedText] of tabs) {
      fireEvent.click(screen.getByRole('tab', { name: tabName }));
      await waitFor(() => expect(screen.getByText(expectedText)).toBeInTheDocument());
    }

    const runButton = screen.getByRole('button', { name: /Run monte carlo/i });
    fireEvent.click(runButton);
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/api/tools/run', { tool: 'monte_carlo', args: [] }));
  });

  it('shows hosted-mode warning and handles missing manifest', async () => {
    apiMock.get.mockImplementation((url) => {
      if (url === '/api/health') return Promise.resolve({ status: 'ok', expensive_tools_enabled: false });
      if (url === '/api/models') return Promise.resolve({ models: ['XGBoost'] });
      if (url.startsWith('/api/models/manifest')) return Promise.resolve({ manifest: null });
      if (url.startsWith('/api/models/precomputed/')) return Promise.resolve({ data: null });
      return Promise.resolve({});
    });

    render(<Models />);
    await waitFor(() => expect(screen.getByText(/Research controls are disabled/)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/No trained model artifact is available/)).toBeInTheDocument());
  });

  it('surfaces initial API errors', async () => {
    apiMock.get.mockRejectedValue(new Error('model API failed'));
    render(<Models />);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('model API failed'));
  });
});
