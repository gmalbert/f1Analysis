import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

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
  it('renders all data-backed next-race sections and supports hiding race details', async () => {
    apiMock.get.mockResolvedValueOnce({
      next_race: { date: '2026-10-04', time: '13:00', fullName: 'Singapore Grand Prix', courseLength: 4.94, turns: 19, laps: 62 },
      race_name: 'Singapore Grand Prix',
      model_mae: 1.25,
      position_mae_by_position: { 1: 0.5 },
      predictions: {
        predictions_by_model: {
          xgboost: {
            model_mae: 1.25,
            predictions: [
              { predicted_rank: 1, predicted_position: 1.2, predicted_position_std: 0.3, constructor: 'McLaren', driverName: 'Driver One' },
              { predicted_rank: 2, predicted_position: 2.4, constructor: 'Ferrari', driverName: 'Driver Two' },
            ],
          },
        },
      },
      past_results: [{ grandPrixYear: 2025, resultsDriverName: 'Driver One', resultsFinalPositionNumber: 1 }],
      dnf_diagnostics: { min: 0.01, max: 0.45, mean: 0.12 },
      dnf_predictions: [{ constructorName: 'McLaren', resultsDriverName: 'Driver One', driverDNFCount: 2, driverDNFPercentage: 5.1, PredictedDNFProbabilityPercentage: 3.2, PredictedDNFProbabilityStd: null }],
      legacy_predictions: [],
      safety_car_predictions: {
        mean: 42.1, min: 10.5, max: 80.4,
        rows: [{ grandPrixName: 'Singapore Grand Prix', grandPrixYear: 2026, PredictedSafetyCarProbabilityPercentage: 55.2, Type: 'Next Race' }],
      },
      race_messages: [{ Year: 2025, Round: 18, SafetyCarStatus: 1, redFlag: 0, yellowFlag: 2, doubleYellowFlag: 0, dnf_count: 3 }],
      driver_performance: [{ resultsDriverName: 'Driver One', average_ending_position: 2.1 }],
      constructor_performance: [{ constructorName: 'McLaren', average_ending_position: 2.5 }],
      fastest_pit_stops: {
        total: 1, pit_lane_time_constant: 21.5,
        rows: [{ year: 2025, round: 18, constructorName: 'McLaren', lap: 25, pitStopSeconds: 2.1, pit_time_stationary: 2.0 }],
      },
      weather: [{ average_temp: 29.4, average_humidity: 78 }],
    });

    render(<NextRace />);

    await waitFor(() => expect(screen.getByText('Predictive Results for Active Drivers')).toBeInTheDocument());
    expect(screen.getByText('Singapore Grand Prix', { selector: 'td' })).toBeInTheDocument();
    expect(screen.getByText(/MAE for Position Predictions: 1.250/)).toBeInTheDocument();
    expect(screen.getByText(/Min: 0.010/)).toBeInTheDocument();
    expect(screen.getByText(/Historical Safety Car Probabilities \(mean\): 42.100/)).toBeInTheDocument();
    expect(screen.getByText(/Pit Time Constant: 21.5/)).toBeInTheDocument();
    expect(screen.getByText(/Weather Data for Singapore Grand Prix/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: /Show Next Race/i }));
    expect(screen.queryByText('Next Race:')).not.toBeInTheDocument();
  });

  it('handles no-next-race response', async () => {
    apiMock.get.mockResolvedValueOnce({ next_race: null });
    render(<NextRace />);
    await waitFor(() => {
      expect(screen.getByText('No upcoming race found in the schedule.')).toBeInTheDocument();
    });
  });

  it('shows the request error state', async () => {
    apiMock.get.mockRejectedValueOnce(new Error('offline'));
    render(<NextRace />);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('offline'));
  });
});
