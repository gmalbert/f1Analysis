# Frontier Enhancement Blueprint

Existing docs already cover race pace, feature engineering, FastF1/Ergast-style sources, MAE optimization, Monte Carlo, deep models, precomputation, and Streamlit performance. The strongest unexplored work is a generative lap/race-state model and decision-focused strategy evaluation.

## Latent lap-time decomposition

Separate driver pace, car pace, fuel burn, tire age/compound, track evolution, traffic, weather, and degradation rather than asking one regressor to absorb them.

```python
lap_time = (driver_skill[driver] + car_pace[constructor, race]
            + fuel_curve(lap) + tire_curve(compound, tire_age)
            + track_evolution(session_time) + traffic_penalty(gap_ahead)
            + weather_effect(track_temp, rain))
```

Fit hierarchically so rookies and new upgrades borrow strength. Preserve posterior/ensemble draws for simulation instead of only point estimates.

## Competing race hazards

Model mechanical DNF, collision, safety car/VSC, rain onset, pit stop, and tire failure as time-varying hazards. Safety-car probability must respond to circuit, weather, field state, and elapsed laps; do not sample a constant race-level rate.

## Strategy policy evaluation

Treat pit decisions as policies under partial information. Evaluate candidate policies with simulation and off-policy sensitivity rather than claiming causal superiority from observed stops (which are heavily confounded by race state).

```python
def choose_action(state, sims, risk=0.25):
    utility = {a: np.mean(v) - risk * np.std(v) for a, v in sims.items()}
    return max(utility, key=utility.get), utility
```

## Data additions

- Timing-loop/telemetry coverage manifests and clock reconciliation.
- Tire-set history, estimated fuel, traffic/dirty-air, pit-lane loss distribution, and track status.
- Upgrade packages and specification changes with effective weekend.
- Weather radar snapshots, marshal-sector incidents, and race-control latency.
- Timestamped market prices for qualifying, podium, points, and head-to-heads.

## UI additions

- Race-state sandbox with editable weather, safety car, degradation, and pit windows.
- Strategy tree showing expected position distribution and downside, not one “optimal” lap.
- Lap-time decomposition and residual diagnostics per driver.
- Data coverage/latency badge for every session.
- Model-versus-market replay frozen at each timestamp.

## Gates

Backtest on leave-one-race and leave-one-season-forward splits. Report MAE/CRPS for lap and finishing distributions, hazard calibration, pit-policy regret in simulation, rank probability score, and market CLV. Slice by wet races, street circuits, rookies, upgrades, red flags, and sparse telemetry. Require deterministic seeded replays from versioned raw snapshots.
