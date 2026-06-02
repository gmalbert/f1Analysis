# Formula 1 Analysis — Model Suggested Enhancements

## Priority 1: MAE Reduction (Target: ≤1.5)

### Qualifying Sector Times
- S1/S2/S3 sector times from FastF1 are already partially integrated. Ensure all three sectors are used as separate features (not aggregated), since S1 and S3 performance can predict overtaking likelihood.

### Tyre Compound Strategy
- Add `expected_tyre_deg_rate` based on circuit category (smooth = Barcelona, abrasive = Bahrain). Tyre strategy shifts race position significantly.

### Safety Car Probability Model
- Build a dedicated safety car probability model per circuit using `all_race_control_messages.csv`.
- Use as a feature input to the main race model (higher SC probability → positions cluster, fewer outliers).

### Neural Network Layer
- Current ensemble: XGBoost + LightGBM + CatBoost. Add a simple MLP post-processing layer to learn non-linear interactions the tree models miss.
- Use `MLPRegressor(hidden_layer_sizes=(64, 32))` with the three base model outputs as features.

## Priority 2: Feature Engineering

### Driver-Constructor Synergy
- Some drivers vastly outperform their car; others underperform. Add a `driver_vs_constructor_norm` ratio: driver's actual avg position vs. constructor's avg position.

### Circuit Category Encoding
- Streets (Monaco, Singapore) vs. high-speed (Monza, Spa) vs. technical (Hungary, Suzuka) behave differently.
- Create a 3-class `circuit_type` feature from circuit characteristics in F1DB.

### Tyre Stint Remaining
- Available from FastF1. Add `tyres_age_at_race` (laps on current compound at race start) as a feature.

## Priority 3: DNF Model

### Dedicated DNF Classifier
- XGBoost binary classifier: `is_dnf` target. Features: `reliability_index` (DNF rate per 100 starts), `constructor_reliability_l3`.
- Use DNF probability to penalise win probability predictions for fragile cars.

## Priority 4: Calibration

- Apply Platt scaling to win probability outputs.
- Track and publish MAE on the Model Performance tab after each race weekend.
