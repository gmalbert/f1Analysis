# Roadmap Part 2: New Data Sources for MAE ≤ 1.5

**Baseline after ROADMAP-1: 1.69 (80/20) / 1.80 (GroupKFold) → Target: ≤ 1.5 | Estimated impact of this section: 0.10–0.18**

> **Tire strategy data — already generated (Feb 2026):** `data_files/tire_strategy_data.csv` (3,452 rows, 2018–2025) and `data_files/race_pace_lap_data.csv` (3,268 rows) exist and are baked into the main CSV. Five tire features (`driver_avg_tire_degradation`, `driver_avg_num_stints`, `driver_soft_tendency`, `track_tire_degradation`, `tire_deg_x_track_deg`) and five pace features (`driver_fuel_corrected_pace`, `driver_race_pace_std`, `track_race_pace_std`, `race_vs_qual_consistency`, `driver_avg_completion_pct`) are in the model. Re-measurement confirmed **no MAE improvement over the 1.687 baseline** — XGBoost already imputed optimal values for the NaN slots. Next step: explore cross-source interaction features (e.g., `tire_deg × apparent_temperature`) that combine these new sources with expanded weather data.

---

## 2A. Expanded Weather Data (Open-Meteo — No API Key Needed)

The current weather pull collects only 4 variables: `average_temp`, `total_precipitation`, `average_humidity`, `average_wind_speed`. The Open-Meteo archive API provides ~20 additional variables at no cost.

### New variables to add to `f1-analysis-weather.py` and `f1-generate-analysis.py`

```python
# Add to the hourly params list in f1-analysis-weather.py:
HOURLY_PARAMS = [
    "temperature_2m",                # already have
    "precipitation",                 # already have
    "relativehumidity_2m",           # already have
    "windspeed_10m",                 # already have
    # --- NEW ADDITIONS ---
    "apparent_temperature",          # "feels like" — affects tire temps
    "windgusts_10m",                 # Peak gusts (affects aero/safety car risk)
    "weathercode",                   # WMO code (exact weather type: fog, storm, etc.)
    "cloudcover",                    # % cloud cover (affects track temp, rubber buildup)
    "surface_pressure",              # Barometric pressure (affects aerodynamic performance)
    "visibility",                    # km visibility (safety car risk indicator)
    "soil_temperature_0cm",          # Surface temp (grip proxy)
    "shortwave_radiation",           # Solar intensity (track surface heating)
]

# Mapped to grouped features in generator (add to f1-generate-analysis.py):
WEATHER_AGGREGATIONS = {
    'apparent_temperature': 'mean',
    'windgusts_10m': 'max',         # Worst gust of the day
    'weathercode': 'max',           # Worst weather code (more severe = higher number)
    'cloudcover': 'mean',
    'surface_pressure': 'mean',
    'visibility': 'min',            # Minimum visibility (worst point of day)
    'soil_temperature_0cm': 'mean',
    'shortwave_radiation': 'sum',   # Total solar energy received
}
```

### New engineered features from weather data

```python
# --- WEATHER-DERIVED ENGINEERED FEATURES ---

# Severe weather flag (weathercode >= 61 = rain events)
df['severe_weather_flag'] = (df['weathercode'].fillna(0) >= 61).astype(int)

# Pure wind flag (gusts > 50 km/h affect downforce-sensitive cars disproportionately)
df['high_wind_gust_flag'] = (df['windgusts_10m'].fillna(0) > 50).astype(int)

# Track surface heating (solar × soil_temp proxy)
df['track_heat_index'] = (
    df['shortwave_radiation'].fillna(0) * df['soil_temperature_0cm'].fillna(20) / 100)

# Safety car weather risk score
df['safety_car_weather_risk'] = (
    df['severe_weather_flag'] * 2 +
    df['high_wind_gust_flag'] +
    (df['total_precipitation'] > 5).astype(int) * 2 +
    (df['average_wind_speed'] > 40).astype(int)
).clip(0, 6)

# Apparent vs actual temp delta (thermal stress)
df['thermal_stress'] = (df['apparent_temperature'] - df['average_temp']).abs().fillna(0)
```

Add to `get_features_and_target()`:
```python
'severe_weather_flag', 'high_wind_gust_flag', 'track_heat_index',
'safety_car_weather_risk', 'thermal_stress',
```

**Estimated MAE impact:** 0.03–0.05  
**Effort:** 3–4 hrs (mainly expanding the hourly pull and adding aggregation logic)

---

## 2B. First-Lap Position Data (Jolpica/Ergast API — Free)

First-lap positions are a powerful predictor of final position (safety card bunching, first-corner incidents). The Ergast/Jolpica API exposes lap-by-lap positions free of charge.

### API endpoint

```
GET https://api.jolpi.ca/ergast/f1/{year}/{round}/laps/1.json
```

Returns an array of `Timings` objects with `driverId` and `position` for lap 1 of every race.

### Step 1 — New pull script: `f1-first-lap-positions.py`

```python
#!/usr/bin/env python3
"""Pull first-lap positions from the Jolpica/Ergast F1 API."""
import requests
import pandas as pd
import time
from pathlib import Path

DATA_DIR = Path('data_files')
OUTPUT_FILE = DATA_DIR / 'first_lap_positions.csv'

BASE_URL = "https://api.jolpi.ca/ergast/f1/{year}/{round}/laps/1.json"

existing = set()
if OUTPUT_FILE.exists():
    old = pd.read_csv(OUTPUT_FILE, sep='\t')
    existing = set(zip(old['year'], old['round']))

all_rows = []

for year in range(2010, 2027):
    for rnd in range(1, 25):
        if (year, rnd) in existing:
            continue
        url = BASE_URL.format(year=year, round=rnd)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                break  # No more rounds this year
            data = response.json()
            races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            if not races:
                break
            race = races[0]
            laps = race.get('Laps', [])
            if not laps:
                continue
            lap1 = laps[0]
            for timing in lap1.get('Timings', []):
                all_rows.append({
                    'year': year,
                    'round': rnd,
                    'driverId': timing.get('driverId'),
                    'first_lap_position': int(timing.get('position', 0)),
                })
            print(f"  {year} R{rnd}: {len(lap1.get('Timings', []))} drivers")
            time.sleep(0.2)  # Be polite to free API
        except Exception as e:
            print(f"  Error {year} R{rnd}: {e}")
            time.sleep(1)

result = pd.DataFrame(all_rows)
if OUTPUT_FILE.exists():
    old = pd.read_csv(OUTPUT_FILE, sep='\t')
    result = pd.concat([old, result], ignore_index=True).drop_duplicates(
        subset=['year', 'round', 'driverId'], keep='last')
result.to_csv(OUTPUT_FILE, sep='\t', index=False)
print(f"Saved {len(result)} rows to {OUTPUT_FILE}")
```

### Step 2 — Feature engineering (add to `f1-generate-analysis.py`)

```python
# --- FIRST LAP POSITION FEATURES ---
first_lap_file = os.path.join(DATA_DIR, 'first_lap_positions.csv')
if os.path.exists(first_lap_file):
    first_lap = pd.read_csv(first_lap_file, sep='\t')
    df = df.merge(first_lap,
        left_on=['grandPrixYear', 'round', 'driverId'],
        right_on=['year', 'round', 'driverId'],
        how='left', suffixes=('', '_fl'))

    # How many positions gained/lost in lap 1
    df['first_lap_delta'] = (
        df['resultsStartingGridPositionNumber'] - df['first_lap_position'])

    # Historical average first lap delta for this driver
    df['driver_avg_first_lap_delta'] = (
        df.groupby('resultsDriverName')['first_lap_delta']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    # Driver's historical first-lap crash/incident rate
    df['driver_first_lap_incident_rate'] = (
        df.groupby('resultsDriverName')['first_lap_position']
        .transform(lambda x: (x.shift(1) > x.shift(1) * 1.3).rolling(5, min_periods=1).mean()))

    print("Created 3 first-lap position features")
```

### Step 3 — Add to `get_features_and_target()`:

```python
'first_lap_delta',
'driver_avg_first_lap_delta',
'driver_first_lap_incident_rate',
```

**Estimated MAE impact:** 0.04–0.07  
**Effort:** 2–3 hrs (API pull + engineering; data available from 2010)

---

## 2C. Circuit Altitude & Atmospheric Density

High-altitude circuits (Mexico City: 2,285m; Interlagos: 792m; Austin: 148m) have dramatically different power unit performance and tire degradation. This is a static lookup — zero API cost.

### Step 1 — Create `data_files/circuit_altitude.csv`

Compile once from Wikipedia/Open-Elevation API. Key circuits:

```
circuitRef	altitude_m	air_density_index
rodriguez	2285	0.74	# Mexico City
interlagos	792	0.92	# Sao Paulo
americas	148	0.99	# Austin
red_bull_ring	660	0.93	# Austria
suzuka	40	1.00
silverstone	153	0.99
monza	160	0.99
monaco	7	1.00
spa	400	0.97
baku	-28	1.00	# Sea level
jeddah	15	1.00
bahrain	5	1.00
albert_park	10	1.00
```

### Step 2 — Add to `f1-generate-analysis.py`

```python
# --- ALTITUDE & AIR DENSITY FEATURES ---
altitude_file = os.path.join(DATA_DIR, 'circuit_altitude.csv')
if os.path.exists(altitude_file):
    altitude_data = pd.read_csv(altitude_file, sep='\t')
    df = df.merge(altitude_data, on='circuitRef', how='left')
    
    # Air density index (approximation: rho/rho_0 = exp(-altitude/8500))
    df['air_density_index'] = df['altitude_m'].apply(
        lambda alt: np.exp(-alt/8500) if pd.notna(alt) else 1.0)
    
    # Altitude category (affects tire compound choices)
    df['is_high_altitude'] = (df['altitude_m'].fillna(0) > 500).astype(int)

    # Constructor power unit efficiency at altitude (Mercedes vs Honda etc.)
    # High-altitude = less ERS dependency advantage for top units
    df['altitude_x_constructor_power'] = (
        df['air_density_index'] * df.get('constructor_raw_pace', 1))

    print("Created 3 altitude/air density features")
```

### Step 3 — Add to `get_features_and_target()`:

```python
'altitude_m', 'air_density_index', 'is_high_altitude',
```

**Estimated MAE impact:** 0.02–0.03 (particularly strong for Mexico/Brazil rounds)  
**Effort:** 1 hr (static CSV + code)

---

## 2D. Penalty / Incident Data

Grid penalties (power unit changes = -5, -10 positions) are currently inferred but not directly modeled. The Jolpica API provides qualifying grid position adjustments.

### Step 1 — Derive from existing data (no new API call)

```python
# In f1-generate-analysis.py (after qualifying merge):
# Grid penalty = difference between qualifying position and starting grid position
df['grid_penalty'] = (
    df['resultsStartingGridPositionNumber'] - df['resultsQualificationPositionNumber']
).clip(0, 20)  # Clip negatives (parc ferme exceptions)

df['had_grid_penalty'] = (df['grid_penalty'] > 0).astype(int)

# Penalty severity category
df['penalty_severity'] = pd.cut(
    df['grid_penalty'],
    bins=[-1, 0, 5, 10, 25],
    labels=['none', 'minor', 'moderate', 'severe']
)
```

### Step 2 — Historical penalty tendency per team (reliability proxy)

```python
# Constructor's penalty rate per season (power unit unreliability metric)
df['constructor_penalty_rate'] = (
    df.groupby(['constructorId', 'grandPrixYear'])['had_grid_penalty']
    .transform(lambda x: x.shift(1).expanding().mean()))
```

Add to `get_features_and_target()`:
```python
'grid_penalty', 'had_grid_penalty', 'constructor_penalty_rate',
```

**Estimated MAE impact:** 0.01–0.03  
**Effort:** 1 hr (derived from existing data)

---

## 2E. Betting Odds (Optional — Premium Data)

Pre-race odds are the market's probability-weighted consensus of all factors. Studies show betting-implied probability correlates 0.72 with final position.

**Recommended source:** The Odds API (https://the-odds-api.com)  
**Free tier:** 500 requests/month (sufficient for ~20 races/season + historical backfill)  
**Paid tier:** $50/month for full historical + real-time

### Feature engineering:

```python
# From The Odds API, store: driver_name, win_odds (decimal), podium_odds
# Implied probability = 1 / decimal_odds (raw, unnormalized)
df['win_probability_implied'] = 1 / df['win_odds'].replace(0, np.nan)
df['podium_probability_implied'] = 1 / df['podium_odds'].replace(0, np.nan)

# Market rank (order by implied win probability within each race)
df['market_rank'] = df.groupby('raceId')['win_probability_implied'].rank(ascending=False)
```

**Estimated MAE impact:** 0.05–0.08  
**Effort:** 4–6 hrs (API integration + historical backfill limited by free tier)  
**Note:** Treat as optional P3 due to cost and access complexity.

---

## Summary

| Data Source | New Features | Est. MAE Impact | Effort | Priority |
|-------------|-------------|----------------|--------|----------|
| Expanded Open-Meteo weather | 5 | 0.03–0.05 | 3–4 hrs | **P1** |
| First-lap positions (Jolpica) | 3 | 0.04–0.07 | 2–3 hrs | **P1** |
| Circuit altitude (static CSV) | 3 | 0.02–0.03 | 1 hr | **P2** |
| Grid penalties (derived) | 3 | 0.01–0.03 | 1 hr | **P2** |
| Betting odds (The Odds API) | 3 | 0.05–0.08 | 4–6 hrs | **P3** |
| **Total estimated** | **17** | **0.15–0.26** | | |

**Net effect: MAE 1.69 → ~1.55–1.65 (this section alone, starting from post-ROADMAP-1 baseline)**  
**Combined with ROADMAP_1: MAE → ~1.55–1.75**
