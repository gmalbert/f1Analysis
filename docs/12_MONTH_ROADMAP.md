# F1 Analysis — 12-Month Feature Roadmap

> Generated: 2026-07-31 | Horizon: August 2026 – July 2027

---

## Executive Summary

This roadmap pushes the F1 Analysis platform toward sub-1.5 MAE for final position
prediction while adding live telemetry, Transformer-based sequence modeling,
automated race-week pipelines, and a fully interactive race strategy simulator.

---

## Q1 (Aug–Oct 2026) — Advanced Feature Engineering

### Feature 1 — OpenF1 Real-Time Telemetry Integration

Replace FastF1 post-race downloads with OpenF1 API for lap-by-lap telemetry
during live sessions. Cache aggressively to avoid re-fetching.

```python
# scripts/fetch_openf1_telemetry.py
import requests, pandas as pd
from pathlib import Path
import time

OPENF1_BASE = "https://api.openf1.org/v1"
DATA_DIR = Path("data_files")

def fetch_lap_data(session_key: int) -> pd.DataFrame:
    """Fetch all lap times for a session from OpenF1."""
    url = f"{OPENF1_BASE}/laps"
    params = {"session_key": session_key, "limit": 5000}
    resp = requests.get(url, params=params, timeout=20)
    df = pd.DataFrame(resp.json())
    out = DATA_DIR / f"openf1_laps_{session_key}.parquet"
    df.to_parquet(out, index=False)
    return df

def fetch_car_data(session_key: int, driver_number: int) -> pd.DataFrame:
    """Fetch raw telemetry (speed, throttle, brake, gear) per driver."""
    resp = requests.get(
        f"{OPENF1_BASE}/car_data",
        params={"session_key": session_key, "driver_number": driver_number},
        timeout=20,
    )
    return pd.DataFrame(resp.json())

def build_max_speed_per_sector(session_key: int, driver_number: int) -> dict:
    """Compute max speed per track sector for corner speed analysis."""
    telemetry = fetch_car_data(session_key, driver_number)
    if "speed" not in telemetry.columns:
        return {}
    return {
        "max_speed": telemetry["speed"].max(),
        "avg_speed": telemetry["speed"].mean(),
        "full_throttle_pct": (telemetry["throttle"] > 95).mean(),
        "heavy_brake_pct": (telemetry["brake"] > 50).mean(),
    }
```

### Feature 2 — Sector Time Analysis Features

Build per-driver sector-time features (S1, S2, S3 relative to session best).
These are strong leading indicators of race pace.

```python
# scripts/sector_time_features.py
import fastf1, pandas as pd, numpy as np
from pathlib import Path

DATA_DIR = Path("data_files")

def build_sector_features(year: int, round_num: int) -> pd.DataFrame:
    """Build sector-time deltas relative to session best."""
    session = fastf1.get_session(year, round_num, "Q")
    session.load()
    laps = session.laps.pick_quicklaps()

    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        col_s = f"{sector}_s"
        laps[col_s] = laps[sector].dt.total_seconds()
        session_best = laps[col_s].min()
        laps[f"{sector}_delta_pct"] = (laps[col_s] - session_best) / session_best

    per_driver = (
        laps.groupby("Driver")
        .agg(
            best_s1=("Sector1Time_s", "min"),
            best_s2=("Sector2Time_s", "min"),
            best_s3=("Sector3Time_s", "min"),
            s1_delta_pct=("Sector1Time_delta_pct", "min"),
            s2_delta_pct=("Sector2Time_delta_pct", "min"),
            s3_delta_pct=("Sector3Time_delta_pct", "min"),
        )
        .reset_index()
    )
    out = DATA_DIR / f"sector_features_{year}_R{round_num}.parquet"
    per_driver.to_parquet(out, index=False)
    return per_driver
```

### Feature 3 — Tyre Compound Lifecycle Model

Model the degradation curve per compound (soft, medium, hard) per circuit.
Predict optimal stint length and pitstop delta.

```python
# scripts/tyre_lifecycle.py
import fastf1, pandas as pd, numpy as np
from scipy.optimize import curve_fit

def exponential_deg(lap_on_tyre: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential tyre degradation model: a + b * exp(c * lap)."""
    return a + b * np.exp(c * lap_on_tyre)

def fit_tyre_model(race_laps: pd.DataFrame, compound: str) -> dict:
    """Fit degradation model for a specific compound."""
    data = race_laps[
        (race_laps["Compound"] == compound) &
        (race_laps["LapTime_s"] < race_laps["LapTime_s"].quantile(0.95))
    ].copy()
    data["TyreLife_num"] = pd.to_numeric(data["TyreLife"], errors="coerce")
    data = data.dropna(subset=["TyreLife_num", "LapTime_s"])

    if len(data) < 10:
        return {"a": data["LapTime_s"].mean(), "b": 0.1, "c": 0.05}

    try:
        popt, _ = curve_fit(
            exponential_deg,
            data["TyreLife_num"].values,
            data["LapTime_s"].values,
            p0=[data["LapTime_s"].min(), 0.1, 0.05],
            bounds=([0, 0, -2], [300, 10, 2]),
            maxfev=5000,
        )
        return {"a": popt[0], "b": popt[1], "c": popt[2], "compound": compound}
    except Exception:
        return {"a": data["LapTime_s"].mean(), "b": 0.1, "c": 0.05}

def predict_deg_at_lap(model: dict, lap: int) -> float:
    return exponential_deg(lap, model["a"], model["b"], model["c"])
```

### Feature 4 — Circuit DNA Features

For each circuit, compute historical characteristics: overtaking difficulty,
safety car probability, pit-lane loss, and compound preference. Feed as
fixed circuit features.

```python
# scripts/circuit_dna.py
import pandas as pd, json
from pathlib import Path

DATA_DIR = Path("data_files")

CIRCUIT_DNA = {
    "monaco": {
        "overtaking_difficulty": 0.95,  # 0 = easy, 1 = impossible
        "sc_probability": 0.75,
        "pit_lane_loss": 22.5,  # seconds
        "preferred_compound": "medium",
        "altitude_m": 7,
        "circuit_length_km": 3.337,
    },
    "monza": {
        "overtaking_difficulty": 0.25,
        "sc_probability": 0.45,
        "pit_lane_loss": 19.8,
        "preferred_compound": "soft",
        "altitude_m": 162,
        "circuit_length_km": 5.793,
    },
    "spa": {
        "overtaking_difficulty": 0.30,
        "sc_probability": 0.65,
        "pit_lane_loss": 23.1,
        "preferred_compound": "hard",
        "altitude_m": 400,
        "circuit_length_km": 7.004,
    },
}

def load_circuit_features(circuit_name: str) -> dict:
    normalized = circuit_name.lower().replace(" ", "_").replace("-", "_")
    return CIRCUIT_DNA.get(normalized, {
        "overtaking_difficulty": 0.5,
        "sc_probability": 0.5,
        "pit_lane_loss": 21.0,
        "preferred_compound": "medium",
        "altitude_m": 200,
        "circuit_length_km": 5.0,
    })
```

### Feature 5 — Qualifying Simulation (Monte Carlo Q1/Q2/Q3)

Monte Carlo simulate qualifying sessions based on practice-lap distributions.
Predict grid position probabilities before sessions begin.

```python
# scripts/qualifying_simulation.py
import numpy as np, pandas as pd
from pathlib import Path

def simulate_qualifying(
    practice_times: pd.DataFrame, n_sim: int = 10_000
) -> pd.DataFrame:
    """
    practice_times: [driver, best_lap_s, lap_std_s]
    Returns P(each driver qualifies in each Q position).
    """
    drivers = practice_times["driver"].tolist()
    means = practice_times["best_lap_s"].values
    stds = practice_times["lap_std_s"].values.clip(min=0.01)

    position_counts = {d: [0] * len(drivers) for d in drivers}

    for _ in range(n_sim):
        sim_times = np.random.normal(means, stds)
        sorted_idx = np.argsort(sim_times)
        for pos, idx in enumerate(sorted_idx):
            position_counts[drivers[idx]][pos] += 1

    results = []
    for driver, counts in position_counts.items():
        row = {"driver": driver}
        for pos, count in enumerate(counts):
            row[f"p_pos_{pos+1}"] = count / n_sim
        row["expected_grid"] = sum((i + 1) * c / n_sim for i, c in enumerate(counts))
        results.append(row)

    return pd.DataFrame(results).sort_values("expected_grid")
```

---

## Q2 (Nov 2026 – Jan 2027) — Advanced Modeling

### Feature 6 — Transformer Sequence Model for Race Position Prediction

Replace XGBoost tabular model with a Transformer that reads lap-by-lap
sequences of each driver's performance during the race.

```python
# scripts/transformer_race_model.py
import torch, torch.nn as nn
import numpy as np, pandas as pd

SEQ_LEN = 20  # laps per sequence

class RaceTransformer(nn.Module):
    def __init__(self, n_drivers: int = 20, n_lap_features: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.lap_embed = nn.Linear(n_lap_features, d_model)
        self.driver_embed = nn.Embedding(n_drivers, d_model)
        self.pos_embed = nn.Embedding(SEQ_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),  # predicted final position
        )

    def forward(self, lap_feats: torch.Tensor, driver_ids: torch.Tensor) -> torch.Tensor:
        B, T, F = lap_feats.shape
        x = self.lap_embed(lap_feats)
        x += self.driver_embed(driver_ids).unsqueeze(1).expand_as(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x += self.pos_embed(positions)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)  # use last lap's encoding

def prepare_race_sequences(pbp: pd.DataFrame) -> tuple:
    """Build (lap_feats, driver_ids, final_positions) tensors."""
    lap_feat_cols = ["laptime_s", "position", "tyre_life", "compound_code",
                     "s1_s", "s2_s", "s3_s", "pit_flag"]
    Xs, Ds, Ys = [], [], []
    for (race_id, driver), driver_df in pbp.groupby(["raceId", "driverId"]):
        driver_df = driver_df.sort_values("lap").head(SEQ_LEN)
        if len(driver_df) < SEQ_LEN:
            continue
        laps = driver_df[lap_feat_cols].values.astype(np.float32)
        Xs.append(laps)
        Ds.append(driver % 20)  # driver index
        Ys.append(driver_df.iloc[-1].get("final_position", 10))
    return torch.FloatTensor(Xs), torch.LongTensor(Ds), torch.FloatTensor(Ys)
```

### Feature 7 — Undercut / Overcut Strategy Optimizer

Given current position, gap to next car, and tyre life, compute the
optimal pit window (undercut vs overcut) for each driver.

```python
# scripts/strategy_optimizer.py
import numpy as np

def compute_undercut_delta(
    leader_lap_time: float,
    follower_lap_time: float,
    pit_loss_seconds: float,
    new_tyre_benefit: float,  # seconds per lap faster on new tyres
    laps_remaining: int,
) -> float:
    """
    Compute expected position delta from undercutting.
    Positive = undercut gains position.
    """
    # After pit: lose pit_loss_seconds but gain new_tyre_benefit per lap
    benefit_per_lap = new_tyre_benefit
    crossover_lap = pit_loss_seconds / max(benefit_per_lap, 0.001)

    if crossover_lap > laps_remaining:
        return -pit_loss_seconds  # no time to recover
    remaining_after_crossover = laps_remaining - crossover_lap
    net_gain = remaining_after_crossover * benefit_per_lap - pit_loss_seconds
    return round(net_gain, 3)

def optimal_pit_window(
    driver_tyre_age: int,
    deg_model: dict,
    pit_loss: float,
    laps_remaining: int,
    current_gap_ahead: float,
) -> dict:
    """Find lap that maximizes net gain from pit stop."""
    best_lap, best_gain = None, -999
    for lap_to_pit in range(1, min(laps_remaining, 20)):
        # Time "spent" on old tyres until pit
        current_deg = deg_model.get("b", 0.1) * np.exp(
            deg_model.get("c", 0.05) * (driver_tyre_age + lap_to_pit)
        )
        new_tyre_benefit = current_deg  # time saved per lap on new vs old
        gain = compute_undercut_delta(
            leader_lap_time=0,
            follower_lap_time=0,
            pit_loss_seconds=pit_loss,
            new_tyre_benefit=new_tyre_benefit,
            laps_remaining=laps_remaining - lap_to_pit,
        )
        if gain > best_gain:
            best_gain, best_lap = gain, driver_tyre_age + lap_to_pit
    return {"optimal_lap": best_lap, "expected_net_gain": best_gain}
```

### Feature 8 — Constructor Championship Probability

Daily Monte Carlo simulation of remaining races. Output each constructor's
probability of winning the championship.

```python
# scripts/championship_simulator.py
import numpy as np, pandas as pd
from collections import defaultdict

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

def simulate_championship(
    remaining_rounds: int,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    model_fn,  # callable: (driver, circuit) -> P(each position 1-20)
    n_sim: int = 10_000,
) -> pd.DataFrame:
    constructors = constructor_standings["constructor"].tolist()
    constructor_wins = defaultdict(int)

    for _ in range(n_sim):
        pts = constructor_standings.set_index("constructor")["points"].to_dict()
        for _ in range(remaining_rounds):
            circuit = "generic"  # in production: iterate over actual circuits
            for driver_row in driver_standings.itertuples():
                probs = model_fn(driver_row.driver, circuit)
                position = np.random.choice(range(1, 21), p=probs)
                race_pts = F1_POINTS.get(position, 0)
                constructor = driver_row.constructor
                pts[constructor] = pts.get(constructor, 0) + race_pts
                if position == 1:
                    pts[constructor] += 1  # fastest lap bonus

        winner = max(pts, key=pts.get)
        constructor_wins[winner] += 1

    return pd.DataFrame([
        {"constructor": c, "championship_prob": constructor_wins[c] / n_sim}
        for c in constructors
    ]).sort_values("championship_prob", ascending=False)
```

### Feature 9 — Wet Weather Performance Classifier

Flag circuits where wet conditions are likely. Classify drivers by wet-weather
performance vs dry-weather performance. Adjust qualifying model.

```python
# scripts/wet_weather_performance.py
import fastf1, pandas as pd

def classify_driver_wet_performance(
    driver_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare driver finishing position distribution in wet vs dry races."""
    wet = driver_results[driver_results["race_condition"] == "wet"]
    dry = driver_results[driver_results["race_condition"] == "dry"]

    stats = []
    for driver in driver_results["driver"].unique():
        d_wet = wet[wet["driver"] == driver]["final_position"].mean()
        d_dry = dry[dry["driver"] == driver]["final_position"].mean()
        if pd.notna(d_wet) and pd.notna(d_dry):
            stats.append({
                "driver": driver,
                "wet_avg_pos": d_wet, "dry_avg_pos": d_dry,
                "wet_improvement": d_dry - d_wet,  # positive = better in wet
            })
    df = pd.DataFrame(stats).sort_values("wet_improvement", ascending=False)
    df["wet_specialist"] = df["wet_improvement"] > 1.5
    return df
```

### Feature 10 — Pre-Race Betting Value Finder

Compute model-implied win probability vs bookmaker odds. Surface positive
EV bets for race winner, podium, points finish, and DNF.

```python
# scripts/betting_value.py
import pandas as pd, numpy as np

def compute_race_betting_edges(
    driver_probs: pd.DataFrame, odds_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare model win probs to bookmaker odds."""
    merged = driver_probs.merge(
        odds_df[["driver", "win_odds_decimal", "podium_odds_decimal"]],
        on="driver", how="left",
    )
    merged["win_implied"] = 1 / merged["win_odds_decimal"]
    merged["podium_implied"] = 1 / merged["podium_odds_decimal"]
    merged["win_edge"] = merged["win_prob"] - merged["win_implied"]
    merged["podium_edge"] = merged["podium_prob"] - merged["podium_implied"]
    merged["win_ev"] = merged["win_edge"] * merged["win_odds_decimal"]
    merged["tier"] = merged["win_ev"].apply(
        lambda e: "HIGH" if e > 0.10 else ("MEDIUM" if e > 0.05 else "LOW")
    )
    return merged.sort_values("win_ev", ascending=False)
```

---

## Q3 (Feb–Apr 2027) — Dashboard & Visualization

### Feature 11 — Race Strategy Simulator (Interactive)

Interactive Streamlit tool: set grid position, tyre compound, pit strategy.
Simulate race outcome with uncertainty bands.

```python
# raceAnalysis_strategy.py (new tab)
import streamlit as st
import numpy as np, plotly.go as go

def render_strategy_simulator(circuit: str) -> None:
    st.title("🏎️ Race Strategy Simulator")
    from scripts.circuit_dna import load_circuit_features
    circuit_data = load_circuit_features(circuit)

    col1, col2, col3 = st.columns(3)
    grid_pos = col1.slider("Starting Grid Position", 1, 20, 5)
    pit_lap = col2.slider("Pit Stop Lap", 10, 50, 25)
    compound = col3.selectbox("Starting Tyre", ["Soft", "Medium", "Hard"])
    n_sim = 1000

    if st.button("Simulate 1,000 Races"):
        final_positions = []
        for _ in range(n_sim):
            pos = grid_pos
            # Random events
            if np.random.random() < circuit_data["sc_probability"] * 0.15:
                pos -= np.random.randint(1, 4)  # safety car gain
            # Tyre advantage after pit
            comp_adj = {"Soft": 0.3, "Medium": 0.15, "Hard": 0.0}[compound]
            pos -= comp_adj * np.random.normal(2, 1)
            final_positions.append(max(1, min(20, int(pos))))

        fig = go.Figure(go.Histogram(x=final_positions, nbinsx=20,
                                      marker_color="#E53935"))
        fig.update_layout(title=f"Simulated Finish Position Distribution (start P{grid_pos})",
                           xaxis_title="Final Position", template="plotly_dark")
        st.plotly_chart(fig, width="stretch")
        st.metric("Expected Position", f"P{np.mean(final_positions):.1f}")
        st.metric("P(Podium)", f"{(np.array(final_positions) <= 3).mean():.1%}")
        st.metric("P(Points)", f"{(np.array(final_positions) <= 10).mean():.1%}")
```

### Feature 12 — Head-to-Head Teammate Comparison

For each constructor, compare teammates on lap delta, qualifying delta,
race incidents, and championship points. Update weekly.

```python
# raceAnalysis_teammates.py
import streamlit as st, pandas as pd, plotly.express as px

def render_teammate_comparison(results: pd.DataFrame) -> None:
    constructors = results["constructor"].unique()
    selected = st.selectbox("Select Constructor", constructors)
    team_data = results[results["constructor"] == selected]
    drivers = team_data["driver"].unique()

    if len(drivers) < 2:
        st.info("Need 2 drivers for comparison.")
        return

    d1, d2 = drivers[0], drivers[1]
    h2h = {
        "qualifying_ahead": (team_data[team_data["driver"] == d1]["grid_pos"] <
                              team_data[team_data["driver"] == d2]["grid_pos"]).mean(),
        "race_ahead": (team_data[team_data["driver"] == d1]["final_position"] <
                        team_data[team_data["driver"] == d2]["final_position"]).mean(),
        "avg_qual_delta": (team_data[team_data["driver"] == d1]["grid_pos"] -
                            team_data[team_data["driver"] == d2]["grid_pos"]).mean(),
    }
    st.subheader(f"{d1} vs {d2}")
    c1, c2 = st.columns(2)
    c1.metric(f"{d1} Quali Ahead %", f"{h2h['qualifying_ahead']:.1%}")
    c2.metric(f"{d1} Race Ahead %", f"{h2h['race_ahead']:.1%}")
    st.metric("Avg Quali Gap (positions)", f"{h2h['avg_qual_delta']:+.2f}")
```

### Feature 13 — Live Race Dashboard (During Grand Prix)

Stream live timing from OpenF1. Display current positions, gap to leader,
tyre info, and predicted final positions updated every 30 seconds.

```python
# scripts/live_race_dashboard.py
import streamlit as st, requests, time
import pandas as pd

OPENF1_BASE = "https://api.openf1.org/v1"

def render_live_race() -> None:
    st.title("🏁 Live Race Tracker")
    placeholder = st.empty()

    # Continuous refresh via st.rerun with timer
    session_key = st.session_state.get("live_session_key")
    if not session_key:
        st.warning("Set live_session_key in session state first.")
        return

    laps = requests.get(
        f"{OPENF1_BASE}/laps", params={"session_key": session_key}, timeout=10
    ).json()
    positions = requests.get(
        f"{OPENF1_BASE}/position", params={"session_key": session_key}, timeout=10
    ).json()

    if not positions:
        st.info("Race not started.")
        return

    df = pd.DataFrame(positions).sort_values("position")
    with placeholder.container():
        st.dataframe(df[["driver_number", "position", "date"]], width="stretch")
    time.sleep(30)
    st.rerun()
```

### Feature 14 — Season Records & Historical Database Explorer

Full-season records browser: fastest laps, most positions gained, worst
qualifying-to-race delta, most DNFs. Cross-season comparisons.

```python
# raceAnalysis_records.py
import streamlit as st, pandas as pd

RECORD_CATEGORIES = {
    "Most Race Wins": lambda df: df.groupby("driver")["win"].sum(),
    "Most Poles": lambda df: df.groupby("driver")["pole"].sum(),
    "Most Fastest Laps": lambda df: df.groupby("driver")["fastest_lap"].sum(),
    "Most DNFs": lambda df: df.groupby("driver")["dnf"].sum(),
    "Best Avg Grid→Race Delta": lambda df: df.groupby("driver").apply(
        lambda g: (g["grid_pos"] - g["final_position"]).mean()
    ),
}

def render_records_page(results: pd.DataFrame) -> None:
    st.title("📚 F1 Historical Records")
    category = st.selectbox("Category", list(RECORD_CATEGORIES.keys()))
    season_filter = st.multiselect("Seasons", sorted(results["season"].unique()),
                                    default=[results["season"].max()])
    filtered = results[results["season"].isin(season_filter)]
    record_fn = RECORD_CATEGORIES[category]
    top = record_fn(filtered).nlargest(20).reset_index()
    top.columns = ["Driver", category]
    st.dataframe(top, width="stretch")
```

### Feature 15 — Rookie vs Veteran Performance Model

Separate models for rookies (first 3 seasons) vs veterans. Rookies have
high variance; separate modeling reduces MAE for those groups.

```python
# f1-generate-analysis.py (new section)
import pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

ROOKIE_THRESHOLD = 3  # seasons

def add_experience_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver experience tier as feature."""
    seasons_driven = df.groupby("driverId")["grandPrixYear"].nunique()
    df["experience_seasons"] = df["driverId"].map(seasons_driven)
    df["is_rookie"] = df["experience_seasons"] <= ROOKIE_THRESHOLD
    df["is_veteran"] = df["experience_seasons"] > 5
    # Rookies have higher predicted variance
    df["experience_variance_factor"] = np.where(
        df["is_rookie"], 1.5, np.where(df["is_veteran"], 0.8, 1.0)
    )
    return df

def train_experience_stratified_models(df: pd.DataFrame, features: list) -> dict:
    """Train separate models for rookies and veterans."""
    models = {}
    for group, name in [(True, "rookie"), (False, "veteran")]:
        subset = df[df["is_rookie"] == group] if group else df[~df["is_rookie"]]
        if len(subset) < 100:
            continue
        X = subset[features].fillna(subset[features].median())
        y = subset["resultsFinalPositionNumber"]
        model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03)
        model.fit(X, y)
        models[name] = model
        print(f"Trained {name} model on {len(subset)} samples.")
    return models
```

---

## Q4 (May–Jul 2027) — Automation & Tooling

### Feature 16 — Automated Race-Week Pipeline

GitHub Action fires on Friday qualifying day. Fetches practice, qualifying,
pre-race odds, weather. Emails race-day prediction report.

```yaml
# .github/workflows/race_week.yml
name: F1 Race Week Pipeline
on:
  schedule:
    - cron: '0 15 * * 5'  # Friday 3 PM UTC (typically Practice 1/2 day)
  workflow_dispatch:

jobs:
  race-week:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - name: Fetch practice data
        run: python f1-generate-analysis.py --session FP1 --session FP2
      - name: Run qualifying prediction
        run: python scripts/qualifying_simulation.py
      - name: Compute championship odds
        run: python scripts/championship_simulator.py
      - name: Send email report
        env:
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/send_rich_email_now.py --force
      - uses: EndBug/add-and-commit@v9
        with:
          message: "Auto: race week data ${{ github.run_number }}"
```

### Feature 17 — Pitstop Decision Support Tool

Interactive tool where user inputs current gap to cars ahead/behind
and tyre age. Returns recommended pit now vs stay out decision with confidence.

```python
# pages/pitstop_advisor.py
import streamlit as st
from scripts.strategy_optimizer import optimal_pit_window

def render_pitstop_advisor() -> None:
    st.title("🛞 Pit Stop Decision Advisor")
    col1, col2, col3 = st.columns(3)
    tyre_age = col1.slider("Current Tyre Age (laps)", 1, 40, 15)
    laps_remaining = col2.slider("Laps Remaining", 1, 50, 30)
    gap_ahead_s = col3.number_input("Gap to car ahead (s)", 0.0, 40.0, 2.5)
    compound = st.selectbox("Current Compound", ["Soft", "Medium", "Hard"])

    deg_models = {"Soft": {"b": 0.4, "c": 0.12}, "Medium": {"b": 0.25, "c": 0.08},
                   "Hard": {"b": 0.15, "c": 0.05}}
    pit_losses = {"Monaco": 22.5, "Silverstone": 21.0, "Monza": 19.5}
    circuit = st.selectbox("Circuit", list(pit_losses.keys()))

    if st.button("Compute Optimal Strategy"):
        result = optimal_pit_window(
            tyre_age, deg_models[compound], pit_losses[circuit],
            laps_remaining, gap_ahead_s
        )
        col_a, col_b = st.columns(2)
        col_a.metric("Optimal Pit Lap", f"Lap {result['optimal_lap']}")
        col_b.metric("Expected Net Gain", f"{result['expected_net_gain']:.1f}s")
        if result["expected_net_gain"] > gap_ahead_s:
            st.success("✅ PIT NOW — Projected to undercut ahead car")
        else:
            st.warning("⚠️ STAY OUT — Pit when gap grows further")
```

### Feature 18 — Driver-Constructor Chemistry Index

Quantify how much better/worse each driver performs for a specific
constructor vs their career baseline. Surface contract-value insights.

```python
# scripts/chemistry_index.py
import pandas as pd

def compute_chemistry_index(results: pd.DataFrame) -> pd.DataFrame:
    """
    Chemistry Index = avg_position_with_constructor - career_avg_position.
    Negative = driver over-performs for constructor (chemistry fit).
    """
    career_avg = results.groupby("driver")["final_position"].mean()
    with_constructor = (
        results.groupby(["driver", "constructor"])["final_position"].mean()
        .reset_index()
        .rename(columns={"final_position": "avg_with_team"})
    )
    with_constructor["career_avg"] = with_constructor["driver"].map(career_avg)
    with_constructor["chemistry_index"] = (
        with_constructor["avg_with_team"] - with_constructor["career_avg"]
    )
    with_constructor["interpretation"] = with_constructor["chemistry_index"].apply(
        lambda c: "🟢 Over-performs" if c < -1.0
        else ("🔴 Under-performs" if c > 1.0 else "🟡 Neutral")
    )
    return with_constructor.sort_values("chemistry_index")
```

### Feature 19 — Race Incident & Safety Car Predictor

Predict probability of safety car / red flag at each circuit. Uses historical
incident rate, circuit category, and weather forecast.

```python
# scripts/incident_predictor.py
import pandas as pd
from xgboost import XGBClassifier
import joblib
from pathlib import Path

FEATURES = [
    "circuit_type",           # encoded: street(0), permanent(1), semi-permanent(2)
    "sc_history_rate",        # historical SC rate at circuit
    "wet_flag",
    "grid_spread",            # variance in qualifying lap times (tight grid = more incidents)
    "race_length_laps",
    "air_temp_c",
    "track_temp_c",
]

def train_incident_model(df: pd.DataFrame) -> None:
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["had_sc"].astype(int)
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05)
    model.fit(X, y)
    joblib.dump(model, Path("models/incident_predictor.joblib"))

def predict_sc_probability(circuit_name: str, wet: bool, grid_spread: float) -> float:
    from scripts.circuit_dna import load_circuit_features
    circuit = load_circuit_features(circuit_name)
    model = joblib.load(Path("models/incident_predictor.joblib"))
    X = pd.DataFrame([{
        "circuit_type": 0 if "Monaco" in circuit_name or "Baku" in circuit_name else 1,
        "sc_history_rate": circuit["sc_probability"],
        "wet_flag": int(wet),
        "grid_spread": grid_spread,
        "race_length_laps": 58,
        "air_temp_c": 22, "track_temp_c": 35,
    }])
    return float(model.predict_proba(X)[0][1])
```

### Feature 20 — Constructor Budget & Car Development Model

Estimate team development rate by tracking circuit-specific performance
improvement rate per team across rounds. Predict end-of-season standings.

```python
# scripts/development_rate.py
import pandas as pd, numpy as np

def compute_development_rate(results: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Measure how much each constructor improves per race (delta in avg position).
    Proxy for car development speed.
    """
    season_df = results[results["season"] == season].sort_values("round")
    rates = []
    for constructor, group in season_df.groupby("constructor"):
        group = group.sort_values("round")
        # Regression: position ~ round_number
        x = group["round"].values
        y = group["avg_position"].values
        if len(x) < 5:
            continue
        coeff = np.polyfit(x, y, 1)[0]  # negative = improving
        rates.append({
            "constructor": constructor, "dev_rate": -coeff,
            "early_avg": group.head(5)["avg_position"].mean(),
            "late_avg": group.tail(5)["avg_position"].mean(),
        })
    df = pd.DataFrame(rates).sort_values("dev_rate", ascending=False)
    df["interpretation"] = df["dev_rate"].apply(
        lambda r: "🚀 Strong development" if r > 0.3
        else ("🐢 Slow development" if r < 0.0 else "📈 Moderate")
    )
    return df
```

### Feature 21 — Weather Risk Score for Each Race

Combine met office data, historical rainfall at circuit, and humidity to
produce a single 0-100 weather-risk score. Higher score = more chaotic race.

```python
# scripts/weather_risk.py
import openmeteo_requests, requests_cache
import numpy as np
from scripts.circuit_dna import CIRCUIT_DNA

CIRCUIT_COORDS = {
    "monaco": (43.7347, 7.4207),
    "silverstone": (52.0708, -1.0167),
    "monza": (45.6156, 9.2811),
    "spa": (50.4372, 5.9718),
}

def compute_weather_risk(circuit: str, race_date: str) -> float:
    """Return 0-100 weather risk score for a race."""
    lat, lon = CIRCUIT_COORDS.get(circuit.lower(), (45.0, 9.0))
    session = requests_cache.CachedSession(".weather_cache", expire_after=3600)
    om = openmeteo_requests.Client(session=session)
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["precipitation_sum", "windspeed_10m_max", "temperature_2m_max"],
        "start_date": race_date, "end_date": race_date,
    }
    resp = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    precip = resp.Daily().Variables(0).ValuesAsNumpy()[0]
    wind = resp.Daily().Variables(1).ValuesAsNumpy()[0]
    temp = resp.Daily().Variables(2).ValuesAsNumpy()[0]

    historical_rain_prob = CIRCUIT_DNA.get(circuit.lower(), {}).get("sc_probability", 0.5)
    risk = min(100.0, (
        precip * 20 +  # mm precipitation
        max(0, wind - 20) * 2 +  # wind above 20 km/h
        historical_rain_prob * 30
    ))
    return round(risk, 1)
```

### Feature 22 — Hall of Fame Elo Rating (All-Time Drivers)

Compute all-time Elo ratings for all F1 drivers across the full F1DB
dataset (1950–present). Rank greatest drivers of all time.

```python
# scripts/all_time_elo.py
import pandas as pd, json
from pathlib import Path

K_FACTOR = 32
HOME_ADV = 0  # no home advantage in F1
INITIAL_ELO = 1500

def compute_alltime_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo for all F1 drivers from 1950-present via pairwise race results."""
    elo = {}
    results = results.sort_values(["year", "round", "final_position"])
    records = []

    for race_id, race in results.groupby(["year", "round"]):
        finishers = race.sort_values("final_position")
        drivers = finishers["driver"].tolist()
        n = len(drivers)

        pre_elo = {d: elo.get(d, INITIAL_ELO) for d in drivers}
        delta = {d: 0.0 for d in drivers}

        for i in range(n):
            for j in range(i + 1, n):
                winner, loser = drivers[i], drivers[j]
                elo_w = pre_elo[winner]
                elo_l = pre_elo[loser]
                exp_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
                delta[winner] += K_FACTOR * (1 - exp_w) / (n - 1)
                delta[loser] += K_FACTOR * (0 - (1 - exp_w)) / (n - 1)

        for d in drivers:
            elo[d] = pre_elo[d] + delta[d]
            records.append({"year": race_id[0], "round": race_id[1],
                             "driver": d, "elo": elo[d]})

    df = pd.DataFrame(records)
    peak_elo = df.groupby("driver")["elo"].max().reset_index().rename(columns={"elo": "peak_elo"})
    final_elo = df.groupby("driver")["elo"].last().reset_index().rename(columns={"elo": "final_elo"})
    return peak_elo.merge(final_elo, on="driver").sort_values("peak_elo", ascending=False)
```

### Feature 23 — Sprint Weekend Handling Improvements

Detect sprint weekends and apply separate sprint race predictions.
Sprint results influence race start compound choice.

```python
# scripts/sprint_integration.py
import pandas as pd
from pathlib import Path

SPRINT_ROUNDS = {2026: [4, 8, 12, 16, 20]}  # update per season announcement

def is_sprint_weekend(year: int, round_num: int) -> bool:
    return round_num in SPRINT_ROUNDS.get(year, [])

def get_sprint_compound_constraint(sprint_results: pd.DataFrame, driver: str) -> str:
    """Sprint race determines which compound is unavailable for main race."""
    sprint_row = sprint_results[sprint_results["driver"] == driver]
    if sprint_row.empty:
        return "unknown"
    sprint_compound = sprint_row.iloc[0].get("sprint_compound", "medium")
    return sprint_compound  # team must use different compound in main race
```

### Feature 24 — Mobile-First Responsive Streamlit Theme

Implement a mobile-first responsive Streamlit layout with sticky race
header, collapsible sections, and touch-friendly controls.

```python
# raceAnalysis.py (theme section)
MOBILE_CSS = """
<style>
@media (max-width: 768px) {
    [data-testid="stSidebar"] { display: none !important; }
    .stDataFrame { font-size: 11px !important; }
    .race-header { position: sticky; top: 0; z-index: 999;
                   background: #1a1a2e; padding: 8px; border-bottom: 1px solid #333; }
    .metric-card { padding: 8px; margin: 4px; border-radius: 8px;
                   background: #16213e; }
}
.race-header { display: flex; justify-content: space-between; align-items: center; }
.tier-badge { padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
.tier-HIGH { background: #1b5e20; color: white; }
.tier-MEDIUM { background: #e65100; color: white; }
.tier-LOW { background: #37474f; color: white; }
</style>
"""

def inject_mobile_theme() -> None:
    import streamlit as st
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
```

---

## Timeline Summary

| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 Aug–Oct 2026 | Advanced features | OpenF1 telemetry, sector times, tyre lifecycle, circuit DNA, qualifying simulation |
| Q2 Nov 2026–Jan 2027 | Advanced modeling | Transformer model, strategy optimizer, championship sim, wet weather, betting value |
| Q3 Feb–Apr 2027 | Dashboard | Race simulator, teammate comparison, live dashboard, records explorer, rookie model |
| Q4 May–Jul 2027 | Automation | Race-week pipeline, pitstop advisor, chemistry index, incident predictor, dev rate, weather risk, all-time Elo, sprint handling, mobile theme |
