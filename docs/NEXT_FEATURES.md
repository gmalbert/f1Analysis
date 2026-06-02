# F1 Analysis — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: MAE Regression Alert (GitHub Actions Email)

**Why:** `scripts/check_mae_regression.py` already exists and computes whether rolling 5-race MAE exceeds the baseline. However, there is no automated alert when regression is detected. Adding an email notification to the existing Actions infrastructure would close the monitoring loop.

**How:**
1. Add `scripts/send_mae_alert.py` using the existing SMTP pattern from `scripts/send_rich_email_now.py`
2. Call `check_mae_regression.py` as a Python import and check its return code / MAE value
3. If rolling MAE > baseline + 0.10, send email: "⚠ MAE Regression: current {X.XX} vs baseline {Y.YY}"
4. Add a step to `.github/workflows/mae-regression-check.yml` to trigger the email on regression detection

**Complexity:** Low

---

## Feature 2: Teammate Head-to-Head Dashboard

**Why:** `teammate_qual_delta` already exists in the generator as a feature, and pairwise teammate comparisons are among the most popular F1 analytics. A dedicated page or tab showing qualifying delta, race pace comparison, and pit stop time by constructor pair would be a high-engagement addition.

**How:**
1. Add a "Teammate H2H" tab to the existing UI in `raceAnalysis.py`
2. For each current constructor, filter to the two active drivers
3. Display: qualifying delta distribution (Plotly violin), race finish position comparison, average pit stop time per race
4. Source data from `f1ForAnalysis.csv` filtered to `grandPrixYear == current_year`
5. Add a season selector to compare teammates across multiple seasons

**Complexity:** Low

---

## Feature 3: Constructor Championship Probability

**Why:** The model predicts individual race finishing positions but has no season-level championship probability output. A Monte Carlo simulation of remaining races → projected points totals with confidence intervals would be the most publicly engaging visualization in the app.

**How:**
1. Build `scripts/championship_simulator.py`: load current standings from `f1db-races-race-results.json`, simulate remaining rounds 10,000 times using the current-season XGBoost predicted finish probabilities
2. Compute: `P(constructor wins championship)` per team, `P(driver wins championship)` per driver
3. Write results to `data_files/championship_probs.json`
4. Add a "Championship Race" tab to `raceAnalysis.py` with a stacked Plotly bar chart per team

**Complexity:** Medium

---

## Feature 4: Tire Strategy Simulator

**Why:** Pit strategy (1-stop vs 2-stop, when to pit) is one of the highest-leverage decision points in F1. `pit_constants.py` already provides track-specific pit lane time loss. Combining this with compound lap-time degradation rates would enable a basic strategy simulator.

**How:**
1. Add `scripts/tire_strategy.py` that models lap time evolution per compound (Soft/Medium/Hard) using published Pirelli degradation estimates
2. For a given race, simulate: total race time for 1-stop (M-H) vs 2-stop (S-M-H) strategies at each possible pit window
3. Output: recommended strategy + estimated position gain/loss vs 1-stop
4. Add a "Strategy Simulator" expander on the Next Race tab in `raceAnalysis.py`

**Complexity:** High

---

## Feature 5: Real-Time Qualifying Integration

**Why:** After each qualifying session, the predicted grid position should update using the actual qualifying result, not the model's pre-qualifying prediction. FastF1 provides qualifying results immediately after each session.

**How:**
1. Create `scripts/fetch_qualifying_realtime.py` that uses FastF1 to check if qualifying session data has been published (checks session status)
2. If `session.results` is available (post-qualifying), overwrite the predicted grid column for that round in `f1ForAnalysis.csv` with actual qualifying positions
3. This significantly improves race prediction accuracy since qualifying position is the #1 predictor
4. Trigger from a GitHub Actions workflow on qualifying Saturday afternoon (EU time)

**Complexity:** High
