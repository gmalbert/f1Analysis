# FastF1 API vs f1db JSON Comparison

## f1db JSON (Current Approach)
**Source**: Static JSON files updated periodically from F1 Database
**Coverage**: 1950-2025 (26,715 records)
**What you get**:
- Final Q1, Q2, Q3 lap times (best time from each session)
- Final qualifying position
- Driver, constructor, race metadata
- Simple time strings (e.g., "1:21.164")

**Pros**:
- ✅ Historical coverage back to 1950
- ✅ No API rate limits
- ✅ Fast to load (already downloaded)
- ✅ Official FIA results
- ✅ Complete coverage for all races

**Cons**:
- ❌ No sector times (S1, S2, S3 breakdowns)
- ❌ No telemetry data (speed, throttle, brake, DRS)
- ❌ No individual lap-by-lap qualifying data
- ❌ No tire compound information per lap
- ❌ No track status (yellow flags, red flags timing)
- ❌ No weather conditions per session
- ❌ Only the BEST lap from each session (Q1/Q2/Q3), not all attempts

---

## FastF1 API (What the script was trying to get)
**Source**: F1's official timing system via FastF1 library
**Coverage**: 2018-present only (telemetry era)
**What you get**:
- **ALL qualifying laps** for each driver (not just best)
- **Sector times** for every lap (Sector 1, 2, 3 split times)
- **Telemetry data**: Speed, throttle position, brake pressure, gear, RPM, DRS status
- **Tire information**: Compound (Soft/Medium/Hard), tire age, stint number
- **Track status**: Yellow flags, red flags, track limits violations
- **Weather data**: Track temp, air temp, humidity, wind per session
- **Individual lap deleted/valid status** (track limit violations)
- **Pit in/out times** for qualifying
- **More precise timing** (microsecond precision vs string parsing)

**Pros**:
- ✅ Extremely granular data (every lap, every sector, telemetry)
- ✅ Can analyze qualifying progression (lap-by-lap improvement)
- ✅ Tire strategy analysis (which compound, how many laps)
- ✅ Can detect deleted laps (track limits)
- ✅ Sector-level performance comparison
- ✅ Weather correlation with lap times

**Cons**:
- ❌ Only 2018-present (no historical data before telemetry era)
- ❌ API rate limits (500 calls/hour - hit during script run)
- ❌ Slower to fetch (30-60 seconds per session)
- ❌ Requires caching infrastructure
- ❌ ~160+ API calls needed for full 2018-2025 coverage

---

## What You're Losing by Using f1db JSON

### Missing Features for MAE Prediction:
1. **Sector consistency**: Can't analyze if a driver is consistently fast in S1 but slow in S3
2. **Qualifying progression**: Can't see if a driver improves lap-to-lap (momentum indicator)
3. **Tire correlation**: Can't correlate tire compound choice with race performance
4. **Track limit issues**: Can't identify drivers who struggle with track limits (DNF risk)
5. **Weather sensitivity**: Can't analyze driver performance in different conditions

### Example Use Cases Lost:
- **"Does this driver improve on their final Q3 attempt?"** (clutch performance)
- **"Average sector 3 time vs competitors"** (late-lap pace)
- **"How many qualifying laps deleted for track limits?"** (precision indicator)
- **"Qualifying on Soft vs Medium compound"** (tire strategy)

---

## Recommendation

**For your current MAE goal (≤1.5):**
- **Start with f1db JSON**: It gives you the essential qualifying time data immediately
- **Qualifying position is the #1 predictor** of race finish - you have that
- **Best qualifying time** is what matters most for grid position

**For future MAE optimization (once you hit ≤1.5):**
- **Consider adding FastF1 sector data** for 2018+ races
- **Focus on sector consistency metrics** (std dev of S1, S2, S3 times)
- **Add tire compound as a categorical feature**
- **Track limit violations as a risk factor**

**Hybrid Approach** (Best of both):
```python
# Use f1db for 1950-2017 (basic times)
# Use FastF1 for 2018-2025 (sector times, telemetry)
# Merge both datasets for complete historical coverage
```

This gives you complete coverage + granular modern data where it matters most.
