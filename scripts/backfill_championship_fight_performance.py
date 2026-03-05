#!/usr/bin/env python3
"""Backfill championship_fight_performance and championship_position in f1ForAnalysis.csv.

Uses f1db-races-driver-standings.json for accurate per-race championship standings
(avoids the leakage issue of computing standings from within the CSV itself).

championship_fight_performance = rolling 3-race mean finishing position (lagged by 1)
for drivers who were in the top-3 of the championship at that race weekend.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path("data_files")
CSV_PATH = DATA_DIR / "f1ForAnalysis.csv"

print("Loading f1ForAnalysis.csv …", flush=True)
df = pd.read_csv(CSV_PATH, sep="\t", low_memory=False)
n_rows = len(df)
print(f"  {n_rows:,} rows, {len(df.columns)} columns")

# ── Load driver standings ──────────────────────────────────────────────────
print("Loading f1db-races-driver-standings.json …", flush=True)
standings_raw = json.load(open(DATA_DIR / "f1db-races-driver-standings.json"))
standings = pd.DataFrame(standings_raw)
print(f"  {len(standings):,} standing entries")

# Normalise to match CSV column names
standings = standings.rename(columns={
    "year":           "grandPrixYear",
    "driverId":       "resultsDriverId",
    "positionNumber": "championship_position_f1db",
    "points":         "driverPoints_f1db",
})

# Keep only columns we need + deduplicate
standings = standings[["raceId", "grandPrixYear", "round",
                        "resultsDriverId", "championship_position_f1db",
                        "driverPoints_f1db"]].drop_duplicates()

# ── Determine join key ─────────────────────────────────────────────────────
# Prefer raceId if present and populated; fall back to year+round
if "raceId" in df.columns and df["raceId"].notna().mean() > 0.5:
    print("  Joining on raceId + resultsDriverId")
    df = df.merge(
        standings[["raceId", "resultsDriverId",
                   "championship_position_f1db", "driverPoints_f1db"]],
        on=["raceId", "resultsDriverId"],
        how="left",
    )
elif "round" in df.columns and "grandPrixYear" in df.columns:
    print("  Joining on grandPrixYear + round + resultsDriverId")
    df = df.merge(
        standings[["grandPrixYear", "round", "resultsDriverId",
                   "championship_position_f1db", "driverPoints_f1db"]],
        on=["grandPrixYear", "round", "resultsDriverId"],
        how="left",
    )
else:
    print("ERROR: Cannot find a suitable join key. Aborting.")
    sys.exit(1)

cov = df["championship_position_f1db"].notna().mean()
print(f"  championship_position_f1db coverage after join: {cov:.1%}")

# ── Overwrite championship_position in CSV ────────────────────────────────
df["championship_position"] = df["championship_position_f1db"]

# ── Recompute championship_fight_performance ──────────────────────────────
# Sort by driver + year + race round so the rolling window is chronological.
# Use raceId as tiebreaker if available.
sort_cols = ["resultsDriverId", "grandPrixYear"]
if "round" in df.columns:
    sort_cols += ["round"]
elif "raceId" in df.columns:
    sort_cols += ["raceId"]
df = df.sort_values(sort_cols).reset_index(drop=True)

in_fight = df["championship_position"] <= 3
n_fight = in_fight.sum()
print(f"  Rows with championship_position <= 3: {n_fight:,}")

df["championship_fight_performance"] = np.nan
if n_fight >= 50:
    df.loc[in_fight, "championship_fight_performance"] = (
        df.loc[in_fight]
        .groupby("resultsDriverId")["resultsFinalPositionNumber"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
else:
    # Relax to top-5 if not enough top-3 rows
    print("  WARNING: < 50 top-3 rows; widening to championship_position <= 5")
    in_fight5 = df["championship_position"] <= 5
    df.loc[in_fight5, "championship_fight_performance"] = (
        df.loc[in_fight5]
        .groupby("resultsDriverId")["resultsFinalPositionNumber"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

cfp_cov = df["championship_fight_performance"].notna().mean()
print(f"  championship_fight_performance coverage: {cfp_cov:.1%}")

# ── Drop helper columns and write back ───────────────────────────────────
df = df.drop(columns=["championship_position_f1db", "driverPoints_f1db"],
             errors="ignore")

# Restore original row order
df = df.sort_index()

print(f"Writing updated CSV ({len(df):,} rows) …", flush=True)
df.to_csv(CSV_PATH, sep="\t", index=False)
print("Done.")
