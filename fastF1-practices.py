import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
import datetime
import numpy as np

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))

# Exclude 'jos-verstappen' as he is not Max Verstappen
drivers = drivers[drivers['id'] != 'jos-verstappen']

results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t')
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
active_drivers = pd.read_csv(path.join(DATA_DIR, 'active_drivers.csv'), sep='\t')

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

def km_to_miles(km):
    return km * 0.621371

all_laps = []
ergast = Ergast(result_type='pandas', auto_cast=True)


def add_teammate_delta(df, group_cols, value_col, new_col):
    """
    Adds a column with the difference between each driver's value_col and their teammate's for each group.
    Only works for teams with exactly 2 drivers per group.
    """

    def teammate_diff(x):
        if len(x) != 2:
            # Return a Series of None with the same length as x
            return pd.Series([None] * len(x), index=x.index)
        val1 = x.iloc[0]
        val2 = x.iloc[1]
        if pd.isna(val1) or pd.isna(val2):
            return pd.Series([None, None], index=x.index)
        return pd.Series([val1 - val2, val2 - val1], index=x.index)
        
        # Calculate the delta and return as a Series with the original index
        # return pd.Series([val1 - val2, val2 - val1], index=x.index)

    # Use 'apply' instead of 'transform' for more complex logic that returns a Series
    deltas = df.groupby(group_cols)[value_col].apply(teammate_diff)
    
    # The result from apply can have a MultiIndex, so we need to reset it 
    # and merge it back into the original dataframe.
    deltas = deltas.reset_index(level=group_cols, drop=True).rename(new_col)

    # If the column already exists from a previous run, drop it before joining.
    if new_col in df.columns:
        df = df.drop(columns=[new_col])
    
    # Join the calculated deltas back to the original DataFrame
    df = df.join(deltas)
    return df

# Check if the processed file exists and read it to avoid reprocessing
csv_path = os.path.join(DATA_DIR, 'all_practice_laps.csv')
if os.path.exists(csv_path):
    processed_df = pd.read_csv(csv_path, sep='\t')
    processed_sessions = set(
        (int(y), int(r), str(s).upper())
        for y, r, s in zip(processed_df['Year'], processed_df['Round'], processed_df['Session'])
        )
else:
    processed_df = pd.DataFrame()
    processed_sessions = set()

# --- MAIN DATA COLLECTION LOOP ---
for year in range(2018, current_year + 1):
# for year in range(2018, current_year + 1):
    season_schedule = ergast.get_race_schedule(season=year)
    total_rounds = len(season_schedule)
    #for round_number in range(2,10):
    for round_number in range(1, total_rounds + 1):    
        for session_type in ['FP1', 'FP2', 'FP3']:
        # for session_type in ['FP1', 'FP2', 'FP3']:
            # session_key = (year, round_number, session_type)
            session_key = (int(year), int(round_number), session_type.upper())
            session_date = pd.to_datetime(season_schedule.iloc[round_number - 1]['raceDate']).date()
            today = datetime.date.today()
            laps_written = False # Flag to track if any laps were written for this session
            if session_key in processed_sessions:
                print(f"Skipping session: {session_key} (already processed)")
                continue
            # Skip if the race is later than today
            if session_date > today:
                print(f"Skipping session: {session_key} (race is in the future)")
                continue
            elif (session_date - datetime.datetime.now().date()).days > 3:
                print(f"Skipping session: {session_key} (more than three days in the future)")
                continue
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load()
                session_drivers = session.drivers

                # --- Clean/Dirty Air Delta Calculation for All Drivers ---
                session_laps = session.laps.pick_accurate()
                telemetry_all = {}
                for drv in session.drivers:
                    abbreviation = session.get_driver(drv).get('Abbreviation', drv)
                    laps_drv = session_laps.pick_drivers(drv)
                    # if not laps_drv.empty:
                    # if len(tel) > 0:
                    if len(laps_drv) > 0:
                        try:
                            # Use all laps' telemetry, not just the fastest lap
                            tel = laps_drv.get_telemetry()
                            if tel is not None and len(tel) > 0:
                                telemetry_all[abbreviation] = tel
                        except Exception as e:
                            print(f"Error getting telemetry for {abbreviation}: {e}")
                            print(f"Skipping {session_type} for {year} round {round_number}: {e}")
                            # Save a marker row so we don't retry this session
                            marker_row = {
                                'Year': year,
                                'Round': round_number,
                                'Session': session_type,
                                'Driver': 'ERROR',
                                'LapTime': pd.NA,
                                'Sector1Time': pd.NA,
                                'Sector2Time': pd.NA,
                                'Sector3Time': pd.NA,
                                'LapTime_sec': pd.NA,
                                'best_s1_sec': pd.NA,
                                'best_s2_sec': pd.NA,
                                'best_s3_sec': pd.NA,
                                # Add any other columns you expect
                            }
                            all_laps.append(marker_row)
                            continue


                def get_air_gap(lap, telemetry_all):
                    drv_tel = lap.get_telemetry()
                    # if drv_tel.empty:
                    if len(drv_tel) == 0: 
                        return None
                    midpoint = drv_tel['Distance'].max() / 2
                    own_point = drv_tel.iloc[(drv_tel['Distance'] - midpoint).abs().argmin()]
                    min_gap = float('inf')
                    for drv, tel in telemetry_all.items():
                        if drv == lap['Driver']:
                            continue

                        idx = (tel['Date'] - own_point['Date']).abs().idxmin()
                        tel_point = tel.loc[idx]
                        other_pos = tel_point[['X', 'Y']]
                        own_pos = own_point[['X', 'Y']]
                        dist = ((own_pos - other_pos) ** 2).sum() ** 0.5
                        min_gap = min(min_gap, dist)
                    return min_gap

                def get_sampled_min_air_gap(lap, telemetry_all, n_points=3):
                    try:
                        drv_tel = lap.get_telemetry()
                        # Exit if telemetry is empty or has no Distance data
                        if drv_tel.empty or 'Distance' not in drv_tel.columns or drv_tel['Distance'].isna().all():
                            return None

                        distances = drv_tel['Distance'].dropna()
                        
                        # Exit if there are not enough data points to create a range
                        if len(distances) < 2 or distances.min() == distances.max():
                            return None

                        # Now safe to calculate sample distances
                        sample_distances = np.linspace(distances.min(), distances.max(), n_points)
                        
                        min_gap = float('inf')
                        own_abbr = lap['Driver']

                        for d in sample_distances:
                            own_point = drv_tel.iloc[(drv_tel['Distance'] - d).abs().argmin()]
                            for drv, tel in telemetry_all.items():
                                if drv == own_abbr:
                                    continue
                                
                                if tel.empty or 'Date' not in tel.columns or 'X' not in tel.columns or 'Y' not in tel.columns:
                                    continue

                                # Find the closest point in time in the other car's telemetry
                                idx = (tel['Date'] - own_point['Date']).abs().idxmin()
                                tel_point = tel.loc[idx]
                                
                                other_pos = tel_point[['X', 'Y']]
                                own_pos = own_point[['X', 'Y']]
                                
                                dist = ((own_pos - other_pos) ** 2).sum() ** 0.5
                                min_gap = min(min_gap, dist)
                        
                        return min_gap if min_gap != float('inf') else None
                    except Exception:
                        # Broad exception to catch any other unexpected telemetry errors
                        return None

                # Store deltas for all drivers in this session
                clean_dirty_deltas = []

                for driver in session.drivers:
                    driver_info = session.get_driver(driver)
                    abbreviation = driver_info.get('Abbreviation', driver)
                    # laps = session.laps.pick_drivers(driver).pick_accurate()
                    laps = session.laps.pick_drivers(driver).pick_quicklaps()
                    air_gaps = []
                    for _, lap in laps.iterlaps():
                        air_gap = get_sampled_min_air_gap(lap, telemetry_all, n_points=3)
                        if air_gap is not None and not pd.isna(lap['LapTime']):
                            air_gaps.append(air_gap)

                    if air_gaps:
                        print(f"{abbreviation} air gap stats: min={min(air_gaps):.2f}, max={max(air_gaps):.2f}, avg={sum(air_gaps)/len(air_gaps):.2f}")
                        print(np.histogram(air_gaps, bins=[0, 5, 10, 20, 30, 50, 100]))
                    clean, dirty = [], []
                    clean_count = 0
                    dirty_count = 0
                    neutral_count = 0

                    for _, lap in laps.iterlaps():
                        air_gap = get_sampled_min_air_gap(lap, telemetry_all, n_points=3)
                        print(f"Driver: {abbreviation}, Lap: {lap['LapNumber']}, Air gap: {air_gap}")

                        if air_gap is None or pd.isna(lap['LapTime']):
                            continue
                        if air_gap > 20:  # the lower the better, so we consider 25 as clean air
                            clean.append(lap['LapTime'].total_seconds())
                            clean_count += 1
                        elif air_gap < 15:
                            dirty.append(lap['LapTime'].total_seconds())
                            dirty_count += 1
                        else:
                            neutral_count += 1
                    print(f"Clean: {clean_count}, Dirty: {dirty_count}, Neutral: {neutral_count}")        
                    if clean and dirty:
                        clean_avg = sum(clean) / len(clean)
                        dirty_avg = sum(dirty) / len(dirty)
                        delta = dirty_avg - clean_avg
                        clean_dirty_deltas.append({
                            'Year': year,
                            'Round': round_number,
                            'Session': session_type,
                            'Driver': abbreviation,  # <--- Always use abbreviation here!
                            'CleanAirAvg': clean_avg,
                            'DirtyAirAvg': dirty_avg,
                            'Delta': delta,
                            'NumCleanLaps': len(clean),
                            'NumDirtyLaps': len(dirty)
                        })
                        print(f"{year} R{round_number} {session_type} {driver}: Clean={clean_avg:.3f}s Dirty={dirty_avg:.3f}s Delta={delta:.3f}s")
                        
                    else:
                        print(f"{year} R{round_number} {session_type} {driver}: Not enough data for delta.")

                # Optionally: Save or merge clean_dirty_deltas into your main DataFrame or output CSV
                # Example: Append to a master list for all sessions, then save at the end
                delta_csv = path.join(DATA_DIR, 'clean_dirty_air_deltas.csv')
                if clean_dirty_deltas:
                    df_clean_dirty = pd.DataFrame(clean_dirty_deltas)
                    df_clean_dirty.to_csv(delta_csv, mode='a', header=not os.path.exists(delta_csv), index=False, sep='\t')

                # --- Continue with your existing fastest lap processing code below ---
                for driver in session_drivers:
                    laps = session.laps.pick_drivers(driver)
                    fastest_lap = laps.pick_fastest()
                    # print(f"Session: {session_type}, Driver: {driver}, Fastest lap found: {fastest_lap is not None and len(fastest_lap) > 0}")

                    if fastest_lap is not None and len(fastest_lap) > 0:
                        fastest_lap = fastest_lap.copy()
                        driver_info = session.get_driver(driver)
                        abbreviation = driver_info.get('Abbreviation', driver)
                        fastest_lap['Driver'] = abbreviation  # <-- Force abbreviation here!
                        fastest_lap['Year'] = session.date.year  
                        fastest_lap['FP_Name'] = session.event['EventName']
                        fastest_lap['Round'] = session.event['RoundNumber']
                        fastest_lap['Session'] = session_type

                        if 'Sector1Time' in laps.columns and not laps['Sector1Time'].isnull().all():
                            laps.loc[:, 'Sector1Time'] = pd.to_timedelta(laps['Sector1Time'], errors='coerce')
                            best_s1 = laps.loc[laps['Sector1Time'].idxmin()]
                            fastest_lap['best_s1'] = best_s1['Sector1Time']
                            # Also assign driver/race info if missing
                            if 'driverId' in best_s1:
                                fastest_lap['driverId'] = best_s1['driverId']
                            if 'raceId' in best_s1:
                                fastest_lap['raceId'] = best_s1['raceId']
                        else:
                            fastest_lap['best_s1'] = pd.NaT

                        if pd.notnull(fastest_lap['best_s1']):
                            fastest_lap['best_s1_sec'] = pd.to_timedelta(fastest_lap['best_s1']).total_seconds()
                        else:
                            fastest_lap['best_s1_sec'] = pd.NA  

                        if 'Sector2Time' in laps.columns and not laps['Sector2Time'].isnull().all():
                            laps.loc[:, 'Sector2Time'] = pd.to_timedelta(laps['Sector2Time'], errors='coerce')
                            best_s2 = laps.loc[laps['Sector2Time'].idxmin()]
                            fastest_lap['best_s2'] = best_s2['Sector2Time']
                            # Also assign driver/race info if missing
                            if 'driverId' in best_s2:
                                fastest_lap['driverId'] = best_s2['driverId']
                            if 'raceId' in best_s2:
                                fastest_lap['raceId'] = best_s2['raceId']
                        else:
                            fastest_lap['best_s2'] = pd.NaT

                        # After assigning fastest_lap['best_s2']
                        if pd.notnull(fastest_lap['best_s2']):
                            fastest_lap['best_s2_sec'] = pd.to_timedelta(fastest_lap['best_s2']).total_seconds()
                        else:
                            fastest_lap['best_s2_sec'] = pd.NA    

                        if 'Sector3Time' in laps.columns and not laps['Sector3Time'].isnull().all():
                            laps.loc[:, 'Sector3Time'] = pd.to_timedelta(laps['Sector3Time'], errors='coerce')
                            best_s3 = laps.loc[laps['Sector3Time'].idxmin()]
                            fastest_lap['best_s3'] = best_s3['Sector3Time']
                            # Also assign driver/race info if missing
                            if 'driverId' in best_s3:
                                fastest_lap['driverId'] = best_s3['driverId']
                            if 'raceId' in best_s3:
                                fastest_lap['raceId'] = best_s3['raceId']
                        else:
                            fastest_lap['best_s3'] = pd.NaT

                        # After assigning fastest_lap['best_s3']
                        if pd.notnull(fastest_lap['best_s3']):
                            fastest_lap['best_s3_sec'] = pd.to_timedelta(fastest_lap['best_s3']).total_seconds()
                        else:
                            fastest_lap['best_s3_sec'] = pd.NA

                        # Only calculate theoretical lap if all sectors are present
                        if pd.notnull(fastest_lap['best_s1']) and pd.notnull(fastest_lap['best_s2']) and pd.notnull(fastest_lap['best_s3']):
                            fastest_lap['best_theory_lap'] = fastest_lap['best_s1'] + fastest_lap['best_s2'] + fastest_lap['best_s3']
                            fastest_lap['best_theory_lap_diff'] = fastest_lap['LapTime'] - (fastest_lap['best_s1'] + fastest_lap['best_s2'] + fastest_lap['best_s3'])
                        else:
                            fastest_lap['best_theory_lap'] = pd.NaT
                            fastest_lap['best_theory_lap_diff'] = pd.NaT

                        all_laps.append(fastest_lap)
                        laps_written = True

                # --- At the end of try, before except ---
                if not laps_written:
                    print(f"No laps found for {session_type} {year} round {round_number}, writing marker row.")
                    marker_row = {
                        'Year': year,
                        'Round': round_number,
                        'Session': session_type,
                        'Driver': 'ERROR',
                        'LapTime': pd.NA,
                        'Sector1Time': pd.NA,
                        'Sector2Time': pd.NA,
                        'Sector3Time': pd.NA,
                        'LapTime_sec': pd.NA,
                        'best_s1_sec': pd.NA,
                        'best_s2_sec': pd.NA,
                        'best_s3_sec': pd.NA,
                        # Add any other columns you expect
                    }
                    all_laps.append(marker_row)

            except Exception as e:
                print(f"Skipping {session_type} for {year} round {round_number}: {e}")
                marker_row = {
                    'Year': year,
                    'Round': round_number,
                    'Session': session_type,
                    'Driver': 'ERROR',
                    'LapTime': pd.NA,
                    'Sector1Time': pd.NA,
                    'Sector2Time': pd.NA,
                    'Sector3Time': pd.NA,
                    'LapTime_sec': pd.NA,
                    'best_s1_sec': pd.NA,
                    'best_s2_sec': pd.NA,
                    'best_s3_sec': pd.NA,
                    # Add any other columns you expect
                }
                all_laps.append(marker_row)
                continue  # Skip to the next session

    if all_laps:
        # new_laps_df = pd.DataFrame(all_laps)
        # ...existing code...
    # if all_laps:
        from collections.abc import Iterable
        print("DEBUG: type(all_laps) =", type(all_laps))

        # Normalize each item into a plain dict/row before building the DataFrame
        rows = []
        for i, item in enumerate(all_laps):
            try:
                if isinstance(item, pd.Series):
                    rows.append(item.to_dict())
                    continue

                if isinstance(item, dict):
                    clean = {}
                    for k, v in item.items():
                        if isinstance(v, pd.Series):
                            clean[k] = v.to_dict()
                        elif hasattr(v, "to_dict") and not isinstance(v, (str, bytes)):
                            try:
                                clean[k] = v.to_dict()
                            except Exception:
                                clean[k] = v
                        else:
                            clean[k] = v
                    rows.append(clean)
                    continue

                if isinstance(item, (list, tuple)):
                    try:
                        rows.append(dict(item))
                        continue
                    except Exception:
                        rows.append({"_raw": item})
                        continue

                rows.append({"_raw": item})
            except Exception as e:
                rows.append({"_error_index": i, "_error": str(e), "_raw_repr": repr(item)})

        try:
            new_laps_df = pd.json_normalize(rows)
        except Exception:
            new_laps_df = pd.DataFrame(rows)

        print("DEBUG: new_laps_df.shape =", getattr(new_laps_df, "shape", None))
    else:
        new_laps_df = pd.DataFrame()    
    # if all_laps:
    # # Defensive conversion for `all_laps` -> DataFrame to avoid dtype/dict shape errors
    #     from collections.abc import Iterable
    #     print("DEBUG: type(all_laps) =", type(all_laps))

    #     if isinstance(all_laps, pd.DataFrame):
    #         new_laps_df = all_laps.copy()
    #     elif isinstance(all_laps, list):
    #         new_laps_df = pd.DataFrame(all_laps)
    #     elif isinstance(all_laps, dict):
    #         try:
    #             new_laps_df = pd.DataFrame.from_dict(all_laps, orient='index')
    #             if new_laps_df.empty:
    #                 raise ValueError("empty after orient='index'")
    #         except Exception:
    #             vals = list(all_laps.values())
    #             if all(isinstance(v, dict) for v in vals):
    #                 new_laps_df = pd.DataFrame(vals)
    #             elif all(isinstance(v, (list, tuple, Iterable)) for v in vals):
    #                 new_laps_df = pd.DataFrame(all_laps)
    #             else:
    #                 new_laps_df = pd.DataFrame([all_laps])
    #     else:
    #         try:
    #             new_laps_df = pd.DataFrame(list(all_laps))
    #         except Exception as e:
    #             # Fallback: make a single-row DF containing the object representation
    #             print(f"WARNING: cannot convert all_laps to table, wrapping as single row: {e}")
    #             new_laps_df = pd.DataFrame([{'raw_all_laps': str(all_laps)}])

    #     print("DEBUG: new_laps_df.shape =", getattr(new_laps_df, "shape", None))
    # else:
    #     new_laps_df = pd.DataFrame()
    # Combine new laps with previous processed laps, keeping all records
    # Debugging: Compare existing, new, and combined records
    if not processed_df.empty:
        processed_df['source'] = 'existing'
    else:
        processed_df = pd.DataFrame()

    if not new_laps_df.empty:
        new_laps_df['source'] = 'new'
    else:
        new_laps_df = pd.DataFrame()

    combined_df = pd.concat([processed_df, new_laps_df], ignore_index=True)

    print("Processed (existing) shape:", processed_df.shape)
    print("New records shape:", new_laps_df.shape)
    print("Combined shape:", combined_df.shape)

    missing_cols = [col for col in ['Year', 'Round', 'Session', 'Driver'] if col not in processed_df.columns]
    if missing_cols:
        print("Missing columns in processed_df:", missing_cols)
        for col in missing_cols:
            processed_df[col] = pd.NA

    # print("Existing sector times:")
    # for col in ['best_s1_sec', 'best_s2_sec', 'best_s3_sec']:
    #     if col not in processed_df.columns:
    #         processed_df[col] = pd.NA
    # print(processed_df[['Year', 'Round', 'Session', 'Driver', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec']].head())

    print("Existing sector times:")
    for col in ['best_s1_sec', 'best_s2_sec', 'best_s3_sec']:
        if col not in processed_df.columns:
            processed_df[col] = pd.NA
    print(processed_df[['Year', 'Round', 'Session', 'Driver', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec']].head())

    # Diagnostic + defensive filler for new_laps_df
    # print("DEBUG: new_laps_df.columns =", list(new_laps_df.columns))
    cols_needed = ['Year', 'Round', 'Session', 'Driver', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec']
    missing = [c for c in cols_needed if c not in new_laps_df.columns]
    if missing:
        # print("DEBUG: missing columns in new_laps_df:", missing)
        for c in missing:
            new_laps_df[c] = pd.NA

    print("New sector times:")
    print(new_laps_df[cols_needed].head())

    print("New sector times:")
    # print(new_laps_df[['Year', 'Round', 'Session', 'Driver', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec']].head())
    

    if not processed_df.empty:
        dfs_to_concat = [df for df in [processed_df, new_laps_df] if not df.empty]
        if dfs_to_concat:
            all_practice_laps_df = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            all_practice_laps_df = pd.DataFrame()
        # all_practice_laps_df = pd.concat([processed_df, new_laps_df], ignore_index=True)

    # Fill missing sector times from any matching row BEFORE dropping duplicates
        for idx, row in all_practice_laps_df.iterrows():
            for sector in ['best_s1_sec', 'best_s2_sec', 'best_s3_sec']:
                if pd.isna(row[sector]):
                    mask = (
                        (all_practice_laps_df['Year'] == row['Year']) &
                        (all_practice_laps_df['Round'] == row['Round']) &
                        (all_practice_laps_df['Session'] == row['Session']) &
                        (all_practice_laps_df['Driver'] == row['Driver']) &
                        (~all_practice_laps_df[sector].isna())
                    )
                    candidates = all_practice_laps_df[mask]
                    if not candidates.empty:
                        all_practice_laps_df.at[idx, sector] = candidates.iloc[0][sector]

        # Now drop duplicates, keeping the row with most sector data
        sector_cols = ['best_s1_sec', 'best_s2_sec', 'best_s3_sec']
        all_practice_laps_df['sector_non_null'] = all_practice_laps_df[sector_cols].notnull().sum(axis=1)
        all_practice_laps_df = all_practice_laps_df.sort_values('sector_non_null', ascending=False)

        all_practice_laps_df = all_practice_laps_df.drop(columns=['sector_non_null'])
        # print(new_laps_df[['Round', 'Session', 'Driver', 'LapTime', 'best_s1', 'best_s2', 'best_s3']].head(20))
    else:
        all_practice_laps_df = new_laps_df

#    Save the combined DataFrame back to CSV
else:
    all_practice_laps_df = processed_df.copy()

# Check which abbreviations are missing from active_drivers
missing_drivers = set(all_practice_laps_df['Driver']) - set(active_drivers['abbreviation'])
print("Drivers missing from active_drivers:", missing_drivers)

# Modify speed to MPH from KM/h
for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
    if col in all_practice_laps_df.columns:
        all_practice_laps_df[f'{col}_mph'] = all_practice_laps_df[col].apply(km_to_miles)

active_drivers['abbreviation'] = active_drivers['abbreviation'].astype(str).str.strip().str.upper()
all_practice_laps_df['Driver'] = all_practice_laps_df['Driver'].astype(str).str.strip().str.upper()

# Merge practice laps with driver names
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_df,
    active_drivers,
    left_on='Driver', 
    right_on='abbreviation',
    how='left'
)

all_practice_laps_with_driver_names.rename(columns={'driverId_y': 'resultsDriverId', 'driverId_x': 'driverId'}, inplace=True)

lap_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec']
for col in lap_cols:
    x_col = f"{col}_x"
    y_col = f"{col}_y"
    if x_col in all_practice_laps_with_driver_names.columns and y_col in all_practice_laps_with_driver_names.columns:
        # Prefer _x, but fill missing with _y
        all_practice_laps_with_driver_names[col] = all_practice_laps_with_driver_names[x_col].combine_first(all_practice_laps_with_driver_names[y_col])
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[x_col, y_col])
    elif x_col in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names[col] = all_practice_laps_with_driver_names[x_col]
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[x_col])
    elif y_col in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names[col] = all_practice_laps_with_driver_names[y_col]
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[y_col])


cols_to_drop = [col for col in all_practice_laps_with_driver_names.columns if col.endswith('_x') or col.endswith('_y')]
if cols_to_drop:
    all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=cols_to_drop)

# Now merge with races
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    races[['id', 'round', 'year']],
    left_on=['Round', 'Year'],
    right_on=['round', 'year'],
    how='left'
).rename(columns={'id': 'raceId'})

all_practice_laps_with_driver_names['Round'] = all_practice_laps_with_driver_names['Round'].astype(int)
all_practice_laps_with_driver_names['Year'] = all_practice_laps_with_driver_names['Year'].astype(int)
races['round'] = races['round'].astype(int)
races['year'] = races['year'].astype(int)

# Drop 'round' and 'year' columns if they exist to avoid merge conflicts
for col in ['round', 'year']:
    if col in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[col])

cols_to_drop = [col for col in all_practice_laps_with_driver_names.columns if col.endswith('_x') or col.endswith('_y')]
if cols_to_drop:
    all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=cols_to_drop)


if 'resultsDriverId' not in all_practice_laps_with_driver_names.columns:
    all_practice_laps_with_driver_names['resultsDriverId'] = all_practice_laps_with_driver_names['driverId']

# Remove duplicate columns (keep only the first occurrence of each column)
all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.loc[:, ~all_practice_laps_with_driver_names.columns.duplicated()]

# Replace resultsDriverId with driverId where resultsDriverId is missing or invalid
mask = (all_practice_laps_with_driver_names['resultsDriverId'].isna()) | (all_practice_laps_with_driver_names['resultsDriverId'] == '-1')
all_practice_laps_with_driver_names.loc[mask, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask, 'driverId']
all_practice_laps_with_driver_names['resultsDriverId'] = all_practice_laps_with_driver_names['resultsDriverId'].astype(str)

# Remove duplicate columns if any
all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.loc[:, ~all_practice_laps_with_driver_names.columns.duplicated()]

# Reset index to avoid alignment issues
all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.reset_index(drop=True)
mask = all_practice_laps_with_driver_names['resultsDriverId'] == '-1'
all_practice_laps_with_driver_names.loc[mask, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask, 'driverId'].astype(str)

# Ensure 'resultsDriverId' is the same type in both DataFrames before merging
all_practice_laps_with_driver_names['resultsDriverId'] = all_practice_laps_with_driver_names['resultsDriverId'].astype(str)
results['resultsDriverId'] = results['resultsDriverId'].astype(str)

all_practice_laps_with_driver_names['Year'] = all_practice_laps_with_driver_names['Year'].astype(int)
results['grandPrixYear'] = results['grandPrixYear'].astype(int)
all_practice_laps_with_driver_names['Round'] = all_practice_laps_with_driver_names['Round'].astype(int)
results['round'] = results['round'].astype(int)


results = results.drop_duplicates(subset=['grandPrixYear', 'round', 'resultsDriverId'])
# ...now do your merge...
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    results[['constructorName', 'grandPrixYear', 'round', 'resultsDriverId']],
    left_on=['Year', 'Round', 'resultsDriverId'],
    right_on=['grandPrixYear', 'round', 'resultsDriverId'],
    how='left'
)


# Rename columns to match the expected format
if 'constructorName' not in all_practice_laps_with_driver_names.columns:
    if 'constructorName_y' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names.rename(columns={'constructorName_y': 'constructorName'}, inplace=True)

# Fill blank or missing constructorName with the value from Team
mask = (all_practice_laps_with_driver_names['constructorName'].isna()) | (all_practice_laps_with_driver_names['constructorName'] == '')
all_practice_laps_with_driver_names.loc[mask, 'constructorName'] = all_practice_laps_with_driver_names.loc[mask, 'Team']

# Manually fix team names
team_name_map = {
    "Red Bull Racing": "Red Bull",
    "Haas F1 Team": "Haas",
    # "Scuderia Ferrari": "Ferrari",
    # Add more mappings as needed
}
all_practice_laps_with_driver_names['constructorName'] = all_practice_laps_with_driver_names['constructorName'].replace(team_name_map)

# Drop unnecessary columns if they exist
cols_to_drop = [
    'raceId.1', 'raceId.2', 'raceId.3', 'raceId_1', 'PitOutTime', 'PitInTime', 'best_s1', 'best_s2', 'best_s3',
    'Time', 'Laptime', 'constructorName_y', 'grandPrixYear_y', 'round.1', 'resultsDriverId_y', 
]

existing_cols = [col for col in cols_to_drop if col in all_practice_laps_with_driver_names.columns]

if existing_cols:
    all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=existing_cols)


for key in ['raceId', 'resultsDriverId']:
    if key in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names[key] = all_practice_laps_with_driver_names[key].fillna(-1)

# Ensure 'FastestPracticeLap_sec' column exists and is correctly named
if 'FastestPracticeLap_sec' not in all_practice_laps_with_driver_names.columns:
    if 'FastestPracticeLap_sec_x' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names.rename(columns={'FastestPracticeLap_sec_x': 'FastestPracticeLap_sec'}, inplace=True)
    elif 'FastestPracticeLap_sec_y' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names.rename(columns={'FastestPracticeLap_sec_y': 'FastestPracticeLap_sec'}, inplace=True)


if 'grandPrixYear_x' in all_practice_laps_with_driver_names.columns:
    all_practice_laps_with_driver_names.rename(columns={'grandPrixYear_x': 'grandPrixYear'}, inplace=True)

mask = all_practice_laps_with_driver_names['grandPrixYear'].isna() | (all_practice_laps_with_driver_names['grandPrixYear'] == '')
all_practice_laps_with_driver_names.loc[mask, 'grandPrixYear'] = all_practice_laps_with_driver_names.loc[mask, 'Year']

# Drop all _x and _y columns at once, only if they exist

cols_to_drop = [col for col in all_practice_laps_with_driver_names.columns if col.endswith('_x') or col.endswith('_y')]
if cols_to_drop:
    all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=cols_to_drop)

# Consolidate duplicate columns that cause the 'ambiguous truth value' error.
# This happens from multiple merges. We keep the first instance of each column.
all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.loc[:,~all_practice_laps_with_driver_names.columns.duplicated()]

# Fill missing sector times from any matching row with same raceId and resultsDriverId
for idx, row in all_practice_laps_with_driver_names.iterrows():
    if pd.notna(row['raceId']) and pd.notna(row['resultsDriverId']):
        for sector in ['best_s1_sec', 'best_s2_sec', 'best_s3_sec']:
            if pd.isna(row[sector]):
                mask = (
                    (all_practice_laps_with_driver_names['Year'] == row['Year']) &
                    (all_practice_laps_with_driver_names['Round'] == row['Round']) &
                    (all_practice_laps_with_driver_names['Session'] == row['Session']) &
                    (all_practice_laps_with_driver_names['Driver'] == row['Driver']) &
                    (all_practice_laps_with_driver_names['raceId'] == row['raceId']) &
                    (all_practice_laps_with_driver_names['resultsDriverId'] == row['resultsDriverId']) &
                    (~all_practice_laps_with_driver_names[sector].isna())
                )
                
                # Defensive: only proceed if mask is boolean Series
                if not isinstance(mask, pd.Series) or mask.dtype != bool:
                    print("Mask is not boolean:", type(mask), mask.dtype)
                    continue
                candidates = all_practice_laps_with_driver_names[mask]

                # FIX: Check type before using .empty
                if isinstance(candidates, pd.DataFrame) and len(candidates) > 0:
                    all_practice_laps_with_driver_names.at[idx, sector] = candidates.iloc[0][sector]
                elif isinstance(candidates, pd.Series) and len(candidates) > 0:
                    all_practice_laps_with_driver_names.at[idx, sector] = candidates.iloc[0]


active_drivers['abbreviation'] = active_drivers['abbreviation'].str.strip().str.upper()
all_practice_laps_with_driver_names['Driver'] = all_practice_laps_with_driver_names['Driver'].str.strip().str.upper()


# Fill missing raceId using Round and Year
race_lookup = races.set_index(['round', 'year'])['id']
mask_raceid = (all_practice_laps_with_driver_names['raceId'].isna()) | (all_practice_laps_with_driver_names['raceId'] == -1)
all_practice_laps_with_driver_names.loc[mask_raceid, 'raceId'] = all_practice_laps_with_driver_names.loc[mask_raceid].apply(
    lambda row: race_lookup.get((row['Round'], row['Year']), -1), axis=1
)

# Ensure all abbreviations and Driver values are uppercase and stripped
active_drivers['abbreviation'] = active_drivers['abbreviation'].str.strip().str.upper()
all_practice_laps_with_driver_names['Driver'] = all_practice_laps_with_driver_names['Driver'].str.strip().str.upper()

# Drop duplicates, keeping the first occurrence
active_drivers = active_drivers.drop_duplicates(subset=['abbreviation'])

# Build lookup
driver_lookup = active_drivers.set_index('abbreviation')['driverId']

# Fill missing resultsDriverId using lookup
mask_driverid = (
    all_practice_laps_with_driver_names['resultsDriverId'].isna()
    | all_practice_laps_with_driver_names['resultsDriverId'].isin(['-1', 'nan', ''])
)
all_practice_laps_with_driver_names.loc[mask_driverid, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask_driverid, 'Driver'].map(driver_lookup)

# If any are still missing, fill with driverId as fallback
mask_still_missing = all_practice_laps_with_driver_names['resultsDriverId'].isna()
all_practice_laps_with_driver_names.loc[mask_still_missing, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask_still_missing, 'driverId']
all_practice_laps_with_driver_names['resultsDriverId'] = all_practice_laps_with_driver_names['resultsDriverId'].astype(str)

# Replace any remaining '-1' with driverId
mask_minus_one = all_practice_laps_with_driver_names['resultsDriverId'] == '-1'
all_practice_laps_with_driver_names.loc[mask_minus_one, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask_minus_one, 'driverId']

# Final bulletproof fill for resultsDriverId
mask_missing = all_practice_laps_with_driver_names['resultsDriverId'].isna() | (all_practice_laps_with_driver_names['resultsDriverId'] == '') | (all_practice_laps_with_driver_names['resultsDriverId'] == '-1')
all_practice_laps_with_driver_names.loc[mask_missing, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask_missing, 'driverId']

# If still missing, fill with Driver abbreviation as last resort
mask_still_missing = all_practice_laps_with_driver_names['resultsDriverId'].isna() | (all_practice_laps_with_driver_names['resultsDriverId'] == '')
all_practice_laps_with_driver_names.loc[mask_still_missing, 'resultsDriverId'] = all_practice_laps_with_driver_names.loc[mask_still_missing, 'Driver']

# Make sure it's a string
all_practice_laps_with_driver_names['resultsDriverId'] = all_practice_laps_with_driver_names['resultsDriverId'].astype(str)

# --- FINAL CALCULATIONS FOR ALL ROWS (old + new) ---
# Ensure LapTime_sec exists for all rows
all_practice_laps_with_driver_names['LapTime_sec'] = pd.to_timedelta(
    all_practice_laps_with_driver_names['LapTime'], errors='coerce'
).dt.total_seconds()

# Calculate FastestPracticeLap_sec for all rows (across FP1, FP2, FP3)
practice_sessions = ['FP1', 'FP2', 'FP3']
all_practice_laps_with_driver_names['FastestPracticeLap_sec'] = (
    all_practice_laps_with_driver_names
    .groupby(['Year', 'Round', 'Driver'])['LapTime_sec']
    .transform('min')
)

all_practice_laps_with_driver_names = (
    all_practice_laps_with_driver_names
    .sort_values(['Year', 'Round', 'Session', 'constructorName', 'LapTime_sec'])
    .drop_duplicates(subset=['Year', 'Round', 'Session', 'Driver', 'constructorName'], keep='first')
)

# Calculate teammate_practice_delta for all valid groups
all_practice_laps_with_driver_names = add_teammate_delta(
    all_practice_laps_with_driver_names,
    ['Year', 'Round', 'Session', 'constructorName'],
    'LapTime_sec',
    'teammate_practice_delta'
)

practice_sessions = ['FP1', 'FP2', 'FP3']
best_constructor_practice = (
    all_practice_laps_with_driver_names[
        all_practice_laps_with_driver_names['Session'].isin(practice_sessions)
    ]
    .sort_values(['Year', 'Round', 'constructorName', 'LapTime_sec'])
    .drop_duplicates(subset=['Year', 'Round', 'constructorName'], keep='first')
    .copy()
)
best_constructor_practice.rename(columns={'LapTime_sec': 'BestConstructorPracticeLap_sec'}, inplace=True)

# Merge best constructor practice lap (across all practice sessions)
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    best_constructor_practice[['Year', 'Round', 'constructorName', 'BestConstructorPracticeLap_sec']],
    on=['Year', 'Round', 'constructorName'],
    how='left'
)
if 'BestConstructorPracticeLap_sec' not in all_practice_laps_with_driver_names.columns:
    if 'BestConstructorPracticeLap_sec_y' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names.rename(columns={'BestConstructorPracticeLap_sec_y': 'BestConstructorPracticeLap_sec'}, inplace=True)
    else:
        all_practice_laps_with_driver_names['BestConstructorPracticeLap_sec'] = pd.NA

# Drop all columns ending with _x, _y, or .1
cols_to_drop = [col for col in all_practice_laps_with_driver_names.columns if col.endswith('_x') or col.endswith('_y') or col.endswith('.1')]
if cols_to_drop:
    all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=cols_to_drop)

# Convert best_theory_lap and best_theory_lap_diff to seconds for all_practice_laps_with_driver_names
all_practice_laps_with_driver_names['best_theory_lap_sec'] = pd.to_timedelta(
    all_practice_laps_with_driver_names['best_theory_lap'], errors='coerce'
).dt.total_seconds()

all_practice_laps_with_driver_names['best_theory_lap_diff_sec'] = pd.to_timedelta(
    all_practice_laps_with_driver_names['best_theory_lap_diff'], errors='coerce'
).dt.total_seconds()


# Save to CSV
all_practice_laps_with_driver_names.to_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t', index=False)
print("Saved all practice laps to all_practice_laps.csv")

# --- Create slimmed-down best lap file for FP1/FP2 ---
practice_sessions = ['FP1', 'FP2']
fp_laps = all_practice_laps_with_driver_names[
    all_practice_laps_with_driver_names['Session'].isin(practice_sessions)
].copy()

# Sort so FP2 comes first, then FP1, then by best lap time (ascending)
fp_laps['Session_priority'] = fp_laps['Session'].map({'FP2': 1, 'FP1': 2})
fp_laps = fp_laps.sort_values(
    ['Year', 'Round', 'Driver', 'FastestPracticeLap_sec', 'Session_priority']
)

# Drop duplicates so you keep only the best FP2 lap if available, otherwise best FP1
fp_laps_deduped = fp_laps.drop_duplicates(
    subset=['Year', 'Round', 'Driver'], keep='first'
).copy()

for sector in ['best_s1_sec', 'best_s2_sec', 'best_s3_sec']:
    # Fill missing sector times with the first non-null value for each group
    fp_laps_deduped[sector] = fp_laps_deduped[sector].combine_first(
        fp_laps.groupby(['Year', 'Round', 'Driver'])[sector].transform('first')
    )

# ...after you define fp_laps_final...

# Ensure abbreviations are uppercase and stripped
results['abbreviation'] = results['abbreviation'].astype(str).str.strip().str.upper()
fp_laps_deduped['Driver'] = fp_laps_deduped['Driver'].astype(str).str.strip().str.upper()

# Merge to fill driverId using raceId and Driver abbreviation
fp_laps_deduped = pd.merge(
    fp_laps_deduped,
    results[['raceId_results', 'abbreviation', 'resultsDriverId']].drop_duplicates(),
    left_on=['raceId', 'Driver'],
    right_on=['raceId_results', 'abbreviation'],
    how='left',
    suffixes=('', '_from_results')
)

# Prefer the driverId from results if available
fp_laps_deduped['driverId'] = fp_laps_deduped['resultsDriverId_from_results'].combine_first(fp_laps_deduped['driverId'])

# print(fp_laps_deduped['driverId'].isnull().sum())
# print(fp_laps_deduped['driverId'].unique())
missing = fp_laps_deduped[fp_laps_deduped['driverId'].isnull()]

# Ensure both are uppercase and stripped
results['abbreviation'] = results['abbreviation'].astype(str).str.strip().str.upper()
missing = missing.copy()
missing['Driver'] = missing['Driver'].astype(str).str.strip().str.upper()

# Try to find matches
for idx, row in missing.iterrows():
    match = results[
        (results['raceId_results'] == row['raceId']) &
        (results['abbreviation'] == row['Driver'])
    ]

active_drivers['abbreviation'] = active_drivers['abbreviation'].astype(str).str.strip().str.upper()
driver_lookup = active_drivers.set_index('abbreviation')['driverId']

mask_missing = fp_laps_deduped['driverId'].isnull()
fp_laps_deduped.loc[mask_missing, 'driverId'] = fp_laps_deduped.loc[mask_missing, 'Driver'].map(driver_lookup)
print("Still missing driverId count:", fp_laps_deduped['driverId'].isnull().sum())

print(fp_laps_deduped[fp_laps_deduped['driverId'].isnull()]['Driver'].unique())
# Select only the columns you want to keep
columns_to_keep = [
    'Year', 'Round', 'raceId', 'Driver', 'driverId', 'LapTime_sec',
    'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'FastestPracticeLap_sec',
    'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
    'best_theory_lap_sec', 'Session'
]
for col in columns_to_keep:
    if col not in fp_laps_deduped.columns:
        # fp_laps_deduped[col] = pd.NA
        fp_laps_deduped.loc[:, col] = pd.NA
fp_laps_final = fp_laps_deduped[columns_to_keep]

fp_laps_final.to_csv(
    path.join(DATA_DIR, 'practice_best_fp1_fp2.csv'), sep='\t', index=False
)
print("Saved slimmed practice file to practice_best_fp1_fp2.csv")