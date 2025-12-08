from matplotlib import pyplot as plt
import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t') 

# Initialize Ergast API
ergast = Ergast(result_type='pandas', auto_cast=True)

# --- NEW: Load existing sessions if file exists ---
existing_sessions = set()
output_file = path.join(DATA_DIR, 'all_race_control_messages.csv')
if path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file, sep='\t')
        # Use Year and Round as unique session identifier
        existing_sessions = set(zip(existing_df['Year'], existing_df['Round']))
    except Exception as e:
        print(f"Could not read existing file: {e}")

# Initialize lists to store race control messages
all_race_control_messages = []


# Helper: make column names unique and coalesce identifier-like candidates
def _coalesce(df, target, candidates):
    for c in candidates:
        if c in df.columns:
            if target not in df.columns:
                df[target] = df[c]
            else:
                df[target] = df[target].fillna(df[c])


def _dedupe_and_coalesce_identifiers(df):
    # Make duplicate column names unique by appending an index (_1, _2, ...)
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    # Find candidate columns for common identifiers
    lower_cols = [c.lower() for c in df.columns]
    def find_candidates(preds):
        return [c for c in df.columns if any(p in c.lower() for p in preds)]

    race_cands = find_candidates(['raceid', 'race_id', ' raceid', ' id'])
    gp_cands = find_candidates(['grand', 'prix', 'grandprix', 'grand_prix'])
    round_cands = find_candidates(['round'])
    year_cands = find_candidates(['year'])

    _coalesce(df, 'raceId', race_cands)
    _coalesce(df, 'grandPrixId', gp_cands)
    _coalesce(df, 'Round', round_cands)
    _coalesce(df, 'Year', year_cands)


def _collapse_named(df, base):
    # Find columns that are the base name or base with an appended suffix and merge them
    candidates = [c for c in df.columns if c == base or c.startswith(f"{base}_")]
    if not candidates:
        return
    # Ensure base exists
    if base not in df.columns:
        df[base] = pd.NA
    for c in candidates:
        if c == base:
            continue
        try:
            df[base] = df[base].fillna(df[c])
        except Exception:
            pass
        try:
            df.drop(columns=[c], inplace=True)
        except Exception:
            pass


# Loop through all seasons and rounds
for i in range(2018, current_year + 1):
    # Get the number of rounds in each season
    season_schedule = ergast.get_race_schedule(season=i)
    total_rounds = len(season_schedule)

    #for round_number in range(1, 3):
    for round_number in range(1, total_rounds + 1):
        # --- NEW: Skip if session already exists ---
        if (i, round_number) in existing_sessions:
            print(f"Skipping {i} round {round_number} (already pulled)")
            continue
        try:
            # Load the race session
            session = fastf1.get_session(i, round_number, 'R')
            session.load(messages=True)

            # Get race control messages as a DataFrame
            race_control_messages = session.race_control_messages
            if isinstance(race_control_messages, pd.DataFrame):
                # Add metadata to the DataFrame
                race_control_messages['Round'] = round_number
                race_control_messages['Year'] = i
                race_control_messages['Event'] = session.event['EventName']
                all_race_control_messages.append(race_control_messages)
        except Exception as e:
            print(f"Failed to load session {i} round {round_number}: {e}")

# Combine all race control messages into a single DataFrame
if all_race_control_messages:
    all_race_control_messages_df = pd.concat(all_race_control_messages, ignore_index=True)
    # If file exists, append new data
    if path.exists(output_file):
        all_race_control_messages_df = pd.concat([existing_df, all_race_control_messages_df], ignore_index=True)
    all_race_control_messages_df.to_csv(output_file, sep='\t', index=False)

    # Save the combined DataFrame to a CSV file (optional)
    # all_race_control_messages_df.to_csv(path.join(DATA_DIR, 'all_race_control_messages.csv'), sep='\t', index=False)

    # Robustly merge race control messages with the races table.
    # The races DataFrame may use slightly different column names for the grand-prix identifier
    # (e.g., 'grandPrixId', 'grand_prix_id', 'grand_prix', etc.). Detect a suitable column name
    # and merge on round/year. Use left join so we can surface missing mappings and handle them.
    race_cols = races.columns.tolist()
    def _find_gp_col(cols):
        # prefer an exact match first
        for cand in ['grandPrixId', 'grand_prix_id', 'grand_prix', 'grandprix', 'grand_prix', 'grand_prixId']:
            if cand in cols:
                return cand
        # fallback: any column containing both 'grand' and 'prix' (case-insensitive)
        for c in cols:
            n = c.lower()
            if 'grand' in n and 'prix' in n:
                return c
        return None

    gp_col = _find_gp_col(race_cols)
    # Build list of columns to grab from races for the merge
    use_cols = [c for c in ['id', 'round', 'year'] if c in races.columns]
    if gp_col and gp_col not in use_cols:
        use_cols.append(gp_col)

    # Ensure duplicate columns are deduped and identifier-like columns are coalesced
    try:
        _dedupe_and_coalesce_identifiers(all_race_control_messages_df)
    except Exception:
        pass

    race_control_messages_with_grandprix = pd.merge(
        all_race_control_messages_df,
        races[use_cols],
        left_on=['Round', 'Year'],
        right_on=['round', 'year'],
        how='left'
    ).drop_duplicates()

    # Collapse any duplicated/variant id-like columns produced by the merge
    _collapse_named(race_control_messages_with_grandprix, 'raceId')
    _collapse_named(race_control_messages_with_grandprix, 'grandPrixId')
    _collapse_named(race_control_messages_with_grandprix, 'Round')
    _collapse_named(race_control_messages_with_grandprix, 'Year')

    # Normalize race id column name to `raceId` (may be `id`, `id_x`, or `id_y` depending on context)
    if 'id' in race_control_messages_with_grandprix.columns:
        race_control_messages_with_grandprix.rename(columns={'id': 'raceId'}, inplace=True)
    elif 'id_x' in race_control_messages_with_grandprix.columns:
        race_control_messages_with_grandprix.rename(columns={'id_x': 'raceId'}, inplace=True)
    elif 'id_y' in race_control_messages_with_grandprix.columns:
        race_control_messages_with_grandprix.rename(columns={'id_y': 'raceId'}, inplace=True)

    # Normalize grand-prix id column name to `grandPrixId`
    if 'grandPrixId' not in race_control_messages_with_grandprix.columns:
        if gp_col and gp_col in race_control_messages_with_grandprix.columns:
            race_control_messages_with_grandprix['grandPrixId'] = race_control_messages_with_grandprix[gp_col]
        # try common fallbacks
        elif 'grand_prix_id' in race_control_messages_with_grandprix.columns:
            race_control_messages_with_grandprix['grandPrixId'] = race_control_messages_with_grandprix['grand_prix_id']
        elif 'grand_prix' in race_control_messages_with_grandprix.columns:
            race_control_messages_with_grandprix['grandPrixId'] = race_control_messages_with_grandprix['grand_prix']

    # If raceId is still missing, attempt to map it from races using (round, year)
    if 'raceId' not in race_control_messages_with_grandprix.columns or race_control_messages_with_grandprix['raceId'].isnull().all():
        mapping = races.set_index(['round', 'year'])['id'] if {'round', 'year', 'id'}.issubset(races.columns) else None
        if mapping is not None:
            # build tuple keys
            keys = list(zip(race_control_messages_with_grandprix['Round'], race_control_messages_with_grandprix['Year']))
            race_control_messages_with_grandprix['raceId'] = [mapping.get(k, pd.NA) for k in keys]
    # end robust merge

    # Coalesce common duplicated/suffixed columns into canonical names to handle older merged files
    _coalesce(race_control_messages_with_grandprix, 'raceId', [c for c in race_control_messages_with_grandprix.columns if 'race' in c.lower() or c.lower()=='id' or c.lower().startswith('id_')])
    _coalesce(race_control_messages_with_grandprix, 'grandPrixId', [c for c in race_control_messages_with_grandprix.columns if ('grand' in c.lower() and 'prix' in c.lower()) or 'grand_prix' in c.lower()])
    _coalesce(race_control_messages_with_grandprix, 'Round', [c for c in race_control_messages_with_grandprix.columns if 'round' in c.lower()])
    _coalesce(race_control_messages_with_grandprix, 'Year', [c for c in race_control_messages_with_grandprix.columns if 'year' in c.lower()])

    # Now drop helper suffixed columns if present to avoid confusion
    for suf in ['_x', '_y']:
        for col in list(race_control_messages_with_grandprix.columns):
            if col.endswith(suf) and col.replace(suf, '') in ['id', 'round', 'year', 'grandPrixId', 'grand_prix', 'grand_prix_id']:
                # keep original if it exists; otherwise remove
                try:
                    race_control_messages_with_grandprix.drop(columns=[col], inplace=True)
                except Exception:
                    pass

    # Diagnostic: detect rows that could not be mapped to a raceId or grandPrixId
    missing_raceid_mask = False
    if 'raceId' in race_control_messages_with_grandprix.columns and 'grandPrixId' in race_control_messages_with_grandprix.columns:
        missing_raceid_mask = race_control_messages_with_grandprix['raceId'].isna() | race_control_messages_with_grandprix['grandPrixId'].isna()
    elif 'raceId' in race_control_messages_with_grandprix.columns:
        missing_raceid_mask = race_control_messages_with_grandprix['raceId'].isna()
    elif 'grandPrixId' in race_control_messages_with_grandprix.columns:
        missing_raceid_mask = race_control_messages_with_grandprix['grandPrixId'].isna()

    if isinstance(missing_raceid_mask, (pd.Series, list)) and missing_raceid_mask.any():
        unmapped = race_control_messages_with_grandprix[missing_raceid_mask].copy()
        # write a diagnostic CSV grouped by Year for quick inspection
        years = sorted(unmapped['Year'].dropna().unique().tolist())
        fname = path.join(DATA_DIR, f"unmapped_race_messages_{years[0] if years else 'unknown'}.csv")
        try:
            unmapped.to_csv(fname, sep='\t', index=False)
            print(f"Wrote unmapped race control messages to: {fname} (rows: {len(unmapped)})")
        except Exception as _e:
            print(f"Could not write unmapped diagnostic file: {_e}")

        # If everything is unmapped, abort further grouping to avoid KeyErrors
        if len(unmapped) == len(race_control_messages_with_grandprix):
            print("All merged race control messages are unmapped (no raceId/grandPrixId). Exiting to avoid groupby KeyError.")
            # Save combined file for debugging and exit early
            debug_fname = path.join(DATA_DIR, 'all_race_control_messages_merged_debug.csv')
            try:
                race_control_messages_with_grandprix.to_csv(debug_fname, sep='\t', index=False)
                print(f"Wrote debug merged file to: {debug_fname}")
            except Exception:
                pass
            raise SystemExit(0)

    # all_race_control_messages_df.to_csv(path.join(DATA_DIR, 'all_race_control_messages.csv'), sep='\t', index=False)
    race_control_messages_with_grandprix.to_csv(path.join(DATA_DIR, 'all_race_control_messages.csv'), sep='\t', index=False)

    # Filter rows where 'Category' is either 'Flag' or 'SafetyCar'
    race_control_messages_with_grandprix = race_control_messages_with_grandprix[
        race_control_messages_with_grandprix['Category'].isin(['Flag', 'SafetyCar'])
    ]

    # Group and aggregate the filtered DataFrame
    
    # df.rename(columns={'old_name': 'new_name'}, inplace=True)
    race_control_messages_with_grandprix.rename(columns={'id': 'raceId'}, inplace=True)

    # Remove sector from the uniqueness so each flag per lap is only counted once
    race_control_messages_with_grandprix_nosector = race_control_messages_with_grandprix.drop_duplicates(
        subset=['raceId', 'Lap', 'Flag']
    )

    race_control_messages_with_grandprix_grouped = race_control_messages_with_grandprix_nosector.groupby(
        ['Round', 'Year', 'raceId', 'grandPrixId']
    ).agg(
        SafetyCarStatus=('Status', lambda x: (x == 'DEPLOYED').sum()),
        redFlag=('Flag', lambda x: (x == 'RED').sum()),
        yellowFlag=('Flag', lambda x: (x == 'YELLOW').sum()),
        doubleYellowFlag=('Flag', lambda x: (x == 'DOUBLE YELLOW').sum()),
    ).reset_index()
    #print(race_control_messages_with_grandprix_grouped.shape)

    # DNF summary (move this inside the block)
    dnf_summary = results[
        results['resultsReasonRetired'].notnull() & (results['resultsReasonRetired'] != '')
    ].groupby('raceId_results').size().reset_index(name='dnf_count')

    race_control_with_dnf = pd.merge(race_control_messages_with_grandprix_grouped, dnf_summary, left_on='raceId', right_on='raceId_results', how='left')
    #print(race_control_with_dnf.shape)
    # Now you can use dnf_summary here
    #print(race_control_with_dnf.head(50))


    race_control_with_dnf.to_csv(path.join(DATA_DIR, 'race_control_messages_grouped_with_dnf.csv'), sep='\t', index=False)

    

else:
    # If no new messages were pulled, but an existing combined file exists, process it
    if 'existing_df' in locals() and not existing_df.empty:
        print("No new race control messages pulled; processing existing combined file.")
        all_race_control_messages_df = existing_df.copy()

        # Run the same downstream processing as when new messages are present
        race_control_messages_with_grandprix = pd.merge(
            all_race_control_messages_df,
            races[[c for c in races.columns if c in ['id','round','year'] or ('grand' in c.lower() and 'prix' in c.lower())]],
            left_on=['Round', 'Year'],
            right_on=['round', 'year'],
            how='left'
        ).drop_duplicates()

        # Ensure duplicate columns are deduped and identifier-like columns are coalesced
        try:
            _dedupe_and_coalesce_identifiers(all_race_control_messages_df)
        except Exception:
            pass

        # Collapse any duplicated/variant id-like columns produced by the merge
        _collapse_named(race_control_messages_with_grandprix, 'raceId')
        _collapse_named(race_control_messages_with_grandprix, 'grandPrixId')
        _collapse_named(race_control_messages_with_grandprix, 'Round')
        _collapse_named(race_control_messages_with_grandprix, 'Year')

        # If there is still an 'id' column (from races), coalesce into 'raceId' then drop it
        if 'id' in race_control_messages_with_grandprix.columns:
            if 'raceId' not in race_control_messages_with_grandprix.columns:
                race_control_messages_with_grandprix.rename(columns={'id': 'raceId'}, inplace=True)
            else:
                race_control_messages_with_grandprix['raceId'] = race_control_messages_with_grandprix['raceId'].fillna(race_control_messages_with_grandprix['id'])
                try:
                    race_control_messages_with_grandprix.drop(columns=['id'], inplace=True)
                except Exception:
                    pass

        # Remove sector from the uniqueness so each flag per lap is only counted once
        race_control_messages_with_grandprix_nosector = race_control_messages_with_grandprix.drop_duplicates(
            subset=['raceId', 'Lap', 'Flag']
        )

        race_control_messages_with_grandprix_grouped = race_control_messages_with_grandprix_nosector.groupby(
            ['Round', 'Year', 'raceId', 'grandPrixId']
        ).agg(
            SafetyCarStatus=('Status', lambda x: (x == 'DEPLOYED').sum()),
            redFlag=('Flag', lambda x: (x == 'RED').sum()),
            yellowFlag=('Flag', lambda x: (x == 'YELLOW').sum()),
            doubleYellowFlag=('Flag', lambda x: (x == 'DOUBLE YELLOW').sum()),
        ).reset_index()

        # DNF summary (move this inside the block)
        dnf_summary = results[
            results['resultsReasonRetired'].notnull() & (results['resultsReasonRetired'] != '')
        ].groupby('raceId_results').size().reset_index(name='dnf_count')

        race_control_with_dnf = pd.merge(race_control_messages_with_grandprix_grouped, dnf_summary, left_on='raceId', right_on='raceId_results', how='left')
        race_control_with_dnf.to_csv(path.join(DATA_DIR, 'race_control_messages_grouped_with_dnf.csv'), sep='\t', index=False)
        print('Wrote race_control_messages_grouped_with_dnf.csv')
    else:
        print("No race control messages were found.")

# Add race messages and DNF summary
    #### way too high. 
    

#response_frame = ergast.get_circuits(season=2022, )
#print(response_frame)
#print(response_frame.columns)

#laps = session.laps
#print(laps.head())  # Ensure it's a DataFrame

#drivers = session.drivers

##stints = laps[["Driver", "DriverNumber", "Stint", "Compound", "LapNumber"]]
#stints = stints.groupby(["Driver", "DriverNumber", "Stint", "Compound"])
#stints = stints.count().reset_index()
#stints = stints.rename(columns={"LapNumber": "StintLength"})
#print(stints)

#hungary = fastf1.get_event(2022, "Hungary")
#print(hungary)

##schedule_2022 = fastf1.get_event_schedule(2022)
#print(schedule_2022)

#hungary_schedule = hungary.get_race()
#print(hungary_schedule)