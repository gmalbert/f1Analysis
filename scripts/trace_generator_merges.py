import pandas as pd
from os import path
import os

DATA_DIR='data_files/'

def load(name):
    p=path.join(DATA_DIR, name)
    print(f"Loading {p}...", end=' ')
    if not os.path.exists(p):
        print('MISSING')
        return None
    if p.endswith('.json'):
        df=pd.read_json(p)
    else:
        df=pd.read_csv(p, sep='\t')
    print('done, rows=', len(df))
    return df

# Load raw sources
races = load('f1db-races.json')
grandPrix = load('f1db-grands-prix.json')
drivers = load('f1db-drivers.json')
race_results = load('f1db-races-race-results.json')
qualifying_json = load('f1db-races-qualifying-results.json')
qualifying_csv = load('all_qualifying_races.csv')
fp1 = load('f1db-races-free-practice-1-results.json')
fp2 = load('f1db-races-free-practice-2-results.json')
fp3 = load('f1db-races-free-practice-3-results.json')
current_practices = load('all_practice_laps.csv')

print('\nStage: races max date')
if races is not None and 'date' in races.columns:
    races['date_parsed']=pd.to_datetime(races['date'], errors='coerce')
    print('races max date:', races['date_parsed'].max())

# merge races with grandPrix
if races is not None and grandPrix is not None:
    rp = races.merge(grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_races', '_grandPrix'])
    if 'date' in rp.columns:
        rp['date_parsed']=pd.to_datetime(rp['date'], errors='coerce')
        print('After merge races+grandPrix, max date:', rp['date_parsed'].max())
    print('rp rows:', len(rp))
else:
    rp=None

# results_and_drivers: merge race_results with drivers
if race_results is not None and drivers is not None:
    results_and_drivers = race_results.merge(drivers, left_on='driverId', right_on='id', how='inner', suffixes=['_results','_drivers'])
    print('results_and_drivers rows:', len(results_and_drivers))
    # if results has a year/date
    if 'date' in results_and_drivers.columns:
        results_and_drivers['date_parsed']=pd.to_datetime(results_and_drivers['date'], errors='coerce')
        print('results_and_drivers max date:', results_and_drivers['date_parsed'].max())
else:
    results_and_drivers=None

# results_and_drivers filtered by recent years? replicate raceNoEarlierThan behavior
current_year = pd.Timestamp.now().year
raceNoEarlierThan = current_year - 10
if results_and_drivers is not None and 'year' in results_and_drivers.columns:
    before = len(results_and_drivers)
    results_and_drivers = results_and_drivers[results_and_drivers['year'] >= raceNoEarlierThan]
    print(f'Filtered results_and_drivers by year>={raceNoEarlierThan}: {before} -> {len(results_and_drivers)}')

# merge with constructors
constructors = load('f1db-constructors.json')
if results_and_drivers is not None and constructors is not None:
    radc = results_and_drivers.merge(constructors, left_on='constructorId', right_on='id', how='inner', suffixes=['_results','_constructors'])
    print('results_and_drivers_and_constructors rows:', len(radc))
else:
    radc=None

# merge with races/grandprix (use rp)
if radc is not None and rp is not None:
    print('rp columns:', rp.columns.tolist())
    # try common right keys used in generator
    candidate_keys = ['raceIdFromGrandPrix', 'id_races', 'raceId', 'id']
    used = None
    for k in candidate_keys:
        if k in rp.columns:
            used = k
            break
    if used is None:
        print('No candidate key found in rp for merging (expected raceIdFromGrandPrix or id_races). Skipping this merge.')
        merged = None
    else:
        merged = radc.merge(rp, left_on='raceId', right_on=used, how='inner', suffixes=['_results','_grandprix'])
        print(f'after merging with grandprix using right key {used}, rows:', len(merged))
        if 'date' in merged.columns:
            merged['date_parsed']=pd.to_datetime(merged['date'], errors='coerce')
            print('merged max date:', merged['date_parsed'].max())
else:
    merged=None

# merge with qualifying (qualifying_json + qualifying_csv)
if merged is not None and (qualifying_json is not None and qualifying_csv is not None):
    try:
        qual = qualifying_json.merge(qualifying_csv[['q1_sec','q1_pos','q2_sec','q2_pos','q3_sec','q3_pos','best_qual_time','teammate_qual_delta','raceId','driverId']], left_on=['raceId','driverId'], right_on=['raceId','driverId'], how='right')
        print('qual shape:', qual.shape)
        # inspect what race dates are covered by qualifying data
        if rp is not None and 'id_races' in rp.columns:
            kval_race_ids = qual['raceId'].unique()
            kval_rp = rp[rp['id_races'].isin(kval_race_ids)]
            if not kval_rp.empty and 'date_parsed' in kval_rp.columns:
                print('qualifying covers races up to:', kval_rp['date_parsed'].max())
        # determine left merge keys in merged
        print('merged columns before qual merge:', merged.columns.tolist())
        # candidate keys for race id in merged (right key used previously)
        race_key_candidates = [
            'raceIdFromGrandPrix', 'id_races', 'raceId_results', 'raceId', 'id'
        ]
        left_race_key = None
        for k in race_key_candidates:
            if k in merged.columns:
                left_race_key = k
                break
        # candidate keys for driver id in merged
        driver_key_candidates = ['resultsDriverId', 'driverId', 'driverId_results', 'id_driver']
        left_driver_key = None
        for k in driver_key_candidates:
            if k in merged.columns:
                left_driver_key = k
                break
        if left_race_key is None or left_driver_key is None:
            print('Could not find suitable keys in merged for qual merge. Skipping.')
            merged_q = None
        else:
            merged_q = merged.merge(qual, left_on=[left_race_key, left_driver_key], right_on=['raceId','driverId'], how='inner', suffixes=['_results','_qualifying'])
            print(f'after merging with qualifying using keys {left_race_key},{left_driver_key}, rows:', len(merged_q))
            if 'date' in merged_q.columns:
                merged_q['date_parsed']=pd.to_datetime(merged_q['date'], errors='coerce')
                print('merged_q max date:', merged_q['date_parsed'].max())
    except Exception as e:
        print('qual merge failed:', e)
else:
    merged_q=None

# merge with practices (fp1, fp2, fp3)
if merged_q is not None and (fp1 is not None and fp2 is not None and fp3 is not None):
    try:
        fp12 = fp1.merge(fp2, on=['raceId','driverId'], how='left', suffixes=['_fp1','_fp2'])
        fp123 = fp12.merge(fp3, on=['raceId','driverId'], how='left')
        merged_pr = merged_q.merge(fp123, left_on=['raceId_results','resultsDriverId'], right_on=['raceId','driverId'], how='left')
        print('after merging practices, rows:', len(merged_pr))
        if 'date' in merged_pr.columns:
            merged_pr['date_parsed']=pd.to_datetime(merged_pr['date'], errors='coerce')
            print('merged_pr max date:', merged_pr['date_parsed'].max())
    except Exception as e:
        print('practices merge failed:', e)
else:
    merged_pr=None

print('\nDone')
