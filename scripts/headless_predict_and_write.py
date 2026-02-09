#!/usr/bin/env python3
"""Headless predictor: build simple predictions from analysis CSV or active drivers.

This script:
- Reads `data_files/f1ForAnalysis.csv` and `data_files/active_drivers.csv`.
- Determines next race via `data_files/f1db-races.json`.
- Builds a driver-level DataFrame for the next race and computes a simple
  ranking-based prediction using available numeric features.
- Writes `data_files/predictions_headless_<race_slug>_<year>.csv`.

This is a lightweight fallback when trained models or prediction artifacts
are not available.
"""
import os
import re
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data_files')


def slugify(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", '-', s)
    s = s.strip('-')
    return s


def compute_next_race_from_calendar():
    races_json = os.path.join(DATA_DIR, 'f1db-races.json')
    if not os.path.exists(races_json):
        return None, None
    with open(races_json, 'r', encoding='utf-8') as fh:
        races = json.load(fh)
    now = pd.to_datetime(datetime.now())
    candidates = []
    for r in races:
        sd = r.get('short_date') or r.get('date') or r.get('race_date') or r.get('raceDate')
        sd_dt = pd.to_datetime(sd, errors='coerce')
        if sd_dt is not None and not pd.isna(sd_dt) and sd_dt >= now:
            name = r.get('grandPrixName') or r.get('name') or r.get('raceName')
            candidates.append((sd_dt, name, r))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]


def build_driver_slice(next_race_name, next_race_date):
    analysis_path = os.path.join(DATA_DIR, 'f1ForAnalysis.csv')
    if os.path.exists(analysis_path):
        df = pd.read_csv(analysis_path, sep='\t')
        # try match by grandPrixName or by short_date
        sel = pd.DataFrame()
        if next_race_name and 'grandPrixName' in df.columns:
            sel = df[df['grandPrixName'].astype(str).str.lower().str.contains(str(next_race_name).lower(), na=False)].copy()
        if sel.empty and next_race_date is not None and 'short_date' in df.columns:
            df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
            sel = df[df['short_date'] == pd.to_datetime(next_race_date)].copy()
        if not sel.empty:
            # collapse to one row per active driver if duplicates exist
            sel = sel.drop_duplicates(subset=['resultsDriverId'])
            return sel

    # fallback: use active_drivers.csv and driver reference
    active_path = os.path.join(DATA_DIR, 'active_drivers.csv')
    drivers = pd.DataFrame()
    if os.path.exists(active_path):
        # active_drivers.csv is tab-separated in this repo
        try:
            drivers = pd.read_csv(active_path, sep='\t')
        except Exception:
            drivers = pd.read_csv(active_path)
        if 'resultsDriverId' in drivers.columns:
            sel = pd.DataFrame({'resultsDriverId': drivers['resultsDriverId'].unique()})
        elif 'id' in drivers.columns:
            sel = pd.DataFrame({'resultsDriverId': drivers['id'].unique()})
        else:
            sel = pd.DataFrame({'resultsDriverId': drivers.iloc[:,0].astype(str).unique()})
        # try to attach names
        if os.path.exists(os.path.join(DATA_DIR, 'active_drivers_interim.csv')):
            try:
                # interim file is tab-separated
                ref = pd.read_csv(os.path.join(DATA_DIR, 'active_drivers_interim.csv'), sep='\t')
                if 'resultsDriverId' in ref.columns and 'resultsDriverName' in ref.columns:
                    sel = sel.merge(ref[['resultsDriverId','resultsDriverName']], on='resultsDriverId', how='left')
            except Exception:
                pass
        return sel

    return pd.DataFrame()


def make_headless_predictions(driver_df):
    if driver_df.empty:
        return driver_df
    df = driver_df.copy()
    # Prefer to preprocess features the same way as raceAnalysis.py
    def load_f1_position_model_features():
        filepath = os.path.join(DATA_DIR, 'f1_position_model_numerical_features.txt')
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                default_numerical = [line.strip() for line in f if line.strip()]
        else:
            default_numerical = []
        monte_carlo_filepath = os.path.join(DATA_DIR, 'f1_position_model_best_features_monte_carlo.txt')
        if os.path.exists(monte_carlo_filepath):
            with open(monte_carlo_filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('Best MAE')]
            numerical = [f for f in lines if f in default_numerical]
            categorical = []
            if numerical or categorical:
                return numerical, categorical
        return default_numerical, []

    numerical_features, categorical_features = load_f1_position_model_features()

    def get_preprocessor_position(X=None):
        global numerical_features, categorical_features
        numerical_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        if not numerical_features and not categorical_features:
            if X is None:
                numerical_features_fallback = []
                categorical_features_fallback = []
            else:
                from pandas.api.types import is_numeric_dtype
                numerical_features_fallback = [col for col in X.columns if is_numeric_dtype(X[col])]
                categorical_features_fallback = [col for col in X.columns if not is_numeric_dtype(X[col])]
            transformers = [
                ('num', Pipeline(steps=[('imputer', numerical_imputer),('scaler', StandardScaler())]), numerical_features_fallback)
            ]
            if categorical_features_fallback:
                transformers.append((
                    'cat', Pipeline(steps=[('imputer', categorical_imputer),('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features_fallback
                ))
        else:
            # Filter out numerical features that are all NaN to avoid sklearn imputation warnings
            if X is not None:
                numerical_features = [col for col in numerical_features if col in X.columns and not X[col].isna().all()]
            
            transformers = [
                ('num', Pipeline(steps=[('imputer', numerical_imputer),('scaler', StandardScaler())]), numerical_features)
            ]
            if categorical_features:
                transformers.append((
                    'cat', Pipeline(steps=[('imputer', categorical_imputer),('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features
                ))
        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor
    # Identify numeric columns to use for ranking; prefer preprocessed feature space
    # Drop identifier columns before preprocessing
    predictor_df = df.drop(columns=[c for c in ['resultsDriverId','resultsDriverName','constructorName'] if c in df.columns], errors='ignore')
    preprocessor = get_preprocessor_position(predictor_df)
    try:
        preprocessor.fit(predictor_df)
        transformed = preprocessor.transform(predictor_df)
        # Wrap into DataFrame so we can compute ranks per transformed feature
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        trans_df = pd.DataFrame(transformed, index=predictor_df.index)
        features = trans_df.columns.tolist()
        use_ranks_df = trans_df
    except Exception:
        # Fallback to original numeric/time-like features
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        time_like = [c for c in df.columns if any(s in c.lower() for s in ('lap','time','sec','best','s1','s2','s3'))]
        features = list(set(num_cols + time_like))
    # Remove id/name columns
    features = [c for c in features if c not in ('resultsDriverId', 'resultsDriverName', 'constructorName')]

    if not features:
        # fallback: sort by name
        df['PredictedFinalPosition'] = range(1, len(df)+1)
        df['PredictedPositionMAE'] = 3.0
        df['PredictedFinalPositionStd'] = 3.0
        return df

    ranks = []
    # If we have a preprocessed DataFrame, rank its columns
    if 'use_ranks_df' in locals():
        for c in use_ranks_df.columns:
            try:
                col = pd.to_numeric(use_ranks_df[c], errors='coerce')
                r = col.rank(method='average', na_option='bottom', ascending=True)
                ranks.append(r)
            except Exception:
                continue
    else:
        for c in features:
            try:
                col = pd.to_numeric(df[c], errors='coerce')
                lower_better = any(s in c.lower() for s in ('time','lap','sec','best','s1','s2','s3','position'))
                if lower_better:
                    r = col.rank(method='average', na_option='bottom', ascending=True)
                else:
                    r = col.rank(method='average', na_option='bottom', ascending=False)
                ranks.append(r)
            except Exception:
                continue

    if not ranks:
        df['PredictedFinalPosition'] = range(1, len(df)+1)
    else:
        ranks_df = pd.concat(ranks, axis=1)
        avg_rank = ranks_df.mean(axis=1)
        df['PredictedFinalPosition'] = avg_rank.rank(method='first', ascending=True)
        # Use the per-driver std across rank contributions as a proxy for prediction uncertainty
        try:
            df['PredictedFinalPositionStd'] = ranks_df.std(axis=1).fillna(0.0)
        except Exception:
            df['PredictedFinalPositionStd'] = 0.0

    # simple MAE estimate: use global median absolute deviation across features as proxy
    try:
        global_mae = float(df.select_dtypes(include=[np.number]).mad().median())
        if np.isnan(global_mae) or global_mae <= 0:
            global_mae = 3.0
    except Exception:
        global_mae = 3.0

    # If std wasn't set above (no ranks), create a reasonable default
    if 'PredictedFinalPositionStd' not in df.columns or df['PredictedFinalPositionStd'].isnull().all():
        df['PredictedFinalPositionStd'] = global_mae

    df['PredictedPositionMAE'] = global_mae
    df['PredictedFinalPosition_Low'] = df['PredictedFinalPosition'] - global_mae
    df['PredictedFinalPosition_High'] = df['PredictedFinalPosition'] + global_mae

    # ensure name columns exist
    # Normalize messy combined id/name fields (some sources concatenate values with tabs)
    if 'resultsDriverId' in df.columns:
        # If values look like a tab-delimited composite, try to extract a trailing numeric id
        def extract_id(val):
            try:
                s = str(val)
                # handle both real tabs and literal "\\t" sequences
                s_clean = s.replace('\\t', '\t')
                m = re.search(r"(\d+)$", s_clean)
                if m:
                    return int(m.group(1))
                # fallback: return original
                return val
            except Exception:
                return val
        df['resultsDriverId'] = df['resultsDriverId'].apply(extract_id)

    # Clean resultsDriverName even if present (handle tab-delimited composites)
    def clean_name(val):
        try:
            s = str(val)
            # normalize literal backslash-t sequences to real tabs then split
            s_norm = s.replace('\\t', '\t')
            if '\t' in s_norm:
                parts = [p.strip() for p in s_norm.split('\t') if p.strip()]
                # prefer the token that looks like 'First Last'
                for p in parts:
                    if ' ' in p:
                        return p
                # else prefer token with alphabetic chars longer than 2
                for p in parts:
                    if re.search(r'[A-Za-z]', p) and len(p) > 2:
                        return p
                return parts[-1]
            return s_norm
        except Exception:
            return str(val)

    if 'resultsDriverName' in df.columns:
        df['resultsDriverName'] = df['resultsDriverName'].apply(clean_name)
    else:
        # try to derive from driver_df or id
        if 'resultsDriverName' in driver_df.columns:
            df['resultsDriverName'] = driver_df['resultsDriverName'].apply(clean_name)
        elif 'resultsDriverId' in df.columns:
            df['resultsDriverName'] = df['resultsDriverId'].astype(str)
        else:
            df['resultsDriverName'] = df.index.astype(str)

    return df


def write_predictions(df, next_race_name, next_race_date):
    if df.empty:
        print('No drivers to predict for')
        return None
    slug = slugify(next_race_name or 'next-race')
    year = pd.to_datetime(next_race_date).year if next_race_date is not None else datetime.now().year
    out_path = os.path.join(DATA_DIR, f'predictions_headless_{slug}_{year}.csv')
    # minimal columns
    cols = ['resultsDriverName', 'resultsDriverId'] if 'resultsDriverId' in df.columns else ['resultsDriverName']
    extras = ['constructorName','PredictedFinalPosition','PredictedFinalPositionStd','PredictedFinalPosition_Low','PredictedFinalPosition_High','PredictedPositionMAE']
    for col in extras:
        if col not in df.columns:
            df[col] = np.nan
    # Ensure numeric dtypes for numeric columns
    num_cols = ['PredictedFinalPosition','PredictedFinalPositionStd','PredictedFinalPosition_Low','PredictedFinalPosition_High','PredictedPositionMAE']
    for nc in num_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors='coerce')

    # Build a cleaned output row-by-row to safely handle composite fields
    def parse_composite(val):
        s = '' if pd.isna(val) else str(val)
        s = s.replace('\\t', '\t')
        # normalize real tabs to a list
        parts = [p.strip() for p in s.split('\t') if p.strip()]
        if not parts:
            return None, ''
        # Try to find a numeric id token at the end
        id_token = None
        name_token = None
        for p in reversed(parts):
            if re.fullmatch(r"\d+", p):
                id_token = int(p)
                break
        # name: prefer token containing a space
        for p in parts:
            if ' ' in p:
                name_token = p
                break
        if name_token is None:
            # fallback to last alpha-looking token
            for p in reversed(parts):
                if re.search(r'[A-Za-z]', p):
                    name_token = p
                    break
        if name_token is None:
            name_token = parts[0]
        return id_token if id_token is not None else parts[-1], name_token

    rows = []
    for _, r in df.iterrows():
        # resultsDriverId field may contain composite data
        raw_id = r.get('resultsDriverId', None)
        raw_name = r.get('resultsDriverName', None)
        parsed_id, parsed_name = None, None
        if raw_name is not None and (('\t' in str(raw_name)) or ('\\t' in str(raw_name))):
            parsed_id, parsed_name = parse_composite(raw_name)
        if parsed_name is None and raw_id is not None and (('\t' in str(raw_id)) or ('\\t' in str(raw_id))):
            parsed_id2, parsed_name2 = parse_composite(raw_id)
            parsed_id = parsed_id or parsed_id2
            parsed_name = parsed_name or parsed_name2
        # final fallbacks
        if parsed_name is None:
            parsed_name = raw_name if raw_name is not None else (raw_id if raw_id is not None else '')
        if parsed_id is None:
            parsed_id = raw_id if raw_id is not None else ''

        row = {
            'resultsDriverId': parsed_id,
            'resultsDriverName': parsed_name,
            'constructorName': r.get('constructorName', '')
        }
        # add numeric prediction fields
        for col in ['PredictedFinalPosition','PredictedFinalPositionStd','PredictedFinalPosition_Low','PredictedFinalPosition_High','PredictedPositionMAE']:
            row[col] = r.get(col, np.nan)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    # If there are duplicate rows per driver (from messy upstream slices), keep the best-ranked row per driver
    try:
        if 'resultsDriverId' in df_out.columns and 'PredictedFinalPosition' in df_out.columns:
            df_out = df_out.sort_values('PredictedFinalPosition').drop_duplicates(subset=['resultsDriverId'], keep='first').reset_index(drop=True)
        elif 'resultsDriverName' in df_out.columns and 'PredictedFinalPosition' in df_out.columns:
            df_out = df_out.sort_values('PredictedFinalPosition').drop_duplicates(subset=['resultsDriverName'], keep='first').reset_index(drop=True)
    except Exception:
        pass

    # If output looks corrupted (lots of missing names or unusually many rows), rebuild a clean driver list from active_drivers.csv
    try:
        valid_names = df_out['resultsDriverName'].notna().sum() if 'resultsDriverName' in df_out.columns else 0
        if df_out.shape[0] > 120 or valid_names < max(5, df_out.shape[0] // 3):
            # rebuild canonical driver list from active_drivers.csv (tab-separated)
            try:
                act = pd.read_csv(active_path, sep='\t')
            except Exception:
                act = pd.read_csv(active_path)
            if 'driverId' in act.columns:
                clean = pd.DataFrame()
                clean['resultsDriverId'] = act['driverId'].astype(str)
                # pick a human-friendly name column
                if 'name' in act.columns:
                    clean['resultsDriverName'] = act['name'].astype(str)
                elif 'fullName' in act.columns:
                    clean['resultsDriverName'] = act['fullName'].astype(str)
                else:
                    clean['resultsDriverName'] = act.iloc[:,0].astype(str)
                clean['PredictedFinalPosition'] = list(range(1, len(clean) + 1))
                clean['PredictedPositionMAE'] = 3.0
                df_out = clean
    except Exception:
        pass
    # Populate constructorName from active_drivers.csv when missing
    try:
        # First, try a canonical mapping file if present
        mapping_path = os.path.join(DATA_DIR, 'driver_to_constructor.csv')
        mapping = {}
        name_map = {}
        if os.path.exists(mapping_path):
            try:
                map_df = pd.read_csv(mapping_path)
                if 'resultsDriverId' in map_df.columns and 'constructorName' in map_df.columns:
                    mapping = dict(zip(map_df['resultsDriverId'].astype(str).tolist(), map_df['constructorName'].astype(str).tolist()))
                if 'resultsDriverName' in map_df.columns and 'constructorName' in map_df.columns:
                    name_map = dict(zip(map_df['resultsDriverName'].astype(str).tolist(), map_df['constructorName'].astype(str).tolist()))
            except Exception:
                mapping = {}
                name_map = {}

        # If mapping provided a value, use it; otherwise fall back to active_drivers.csv
        if mapping or name_map:
            def fill_from_mapping(row):
                cur = row.get('constructorName', '')
                if cur and str(cur).strip():
                    return cur
                rid = row.get('resultsDriverId', '')
                rname = row.get('resultsDriverName', '')
                try:
                    if pd.notna(rid):
                        key = str(int(rid)) if (isinstance(rid, (int, float)) and not pd.isna(rid)) else str(rid)
                        if key in mapping and mapping[key]:
                            return mapping[key]
                except Exception:
                    pass
                try:
                    if rname and str(rname) in name_map and name_map[str(rname)]:
                        return name_map[str(rname)]
                except Exception:
                    pass
                return ''

            df_out['constructorName'] = df_out.apply(fill_from_mapping, axis=1)
        else:
            active_path = os.path.join(DATA_DIR, 'active_drivers.csv')
            if os.path.exists(active_path):
                # active_drivers.csv is tab-separated in this repo
                try:
                    active_df = pd.read_csv(active_path, sep='\t')
                except Exception:
                    active_df = pd.read_csv(active_path)
                # try possible column names for id and constructor
                id_cols = [c for c in ['resultsDriverId', 'id', 'driverId'] if c in active_df.columns]
                ctor_cols = [c for c in ['constructorName', 'constructor', 'team', 'constructorId'] if c in active_df.columns]
                if id_cols and ctor_cols:
                    idc = id_cols[0]
                    ct = ctor_cols[0]
                    # build mapping
                    try:
                        mapping = dict(zip(active_df[idc].astype(str).tolist(), active_df[ct].astype(str).tolist()))
                    except Exception:
                        mapping = {}
                    # also build name-based mapping if name column exists
                    name_map = {}
                    name_cols = [c for c in ['resultsDriverName','driverName','name'] if c in active_df.columns]
                    if name_cols:
                        nc = name_cols[0]
                        try:
                            name_map = dict(zip(active_df[nc].astype(str).tolist(), active_df[ct].astype(str).tolist()))
                        except Exception:
                            name_map = {}
                    # apply mapping
                    def fill_ctor(row):
                        cur = row.get('constructorName', '')
                        if cur and str(cur).strip():
                            return cur
                        rid = row.get('resultsDriverId', '')
                        rname = row.get('resultsDriverName', '')
                        # try id lookup (stringify)
                        try:
                            if pd.notna(rid):
                                key = str(int(rid)) if (isinstance(rid, (int, float)) and not pd.isna(rid)) else str(rid)
                                if key in mapping and mapping[key]:
                                    return mapping[key]
                        except Exception:
                            pass
                        # try name lookup
                        try:
                            if rname and str(rname) in name_map:
                                return name_map[str(rname)]
                        except Exception:
                            pass
                        return ''

                    df_out['constructorName'] = df_out.apply(fill_ctor, axis=1)
    except Exception:
        pass
    # If constructorName still missing, try active_drivers_interim.csv which often contains constructorName
    try:
        if df_out['constructorName'].isna().all() or (df_out['constructorName'].astype(str).str.strip() == '').all():
            interim_path = os.path.join(DATA_DIR, 'active_drivers_interim.csv')
            if os.path.exists(interim_path):
                try:
                    ref = pd.read_csv(interim_path, sep='\t')
                except Exception:
                    ref = pd.read_csv(interim_path)
                if 'resultsDriverId' in ref.columns and 'constructorName' in ref.columns:
                    mapping2 = dict(zip(ref['resultsDriverId'].astype(str).tolist(), ref['constructorName'].astype(str).tolist()))
                    # name-based mapping as well
                    name_map2 = {}
                    if 'resultsDriverName' in ref.columns:
                        try:
                            name_map2 = dict(zip(ref['resultsDriverName'].astype(str).tolist(), ref['constructorName'].astype(str).tolist()))
                        except Exception:
                            name_map2 = {}
                    def fill_from_interim(row):
                        cur = row.get('constructorName', '')
                        if cur and str(cur).strip():
                            return cur
                        rid = row.get('resultsDriverId', '')
                        rname = row.get('resultsDriverName', '')
                        try:
                            key = str(int(rid)) if (isinstance(rid, (int, float)) and not pd.isna(rid)) else str(rid)
                        except Exception:
                            key = str(rid)
                        val = mapping2.get(key, '')
                        if val:
                            return val
                        # try name-based lookup (direct match)
                        if rname and str(rname) in name_map2:
                            return name_map2.get(str(rname), '')
                        # try normalized name lookup (lower)
                        try:
                            lower_map = {k.lower(): v for k, v in name_map2.items()}
                            if rname and str(rname).lower() in lower_map:
                                return lower_map.get(str(rname).lower(), '')
                        except Exception:
                            pass
                        # try normalized name lookup (strip accents and punctuation)
                        try:
                            import unicodedata as _ud
                            def _norm(s):
                                s = str(s).lower()
                                s = _ud.normalize('NFKD', s)
                                s = ''.join(ch for ch in s if not _ud.combining(ch))
                                s = re.sub(r"[^a-z0-9]+", ' ', s).strip()
                                return s
                            norm_map = { _norm(k): v for k, v in name_map2.items() }
                            if rname and _norm(rname) in norm_map:
                                return norm_map.get(_norm(rname), '')
                        except Exception:
                            pass
                        return ''
                    df_out['constructorName'] = df_out.apply(fill_from_interim, axis=1)
    except Exception:
        pass

    # Final fallback: use f1ForAnalysis.csv historic rows to map driver id/name -> most common constructor
    try:
        if df_out['constructorName'].isna().all() or (df_out['constructorName'].astype(str).str.strip() == '').all():
            analysis_csv = os.path.join(DATA_DIR, 'f1ForAnalysis.csv')
            if os.path.exists(analysis_csv):
                # read id, name and constructor when available
                usecols = []
                # attempt to read the common columns; pd.read_csv will error if usecols contains missing ones,
                # so wrap in try/except and progressively fall back
                try:
                    hist = pd.read_csv(analysis_csv, sep='\t', usecols=['resultsDriverId','resultsDriverName','constructorName'])
                except Exception:
                    try:
                        hist = pd.read_csv(analysis_csv, sep='\t', usecols=['resultsDriverName','constructorName'])
                    except Exception:
                        hist = pd.read_csv(analysis_csv, sep='\t')

                # build mappings by id and by name
                try:
                    hist = hist.dropna(subset=['constructorName'])
                except Exception:
                    hist = hist

                mapping_by_id = {}
                mapping_by_name = {}
                try:
                    if 'resultsDriverId' in hist.columns:
                        mapping_by_id = hist.dropna(subset=['resultsDriverId']).groupby('resultsDriverId')['constructorName'].agg(lambda x: x.value_counts().index[0]).to_dict()
                except Exception:
                    mapping_by_id = {}
                try:
                    if 'resultsDriverName' in hist.columns:
                        mapping_by_name = hist.dropna(subset=['resultsDriverName']).groupby('resultsDriverName')['constructorName'].agg(lambda x: x.value_counts().index[0]).to_dict()
                except Exception:
                    mapping_by_name = {}

                def fill_from_history(row):
                    cur = row.get('constructorName', '')
                    if cur and str(cur).strip():
                        return cur
                    rid = row.get('resultsDriverId', '')
                    rname = row.get('resultsDriverName', '')
                    # try id mapping first
                    try:
                        if pd.notna(rid):
                            key = str(int(rid)) if (isinstance(rid, (int, float)) and not pd.isna(rid)) else str(rid)
                            if key in mapping_by_id and mapping_by_id[key]:
                                return mapping_by_id[key]
                    except Exception:
                        pass
                    # try name mapping
                    try:
                        if rname and str(rname) in mapping_by_name:
                            return mapping_by_name.get(str(rname), '')
                        # try lower-case name match
                        ln = str(rname).lower() if rname else ''
                        lower_map = {k.lower(): v for k, v in mapping_by_name.items()}
                        if ln in lower_map:
                            return lower_map.get(ln, '')
                    except Exception:
                        pass
                    return ''

                df_out['constructorName'] = df_out.apply(fill_from_history, axis=1)
    except Exception:
        pass
    # Ensure numeric dtypes
    for nc in ['PredictedFinalPosition','PredictedFinalPositionStd','PredictedFinalPosition_Low','PredictedFinalPosition_High','PredictedPositionMAE']:
        if nc in df_out.columns:
            df_out[nc] = pd.to_numeric(df_out[nc], errors='coerce')

    # Sanitize string columns
    for c in df_out.select_dtypes(include=['object']).columns:
        df_out[c] = df_out[c].astype(str).str.replace('\\t', ' ', regex=False).str.replace('\t', ' ', regex=False).str.replace('\n', ' ', regex=False)

    # Try to extract a clean numeric driver id and a human-friendly name
    import re as _re
    def extract_id_from_text(s):
        if s is None:
            return ''
        s = str(s)
        # split on literal backslash-t or real tab, then find last pure-numeric token
        parts = _re.split(r"\\t|\t|\s+", s)
        parts = [p for p in parts if p]
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        # fallback: any trailing numeric at end of string
        m = _re.search(r"(\d+)$", s)
        if m:
            return int(m.group(1))
        return s

    name_pattern = _re.compile(r"([A-ZÄÖÜÀÈÉ][a-zA-Z'`\-]+)\s+([A-ZÄÖÜÀÈÉ][a-zA-Z'`\-]+)")
    def extract_name_from_text(s):
        if s is None:
            return ''
        s = str(s)
        # Normalize literal and real tabs into separators, then split
        parts = _re.split(r"\\t|\t|\s+", s)
        parts = [p.strip() for p in parts if p and not p.isdigit()]
        # 1) prefer existing multi-word token with capitalized words
        for p in parts:
            if name_pattern.search(p):
                return name_pattern.search(p).group(0)
        # 2) prefer a token that looks like 'first-last' (slug) -> title case
        for p in parts:
            if '-' in p and p.lower() == p:
                return p.replace('-', ' ').title()
        # 3) fallback: take first two alphabetic tokens
        words = [re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'`\-]", '', t) for t in parts if re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t)]
        if len(words) >= 2:
            return f"{words[0]} {words[1]}"
        if words:
            return words[0]
        # last resort: return original cleaned string
        return s.replace('\\t', ' ').replace('\t', ' ')

    df_out['resultsDriverId'] = df_out.apply(lambda r: extract_id_from_text(r.get('resultsDriverId', '')), axis=1)
    df_out['resultsDriverName'] = df_out.apply(lambda r: extract_name_from_text(r.get('resultsDriverName', r.get('resultsDriverId', ''))), axis=1)

    # Write tab-separated file to match repository conventions
    df_out.to_csv(out_path, index=False, sep='\t')
    return out_path


def main():
    next_race_name, next_race_date = compute_next_race_from_calendar()
    print('Next race:', next_race_name, next_race_date)
    driver_slice = build_driver_slice(next_race_name, next_race_date)
    if driver_slice.empty:
        print('Driver slice empty; aborting headless prediction')
        return
    preds = make_headless_predictions(driver_slice)
    out = write_predictions(preds, next_race_name, next_race_date)
    print('Wrote predictions to', out)
    print('\nPreview:')
    # Read back the written TSV to show the cleaned output as preview
    try:
        clean_df = pd.read_csv(out, sep='\t')
        cols = [c for c in ['resultsDriverName', 'constructorName', 'PredictedFinalPosition', 'PredictedPositionMAE'] if c in clean_df.columns]
        if len(cols) > 0:
            print(clean_df[cols].sort_values('PredictedFinalPosition' if 'PredictedFinalPosition' in clean_df.columns else cols[0]).head(20).to_string(index=False))
        else:
            print(clean_df.head(10).to_string(index=False))
    except Exception as _e:
        print('Could not read back cleaned TSV for preview:', str(_e))


if __name__ == '__main__':
    main()
