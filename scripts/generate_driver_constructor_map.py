#!/usr/bin/env python3
"""Generate a canonical mapping of driver -> most common constructor.

Produces `data_files/driver_to_constructor.csv` with columns:
- resultsDriverId, resultsDriverName, constructorName, count

This script uses `data_files/f1ForAnalysis.csv`, `data_files/active_drivers.csv`,
and `data_files/active_drivers_interim.csv` (if present) to build a robust mapping.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data_files')


def canonicalize(s):
    if pd.isna(s):
        return ''
    return str(s).strip()


def build_from_analysis(analysis_path):
    if not os.path.exists(analysis_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(analysis_path, sep='\t', usecols=['resultsDriverId','resultsDriverName','constructorName'])
    except Exception:
        try:
            df = pd.read_csv(analysis_path, sep='\t')
        except Exception:
            return pd.DataFrame()
    # keep only rows with constructor
    if 'constructorName' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['constructorName'])
    df['resultsDriverId'] = df.get('resultsDriverId', df.get('driverId', pd.Series(['']*len(df)))).astype(str)
    df['resultsDriverName'] = df.get('resultsDriverName', df.get('resultsDriverName', df.get('driverName', df.get('name', pd.Series(['']*len(df)))))).astype(str)
    df['constructorName'] = df['constructorName'].astype(str)
    df['constructorName'] = df['constructorName'].str.replace('\t',' ', regex=False).str.strip()
    gp = df.groupby(['resultsDriverId','resultsDriverName','constructorName']).size().reset_index(name='count')
    # pick most common constructor per driver id
    gp_sorted = gp.sort_values(['resultsDriverId','count'], ascending=[True,False])
    top = gp_sorted.groupby('resultsDriverId').first().reset_index()
    top = top[['resultsDriverId','resultsDriverName','constructorName','count']]
    top['resultsDriverId'] = top['resultsDriverId'].astype(str)
    return top


def build_from_active(active_path):
    if not os.path.exists(active_path):
        return pd.DataFrame()
    try:
        ad = pd.read_csv(active_path, sep='\t')
    except Exception:
        try:
            ad = pd.read_csv(active_path)
        except Exception:
            return pd.DataFrame()
    # attempt to find id, name and constructor columns
    id_col = None
    name_col = None
    ctor_col = None
    for c in ['resultsDriverId','driverId','id']:
        if c in ad.columns:
            id_col = c
            break
    for c in ['resultsDriverName','driverName','name','fullName']:
        if c in ad.columns:
            name_col = c
            break
    for c in ['constructorName','constructor','team']:
        if c in ad.columns:
            ctor_col = c
            break
    if ctor_col is None or id_col is None:
        return pd.DataFrame()
    df = ad[[id_col, name_col, ctor_col]].copy()
    df.columns = ['resultsDriverId','resultsDriverName','constructorName']
    df['resultsDriverId'] = df['resultsDriverId'].astype(str)
    df['constructorName'] = df['constructorName'].astype(str)
    df = df.groupby(['resultsDriverId','resultsDriverName','constructorName']).size().reset_index(name='count')
    df_sorted = df.sort_values(['resultsDriverId','count'], ascending=[True,False]).groupby('resultsDriverId').first().reset_index()
    return df_sorted[['resultsDriverId','resultsDriverName','constructorName','count']]


def main():
    analysis_path = os.path.join(DATA_DIR, 'f1ForAnalysis.csv')
    active_path = os.path.join(DATA_DIR, 'active_drivers.csv')
    interim_path = os.path.join(DATA_DIR, 'active_drivers_interim.csv')
    parts = []
    a = build_from_analysis(analysis_path)
    if not a.empty:
        parts.append(a)
    b = build_from_active(active_path)
    if not b.empty:
        parts.append(b)
    # interim can provide additional mappings
    if os.path.exists(interim_path):
        try:
            c = build_from_active(interim_path)
            if not c.empty:
                parts.append(c)
        except Exception:
            pass

    if not parts:
        print('No source data found to build driver->constructor mapping')
        return

    merged = pd.concat(parts, ignore_index=True)
    # for duplicate driver ids prefer highest count (already per-part), but aggregated across parts we take most frequent constructor
    agg = merged.groupby(['resultsDriverId','constructorName'])['count'].sum().reset_index()
    agg_sorted = agg.sort_values(['resultsDriverId','count'], ascending=[True,False])
    top = agg_sorted.groupby('resultsDriverId').first().reset_index()

    # attach a representative name where possible
    names = merged.groupby('resultsDriverId')['resultsDriverName'].agg(lambda x: x.dropna().astype(str).mode().iloc[0] if len(x.dropna())>0 else '').to_dict()
    top['resultsDriverName'] = top['resultsDriverId'].map(names).fillna('')

    out_path = os.path.join(DATA_DIR, 'driver_to_constructor.csv')
    top = top[['resultsDriverId','resultsDriverName','constructorName','count']]
    top.to_csv(out_path, index=False)
    print('Wrote mapping to', out_path)


if __name__ == '__main__':
    main()
