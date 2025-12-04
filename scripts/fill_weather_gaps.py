#!/usr/bin/env python3
"""Fill gaps in weather data using interpolation and track-level fallbacks.

This script operates on `data_files/f1WeatherData_AllData.csv` (hourly) and
`data_files/f1WeatherData_Grouped.csv` (daily/grouped). It will:
 - interpolate numeric hourly fields per grandPrixId
 - for any remaining NaNs, fill from grouped track averages where available

Usage:
  python scripts/fill_weather_gaps.py --hourly data_files/f1WeatherData_AllData.csv

Output files are written with `.imputed.csv` suffix by default.
"""
from __future__ import annotations

import argparse
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def read_maybe_tab(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as fh:
        sample = fh.read(2048)
    sep = '\t' if '\t' in sample else ','
    return pd.read_csv(path, sep=sep, low_memory=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fill weather data gaps')
    parser.add_argument('--hourly', default='data_files/f1WeatherData_AllData.csv')
    parser.add_argument('--grouped', default='data_files/f1WeatherData_Grouped.csv')
    parser.add_argument('--output-hourly', default=None)
    parser.add_argument('--output-grouped', default=None)
    args = parser.parse_args(argv)

    out_hourly = args.output_hourly or args.hourly.replace('.csv', '.imputed.csv')
    out_grouped = args.output_grouped or args.grouped.replace('.csv', '.imputed.csv')

    if not os.path.exists(args.hourly):
        print(f"Hourly file {args.hourly} not found — nothing to do")
        return 1

    df_hourly = read_maybe_tab(args.hourly)
    df_grouped = read_maybe_tab(args.grouped) if os.path.exists(args.grouped) else None

    # pick numeric cols to interpolate
    num_cols = df_hourly.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("No numeric columns found in hourly weather file")
        df_hourly.to_csv(out_hourly, index=False)
        return 0

    # group by grandPrixId or similar
    gp_col = None
    for cand in ('grandPrixId', 'gpid', 'raceId', 'grand_prix_id'):
        if cand in df_hourly.columns:
            gp_col = cand
            break

    if gp_col is None:
        # fallback: interpolate globally
        df_hourly[num_cols] = df_hourly[num_cols].interpolate(limit_direction='both')
    else:
        def interp_grp(g):
            return g.interpolate(limit_direction='both')
        # Only operate on numeric columns to avoid applying to grouping columns
        df_hourly[num_cols] = df_hourly.groupby(gp_col, group_keys=False)[num_cols].apply(interp_grp)

    # Attempt Open-Meteo backfill for remaining NaNs if coordinates and datetime exist
    api_url = 'https://archive-api.open-meteo.com/v1/archive'
    if gp_col and 'datetime' in df_hourly.columns and 'latitude' in df_hourly.columns and 'longitude' in df_hourly.columns:
        # normalize datetime column to naive timestamps for matching
        df_hourly['__dt'] = pd.to_datetime(df_hourly['datetime'], infer_datetime_format=True).dt.floor('min')
        # for each group with NaNs, fetch by date ranges where data missing
        grouped = df_hourly.groupby(gp_col)
        for gp, gdf in grouped:
            if not gdf[num_cols].isna().any().any():
                continue
            lat = gdf['latitude'].dropna().unique()
            lon = gdf['longitude'].dropna().unique()
            if len(lat) == 0 or len(lon) == 0:
                continue
            lat = float(lat[0]); lon = float(lon[0])
            # find dates with missing values
            gdf = gdf.copy()
            gdf['date_only'] = gdf['__dt'].dt.date
            missing_dates = sorted(set(gdf.loc[gdf[num_cols].isna().any(axis=1), 'date_only'].tolist()))
            for miss_date in missing_dates:
                start = miss_date.isoformat()
                end = (miss_date + timedelta(days=0)).isoformat()
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': start,
                    'end_date': end,
                    'hourly': 'temperature_2m,precipitation,relativehumidity_2m,windspeed_10m',
                    'timezone': 'UTC'
                }
                try:
                    r = requests.get(api_url, params=params, timeout=15)
                    if r.status_code != 200:
                        time.sleep(1)
                        continue
                    payload = r.json()
                    hourly = payload.get('hourly', {})
                    times = hourly.get('time', [])
                    if not times:
                        continue
                    df_api = pd.DataFrame({
                        'datetime': pd.to_datetime(times),
                        'temperature_2m': hourly.get('temperature_2m'),
                        'precipitation': hourly.get('precipitation'),
                        'relativehumidity_2m': hourly.get('relativehumidity_2m'),
                        'windspeed_10m': hourly.get('windspeed_10m')
                    })
                    df_api['__dt'] = df_api['datetime'].dt.floor('min')
                    # mapping to local column names
                    mapping = {
                        'average_temp': 'temperature_2m',
                        'temperature_2m': 'temperature_2m',
                        'total_precipitation': 'precipitation',
                        'precipitation': 'precipitation',
                        'average_humidity': 'relativehumidity_2m',
                        'relativehumidity_2m': 'relativehumidity_2m',
                        'average_wind_speed': 'windspeed_10m',
                        'windspeed_10m': 'windspeed_10m'
                    }
                    # fill missing rows in df_hourly for this gp and date
                    for api_col in ['temperature_2m', 'precipitation', 'relativehumidity_2m', 'windspeed_10m']:
                        # find matching target cols in df_hourly
                        target_cols = [c for c, v in mapping.items() if v == api_col and c in df_hourly.columns]
                        if not target_cols:
                            continue
                        # merge on datetime
                        mask = (df_hourly[gp_col] == gp) & (df_hourly['__dt'].dt.date == miss_date)
                        if not mask.any():
                            continue
                        left = df_hourly.loc[mask]
                        merged = left.merge(df_api[['__dt', api_col]], left_on='__dt', right_on='__dt', how='left', suffixes=('', '_api'))
                        for tgt in target_cols:
                            df_hourly.loc[mask, tgt] = df_hourly.loc[mask, tgt].fillna(merged[api_col].values)
                except Exception:
                    time.sleep(1)
                    continue
        # drop helper column
        df_hourly.drop(columns=['__dt'], inplace=True, errors='ignore')

    # remaining NaNs: fill from grouped file (track averages) if available
    if df_grouped is not None:
        gp_map = {}
        # attempt to map by grandPrixId or short_date
        for col in ('grandPrixId', 'gpid', 'raceId'):
            if col in df_grouped.columns:
                gp_map['id_col'] = col
                break
        # compute grouped medians for numeric cols
        grouped_median = df_grouped.select_dtypes(include=[np.number]).median()
        for col in num_cols:
            if df_hourly[col].isna().any():
                fill_val = grouped_median.get(col, np.nan)
                if pd.notna(fill_val):
                    df_hourly[col] = df_hourly[col].fillna(fill_val)

    # still remaining NaNs -> global median
    df_hourly = df_hourly.fillna(df_hourly.select_dtypes(include=[np.number]).median())

    df_hourly.to_csv(out_hourly, index=False)
    print(f"Wrote imputed hourly weather to {out_hourly}")

    # update grouped file by recomputing simple daily aggregates from imputed hourly
    try:
        if gp_col and 'datetime' in df_hourly.columns:
            agg = df_hourly.groupby(gp_col).agg({c: 'mean' for c in num_cols}).reset_index()
            # preserve non-numeric columns when possible
            if df_grouped is not None:
                keep_cols = [c for c in df_grouped.columns if c not in agg.columns]
                merged = df_grouped[keep_cols].merge(agg, left_on=gp_map.get('id_col', gp_col), right_on=gp_col, how='left')
            else:
                merged = agg
            merged.to_csv(out_grouped, index=False)
            print(f"Wrote imputed grouped weather to {out_grouped}")
    except Exception as e:
        print(f"Could not recompute grouped weather: {e}")

    return 0


def fill_weather_gaps(hourly_path: str = 'data_files/f1WeatherData_AllData.csv', grouped_path: str = 'data_files/f1WeatherData_Grouped.csv', output_hourly: str | None = None, output_grouped: str | None = None, backfill: bool = True) -> pd.DataFrame:
    """Wrapper to call the main logic programmatically.
    Returns the imputed hourly DataFrame. Files are written to disk.
    """
    class Args:
        pass
    a = Args()
    a.hourly = hourly_path
    a.grouped = grouped_path
    a.output_hourly = output_hourly
    a.output_grouped = output_grouped
    # reuse main's behavior
    return_data = main(["--hourly", a.hourly, "--grouped", a.grouped, "--output-hourly", a.output_hourly or '', "--output-grouped", a.output_grouped or ''])
    # read the produced file to return dataframe
    outf = a.output_hourly or (a.hourly.replace('.csv', '.imputed.csv'))
    try:
        df = read_maybe_tab(outf)
    except Exception:
        df = pd.DataFrame()
    return df


if __name__ == '__main__':
    raise SystemExit(main())


if __name__ == '__main__':
    raise SystemExit(main())
