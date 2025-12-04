#!/usr/bin/env python3
"""Normalize sprint weekend artifacts.

This tiny utility marks sprint weekends in the practice/qualifying datasets and
writes out a normalized copy. It uses `data_files/all_sprint_races.csv` to
detect sprint weekends. If that file is missing, the script will only emit a
warning and copy the input to the output.

Usage:
  python scripts/handle_sprint_weekends.py --input data_files/all_practice_results.csv

Output:
  Writes `<input>.normalized.csv` by default (tab/CSV autodetect retained).
"""
from __future__ import annotations

import argparse
import pandas as pd
import os


def read_maybe_tab(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as fh:
        sample = fh.read(2048)
    sep = '\t' if '\t' in sample else ','
    return pd.read_csv(path, sep=sep, low_memory=False)


def normalize_sprint_weekends(input_path: str, sprints_path: str = 'data_files/all_sprint_races.csv', output_path: str | None = None) -> pd.DataFrame:
    out = output_path or input_path.replace('.csv', '.normalized.csv')
    df = read_maybe_tab(input_path)

    if os.path.exists(sprints_path):
        sprints = read_maybe_tab(sprints_path)
        sprint_race_ids = set()
        for col in ('raceId', 'RaceId', 'race_id'):
            if col in sprints.columns:
                sprint_race_ids.update(sprints[col].dropna().astype(int).tolist())
                break
        # detect which race id column exists in the input and use it safely
        race_col = None
        for col in ('raceId', 'RaceId', 'race_id'):
            if col in df.columns:
                race_col = col
                break

        if race_col is not None:
            df['is_sprint_weekend'] = df[race_col].apply(lambda x: int(x) in sprint_race_ids if pd.notna(x) else False)
        else:
            # no race id column present; conservative default = False
            df['is_sprint_weekend'] = False

        # add a conservative qualifying type marker
        df['qualifying_type'] = df['is_sprint_weekend'].apply(lambda v: 'sprint' if v else 'standard')
        print(f"Marked {df['is_sprint_weekend'].sum()} rows as sprint weekends")
    else:
        print(f"Warning: sprint file {sprints_path} not found — copying input to output")

    df.to_csv(out, index=False)
    print(f"Wrote normalized file to {out}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Normalize sprint weekend data')
    parser.add_argument('--input', default='data_files/all_practice_results.csv')
    parser.add_argument('--sprints', default='data_files/all_sprint_races.csv')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    normalize_sprint_weekends(args.input, args.sprints, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


if __name__ == '__main__':
    raise SystemExit(main())
