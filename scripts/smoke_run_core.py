#!/usr/bin/env python3
"""Smoke test: import critical helper functions and run a lightweight end-to-end check.

This script imports functions from `scripts/headless_predict_and_write.py` (which
is import-safe) and runs them against the local `data_files/` to validate key
pipelines (next-race detection, building driver slice, making headless
predictions, and writing the predictions TSV). It also checks the repaired
race-control messages grouped output for basic sanity.
"""
import traceback
import os
import sys
import pandas as pd

# Ensure repo root is on sys.path so `scripts.*` imports work when executed directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.headless_predict_and_write import (
    compute_next_race_from_calendar,
    build_driver_slice,
    make_headless_predictions,
    write_predictions,
)

DATA_DIR = os.path.join(ROOT, 'data_files')


def check_file(path):
    print(f"Checking file: {path}")
    if not os.path.exists(path):
        print("  MISSING")
        return False
    try:
        df = pd.read_csv(path, sep='\t', nrows=5)
        print(f"  OK — columns: {list(df.columns)[:6]} ; rows preview: {len(df)}")
        return True
    except Exception as e:
        print("  ERROR reading file:", e)
        return False


def main():
    try:
        # 1) Basic data-file sanity checks for repaired artifacts
        print('\n== Data file quick checks ==')
        check_file(os.path.join(DATA_DIR, 'all_race_control_messages.csv'))
        check_file(os.path.join(DATA_DIR, 'race_control_messages_grouped_with_dnf.csv'))

        # 2) Compute next race candidate
        print('\n== Next race detection ==')
        next_name, next_date = compute_next_race_from_calendar()
        print('Next race detection ->', next_name, next_date)

        # 3) Build driver slice for next race
        print('\n== Build driver slice ==')
        drivers = build_driver_slice(next_name, next_date)
        print('Driver slice shape:', drivers.shape)
        if drivers.empty:
            print('Driver slice is empty — trying active drivers fallback inside builder')

        # 4) Make headless predictions
        print('\n== Make headless predictions ==')
        preds = make_headless_predictions(drivers)
        if preds is None:
            print('Predictions function returned None')
        else:
            print('Predictions shape:', getattr(preds, 'shape', 'n/a'))
            # show a small preview if available
            try:
                cols = [c for c in ['resultsDriverName','PredictedFinalPosition','PredictedPositionMAE'] if c in preds.columns]
                print(preds[cols].head(10).to_string(index=False))
            except Exception:
                print('Could not pretty-print predictions')

        # 5) Write predictions to TSV (if possible)
        print('\n== Write predictions (headless) ==')
        out = write_predictions(preds, next_name or 'next-race', next_date)
        print('Wrote predictions to:', out)

        # 6) Read back the predictions file (sanity)
        if out and os.path.exists(out):
            try:
                back = pd.read_csv(out, sep='\t')
                print('Read back predictions — rows:', back.shape[0], 'cols:', back.shape[1])
                print(back.head(5).to_string(index=False))
            except Exception as e:
                print('Could not read back predictions file:', e)

        print('\nSMOKE TEST: Completed without uncaught exceptions.')

    except Exception:
        print('SMOKE TEST: Unhandled exception:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
