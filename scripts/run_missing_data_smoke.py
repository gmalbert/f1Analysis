#!/usr/bin/env python3
"""Run smoke checks for the missing-data helper scripts.

This script calls the three helpers and reports pass/fail for each. It exits
with non-zero when any helper returns a non-zero exit code.
"""
from __future__ import annotations

import subprocess
import sys
from os import path


def run_script(script_path, args=None):
    cmd = [sys.executable, script_path] + (args or [])
    print('> ' + ' '.join(cmd))
    res = subprocess.run(cmd)
    return res.returncode


def main():
    base = path.join('scripts')
    scripts = [
        (path.join(base, 'impute_missing_practice.py'), ['--input', path.join('data_files', 'all_practice_results.csv')]),
        (path.join(base, 'handle_sprint_weekends.py'), ['--input', path.join('data_files', 'all_practice_results.csv')]),
        (path.join(base, 'fill_weather_gaps.py'), ['--hourly', path.join('data_files', 'f1WeatherData_AllData.csv'), '--grouped', path.join('data_files', 'f1WeatherData_Grouped.csv')]),
    ]

    failures = []
    for script, args in scripts:
        rc = run_script(script, args)
        if rc != 0:
            failures.append((script, rc))

    if failures:
        print('Failures:')
        for s, rc in failures:
            print(f'  {s} -> exit {rc}')
        return 2
    print('All missing-data helper scripts passed (or exited 0).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
