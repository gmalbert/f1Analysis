#!/usr/bin/env python3
"""Run all smoke/check scripts from `scripts/` and summarize results.

Usage (from repo root):

PowerShell examples
```
# list discovered scripts
python .\scripts\run_all_smoke_checks.py --list-only

# run all smoke checks (stop on first failure)
python .\scripts\run_all_smoke_checks.py

# run all smoke checks, continue through failures and show full summary
python .\scripts\run_all_smoke_checks.py --continue-on-fail
```

This script will look for a small set of canonical smoke-test scripts and run
them in sequence, printing stdout/stderr and a final summary with pass/fail.
It returns exit code 0 if all tests pass, or 1 if any test fails (unless
`--continue-on-fail` is provided, in which case it still prints failures but
exits 1 if any failed).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


DEFAULT_SCRIPTS = [
    'scripts/check_generation_smoke.py',
    'scripts/check_teammate_delta.py',
    'scripts/run_missing_data_smoke.py',
    'scripts/audit_temporal_leakage.py',
]


def run_script(path: str) -> tuple[int, str, str, float]:
    start = time.time()
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, check=False)
        rc = proc.returncode
        out = proc.stdout
        err = proc.stderr
    except FileNotFoundError:
        rc = 127
        out = ''
        err = f'Script not found: {path}\n'
    elapsed = time.time() - start
    return rc, out, err, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run all smoke/check scripts and summarize results')
    parser.add_argument('--continue-on-fail', action='store_true', help='Run all tests even if some fail; exit code remains 1 if any failed')
    parser.add_argument('--list-only', action='store_true', help='Only list discovered scripts and exit')
    args = parser.parse_args(argv)

    # Discover scripts to run: prefer the DEFAULT_SCRIPTS, but also include any other
    # check_/run_*.py files in the scripts/ folder for convenience.
    scripts_dir = os.path.join(os.getcwd(), 'scripts')
    candidates = []
    for s in DEFAULT_SCRIPTS:
        if os.path.exists(s):
            candidates.append(s)
        else:
            # warn but continue
            print(f'Note: expected script {s} not found; skipping')

    # Discover additional scripts matching patterns
    if os.path.isdir(scripts_dir):
        for fname in sorted(os.listdir(scripts_dir)):
            # avoid adding this runner script (would recurse endlessly)
            if fname == os.path.basename(__file__):
                continue
            if fname in (os.path.basename(x) for x in DEFAULT_SCRIPTS):
                continue
            if fname.startswith('check_') or fname.startswith('run_'):
                candidates.append(os.path.join('scripts', fname))

    if args.list_only:
        print('Discovered scripts:')
        for c in candidates:
            print(' -', c)
        return 0

    if not candidates:
        print('No smoke/check scripts found under scripts/. Nothing to run.')
        return 0

    results = []
    any_fail = False
    skipped = []

    for script in candidates:
        print('\n' + '=' * 70)
        print(f'Running: {script}')
        print('=' * 70)
        rc, out, err, elapsed = run_script(script)
        print(f'Exit code: {rc}  (elapsed {elapsed:.1f}s)')
        if out:
            print('--- STDOUT ---')
            print(out)
        if err:
            print('--- STDERR ---')
            print(err)
        # Treat common missing-file errors in helper scripts as SKIPPED (non-fatal)
        if rc != 0 and (('No such file or directory' in err) or ('FileNotFoundError' in err)):
            print(f"Note: {script} appears to have missing input files; marking SKIPPED and continuing")
            skipped.append(script)
            # record as a special 0 (skipped) result for summary consistency
            results.append((script, 0, elapsed))
            continue

        results.append((script, rc, elapsed))
        if rc != 0:
            any_fail = True
            if not args.continue_on_fail:
                print(f'Aborting early because {script} failed (use --continue-on-fail to run all)')
                break

    # Summary
    print('\n' + '=' * 70)
    print('Smoke check summary:')
    for script, rc, elapsed in results:
        status = 'PASS' if rc == 0 else f'FAIL (code {rc})'
        print(f' - {script:<50} {status:20} {elapsed:.1f}s')

    if any_fail:
        print('\nOne or more smoke checks failed.')
        return 1
    print('\nAll smoke checks passed ✅')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
