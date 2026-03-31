#!/usr/bin/env python3
"""
Full F1 Analysis Pipeline Runner
=================================
Run this script after adding/updating data files (JSON, CSVs) to regenerate
all analysis artifacts in the correct order.

Usage:
    python run_full_pipeline.py                    # run everything
    python run_full_pipeline.py --skip-generation  # skip data generation (use existing CSVs)
    python run_full_pipeline.py --skip-training    # skip model training
    python run_full_pipeline.py --skip-features    # skip feature selection suite
    python run_full_pipeline.py --skip-smoke       # skip smoke/data-quality checks
    python run_full_pipeline.py --quick            # generation + training only (no feature selection)

Stages (in order):
  1. Data Generation       f1-generate-analysis.py
  2. Smoke Checks          scripts/run_all_smoke_checks.py
  3. Model Training        train_xgboost / train_lightgbm / train_catboost / train_ensemble
  4. Race Predictions      scripts/precompute/generate_race_predictions.py
  5. Feature Selection     rfe / boruta / shap / permutation
  6. Position Analysis     scripts/position_group_analysis.py + precompute variant
  7. Export Artifacts      scripts/export_feature_selection.py
"""

import argparse
import io
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*MemoryCacheStorageManager*")

logging.getLogger("streamlit").setLevel(logging.WARNING)

# Force UTF-8 stdout on Windows to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Colour helpers (no dependencies) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def header(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def info(msg):
    print(f"{CYAN}[{_ts()}] {msg}{RESET}")

def ok(msg):
    print(f"{GREEN}[{_ts()}] [OK]   {msg}{RESET}")

def warn(msg):
    print(f"{YELLOW}[{_ts()}] [WARN] {msg}{RESET}")

def fail(msg):
    print(f"{RED}[{_ts()}] [FAIL] {msg}{RESET}")

# ── Step runner ────────────────────────────────────────────────────────────────

def run_step(label: str, cmd: list[str], *, critical: bool = True, env_extra: dict | None = None) -> bool:
    """Run a subprocess step and return True on success."""
    import os
    info(f"Starting: {label}")
    t0 = time.time()

    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_LOG_LEVEL", "error")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    if env_extra:
        env.update(env_extra)

    try:
        result = subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        warn(f"{label} interrupted by user  ({elapsed:.0f}s)")
        raise
    elapsed = time.time() - t0

    if result.returncode == 0:
        ok(f"{label}  ({elapsed:.0f}s)")
        return True
    else:
        msg = f"{label} failed with exit code {result.returncode}  ({elapsed:.0f}s)"
        if critical:
            fail(msg)
        else:
            warn(msg + "  [non-critical, continuing]")
        return False


# ── Stages ────────────────────────────────────────────────────────────────────

def stage_data_generation(py: str) -> bool:
    header("Stage 1 · Data Generation")
    ok = run_step(
        "f1-generate-analysis.py",
        [py, "f1-generate-analysis.py"],
        critical=True,
    )
    if ok:
        # Patch best_qual_time & teammate_qual_delta into qualifying CSV
        # (non-critical: failure here does not block subsequent stages)
        run_step(
            "compute_teammate_delta.py",
            [py, "scripts/compute_teammate_delta.py"],
            critical=False,
        )
    return ok


def stage_smoke_checks(py: str) -> bool:
    header("Stage 2 · Smoke / Data-Quality Checks")
    return run_step(
        "run_all_smoke_checks.py",
        [py, "scripts/run_all_smoke_checks.py", "--continue-on-fail"],
        critical=False,   # smoke failures are warnings, not blockers
    )


def stage_model_training(py: str) -> dict:
    header("Stage 3 · Model Training")
    results = {}

    # XGBoost, LightGBM, CatBoost can train independently; run them sequentially
    # (parallel training locally would thrash RAM/CPU and give unclear output)
    for name, script in [
        ("XGBoost",   "scripts/precompute/train_xgboost.py"),
        ("LightGBM",  "scripts/precompute/train_lightgbm.py"),
        ("CatBoost",  "scripts/precompute/train_catboost.py"),
    ]:
        results[name] = run_step(
            f"Train {name}",
            [py, script],
            critical=False,   # individual model failures don't block other models
        )

    # Ensemble only makes sense after the base models
    base_ok = any(results.values())
    if base_ok:
        results["Ensemble"] = run_step(
            "Train Ensemble",
            [py, "scripts/precompute/train_ensemble.py"],
            critical=False,
        )
    else:
        warn("All base models failed — skipping Ensemble training")
        results["Ensemble"] = False

    return results


def stage_predictions(py: str) -> bool:
    header("Stage 4 · Race Predictions")
    return run_step(
        "generate_race_predictions.py",
        [py, "scripts/precompute/generate_race_predictions.py",
         "--output", "data_files/precomputed/race_predictions.json"],
        critical=False,
    )


def stage_feature_selection(py: str) -> dict:
    header("Stage 5 · Feature Selection Suite")
    results = {}
    steps = [
        ("RFE",                 ["scripts/precompute/rfe_features.py",
                                  "--output", "data_files/precomputed/rfe_results.json"]),
        ("Boruta",              ["scripts/precompute/boruta_features.py",
                                  "--max-iter", "200",
                                  "--output", "data_files/precomputed/boruta_results.json"]),
        ("SHAP Analysis",       ["scripts/precompute/shap_analysis.py",
                                  "--output", "data_files/precomputed/shap_results.json"]),
        ("Permutation Importance", ["scripts/precompute/permutation_importance.py",
                                    "--output", "data_files/precomputed/permutation_results.json"]),
    ]
    for label, extra_args in steps:
        results[label] = run_step(label, [py] + extra_args, critical=False)
    return results


def stage_position_analysis(py: str) -> dict:
    header("Stage 6 · Position Group Analysis")
    results = {}
    results["position_group_analysis"] = run_step(
        "position_group_analysis.py",
        [py, "scripts/position_group_analysis.py"],
        critical=False,
    )
    results["position_group_analysis_precompute"] = run_step(
        "position_group_analysis_precompute.py",
        [py, "scripts/precompute/position_group_analysis_precompute.py"],
        critical=False,
    )
    return results


def stage_export_artifacts(py: str) -> bool:
    header("Stage 7 · Export Feature-Selection Artifacts")
    # Run without flags: the script generates both CSV and HTML by default
    return run_step(
        "export_feature_selection.py",
        [py, "scripts/export_feature_selection.py"],
        critical=False,
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results: dict, total_elapsed: float):
    header("Pipeline Summary")
    passed = failed = skipped = 0
    for stage, result in all_results.items():
        if result is None:
            print(f"  {YELLOW}SKIPPED {RESET}  {stage}")
            skipped += 1
        elif result is True or (isinstance(result, dict) and all(result.values())):
            print(f"  {GREEN}PASSED  {RESET}  {stage}")
            passed += 1
        elif isinstance(result, dict):
            n_ok  = sum(1 for v in result.values() if v)
            n_all = len(result)
            colour = GREEN if n_ok == n_all else (YELLOW if n_ok > 0 else RED)
            print(f"  {colour}PARTIAL {RESET}  {stage}  ({n_ok}/{n_all} steps ok)")
            if n_ok == n_all:
                passed += 1
            else:
                failed += 1
        else:
            print(f"  {RED}FAILED  {RESET}  {stage}")
            failed += 1

    total_mins = int(total_elapsed // 60)
    total_secs = int(total_elapsed % 60)
    print(f"\n  Total time: {total_mins}m {total_secs}s")
    print(f"  {GREEN}{passed} passed{RESET}  |  {YELLOW}{skipped} skipped{RESET}  |  {RED}{failed} failed{RESET}\n")
    return failed == 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full F1 analysis pipeline after data file updates."
    )
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip Stage 1 (use existing f1ForAnalysis.csv)")
    parser.add_argument("--skip-smoke",      action="store_true",
                        help="Skip Stage 2 smoke checks")
    parser.add_argument("--skip-training",   action="store_true",
                        help="Skip Stage 3 model training")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip Stage 4 race predictions")
    parser.add_argument("--skip-features",   action="store_true",
                        help="Skip Stage 5 feature selection suite")
    parser.add_argument("--skip-position",   action="store_true",
                        help="Skip Stage 6 position group analysis")
    parser.add_argument("--skip-export",     action="store_true",
                        help="Skip Stage 7 artifact export")
    parser.add_argument("--quick",           action="store_true",
                        help="Shorthand: generation + smoke + training only")
    args = parser.parse_args()

    if args.quick:
        args.skip_predictions = True
        args.skip_features    = True
        args.skip_position    = True
        args.skip_export      = True

    # Prefer the venv Python if we're not already inside it
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists() and "venv" not in sys.executable.lower():
        py = str(venv_python.resolve())
    else:
        py = sys.executable

    print(f"\n{BOLD}F1 Analysis — Full Pipeline Runner{RESET}")
    print(f"Python: {py}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pipeline_start = time.time()
    all_results: dict = {}

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.skip_generation:
        warn("Stage 1 (Data Generation) skipped by --skip-generation flag")
        all_results["1 · Data Generation"] = None
    else:
        ok_ = stage_data_generation(py)
        all_results["1 · Data Generation"] = ok_
        if not ok_:
            fail("Data generation failed — aborting pipeline")
            print_summary(all_results, time.time() - pipeline_start)
            sys.exit(1)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if args.skip_smoke:
        warn("Stage 2 (Smoke Checks) skipped")
        all_results["2 · Smoke Checks"] = None
    else:
        all_results["2 · Smoke Checks"] = stage_smoke_checks(py)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if args.skip_training:
        warn("Stage 3 (Model Training) skipped by --skip-training flag")
        all_results["3 · Model Training"] = None
    else:
        all_results["3 · Model Training"] = stage_model_training(py)

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    if args.skip_predictions:
        warn("Stage 4 (Race Predictions) skipped")
        all_results["4 · Race Predictions"] = None
    else:
        all_results["4 · Race Predictions"] = stage_predictions(py)

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    if args.skip_features:
        warn("Stage 5 (Feature Selection) skipped")
        all_results["5 · Feature Selection"] = None
    else:
        all_results["5 · Feature Selection"] = stage_feature_selection(py)

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    if args.skip_position:
        warn("Stage 6 (Position Analysis) skipped")
        all_results["6 · Position Analysis"] = None
    else:
        all_results["6 · Position Analysis"] = stage_position_analysis(py)

    # ── Stage 7 ──────────────────────────────────────────────────────────────
    if args.skip_export:
        warn("Stage 7 (Export Artifacts) skipped")
        all_results["7 · Export Artifacts"] = None
    else:
        all_results["7 · Export Artifacts"] = stage_export_artifacts(py)

    # ── Final summary ─────────────────────────────────────────────────────────
    success = print_summary(all_results, time.time() - pipeline_start)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(130)
