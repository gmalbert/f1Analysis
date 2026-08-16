from __future__ import annotations

import pandas as pd

from f1bet.evaluation import (
    REQUIRED_ABLATION_VARIANTS,
    evaluate_probability_variants,
    probability_slice_report,
    validate_ablation_coverage,
)


def _records() -> pd.DataFrame:
    rows = []
    for event in range(8):
        for driver in range(2):
            rows.append(
                {
                    "event_id": f"{2024 + event // 4}-R{event % 4 + 1:02d}-R",
                    "fold_id": event // 2,
                    "outcome": int(driver == event % 2),
                    "raw": 0.7 if driver == event % 2 else 0.3,
                    "calibrated": 0.8 if driver == event % 2 else 0.2,
                    "circuit_type": "street" if event % 2 else "technical",
                    "is_wet_race": event % 3 == 0,
                    "grid_position": driver + 1 + event,
                    "is_rookie": driver == 1,
                    "constructor_id": f"team-{driver}",
                    "source_coverage": 0.9 + 0.01 * event,
                }
            )
    return pd.DataFrame(rows)


def test_probability_variants_share_folds_and_report_clustered_intervals() -> None:
    report = evaluate_probability_variants(
        _records(),
        {"raw_probabilities": "raw", "calibrated_probabilities": "calibrated"},
        n_bins=4,
        n_bootstrap=100,
    )
    assert set(report.variant) == {"raw_probabilities", "calibrated_probabilities"}
    assert report.groupby("variant").fold_id.nunique().eq(4).all()
    assert report["brier_ci_low"].notna().all()


def test_ablation_coverage_requires_every_variant_on_the_same_folds() -> None:
    complete = pd.DataFrame(
        [
            {"variant": variant, "fold_id": fold, "search_trials": 7}
            for variant in REQUIRED_ABLATION_VARIANTS
            for fold in (0, 1)
        ]
    )
    coverage = validate_ablation_coverage(complete)
    assert coverage.complete and coverage.search_trials == 7
    incomplete = validate_ablation_coverage(complete[complete.variant != "no_weather"])
    assert not incomplete.complete and incomplete.missing_variants == ("no_weather",)


def test_required_slice_report_has_all_dimensions() -> None:
    records = _records().rename(columns={"calibrated": "probability"})
    result = probability_slice_report(records, n_bins=4, n_bootstrap=100)
    assert result.missing_dimensions == ()
    assert set(result.report.dimension) == {
        "season",
        "circuit_archetype",
        "wet_dry",
        "grid_band",
        "rookie",
        "constructor",
        "data_coverage",
    }
