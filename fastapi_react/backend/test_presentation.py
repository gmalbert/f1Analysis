"""Integration coverage for request-isolated native React presentation data."""

import datetime as dt
import io
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.data import load_streamlit_raw_data
from app.services.presentation import Presentation, clean, render_view, scalar, table_rows


def walk(nodes):
    for node in nodes:
        yield node
        yield from walk(node.get("children", []))


@pytest.mark.parametrize("page", range(1, 8))
def test_every_reference_page_serializes(page):
    result = TestClient(app).post("/api/views", json={"page": page, "values": {}})
    assert result.status_code == 200, result.text
    payload = result.json()
    assert len(payload["tabs"]) == 7
    assert any(node["type"] == "heading" for node in payload["nodes"])
    assert "NaN" not in json.dumps(payload)


def test_filters_and_all_model_panels():
    result = render_view(1, {"filter_results_main": True})
    assert result["sidebar"]
    table = next(node for node in walk(result["nodes"]) if node["type"] == "table")
    # Default filters must preserve every source race entry as the dataset grows.
    source = load_streamlit_raw_data()
    assert not source.empty
    assert len(table["rows"]) == len(source)
    identity_columns = ["grandPrixYear", "grandPrixName", "resultsDriverName"]
    indexes = [next(i for i, column in enumerate(table["columns"]) if column["key"] == key)
               for key in identity_columns]
    assert Counter(tuple(row[i] for i in indexes) for row in table["rows"]) == Counter(
        source[identity_columns].itertuples(index=False, name=None)
    )
    assert len(table["columns"]) == 34
    assert sum(c["key"] == "positionsGained" for c in table["columns"]) == 2
    for subtab in range(7):
        result = render_view(5, {"_tabs:📊 Model Performance": subtab})
        assert result["nodes"]
        assert not any(n["type"] == "notice" and n["severity"] == "error" for n in walk(result["nodes"]))


def test_formatting_and_cache_isolation():
    ui = Presentation(1, {})
    frame = pd.DataFrame(
        {"number": [1.25, np.nan], "flag": [True, False], "date": [dt.date(2026, 1, 1), None]}
    )
    ui.dataframe(
        frame,
        hide_index=False,
        column_config={"number": ui.column_config.NumberColumn("Amount", format="%.2f"), "date": None},
    )
    table = ui.nodes[-1]
    assert [c["label"] for c in table["columns"]] == ["Amount", "flag"]
    assert table["rows"] == [[1.25, True], [None, False]]
    assert not table["hide_index"]

    ui.dataframe(frame, column_order=["flag", "number", "flag"])
    assert [c["key"] for c in ui.nodes[-1]["columns"]] == ["flag", "number", "flag"]

    @ui.cache_data
    def isolated_test_cache():
        return frame

    first = isolated_test_cache()
    first.iloc[0, 0] = 999
    assert isolated_test_cache().iloc[0, 0] == 1.25
    assert clean({"date": dt.date(2026, 1, 1), "nan": np.nan}) == {"date": "2026-01-01", "nan": None}


def test_controls_downloads_and_charts():
    ui = Presentation(
        1, {"date": ["2020-01-01", "2021-01-01"], "pick": "b", "flag": True, "upload": "a,b\n1,2\n"}, "go"
    )
    assert ui.checkbox("flag")
    assert ui.selectbox("pick", ["a", "b"]) == "b"
    assert ui.slider(
        "date", dt.date(2019, 1, 1), dt.date(2026, 1, 1), (dt.date(2019, 1, 1), dt.date(2026, 1, 1))
    ) == (dt.date(2020, 1, 1), dt.date(2021, 1, 1))
    assert ui.button("go")
    assert not ui.button("disabled", disabled=True)
    assert ui.file_uploader("upload").read() == "a,b\n1,2\n"
    ui.download_button("download", io.BytesIO(b"abc"), "test.csv")
    assert ui.nodes[-1]["data"] == "YWJj"
    for method in [ui.scatter_chart, ui.line_chart, ui.bar_chart]:
        method(pd.DataFrame({"x": [1, 2], "y": [2, 3]}), x="x", y="y")
    assert all(n["spec"] for n in ui.nodes if n["type"] == "vega")


def test_batched_table_values_preserve_numeric_precision_missing_types_dates_and_duplicates():
    frame = pd.DataFrame(
        {
            "float": [np.nextafter(1.0, 2.0), np.inf, -np.inf, np.nan],
            "integer": pd.Series([2**60 + 1, None, -2**60, 0], dtype="Int64"),
            "boolean": pd.Series([True, False, None, True], dtype="boolean"),
            "timestamp": pd.to_datetime(["2026-01-01T12:34:56.123456789", None, None, None]),
            "mixed": [np.float64(1.25), dt.date(2026, 1, 2), np.inf, pd.NA],
            "category": pd.Categorical(["a", "b", None, "a"]),
            "nested": [["a", "b"], {"value": 2}, [], None],
        }
    )
    selected = frame[["integer", "float", "boolean", "timestamp", "mixed", "category", "nested", "integer"]]
    expected = [[scalar(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    assert table_rows(selected) == expected
    assert table_rows(selected)[0][0] == 2**60 + 1
    json.dumps(table_rows(selected), allow_nan=False)
    assert table_rows(frame.iloc[:0]) == []
    assert table_rows(pd.DataFrame(index=range(2))) == [[], []]


def test_betting_calculator_remains_available_without_upload_tools():
    # Old browser state/actions must not restore disabled tools or hide the calculator.
    result = render_view(7, {
        "_tabs:Value & stake": 1, "Simulations": 1000,
        "f1bet_field_upload": {"name": "old.csv", "content": "invalid csv"},
    }, "run_f1bet_simulation")
    nodes = list(walk(result["nodes"]))
    assert not any(n["type"] in {"upload", "table"} for n in nodes)
    assert len([n for n in nodes if n["type"] == "metric"]) == 4
    assert not any(n.get("label") in {"Field simulation", "Paper replay", "Calibration"} for n in nodes)
    baseline = render_view(7, {})
    changed = render_view(7, {"Model probability": 0.8})
    a = [n["value"] for n in walk(baseline["nodes"]) if n["type"] == "metric"]
    b = [n["value"] for n in walk(changed["nodes"]) if n["type"] == "metric"]
    assert a != b


def test_shared_audit_has_the_structured_callable_expected_by_both_apps(monkeypatch):
    from scripts import audit_temporal_leakage as audit

    frame = pd.DataFrame(
        {
            "resultsFinalPositionNumber": list(range(1, 21)) * 2,
            "future_result": list(range(1, 21)) * 2,
            "constant": [0] * 40,
        }
    )
    monkeypatch.setattr(audit.pd, "read_csv", lambda *args, **kwargs: frame)
    monkeypatch.setattr(audit.pd, "read_json", lambda *args, **kwargs: pd.DataFrame())
    report = audit.run_audit(40)
    issues = report[report["feature"] == "future_result"]["issue_type"].tolist()
    assert "name_pattern" in issues
    assert "high_correlation" in issues
    assert "exact_equality" in issues
    assert "constant" not in report["feature"].tolist()
    with pytest.raises(ValueError, match="positive integer"):
        audit.run_audit(-1)
