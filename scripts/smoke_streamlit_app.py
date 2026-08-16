"""Exercise the real Streamlit entrypoint and the betting calculator offline."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "f1analysis-matplotlib"))

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
    entrypoint = Path(__file__).resolve().parents[1] / "raceAnalysis.py"
    app = AppTest.from_file(str(entrypoint), default_timeout=120).run()
    if app.exception:
        raise AssertionError("initial app run failed: " + "; ".join(item.message for item in app.exception))
    tabs = {item.label for item in app.tabs}
    required_tabs = {
        "📐 Betting Research",
        "Value & stake",
        "Field simulation",
        "Paper replay",
        "Calibration",
        "Release gates",
    }
    missing_tabs = required_tabs - tabs
    if missing_tabs:
        raise AssertionError(f"missing Streamlit tabs: {sorted(missing_tabs)}")

    probability = next(item for item in app.number_input if item.label == "Model probability")
    probability.set_value(0.75)
    app.run()
    if app.exception:
        raise AssertionError("calculator rerun failed: " + "; ".join(item.message for item in app.exception))
    metrics = {item.label: item.value for item in app.metric}
    paper_stake = next(value for label, value in metrics.items() if label.startswith("Paper stake"))
    if metrics.get("Raw EV / unit") != "+57.50%" or paper_stake != "$100.00":
        raise AssertionError(f"unexpected calculator result: EV={metrics.get('Raw EV / unit')}, stake={paper_stake}")
    print("Streamlit real-entrypoint smoke passed: betting tabs rendered and calculator reran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
