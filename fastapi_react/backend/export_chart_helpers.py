"""Vendor the reference's pure chart builder, without its server dependency.

Offline maintenance tool. Upstream license headers are retained; modifications
replace the dataframe/error adapters and nothing in the chart calculations.
"""

from pathlib import Path

import streamlit

source = Path(streamlit.__file__).parent / "elements" / "lib"
target = Path(__file__).parent / "app" / "services"
colors = (source / "color_util.py").read_text(encoding="utf-8")
colors = colors.replace(
    "from streamlit.errors import StreamlitInvalidColorError",
    "from app.services.chart_adapters import InvalidColorError as StreamlitInvalidColorError",
)
chart = (source / "built_in_chart_utils.py").read_text(encoding="utf-8")
chart = chart.replace(
    "from streamlit import dataframe_util, type_util",
    "from app.services.chart_adapters import dataframe_util, type_util",
)
chart = chart.replace(
    "from streamlit.elements.lib.color_util import", "from app.services.chart_colors import"
)
chart = chart.replace(
    "from streamlit.errors import Error, StreamlitAPIException",
    "from app.services.chart_adapters import ChartError as Error, ChartError as StreamlitAPIException",
)
chart = chart.replace(
    "    from streamlit.dataframe_util import Data\n    from streamlit.elements.lib.layout_utils import (\n        Height,\n        Width,\n    )",
    "    Data = Any\n    Height = int | str\n    Width = int | str",
)
for name, text in [("chart_colors.py", colors), ("chart_builder.py", chart)]:
    (target / name).write_text(
        "# Adapted offline from Streamlit "
        + streamlit.__version__
        + "; see export_chart_helpers.py.\n"
        + text,
        encoding="utf-8",
    )
