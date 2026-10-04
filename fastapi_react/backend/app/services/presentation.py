"""Request-isolated Python view data for the native React interface.

The offline-exported views call this small presentation protocol. It carries
values, column configurations, charts and widget state, not Python objects or
executable browser code. Prediction pages only load offline-trained artifacts;
the explicit bin-count experiment retains the reference's opt-in computation.
Administrative audit actions use the shared structured audit implementation.
"""

from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import io
import json
import logging
import pickle
import threading
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from app.config import DATA_DIR, REPO_ROOT

_CACHE: OrderedDict[Any, Any] = OrderedDict()
_LOCK = threading.RLock()
_MODEL_LOCK = threading.RLock()
_RENDER_LOCK = threading.RLock()
_VIEW_FILE = Path(__file__).with_name("reference_views.py")
_CODE = compile(_VIEW_FILE.read_text(encoding="utf-8"), str(_VIEW_FILE), "exec")


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [clean(v) for v in value]
    return scalar(value)


def table_rows(frame: pd.DataFrame) -> list[list[Any]]:
    """Normalize whole numeric columns without rounding values or visiting each cell.

    Mixed/text/date columns retain scalar normalization. An object matrix keeps
    Python integers, booleans and float precision when converted back to rows.
    """
    values = frame.to_numpy(dtype=object, copy=True)
    for index, (_name, series) in enumerate(frame.items()):
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
            values[~np.isfinite(numeric), index] = None
        else:
            values[:, index] = np.fromiter(
                (scalar(value) for value in values[:, index]), dtype=object, count=len(frame)
            )
    rows: list[list[Any]] = values.tolist()
    return rows


class State(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class ColumnConfig:
    def __getattr__(self, kind: str) -> Any:
        def column(label: str | None = None, **kwargs: Any) -> dict[str, Any]:
            return {"label": label, "kind": kind, **clean(kwargs)}

        return column


class Container:
    def __init__(self, ui: Presentation, node: dict[str, Any]):
        self.ui = ui
        self.node = node

    def __enter__(self) -> Container:
        self.ui.stack.append(self.node["children"])
        self.ui.visibility.append(
            True if self.node.get("sidebar") else self.ui.visible and not self.node.get("hidden", False)
        )
        return self

    def __exit__(self, *_args: Any) -> None:
        self.ui.stack.pop()
        self.ui.visibility.pop()

    def __getattr__(self, method: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            with self:
                return getattr(self.ui, method)(*args, **kwargs)

        return call


class Presentation:
    def __init__(self, page: int, values: dict[str, Any], action: str | None = None):
        self.page = page
        self.values = dict(values)
        self.action = action
        self.nodes: list[dict[str, Any]] = []
        self.sidebar_nodes: list[dict[str, Any]] = []
        self.stack = [self.nodes]
        self.visibility = [True]
        self.sidebar = Container(self, {"children": self.sidebar_nodes, "sidebar": True})
        self.column_config = ColumnConfig()
        self.session_state = State()
        self.model_type = values.get("Select Model Type", "XGBoost")
        self.namespace: dict[str, Any] = {}
        self.widgets: dict[str, Any] = {}
        self.root_tabs = False

    @property
    def visible(self) -> bool:
        return self.visibility[-1]

    def add(self, kind: str, **props: Any) -> dict[str, Any]:
        node = {"type": kind, "id": f"n{len(self.stack[-1])}", **props}
        if self.visible:
            self.stack[-1].append(node)
        return node

    def group(self, kind: str, **props: Any) -> Container:
        return Container(self, self.add(kind, children=[], **props))

    def tabs(self, labels: list[str]) -> list[Container]:
        root = not self.root_tabs
        self.root_tabs = True
        node = self.add("tabs", labels=labels, root=root, children=[])
        selected = self.page - 1 if root else int(self.values.get("_tabs:" + labels[0], 0))
        tabs = []
        for i, label in enumerate(labels):
            child = {"type": "tab", "label": label, "children": [], "index": i, "hidden": i != selected}
            node["children"].append(child)
            tabs.append(Container(self, child))
        return tabs

    def columns(self, widths: Any, **kwargs: Any) -> list[Container]:
        widths = [1] * widths if isinstance(widths, int) else list(widths)
        node = self.add("columns", widths=widths, children=[])
        columns = []
        for width in widths:
            child = {"type": "column", "width": width, "children": []}
            node["children"].append(child)
            columns.append(Container(self, child))
        return columns

    def expander(self, label: str, expanded: bool = False, **kwargs: Any) -> Container:
        return self.group("expander", label=label, expanded=expanded)

    def spinner(self, *_args: Any, **_kwargs: Any) -> Container:
        return Container(self, {"children": self.stack[-1]})

    def set_page_config(self, **_kwargs: Any) -> None:
        pass

    def stop(self) -> None:
        raise ValueError("The analysis could not load its required data.")

    def title(self, text: str) -> None:
        self.add("heading", text=text, level=1)

    def header(self, text: str) -> None:
        self.add("heading", text=text, level=2)

    def subheader(self, text: str) -> None:
        self.add("heading", text=text, level=3)

    def caption(self, text: str) -> None:
        self.add("caption", text=str(text))

    def write(self, *args: Any, **_kwargs: Any) -> None:
        for value in args:
            if isinstance(value, pd.DataFrame):
                self.dataframe(value)
            elif isinstance(value, (dict, list, np.ndarray)):
                self.json(value)
            elif value is not None:
                self.markdown(str(value))

    def markdown(self, text: str, unsafe_allow_html: bool = False, **_kwargs: Any) -> None:
        if "<style>" in text:
            return
        self.add("html" if unsafe_allow_html else "markdown", text=text)

    def text(self, text: str) -> None:
        self.add("text", text=str(text))

    def code(self, text: str, **_kwargs: Any) -> None:
        self.add("code", text=str(text))

    def json(self, value: Any, **_kwargs: Any) -> None:
        self.add("json", value=clean(value))

    def divider(self) -> None:
        self.add("divider")

    def info(self, text: str, icon: str | None = None, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="info", icon=icon)

    def warning(self, text: str, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="warning")

    def error(self, text: str, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="error")

    def success(self, text: str, **_kwargs: Any) -> None:
        self.add("notice", text=text, severity="success")

    def metric(self, label: str, value: Any, delta: Any = None, **_kwargs: Any) -> None:
        self.add("metric", label=label, value=clean(value), delta=clean(delta))

    def widget(self, kind: str, label: str, default: Any, key: str | None = None, **props: Any) -> Any:
        widget_id = key or label
        value = self.values.get(widget_id, default)
        self.widgets[widget_id] = clean(value)
        self.add(kind, label=label, key=widget_id, value=clean(value), **clean(props))
        return value

    def checkbox(self, label: str, value: bool = False, key: str | None = None, **kwargs: Any) -> bool:
        return bool(self.widget("checkbox", label, value, key, disabled=kwargs.get("disabled", False)))

    def selectbox(
        self, label: str, options: Any, index: int = 0, key: str | None = None, **kwargs: Any
    ) -> Any:
        options = list(options)
        value = self.values.get(key or label, options[index] if options else None)
        if value not in options:
            value = options[0] if options else None
        self.values[key or label] = value
        return self.widget("select", label, value, key, options=options, help=kwargs.get("help"))

    def multiselect(
        self, label: str, options: Any, default: Any = None, key: str | None = None, **kwargs: Any
    ) -> Any:
        return self.widget("multiselect", label, default or [], key, options=list(options))

    def number_input(
        self,
        label: str,
        min_value: Any = None,
        max_value: Any = None,
        value: Any = None,
        step: Any = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        value = value if value is not None else (min_value if min_value is not None else 0)
        return self.widget(
            "number",
            label,
            value,
            key,
            min=min_value,
            max=max_value,
            step=step or (1 if isinstance(value, int) else 0.01),
            format=kwargs.get("format") or ("%d" if isinstance(value, int) else "%.2f"),
        )

    def slider(
        self,
        label: str,
        min_value: Any = None,
        max_value: Any = None,
        value: Any = None,
        step: Any = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        default = value if value is not None else min_value
        result = self.widget(
            "slider",
            label,
            default,
            key,
            min=min_value,
            max=max_value,
            step=step or 1,
            format=kwargs.get("format"),
        )
        is_date = isinstance(min_value, dt.date)
        if isinstance(default, tuple):
            if is_date:
                return tuple(dt.date.fromisoformat(str(v)[:10]) if isinstance(v, str) else v for v in result)
            return tuple(result)
        return result

    def button(self, label: str, key: str | None = None, **kwargs: Any) -> bool:
        disabled = kwargs.get("disabled", False)
        widget_id = key or label
        self.add("button", label=label, key=widget_id, disabled=disabled, help=kwargs.get("help"))
        return self.action == widget_id and not disabled

    def file_uploader(self, label: str, type: Any = None, key: str | None = None, **kwargs: Any) -> Any:
        widget_id = key or label
        csv = self.values.get(widget_id)
        self.add(
            "upload", label=label, key=widget_id, filename=csv.get("name") if isinstance(csv, dict) else None
        )
        if isinstance(csv, dict):
            csv = csv.get("content")
        return io.StringIO(csv) if isinstance(csv, str) else None

    def download_button(
        self, label: str, data: Any, file_name: str = "download.txt", mime: str | None = None, **kwargs: Any
    ) -> None:
        if hasattr(data, "read"):
            data = data.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.add(
            "download",
            label=label,
            filename=file_name,
            mime=mime or "application/octet-stream",
            data=base64.b64encode(data).decode("ascii"),
        )

    def dataframe(
        self,
        frame: Any,
        column_config: dict[str, Any] | None = None,
        column_order: list[str] | None = None,
        hide_index: bool | None = None,
        width: Any = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.visible:
            return
        config = column_config or {}
        styles = {}
        formats = {}
        styler = frame if hasattr(frame, "_compute") and hasattr(frame, "data") else None
        if styler is not None:
            frame = styler.data
            try:
                styler._compute()
                styles = {f"{r}:{c}": dict(style) for (r, c), style in styler.ctx.items()}
                formats = {f"{r}:{c}": fn(frame.iloc[r, c]) for (r, c), fn in styler._display_funcs.items()}
            except Exception:
                logging.getLogger(__name__).exception("Could not apply dataframe styles")
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        # An explicit order is also the displayed column selection. Preserve
        # duplicates: the reference includes positionsGained twice in Explorer.
        cols = list(column_order) if column_order is not None else list(frame.columns)
        cols = [c for c in cols if c in frame and config.get(c, "visible") is not None]
        definitions = []
        for column in cols:
            definition = config.get(column, {})
            definition = definition if isinstance(definition, dict) else {"label": definition}
            series = frame[column]
            inferred = (
                "CheckboxColumn"
                if pd.api.types.is_bool_dtype(series)
                else "NumberColumn"
                if pd.api.types.is_numeric_dtype(series)
                else "DateColumn"
                if pd.api.types.is_datetime64_any_dtype(series)
                else "TextColumn"
            )
            definitions.append(
                {
                    "key": str(column),
                    "label": definition.get("label") or str(column),
                    "kind": inferred,
                    **definition,
                }
            )
        values = table_rows(frame[cols])
        # Preserve original row/column positions for Styler formatting/highlights.
        source_positions = {c: frame.columns.get_loc(c) for c in cols}
        cell_styles = (
            [[styles.get(f"{r}:{source_positions[c]}", {}) for c in cols] for r in range(len(frame))]
            if styles
            else None
        )
        display = (
            [[formats.get(f"{r}:{source_positions[c]}") for c in cols] for r in range(len(frame))]
            if formats
            else None
        )
        self.add(
            "table",
            columns=definitions,
            rows=values,
            index=clean(list(frame.index)),
            index_name=frame.index.name,
            hide_index=bool(hide_index),
            width=width,
            height=height or min(400, 35 * (len(frame) + 1) + 3),
            styles=cell_styles,
            display=display,
        )

    def chart(
        self,
        kind: str,
        data: Any,
        x: str | None = None,
        y: Any = None,
        x_label: str | None = None,
        y_label: str | None = None,
        color: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self.visible:
            return
        import altair as alt

        from app.services.chart_builder import ChartType, generate_chart

        alt.data_transformers.disable_max_rows()
        chart = generate_chart(
            {"scatter": ChartType.SCATTER, "line": ChartType.LINE, "bar": ChartType.VERTICAL_BAR}[kind],
            data,
            x_from_user=x,
            y_from_user=y,
            x_axis_label=x_label,
            y_axis_label=y_label,
            color_from_user=color,
            size_from_user=kwargs.get("size"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            stack=kwargs.get("stack"),
            sort_from_user=kwargs.get("sort", False),
        )
        self.altair_chart(chart, width=kwargs.get("width"))

    def scatter_chart(self, data: Any, **kwargs: Any) -> None:
        self.chart("scatter", data, **kwargs)

    def line_chart(self, data: Any, **kwargs: Any) -> None:
        self.chart("line", data, **kwargs)

    def bar_chart(self, data: Any, **kwargs: Any) -> None:
        self.chart("bar", data, **kwargs)

    def altair_chart(self, chart: Any, **kwargs: Any) -> None:
        if not self.visible:
            return
        import altair as alt

        with alt.theme.enable("none"):
            spec = chart.to_dict()
        self.add("vega", spec=clean(spec), width=kwargs.get("width"))

    def plotly_chart(self, figure: Any, **kwargs: Any) -> None:
        self.add("plotly", spec=json.loads(figure.to_json()))

    def pyplot(self, figure: Any, **kwargs: Any) -> None:
        if not self.visible:
            import matplotlib.pyplot as plt

            plt.close(figure)
            return
        stream = io.BytesIO()
        figure.savefig(stream, format="png", bbox_inches="tight", dpi=200)
        import matplotlib.pyplot as plt

        plt.close(figure)
        self.add(
            "image",
            src="data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii"),
            width="stretch",
        )

    def image(self, image: Any, width: Any = None, **kwargs: Any) -> None:
        if not self.visible:
            return
        path = Path(image)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            self.warning(f"Image not found: {path.name}")
            return
        if path.name == "gridlocked-logo-with-text.png":
            self.add("image", src="/api/brand/logo", width=width)
            return
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        self.add(
            "image",
            src=f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii"),
            width=width,
        )

    def cache_data(self, func: Any = None, **_kwargs: Any) -> Any:
        def decorate(fn: Any) -> Any:
            @wraps(fn)
            def cached(*args: Any, **kwargs: Any) -> Any:
                key = (fn.__name__, hashlib.sha256(pickle.dumps((args, kwargs), protocol=5)).hexdigest())
                with _LOCK:
                    if key not in _CACHE:
                        _CACHE[key] = fn(*args, **kwargs)
                        if len(_CACHE) > 64:
                            _CACHE.popitem(last=False)
                    return copy.deepcopy(_CACHE[key])

            return cached

        return decorate(func) if func else decorate

    cache_resource = cache_data

    def load_model(self, name: str, model_type: str | None, fingerprint: dict[str, Any], version: str) -> Any:
        from model_artifacts import artifact_matches_fingerprint

        directory = {
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "CatBoost": "catboost",
            "Ensemble (XGBoost + LightGBM + CatBoost)": "ensemble",
            "Position Group": "position_group",
            "Track-Weighted Ensemble": "track_weighted",
        }
        dirs = (
            [directory[model_type], ""]
            if model_type in directory
            else ["xgboost", "lightgbm", "catboost", "ensemble", ""]
        )
        stale = None
        for folder in dirs:
            path = DATA_DIR / "models" / folder / f"{name}.pkl"
            if not path.is_file():
                continue
            key = ("model", str(path), path.stat().st_mtime_ns)
            with _MODEL_LOCK:
                if key not in _CACHE:
                    namespace = self.namespace

                    class Unpickler(pickle.Unpickler):
                        def __init__(self, source: Any, namespace: dict[str, Any]):
                            super().__init__(source)
                            self.view_namespace = namespace

                        def find_class(self, module: str, cls: str) -> Any:
                            if module in {"raceAnalysis", "__main__"} and cls in self.view_namespace:
                                return self.view_namespace[cls]
                            return super().find_class(module, cls)

                    with path.open("rb") as source:
                        _CACHE[key] = Unpickler(source, namespace).load()
                artifact = dict(_CACHE[key])
            if artifact.get("cache_version") != version:
                continue
            if (
                name == "position_model"
                and model_type
                and artifact.get("model_type")
                not in self.namespace["_MODEL_TYPE_ARTIFACT_LABELS"].get(model_type, {model_type})
            ):
                continue
            artifact["_artifact_path"] = str(path)
            manifest_name = {
                "position_model": "manifest.json",
                "dnf_model": "dnf_manifest.json",
                "safetycar_model": "safetycar_manifest.json",
            }.get(name)
            manifest_path = path.parent / manifest_name if manifest_name else None
            if manifest_path and manifest_name and not manifest_path.exists():
                manifest_path = DATA_DIR / "models" / manifest_name
            if manifest_path and manifest_path.exists():
                from f1bet.artifacts import ModelManifest

                try:
                    manifest = ModelManifest.load(manifest_path)
                    if name == "position_model":
                        feature_names = tuple(
                            str(v) for v in getattr(artifact.get("preprocessor"), "feature_names_in_", ())
                        )
                        if (
                            manifest.schema_version != "legacy-wide-v1"
                            or manifest.feature_names != feature_names
                        ):
                            continue
                    artifact["_manifest_status"] = (
                        "verified" if manifest.data_sha256 == fingerprint.get("data_sha256") else "stale"
                    )
                except (ValueError, KeyError, TypeError):
                    continue
            else:
                artifact["_manifest_status"] = "legacy-missing"
            match = artifact_matches_fingerprint(artifact, fingerprint)
            artifact["_artifact_status"] = "current" if match else "legacy" if match is None else "stale"
            if match is not False:
                return artifact
            stale = stale or artifact
        return stale

    def dnf_diagnostics(self, data: pd.DataFrame) -> np.ndarray:
        path = Path(__file__).with_name("dnf_diagnostics.json")
        if not path.is_file():
            raise ValueError("Export DNF diagnostic probabilities offline before serving Next Race.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(
            (DATA_DIR / "f1ForAnalysis.csv").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if payload["data_sha256"] != digest or len(payload["probabilities"]) != len(data):
            raise ValueError("Re-export DNF diagnostics for the current analysis dataset.")
        return np.array(payload["probabilities"])


def render_view(page: int, values: dict[str, Any], action: str | None = None) -> dict[str, Any]:
    ui = Presentation(page, values, action)
    namespace = {
        "ui": ui,
        "__name__": "react_reference_views",
        "__file__": str(REPO_ROOT / "raceAnalysis.py"),
        "repository_data_dir": DATA_DIR,
        "repository_root": REPO_ROOT,
    }
    namespace["view_namespace"] = lambda: namespace
    ui.namespace = namespace
    # Each request owns its variables, widgets and model selection. Cached data
    # is immutable to the caller: source view mutations receive a private copy.
    # Matplotlib and Altair maintain process-wide rendering state.
    with _RENDER_LOCK:
        exec(_CODE, namespace)  # noqa: S102 - fixed, checked-in module; no user-supplied code.
    root = next((n for n in ui.nodes if n["type"] == "tabs" and n.get("root")), None)
    page_nodes = root["children"][page - 1]["children"] if root else []
    shell = ui.nodes[: ui.nodes.index(root)] if root else []
    return {
        "shell": shell,
        "tabs": root["labels"] if root else [],
        "nodes": page_nodes,
        "sidebar": ui.sidebar_nodes,
        "widgets": ui.widgets,
    }
