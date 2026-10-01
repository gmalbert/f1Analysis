"""Small pandas adapters for the vendored pure Vega-Lite chart builder."""

from typing import Any

import pandas as pd


class ChartError(ValueError):
    pass


class InvalidColorError(ChartError):
    def __init__(self, color: Any):
        super().__init__(f"Invalid chart color: {color}")


class DataframeAdapter:
    @staticmethod
    def convert_anything_to_pandas_df(data: Any, ensure_copy: bool = False) -> pd.DataFrame:
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return frame.copy(deep=True) if ensure_copy else frame

    @staticmethod
    def convert_anything_to_list(data: Any) -> list[Any]:
        return list(data)

    @staticmethod
    def fix_arrow_incompatible_column_types(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        for column in frame.select_dtypes(include=["object"]).columns:
            if pd.api.types.infer_dtype(frame[column], skipna=True) in {"mixed", "mixed-integer", "complex"}:
                frame[column] = frame[column].astype("string")
        return frame


class TypeAdapter:
    @staticmethod
    def is_altair_version_less_than(version: str) -> bool:
        return False  # Runtime dependency requires Altair 6 or later.


dataframe_util = DataframeAdapter()
type_util = TypeAdapter()
