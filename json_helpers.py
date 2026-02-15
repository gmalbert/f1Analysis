"""
JSON serialization helpers for repository-wide precompute outputs.

Provides a `json_default` function suitable for `json.dump(..., default=...)`
that converts common numpy / pandas scalar and array types to native
Python types (int/float/bool/list/ISO strings). Also exposes `safe_dump`
which writes JSON using the helper by default.
"""
from __future__ import annotations

from pathlib import Path
import json
import datetime
import numpy as np
import pandas as pd


def json_default(obj):
    """Return a JSON-serializable representation for known types.

    Intended for use as the ``default`` parameter to ``json.dump``/``dumps``.
    """
    # numpy / pandas scalar types
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # numpy arrays -> lists
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    # pandas / datetime -> ISO string
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()

    # pandas NA
    if obj is pd.NA:
        return None

    # Fallback for small containers
    if isinstance(obj, (list, tuple, set)):
        return list(obj)

    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def safe_dump(obj, fp_or_path, **kwargs):
    """Write `obj` to `fp_or_path` using `json_default` for non-serializable types.

    `fp_or_path` may be a file-like object or a path-like (str / Path).
    Additional keyword args are forwarded to ``json.dump``.
    """
    if isinstance(fp_or_path, (str, Path)):
        with open(fp_or_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, default=json_default, **kwargs)
    else:
        json.dump(obj, fp_or_path, default=json_default, **kwargs)
