from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_REPO_ROOT = HERE.parents[3]
REPO_ROOT = Path(os.environ.get("F1_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
DATA_DIR = REPO_ROOT / "data_files"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"
MODELS_DIR = DATA_DIR / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENABLE_EXPENSIVE_TOOLS = os.environ.get("ENABLE_EXPENSIVE_TOOLS", "0").strip().lower() in {"1", "true", "yes"}
MAX_TABLE_ROWS = int(os.environ.get("MAX_TABLE_ROWS", "1000"))
CACHE_VERSION = os.environ.get("F1_CACHE_VERSION", "v3.3")

MODEL_TYPES = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Ensemble (XGBoost + LightGBM + CatBoost)",
    "Position Group",
    "Track-Weighted Ensemble",
]
