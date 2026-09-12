from __future__ import annotations

import subprocess
import sys
from typing import Any

from app.config import ENABLE_EXPENSIVE_TOOLS, REPO_ROOT

TOOLS = {
    "monte_carlo": "scripts/precompute/monte_carlo_features.py",
    "rfe": "scripts/precompute/rfe_features.py",
    "boruta": "scripts/precompute/boruta_features.py",
    "shap": "scripts/precompute/shap_analysis.py",
    "permutation": "scripts/precompute/permutation_importance.py",
    "temporal_leakage": "scripts/audit_temporal_leakage.py",
    "hyperparameter_grid": "scripts/precompute/hyperparameter_grid_search.py",
    "hyperparameter_bayesian": "scripts/precompute/hyperparameter_bayesian.py",
}

def run_tool(name: str, args: list[str]) -> dict[str, Any]:
    if not ENABLE_EXPENSIVE_TOOLS:
        raise PermissionError(
            "Expensive/manual analysis tools are disabled. Set ENABLE_EXPENSIVE_TOOLS=1 only on a test host."
        )
    relative = TOOLS.get(name)
    if not relative:
        raise KeyError(name)
    script = REPO_ROOT / relative
    if not script.exists():
        raise FileNotFoundError(str(script))
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=1800, check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-100_000:],
        "stderr": completed.stderr[-100_000:],
    }
