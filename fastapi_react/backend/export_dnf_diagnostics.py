"""Rebuild diagnostic probabilities offline using the reference algorithm."""

import ast
import hashlib
import json
from pathlib import Path

from app.config import DATA_DIR, REPO_ROOT
from app.services.presentation import _CODE, Presentation

ui = Presentation(1, {})
namespace = {
    "ui": ui,
    "__name__": "react_reference_views",
    "__file__": str(REPO_ROOT / "raceAnalysis.py"),
    "repository_data_dir": DATA_DIR,
    "repository_root": REPO_ROOT,
}
namespace["view_namespace"] = lambda: namespace
ui.namespace = namespace
exec(_CODE, namespace)  # noqa: S102 - fixed, checked-in offline view source.
source = ast.parse((REPO_ROOT / "raceAnalysis.py").read_text(encoding="utf-8"))
function = next(
    n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "get_dnf_diagnostic_probs"
)
function.decorator_list = []
exec(compile(ast.Module(body=[function], type_ignores=[]), "reference_dnf_algorithm", "exec"), namespace)  # noqa: S102 - original diagnostic function, not user input.
probabilities = namespace["get_dnf_diagnostic_probs"](namespace["CACHE_VERSION"])
payload = {
    "data_sha256": hashlib.sha256(
        (DATA_DIR / "f1ForAnalysis.csv").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest(),
    "probabilities": probabilities.tolist(),
}
(Path(__file__).parent / "app/services/dnf_diagnostics.json").write_text(
    json.dumps(payload), encoding="utf-8"
)
print(f"Exported {len(probabilities)} diagnostic probabilities.")
