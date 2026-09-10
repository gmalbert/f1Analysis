from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_production_parquet import build_parquet


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = REPO_ROOT / "raceAnalysis.py"


class InfrastructurePolicyTests(unittest.TestCase):
    def test_runtime_has_no_top_level_research_imports(self) -> None:
        tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8-sig"))
        forbidden = {"fastf1", "shap", "boruta", "seaborn", "optuna"}
        imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertTrue(forbidden.isdisjoint(imports))

    def test_research_actions_are_opt_in(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8-sig")
        self.assertIn("RESEARCH_MODE =", source)
        for marker in (
            'Run Monte Carlo Search',
            'Run RFE',
            'Run Boruta',
            'Run RFE to Minimize MAE',
            'Start Hyperparameter Tuning',
        ):
            guarded_lines = [
                line for line in source.splitlines()
                if marker in line and "st.button" in line
            ]
            self.assertTrue(guarded_lines, marker)
            self.assertTrue(
                all("RESEARCH_MODE and st." in line for line in guarded_lines),
                marker,
            )

    def test_parquet_builder_preserves_schema_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            destination = root / "output.parquet"
            source.write_text(
                "driver\tposition\tactive\n"
                "A\t1\tTrue\n"
                "B\t2\tFalse\n",
                encoding="utf-8",
            )

            rows, columns, _ = build_parquet(source, destination)
            result = pd.read_parquet(destination)

            self.assertEqual((rows, columns), (2, 3))
            self.assertEqual(list(result.columns), ["driver", "position", "active"])
            self.assertEqual(result["position"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
