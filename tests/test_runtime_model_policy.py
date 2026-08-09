from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = REPO_ROOT / "raceAnalysis.py"


def function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def with_blocks_for_name(tree: ast.AST, context_name: str) -> list[ast.With]:
    matches: list[ast.With] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == context_name
            for item in node.items
        ):
            matches.append(node)
    return matches


class RuntimeModelPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8-sig"))
        cls.functions = function_nodes(cls.tree)

    def test_request_model_getters_never_call_training_functions(self) -> None:
        forbidden = {
            "train_and_evaluate_model",
            "train_and_evaluate_dnf_model",
            "train_and_evaluate_safetycar_model",
        }
        for function_name in ("get_trained_model", "get_dnf_model", "get_safetycar_model"):
            with self.subTest(function=function_name):
                self.assertTrue(
                    forbidden.isdisjoint(called_names(self.functions[function_name])),
                    f"{function_name} must load workflow artifacts, not train models",
                )

    def test_model_loader_uses_shared_resource_cache(self) -> None:
        loader = self.functions["_load_pretrained_model_resource"]
        decorators = [ast.unparse(decorator) for decorator in loader.decorator_list]
        self.assertTrue(any("st.cache_resource" in decorator for decorator in decorators))
        self.assertTrue(any("max_entries=8" in decorator for decorator in decorators))

    def test_main_dataframe_cache_is_bounded(self) -> None:
        loader = self.functions["load_data"]
        decorators = [ast.unparse(decorator) for decorator in loader.decorator_list]
        self.assertTrue(any("st.cache_data" in decorator for decorator in decorators))
        self.assertTrue(any("max_entries=1" in decorator for decorator in decorators))

    def test_diagnostic_tabs_do_not_run_training_or_validation(self) -> None:
        forbidden = {"fit", "fit_transform", "permutation_importance", "cross_val_score"}
        for tab_name in ("tab_feat", "tab_hist"):
            blocks = with_blocks_for_name(self.tree, tab_name)
            self.assertEqual(len(blocks), 1, f"Expected one {tab_name} render block")
            self.assertTrue(
                forbidden.isdisjoint(called_names(blocks[0])),
                f"{tab_name} must render workflow artifacts instead of computing diagnostics",
            )

    def test_diagnostic_artifact_loaders_are_bounded(self) -> None:
        for loader_name in (
            "load_precomputed_permutation",
            "load_precomputed_historical_validation",
        ):
            loader = self.functions[loader_name]
            decorators = [ast.unparse(decorator) for decorator in loader.decorator_list]
            self.assertTrue(any("st.cache_data" in decorator for decorator in decorators))
            self.assertTrue(any("max_entries=1" in decorator for decorator in decorators))


if __name__ == "__main__":
    unittest.main()
