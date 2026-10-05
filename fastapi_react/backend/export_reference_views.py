"""One-way conversion of the reference views to the React presentation protocol.

Run offline after intentionally updating the reference. The generated Python
uses no Streamlit server, session, imports, or browser runtime. Keeping the
view calculations together preserves model feature ordering and formatting.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "app" / "services"


class Convert(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and (
            node.value in {"scripts", "data_files"}
            or (node.value.startswith(("scripts/", "data_files/"))
            and " " not in node.value)
        ):
            return ast.Call(
                func=ast.Name(id="str", ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Name(id="repository_root", ctx=ast.Load()),
                        op=ast.Div(),
                        right=ast.Constant(node.value),
                    )
                ],
                keywords=[],
            )
        return node

    def visit_Name(self, node):
        if node.id == "st":
            node.id = "ui"
        return node

    def visit_Import(self, node):
        node.names = [name for name in node.names if name.name != "streamlit"]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        if node.module == "footer":
            return None
        if node.module == "f1bet.streamlit_page":
            return ast.parse("from app.services.betting_view import render_betting_research").body[0]
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "render_betting_research":
                node.args.insert(0, ast.Name(id="ui", ctx=ast.Load()))
            elif node.func.id == "get_trained_model" and not any(
                k.arg == "model_type" for k in node.keywords
            ):
                node.keywords.append(
                    ast.keyword(
                        arg="model_type",
                        value=ast.Attribute(
                            value=ast.Name(id="ui", ctx=ast.Load()), attr="model_type", ctx=ast.Load()
                        ),
                    )
                )
            elif node.func.id == "globals":
                node.func.id = "view_namespace"
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == "DATA_DIR" for t in node.targets):
            node.value = ast.parse("str(repository_data_dir)", mode="eval").body
        if any(isinstance(t, ast.Name) and t.id == "RESEARCH_MODE" for t in node.targets):
            node.value = ast.Constant(False)
        return node


tree = ast.parse((ROOT / "raceAnalysis.py").read_text(encoding="utf-8"))
# This shared panel is defined inside the source's Models tab but also used by
# Data & Debug. Move its declaration outside the conditional page execution.
shared_audit = next(
    n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "leakage_audit_ui"
)
tree.body.insert(0, shared_audit)
# Preserve pure calculations and UI declarations. Runtime training functions
# have no place in the public API: fail closed even if accidentally invoked.
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "train_and_evaluate_model",
        "train_and_evaluate_dnf_model",
        "train_and_evaluate_safetycar_model",
        "monte_carlo_feature_selection",
        "run_rfe_feature_selection",
        "run_boruta_feature_selection",
        "rfe_minimize_mae",
    }:
        node.body = ast.parse(
            "raise RuntimeError('Run offline training workflows to update model artifacts.')"
        ).body
    if isinstance(node, ast.FunctionDef) and node.name == "get_dnf_diagnostic_probs":
        node.body = ast.parse("return ui.dnf_diagnostics(data)").body
    if isinstance(node, ast.FunctionDef) and node.name == "load_pretrained_model":
        node.body = ast.parse(
            "return ui.load_model(model_name, model_type, get_data_fingerprint('f1SafetyCarFeatures.csv' if model_name == 'safetycar_model' else 'f1ForAnalysis.csv'), CACHE_VERSION)"
        ).body

tree = Convert().visit(tree)
body = []
for node in tree.body:
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "add_betting_oracle_footer"
    ):
        continue
    if (
        isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Name)
    ):
        name = node.items[0].context_expr.id
        if name in {"tab2", "tab3", "tab4", "tab5", "tab6", "tab7"}:
            node = ast.If(
                test=ast.parse(f"ui.page == {int(name[-1])}", mode="eval").body, body=[node], orelse=[]
            )
    body.append(node)
tree.body = body
ast.fix_missing_locations(tree)
(OUT / "reference_views.py").write_text(
    "# Generated by export_reference_views.py; review source changes before re-export.\n"
    "# This module is executed in an isolated request namespace by presentation.py.\n"
    + ast.unparse(tree)
    + "\n",
    encoding="utf-8",
)
betting = Convert().visit(ast.parse((ROOT / "f1bet" / "streamlit_page.py").read_text(encoding="utf-8")))
for node in betting.body:
    if isinstance(node, ast.ImportFrom) and node.level:
        node.module = "f1bet." + node.module
        node.level = 0
    if isinstance(node, ast.FunctionDef) and node.name == "render_betting_research":
        node.args.args.insert(0, ast.arg(arg="ui"))
        # Public React deployment exposes only the calculator. Keep upload-based
        # simulation, replay and calibration in the offline research source.
        calculator = next(
            item for item in node.body
            if isinstance(item, ast.With)
            and isinstance(item.items[0].context_expr, ast.Name)
            and item.items[0].context_expr.id == "calculator"
        )
        node.body = [node.body[0], *ast.parse("ui.subheader('Value & stake')").body, *calculator.body]
betting.body = [
    node for node in betting.body
    if not (isinstance(node, ast.FunctionDef) and node.name == "_simulation_template")
    and not (isinstance(node, ast.ImportFrom) and node.module in {
        "f1bet.backtest", "f1bet.calibration", "f1bet.simulation"
    })
]
ast.fix_missing_locations(betting)
(OUT / "betting_view.py").write_text(
    "# Generated offline; no Streamlit dependency.\n" + ast.unparse(betting) + "\n", encoding="utf-8"
)
print("Exported React presentation views.")
