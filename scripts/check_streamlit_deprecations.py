"""Fail CI when removed Streamlit keyword arguments are used in Python code."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".venv-mac", "__pycache__"}
FORBIDDEN_KEYWORDS = {"use_container_width"}


def python_files() -> list[Path]:
    return [
        path
        for path in REPO_ROOT.rglob("*.py")
        if not SKIP_DIRS.intersection(path.relative_to(REPO_ROOT).parts)
    ]


def main() -> int:
    violations: list[str] = []

    for path in python_files():
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: syntax error: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg in FORBIDDEN_KEYWORDS:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{keyword.value.lineno}: "
                        f"deprecated Streamlit keyword {keyword.arg!r}"
                    )

    if violations:
        print("Streamlit API compatibility check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Streamlit API compatibility check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
