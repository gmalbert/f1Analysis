"""Validate GitHub Actions syntax, least privilege, and immutable action pins."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml


FULL_SHA = re.compile(r"[0-9a-f]{40}")
USES = re.compile(r"\buses:\s*[^@\s]+@([^\s#]+)")


def validate_workflows(root: Path) -> list[str]:
    errors: list[str] = []
    paths = sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml"))
    if not paths:
        return [f"no workflow files found in {root}"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            document = yaml.load(text, Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            errors.append(f"{path}: invalid YAML: {exc}")
            continue
        if not isinstance(document, dict) or "jobs" not in document:
            errors.append(f"{path}: workflow must define jobs")
        permissions = document.get("permissions") if isinstance(document, dict) else None
        if not isinstance(permissions, dict) or permissions.get("contents") not in {"read", "none"}:
            errors.append(f"{path}: top-level contents permission must be read or none")
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = USES.search(line)
            if match and not FULL_SHA.fullmatch(match.group(1)):
                errors.append(f"{path}:{line_number}: action is not pinned to a full SHA")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".github/workflows"))
    args = parser.parse_args()
    errors = validate_workflows(args.root)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated workflow security in {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

