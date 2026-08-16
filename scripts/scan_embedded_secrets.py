"""Fail on credential-shaped plaintext assignments in source-controlled code."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|password|authorization)\b\s*[=:]\s*"
    r"[\"']([^\"']{12,})[\"']"
)
PLACEHOLDERS = ("example", "placeholder", "replace", "changeme", "[redacted]", "${{")


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    extensions = {".py", ".yml", ".yaml", ".json", ".toml", ".ini"}
    ignored_parts = {".git", ".venv", "venv", "node_modules", "data_files"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for match in ASSIGNMENT.finditer(line):
                candidate = match.group(2).lower()
                if not any(placeholder in candidate for placeholder in PLACEHOLDERS):
                    findings.append(f"{path}:{line_number}: possible embedded {match.group(1)}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    findings = scan(args.root)
    if findings:
        print("\n".join(findings))
        return 1
    print("No credential-shaped plaintext assignments found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

