from __future__ import annotations

from pathlib import Path

from scripts.scan_embedded_secrets import scan
from scripts.validate_workflow_security import validate_workflows


ROOT = Path(__file__).resolve().parents[1]


def test_workflows_are_least_privilege_and_immutably_pinned() -> None:
    assert validate_workflows(ROOT / ".github" / "workflows") == []


def test_repository_has_no_credential_shaped_plaintext_assignments() -> None:
    assert scan(ROOT) == []

