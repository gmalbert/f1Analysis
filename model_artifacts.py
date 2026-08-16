"""Content fingerprints shared by model-training jobs and the Streamlit app."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any, MutableMapping


FINGERPRINT_ALGORITHM = "sha256"


def build_data_fingerprint(data_path: str | Path) -> dict[str, Any]:
    """Return stable content metadata for a model's source dataset.

    Line endings are normalized to LF before hashing so the digest is identical
    across checkouts (.gitattributes ``text=auto`` stores LF but Windows working
    copies may be CRLF) and therefore matches on GitHub Actions and Streamlit
    Cloud as well as local Windows training runs.
    """
    path = Path(data_path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        data = source.read()
    digest.update(data.replace(b"\r\n", b"\n"))

    return {
        "data_file": path.name,
        "data_sha256": digest.hexdigest(),
        "data_size": len(data.replace(b"\r\n", b"\n")),
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
    }


def stamp_artifact(
    artifact: MutableMapping[str, Any],
    fingerprint: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Attach a previously computed dataset fingerprint to an artifact."""
    for key in ("data_file", "data_sha256", "data_size", "fingerprint_algorithm"):
        artifact[key] = fingerprint[key]
    return artifact


def artifact_matches_fingerprint(
    artifact: MutableMapping[str, Any],
    fingerprint: MutableMapping[str, Any],
) -> bool | None:
    """Return True/False for fingerprinted artifacts, or None for legacy ones."""
    artifact_digest = artifact.get("data_sha256")
    if not artifact_digest:
        return None
    if artifact.get("fingerprint_algorithm", FINGERPRINT_ALGORITHM) != FINGERPRINT_ALGORITHM:
        return False
    if artifact.get("data_file") not in (None, fingerprint["data_file"]):
        return False
    return hmac.compare_digest(str(artifact_digest), str(fingerprint["data_sha256"]))
