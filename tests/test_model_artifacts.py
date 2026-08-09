from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_artifacts import (
    artifact_matches_fingerprint,
    build_data_fingerprint,
    stamp_artifact,
)


class ModelArtifactFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_content_based(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.csv")
            second = Path(directory, "second.csv")
            first.write_bytes(b"same-content\n")
            second.write_bytes(b"same-content\n")

            first_fingerprint = build_data_fingerprint(first)
            second_fingerprint = build_data_fingerprint(second)

            self.assertEqual(first_fingerprint["data_sha256"], second_fingerprint["data_sha256"])

    def test_stamped_artifact_matches_its_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "f1ForAnalysis.csv")
            data_path.write_bytes(b"race-data\n")
            fingerprint = build_data_fingerprint(data_path)
            artifact: dict[str, object] = {"model": "placeholder"}

            stamp_artifact(artifact, fingerprint)

            self.assertTrue(artifact_matches_fingerprint(artifact, fingerprint))

    def test_changed_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "f1ForAnalysis.csv")
            data_path.write_bytes(b"old-data\n")
            artifact: dict[str, object] = {}
            stamp_artifact(artifact, build_data_fingerprint(data_path))
            data_path.write_bytes(b"new-data\n")

            self.assertFalse(
                artifact_matches_fingerprint(artifact, build_data_fingerprint(data_path))
            )

    def test_legacy_artifact_has_unknown_status(self) -> None:
        fingerprint = {
            "data_file": "f1ForAnalysis.csv",
            "data_sha256": "a" * 64,
            "data_size": 1,
            "fingerprint_algorithm": "sha256",
        }
        self.assertIsNone(artifact_matches_fingerprint({}, fingerprint))


if __name__ == "__main__":
    unittest.main()
