"""Append-only CSV ledgers with contract validation and atomic replacement."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pandas as pd

from .contracts import DatasetContract


class LedgerStore:
    def __init__(
        self,
        path: str | Path,
        *,
        contract: DatasetContract,
        id_column: str,
    ) -> None:
        self.path = Path(path)
        self.contract = contract
        self.id_column = id_column

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=[rule.name for rule in self.contract.rules])
        return pd.read_csv(self.path)

    def append(self, records: Iterable[dict[str, Any] | object]) -> int:
        rows = []
        for record in records:
            if hasattr(record, "as_record"):
                rows.append(record.as_record())
            elif is_dataclass(record):
                rows.append(asdict(record))
            elif isinstance(record, dict):
                rows.append(dict(record))
            else:
                raise TypeError(f"unsupported ledger record: {type(record).__name__}")
        incoming = pd.DataFrame(rows)
        if incoming.empty:
            return 0
        existing = self.read()
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
        if self.id_column not in combined:
            raise KeyError(f"ledger requires id column {self.id_column!r}")
        combined = combined.drop_duplicates(self.id_column, keep="first")
        report = self.contract.validate(combined)
        report.raise_for_errors()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Use a sibling temporary file so os.replace remains atomic on Windows.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv", dir=self.path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            combined.to_csv(handle, index=False)
        temp_path.replace(self.path)
        return len(combined) - len(existing)

    def write_metadata(self, **metadata: Any) -> Path:
        destination = self.path.with_suffix(self.path.suffix + ".metadata.json")
        payload = {
            "contract": self.contract.name,
            "schema_version": self.contract.schema_version,
            **metadata,
        }
        destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return destination
