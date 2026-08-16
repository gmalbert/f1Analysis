"""Stable driver/constructor identity resolution with ambiguity reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import re
import unicodedata
from typing import Iterable

import pandas as pd


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    canonical_id: str
    alias: str
    provider: str = "global"
    provider_id: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None

    def __post_init__(self) -> None:
        if not self.canonical_id.strip() or not normalize_label(self.alias) or not self.provider.strip():
            raise ValueError("canonical_id, alias, and provider are required")
        if self.valid_from is not None and self.valid_to is not None and self.valid_from > self.valid_to:
            raise ValueError("valid_from cannot be later than valid_to")

    def active_at(self, value: date | datetime | None) -> bool:
        if value is None:
            return True
        target = value.date() if isinstance(value, datetime) else value
        return (self.valid_from is None or self.valid_from <= target) and (
            self.valid_to is None or target <= self.valid_to
        )


@dataclass(slots=True)
class IdentityResolver:
    aliases: dict[str, str] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    records: list[IdentityRecord] = field(default_factory=list)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        canonical_col: str,
        alias_cols: Iterable[str],
    ) -> "IdentityResolver":
        resolver = cls()
        for _, row in frame.iterrows():
            canonical = str(row[canonical_col]).strip()
            if not canonical or canonical.lower() == "nan":
                continue
            for column in alias_cols:
                value = row.get(column)
                if pd.isna(value):
                    continue
                resolver.add(canonical, str(value))
            resolver.add(canonical, canonical)
        return resolver

    def add(
        self,
        canonical_id: str,
        alias: str,
        *,
        provider: str = "global",
        provider_id: str | None = None,
        valid_from: date | None = None,
        valid_to: date | None = None,
    ) -> None:
        key = normalize_label(alias)
        if not key:
            return
        record = IdentityRecord(
            canonical_id=str(canonical_id).strip(),
            alias=str(alias),
            provider=provider,
            provider_id=provider_id,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        self.records.append(record)
        if provider != "global" or valid_from is not None or valid_to is not None:
            return
        existing = self.aliases.get(key)
        if existing is not None and existing != canonical_id:
            self.ambiguous.add(key)
            self.aliases.pop(key, None)
            return
        if key not in self.ambiguous:
            self.aliases[key] = canonical_id

    def resolve(
        self,
        value: object,
        *,
        strict: bool = True,
        provider: str = "global",
        at: date | datetime | None = None,
    ) -> str | None:
        key = normalize_label(value)
        scoped = {
            record.canonical_id
            for record in self.records
            if record.provider == provider
            and record.active_at(at)
            and (normalize_label(record.alias) == key or (record.provider_id is not None and normalize_label(record.provider_id) == key))
        }
        if provider != "global" or at is not None:
            if len(scoped) > 1:
                if strict:
                    raise ValueError(f"ambiguous entity alias: {value!r}")
                return None
            if len(scoped) == 1:
                return next(iter(scoped))
            if strict:
                raise KeyError(f"unknown entity alias: {value!r}")
            return None
        if key in self.ambiguous:
            if strict:
                raise ValueError(f"ambiguous entity alias: {value!r}")
            return None
        resolved = self.aliases.get(key)
        if resolved is None and strict:
            raise KeyError(f"unknown entity alias: {value!r}")
        return resolved

    def resolution_report(
        self,
        values: Iterable[object],
        *,
        provider: str = "global",
        at: date | datetime | None = None,
    ) -> pd.DataFrame:
        rows = []
        for value in values:
            key = normalize_label(value)
            try:
                canonical = self.resolve(value, strict=True, provider=provider, at=at)
                status = "matched"
            except ValueError:
                canonical, status = None, "ambiguous"
            except KeyError:
                canonical, status = None, "unknown"
            rows.append(
                {
                    "input": value,
                    "normalized": key,
                    "provider": provider,
                    "canonical_id": canonical,
                    "status": status,
                }
            )
        return pd.DataFrame(rows)
