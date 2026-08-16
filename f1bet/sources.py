"""Retrying, secret-safe JSON source adapters with raw snapshot persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import requests


class SourceError(RuntimeError):
    pass


@dataclass(slots=True)
class JsonClient:
    base_url: str
    timeout_seconds: float = 20.0
    max_attempts: int = 3
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("timeout_seconds and max_attempts must be positive")
        self.session = self.session or requests.Session()

    def get(self, endpoint: str, params: Mapping[str, Any] | None = None) -> Any:
        assert self.session is not None
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.session.get(url, params=dict(params or {}), timeout=self.timeout_seconds)
                if response.status_code == 429:
                    delay = min(float(response.headers.get("Retry-After", 1)), 5.0)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(2**attempt, 4))
        raise SourceError(f"source request failed after {self.max_attempts} attempts") from error


def persist_raw_snapshot(
    payload: Any,
    destination: str | Path,
    *,
    source: str,
    request_metadata: Mapping[str, Any] | None = None,
    captured_at: datetime | None = None,
) -> Path:
    if not str(source).strip():
        raise ValueError("source is required")
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("captured_at must be timezone-aware")
    sanitized_request = _sanitize_metadata(dict(request_metadata or {}))
    content = {
        "source": source,
        "request": sanitized_request,
        "payload": payload,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    path = Path(destination)
    if (path.exists() and path.is_dir()) or not path.suffix:
        path = path / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "source": source,
        "captured_at": captured.astimezone(timezone.utc).isoformat(),
        "content_sha256": digest,
        "request": sanitized_request,
        "payload": payload,
    }
    serialized = json.dumps(envelope, indent=2, sort_keys=True, default=str)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") != digest:
            raise FileExistsError(f"immutable raw snapshot already exists: {path}") from None
    return path


def _sanitize_metadata(value: Any, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    secret_tokens = ("api_key", "apikey", "token", "secret", "password", "authorization")
    if any(token in normalized for token in secret_tokens):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(child_key): _sanitize_metadata(child_value, str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata(item, key) for item in value]
    return value


class OpenF1Client(JsonClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://api.openf1.org/v1", **kwargs)

    def laps(self, session_key: int, driver_number: int | None = None) -> Any:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self.get("laps", params)


class JolpicaClient(JsonClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://api.jolpi.ca/ergast/f1", **kwargs)

    def results(self, season: int, round_number: int) -> Any:
        return self.get(f"{season}/{round_number}/results.json")


class OddsApiClient(JsonClient):
    """The Odds API v4 adapter; credentials are read only from the environment."""

    def __init__(self, api_key_env: str = "THE_ODDS_API_KEY", **kwargs: Any) -> None:
        super().__init__("https://api.the-odds-api.com/v4", **kwargs)
        self.api_key_env = api_key_env

    def _key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise SourceError(f"missing required environment variable {self.api_key_env}")
        return value

    def sports(self) -> Any:
        return self.get("sports", {"apiKey": self._key()})

    def odds(self, sport_key: str, *, regions: str = "us", markets: str = "h2h") -> Any:
        return self.get(
            f"sports/{sport_key}/odds",
            {
                "apiKey": self._key(),
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
