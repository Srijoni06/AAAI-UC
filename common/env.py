"""Tiny .env loader (no python-dotenv dependency).

Reads ``KEY=VALUE`` lines from a ``.env`` file at the repo root into
``os.environ``. By default values in ``.env`` override what is already in the
environment (handy when a shell has a stale placeholder); pass
``override=False`` to keep existing values.

``load_dotenv`` returns a report of what it did so callers can show where a
config value actually came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DotenvReport:
    path: Path
    exists: bool
    override: bool
    # key -> value parsed from the .env file (regardless of whether applied)
    parsed: dict[str, str] = field(default_factory=dict)
    # keys whose os.environ value was actually set/changed by this call
    applied: list[str] = field(default_factory=list)
    # keys present in .env but left untouched because a real env var already
    # existed and override=False
    skipped: list[str] = field(default_factory=list)
    # key -> value it held in the environment before .env replaced it
    overridden: dict[str, str] = field(default_factory=dict)

    def source_of(self, key: str) -> str:
        """Where the effective value of ``key`` came from."""
        if key in self.overridden:
            return (
                f".env ({self.path}), overriding shell value "
                f"{self.overridden[key]!r}"
            )
        if key in self.applied:
            return f".env ({self.path})"
        if key in self.parsed and key in self.skipped:
            return "shell environment (.env present but not overriding)"
        if key in os.environ:
            return "shell environment"
        return "unset (using code default)"


def load_dotenv(path: str | Path | None = None, *, override: bool = True) -> DotenvReport:
    env_path = Path(path) if path is not None else _REPO_ROOT / ".env"
    report = DotenvReport(path=env_path, exists=env_path.exists(), override=override)
    if not report.exists:
        return report
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        report.parsed[key] = value
        if override or key not in os.environ:
            prev = os.environ.get(key)
            if prev is not None and prev != value:
                report.overridden[key] = prev
            os.environ[key] = value
            report.applied.append(key)
        else:
            report.skipped.append(key)
    return report
