"""Tiny .env loader (no python-dotenv dependency).

Reads ``KEY=VALUE`` lines from a ``.env`` file at the repo root into
``os.environ``. By default values in ``.env`` override what is already in the
environment (handy when a shell has a stale placeholder); pass
``override=False`` to keep existing values.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None, *, override: bool = True) -> None:
    env_path = Path(path) if path is not None else _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
