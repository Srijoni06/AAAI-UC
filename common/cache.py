"""On-disk response cache for LLM calls.

Keyed by a SHA-256 of (backend, model, system, temperature, prompt) so a cache
hit is only reused for a byte-identical request. Stored as a single JSON object
at ``.cache/llm_cache.json`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / ".cache" / "llm_cache.json"


class LLMCache:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def key(
        *, backend: str, model: str, system: str, temperature: float, prompt: str
    ) -> str:
        blob = json.dumps(
            {
                "backend": backend,
                "model": model,
                "system": system,
                "temperature": temperature,
                "prompt": prompt,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        return entry["response"] if entry else None

    def set(self, key: str, response: str, meta: dict) -> None:
        self._data[key] = {"response": response, **meta}
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._data)
