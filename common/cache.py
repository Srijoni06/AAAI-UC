"""On-disk response cache for LLM calls.

Keyed by a SHA-256 of (backend, model, system, temperature, prompt) so a cache
hit is only reused for a byte-identical request. Stored as a single JSON object
at ``.cache/llm_cache.json`` (gitignored).

An optional ``sample_id`` widens the key: with no ``sample_id`` the cache
behaves as before (identical request -> one cached answer forever), but passing
``sample_id=0, 1, 2, ...`` lets the *same* prompt be re-sampled and each draw
cached under its own key. Repeated-sampling / self-consistency logic needs this;
without it, asking the same question twice would always return the first answer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

SampleId = Union[int, str]

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
        *,
        backend: str,
        model: str,
        system: str,
        temperature: float,
        prompt: str,
        sample_id: Optional[SampleId] = None,
    ) -> str:
        payload = {
            "backend": backend,
            "model": model,
            "system": system,
            "temperature": temperature,
            "prompt": prompt,
        }
        # Only widen the key when a sample is explicitly requested, so existing
        # cache entries (and the default single-answer behaviour) are untouched.
        if sample_id is not None:
            payload["sample_id"] = sample_id
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
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
