"""Checkpoint/resume protocol for ``eval.run_comparison``.

This is layered on top of ``common.cache.LLMCache``, which already makes a
fully-cached rerun near-instant regardless of how expensive the original judge
calls were (empirically: ~40s instead of hours, for a run whose every call was
already cached). This checkpoint is a second, *harness-level* layer: it tracks
which documents have finished detection and which ``(condition, document)``
pairs have finished resolution, and persists their actual results - so a
resumed run skips re-invoking detection/resolution logic for completed work
entirely, instead of relying on the LLM cache alone to make that reprocessing
cheap.

File: ``<out_dir>/checkpoint.json``. Removed automatically when
``run_comparison()`` completes successfully, so a genuinely new run never
mistakes a leftover file for resumable state.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def fingerprint(**parts: object) -> str:
    """Stable hash of the run configuration a checkpoint is only valid for.

    Any of these changing between runs (backend, threshold, agent/judge/
    reconciler model, or the exact set of documents) makes a saved checkpoint
    unsafe to resume from - it would silently merge results computed under a
    different configuration - so a mismatch is treated as "no usable
    checkpoint" rather than an error.
    """
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Checkpoint:
    config_fingerprint: str
    completed_docs: set[str] = field(default_factory=set)
    pair_specs_by_doc: dict[str, list[dict]] = field(default_factory=dict)
    pair_records_by_doc: dict[str, list[dict]] = field(default_factory=dict)
    # condition name -> set of doc_ids whose resolution is fully scored
    completed_conditions: dict[str, set[str]] = field(default_factory=dict)
    # condition name -> {doc_id: decision dict}
    decisions_by_condition: dict[str, dict[str, dict]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config_fingerprint": self.config_fingerprint,
            "completed_docs": sorted(self.completed_docs),
            "pair_specs_by_doc": self.pair_specs_by_doc,
            "pair_records_by_doc": self.pair_records_by_doc,
            "completed_conditions": {k: sorted(v) for k, v in self.completed_conditions.items()},
            "decisions_by_condition": self.decisions_by_condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(
            config_fingerprint=d["config_fingerprint"],
            completed_docs=set(d.get("completed_docs", [])),
            pair_specs_by_doc=dict(d.get("pair_specs_by_doc", {})),
            pair_records_by_doc=dict(d.get("pair_records_by_doc", {})),
            completed_conditions={k: set(v) for k, v in d.get("completed_conditions", {}).items()},
            decisions_by_condition={k: dict(v) for k, v in d.get("decisions_by_condition", {}).items()},
        )


def checkpoint_path(out_dir: "str | Path") -> Path:
    return Path(out_dir) / "checkpoint.json"


def load_checkpoint(path: Path, expected_fingerprint: str) -> Optional[Checkpoint]:
    """Load a checkpoint if present, readable, and valid for this exact config.

    Returns ``None`` (meaning: start fresh) on a missing file, a corrupt file,
    or a fingerprint mismatch - never raises, since any of those just means
    there is nothing safe to resume from.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("config_fingerprint") != expected_fingerprint:
        return None
    return Checkpoint.from_dict(data)


def save_checkpoint(ckpt: Checkpoint, path: Path) -> None:
    """Atomic write (temp file + ``os.replace``) so a crash mid-write can't corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ckpt.to_dict(), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def clear_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()
