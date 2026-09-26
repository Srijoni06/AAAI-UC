"""Tests for eval/checkpoint.py: the low-level save/load/fingerprint primitives.

End-to-end resume behavior (partial checkpoint detected and resumed, --fresh
ignoring one, cleanup on success) is covered in tests/test_run_comparison.py,
against the real eval.run_comparison harness.
"""

from __future__ import annotations

import json

import pytest

from eval.checkpoint import Checkpoint, checkpoint_path, clear_checkpoint, fingerprint, load_checkpoint, save_checkpoint


def test_fingerprint_is_deterministic_and_order_independent():
    a = fingerprint(backend="fake", threshold=-1.0, doc_ids=["d1", "d2"])
    b = fingerprint(threshold=-1.0, backend="fake", doc_ids=["d1", "d2"])
    assert a == b


def test_fingerprint_changes_with_config():
    base = fingerprint(backend="fake", threshold=-1.0, doc_ids=["d1"])
    assert fingerprint(backend="real", threshold=-1.0, doc_ids=["d1"]) != base
    assert fingerprint(backend="fake", threshold=0.0, doc_ids=["d1"]) != base
    assert fingerprint(backend="fake", threshold=-1.0, doc_ids=["d1", "d2"]) != base


def test_save_load_roundtrip(tmp_path):
    path = checkpoint_path(tmp_path)
    ckpt = Checkpoint(config_fingerprint="fp1")
    ckpt.completed_docs.add("doc-a")
    ckpt.pair_specs_by_doc["doc-a"] = [{"item_a_id": "1", "item_b_id": "2"}]
    ckpt.pair_records_by_doc["doc-a"] = [{"doc_id": "doc-a", "outcome": "TP"}]
    ckpt.completed_conditions["null"] = {"doc-a"}
    ckpt.decisions_by_condition["null"] = {"doc-a": {"topic": "t", "correct": True}}

    save_checkpoint(ckpt, path)
    loaded = load_checkpoint(path, "fp1")

    assert loaded is not None
    assert loaded.completed_docs == {"doc-a"}
    assert loaded.pair_specs_by_doc == ckpt.pair_specs_by_doc
    assert loaded.pair_records_by_doc == ckpt.pair_records_by_doc
    assert loaded.completed_conditions == {"null": {"doc-a"}}
    assert loaded.decisions_by_condition == {"null": {"doc-a": {"topic": "t", "correct": True}}}


def test_load_returns_none_when_file_missing(tmp_path):
    assert load_checkpoint(checkpoint_path(tmp_path), "fp1") is None


def test_load_returns_none_on_fingerprint_mismatch(tmp_path):
    path = checkpoint_path(tmp_path)
    save_checkpoint(Checkpoint(config_fingerprint="fp1"), path)
    assert load_checkpoint(path, "fp2") is None
    # the file itself is untouched by a mismatched load - still there, still fp1
    assert json.loads(path.read_text())["config_fingerprint"] == "fp1"


def test_load_returns_none_on_corrupt_file(tmp_path):
    path = checkpoint_path(tmp_path)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_checkpoint(path, "fp1") is None


def test_save_is_atomic_no_leftover_tmp_file(tmp_path):
    path = checkpoint_path(tmp_path)
    save_checkpoint(Checkpoint(config_fingerprint="fp1"), path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_clear_checkpoint_removes_file(tmp_path):
    path = checkpoint_path(tmp_path)
    save_checkpoint(Checkpoint(config_fingerprint="fp1"), path)
    assert path.exists()
    clear_checkpoint(path)
    assert not path.exists()


def test_clear_checkpoint_is_a_noop_when_absent(tmp_path):
    path = checkpoint_path(tmp_path)
    clear_checkpoint(path)  # must not raise
    assert not path.exists()
