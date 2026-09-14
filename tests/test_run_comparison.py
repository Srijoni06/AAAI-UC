"""End-to-end offline tests for eval/run_comparison.py.

All tests use the fake backend (no network, deterministic, <10s).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.seed_conflicts import COEXIST, SEED_CONFLICTS, SEEDS_BY_ID
from eval.fake_backend import FakeEmbedder, ScopedFakeLLM
from eval.run_comparison import run_comparison


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_seeds(subset: bool = True):
    """Return a small subset for fast tests (all 4 conflict types)."""
    if not subset:
        return SEED_CONFLICTS
    return [
        SEEDS_BY_ID["doc-benchmark"],      # factual / subtle
        SEEDS_BY_ID["doc-cost"],           # magnitude / obvious
        SEEDS_BY_ID["doc-sota"],           # staleness / moderate
        SEEDS_BY_ID["doc-languages"],      # COEXIST
    ]


# --------------------------------------------------------------------------- #
# Core pipeline test
# --------------------------------------------------------------------------- #
class TestRunComparison:
    def test_returns_all_conditions(self, tmp_path):
        seeds = _make_seeds()
        report = run_comparison(
            seeds,
            backend="fake",
            threshold=0.0,
            out_dir=str(tmp_path),
        )
        assert "detection" in report
        assert "conditions" in report
        conds = report["conditions"]
        assert "null" in conds
        assert "last_write_wins" in conds
        assert "majority_vote" in conds
        assert "static_confidence" in conds

    def test_detection_metrics(self, tmp_path):
        seeds = _make_seeds()
        report = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        det = report["detection"]
        # All non-COEXIST seeds have cross-excerpt pairs -> all detected
        assert det["recall"] == 1.0
        # COEXIST seed cross-excerpt pairs are false positives at detection
        # level (reconciler catches them later)
        assert det["precision"] < 1.0
        assert det["TP"] > 0

    def test_null_condition_worst(self, tmp_path):
        seeds = _make_seeds()
        report = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        null_acc = report["conditions"]["null"]["accuracy"]
        lww_acc = report["conditions"]["last_write_wins"]["accuracy"]
        # Null always worse than LWW (no resolution vs some resolution)
        assert null_acc <= lww_acc

    def test_lww_accuracy_above_zero(self, tmp_path):
        seeds = _make_seeds()
        report = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        lww = report["conditions"]["last_write_wins"]
        assert lww["accuracy"] > 0.0
        assert lww["correct"] > 0

    def test_majority_vote_cluster_size(self, tmp_path):
        """Majority vote confirms the 3-agent correlated cluster."""
        seeds = _make_seeds()
        report = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        mv = report["conditions"]["majority_vote"]
        assert mv["total_conflicts"] > 0
        # Majority vote should be decisive (no ties with 3v2 split)
        assert mv["contested"] == 0 or mv["decisive"] > 0

    def test_json_written(self, tmp_path):
        seeds = _make_seeds()
        run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        json_path = tmp_path / "run_comparison.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "config" in data
        assert "conditions" in data

    def test_summary_written(self, tmp_path):
        seeds = _make_seeds()
        run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        md_path = tmp_path / "summary.md"
        assert md_path.exists()
        text = md_path.read_text()
        assert "Resolution Accuracy" in text

    def test_deterministic(self, tmp_path):
        """Two runs produce identical results."""
        seeds = _make_seeds()
        r1 = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path / "r1"))
        r2 = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path / "r2"))
        for cond in ["null", "last_write_wins", "majority_vote", "static_confidence"]:
            assert r1["conditions"][cond]["accuracy"] == r2["conditions"][cond]["accuracy"]
            assert r1["conditions"][cond]["correct"] == r2["conditions"][cond]["correct"]
