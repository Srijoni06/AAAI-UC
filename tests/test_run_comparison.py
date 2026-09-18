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

    def test_null_resolves_fewer_conflicts_than_lww(self, tmp_path):
        seeds = _make_seeds()
        report = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))
        null = report["conditions"]["null"]
        lww = report["conditions"]["last_write_wins"]
        # Null applies no resolver, so every genuine CREDIBILITY conflict stays
        # CONTESTED; only COORDINATION cases (kept as-is) are ever "decisive".
        # NOTE: comparing *accuracy* (null_acc <= lww_acc) is not a safe
        # invariant here - null's accuracy is computed over a tiny decisive
        # subset (often just 1 trivially-correct COORDINATION case), so it can
        # exceed LWW's accuracy over its full, harder set once the anchor
        # excerpt rotates per seed instead of always landing on excerpt 0.
        assert null["contested"] > 0
        assert lww["contested"] == 0
        assert null["decisive"] < lww["decisive"]

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


# --------------------------------------------------------------------------- #
# Pair-level detection output (results/detection_pairs.json)
# --------------------------------------------------------------------------- #
class TestDetectionPairsOutput:
    def _run(self, tmp_path, **kw):
        report = run_comparison(_make_seeds(), backend="fake", out_dir=str(tmp_path), **kw)
        data = json.loads((tmp_path / "detection_pairs.json").read_text(encoding="utf-8"))
        return report, data

    def test_default_threshold_sends_every_pair_to_the_judge(self):
        import inspect

        # cosine spans [-1, 1]; 0.0 silently dropped negative-similarity pairs
        assert inspect.signature(run_comparison).parameters["threshold"].default == -1.0

    def test_record_outcomes_match_reported_metrics(self, tmp_path):
        report, data = self._run(tmp_path)
        det = report["detection"]
        by_outcome = {o: sum(p["outcome"] == o for p in data["pairs"]) for o in ("TP", "FP", "FN", "TN")}
        assert (by_outcome["TP"], by_outcome["FP"], by_outcome["FN"]) == (det["TP"], det["FP"], det["FN"])
        assert data["counts"] == det

    def test_every_same_topic_pair_is_recorded_with_claim_text(self, tmp_path):
        _, data = self._run(tmp_path)
        assert len(data["pairs"]) == 4 * 10  # 4 docs x C(5 agents, 2)
        for p in data["pairs"]:
            assert p["claim_1"]["text"] and p["claim_2"]["text"]
            assert p["claim_1"]["agent_id"] != p["claim_2"]["agent_id"]
            assert p["doc_id"] and p["conflict_type"] and p["difficulty"]
            assert p["judged"] is True and p["verdict"] in {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}

    def test_records_use_deterministic_claim_order(self, tmp_path):
        _, data = self._run(tmp_path)
        for p in data["pairs"]:
            k1 = (p["claim_1"]["agent_id"], p["claim_1"]["text"])
            k2 = (p["claim_2"]["agent_id"], p["claim_2"]["text"])
            assert k1 <= k2

    def test_pairs_below_stage1_threshold_are_recorded_as_never_judged(self, tmp_path):
        # cosine can't exceed 1.0, so a threshold of 2.0 drops every pair at Stage 1
        report, data = self._run(tmp_path, threshold=2.0)
        fns = [p for p in data["pairs"] if p["outcome"] == "FN"]
        assert fns and report["detection"]["FN"] == len(fns)
        assert all(p["judged"] is False and p["verdict"] is None for p in data["pairs"])
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "never judged" in summary

    def test_summary_lists_missed_pairs_with_claim_text(self, tmp_path):
        _, data = self._run(tmp_path, threshold=2.0)
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "Missed contradictions" in summary
        fn = next(p for p in data["pairs"] if p["outcome"] == "FN")
        assert fn["claim_1"]["text"] in summary and fn["claim_2"]["text"] in summary
