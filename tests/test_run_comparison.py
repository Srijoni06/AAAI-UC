"""End-to-end offline tests for eval/run_comparison.py.

All tests use the fake backend (no network, deterministic, <10s).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from domain.seed_conflicts import COEXIST, SEED_CONFLICTS, SEEDS_BY_ID, ConflictType, Difficulty, Excerpt, SeedConflict
from eval.fake_backend import FakeEmbedder, RuleJudgeLLM, ScopedFakeLLM
from eval.run_comparison import (
    _cluster_correctness,
    _condition_result_from_decisions,
    _detect_one_doc,
    run_comparison,
)
from memory.detector import ConflictDetector
from memory.store import MemoryItem


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
# gold_positive: per-pair coexist_pairs, not just the seed-level COEXIST flag
# --------------------------------------------------------------------------- #
class TestGoldPositiveCoexistPairs:
    def _item(self, agent_id: str, excerpt_id: str, content: str) -> MemoryItem:
        return MemoryItem(
            agent_id=agent_id,
            topic="t",
            content=content,
            metadata={"excerpt_id": excerpt_id, "question": "q?"},
        )

    def _records_by_pair(self, records: list[dict]) -> dict[frozenset, dict]:
        return {
            frozenset([r["claim_1"]["excerpt_id"], r["claim_2"]["excerpt_id"]]): r
            for r in records
        }

    def test_coexist_pair_excluded_others_still_gold_positive(self):
        """3 excerpts, one coexist_pairs exception between e0/e1 only: that
        pair must NOT be gold_positive, but e0/e2 and e1/e2 (genuinely
        cross-excerpt, no exception) must still be gold_positive."""
        seed = SeedConflict(
            doc_id="doc-synthetic",
            title="t",
            topic="t",
            question="q?",
            excerpts=(
                Excerpt("e0", "s0", "text zero"),
                Excerpt("e1", "s1", "text one"),
                Excerpt("e2", "s2", "text two"),
            ),
            gold_excerpt_id="e2",
            gold_answer="two",
            conflict_type=ConflictType.FACTUAL,
            difficulty=Difficulty.MODERATE,
            coexist_pairs=frozenset({frozenset({"e0", "e1"})}),
        )
        items = [
            self._item("agent_A", "e0", "Claim A"),
            self._item("agent_B", "e1", "Claim B"),
            self._item("agent_C", "e2", "Claim C"),
        ]
        detector = ConflictDetector(llm=RuleJudgeLLM(), embedder=FakeEmbedder(), similarity_threshold=-1.0)
        _specs, records = _detect_one_doc(seed, items, detector)

        by_pair = self._records_by_pair(records)
        assert by_pair[frozenset({"e0", "e1"})]["gold_positive"] is False
        assert by_pair[frozenset({"e0", "e2"})]["gold_positive"] is True
        assert by_pair[frozenset({"e1", "e2"})]["gold_positive"] is True

    def test_doc_languages_all_pairs_still_coexist(self):
        """doc-languages' seed-level COEXIST behaviour (every pair coexists)
        keeps working now that it's expressed via coexist_pairs."""
        seed = SEEDS_BY_ID["doc-languages"]
        assert seed.gold_excerpt_id == COEXIST  # unchanged
        items = [
            self._item("agent_A", "abstract", "Claim about 90 languages"),
            self._item("agent_B", "eval", "Claim about 46 languages"),
        ]
        detector = ConflictDetector(llm=RuleJudgeLLM(), embedder=FakeEmbedder(), similarity_threshold=-1.0)
        _specs, records = _detect_one_doc(seed, items, detector)

        assert len(records) == 1
        assert records[0]["gold_positive"] is False


# --------------------------------------------------------------------------- #
# _cluster_correctness: per-cluster correctness map fed to PeerMemory.update(),
# so correlation tracking can be scoped to each pair's own answer cluster
# instead of the single topic-level resolver-decision flag.
# --------------------------------------------------------------------------- #
class TestClusterCorrectness:
    def _item(self, agent_id: str, excerpt_id: str) -> MemoryItem:
        return MemoryItem(
            agent_id=agent_id,
            topic="t",
            content=f"claim by {agent_id}",
            metadata={"source_id": f"doc#{excerpt_id}", "excerpt_id": excerpt_id},
        )

    def test_maps_each_cluster_independently_of_resolver_pick(self):
        items = [
            self._item("agent_A", "eAnchor"),
            self._item("agent_C", "eAnchor"),
            self._item("agent_B", "eOther"),
        ]
        result = _cluster_correctness(items, gold="eOther")
        assert result == {"doc#eAnchor": False, "doc#eOther": True}

    def test_coexist_gold_marks_every_cluster_correct(self):
        items = [self._item("agent_A", "abstract"), self._item("agent_B", "eval")]
        result = _cluster_correctness(items, gold=COEXIST)
        assert result == {"doc#abstract": True, "doc#eval": True}


# --------------------------------------------------------------------------- #
# resolution-only vs. end-to-end accuracy: reporting BOTH explicitly instead
# of silently excluding documents where detection found nothing to resolve.
# --------------------------------------------------------------------------- #
class TestEndToEndAccuracy:
    def test_end_to_end_lower_than_resolution_only_when_detection_misses(self):
        """2 decided-correct docs out of 3 total: resolution-only accuracy
        (2/2 decisive) must be higher than end-to-end accuracy (2/3 total),
        since the undetected 3rd doc counts as incorrect end-to-end but is
        simply absent from the resolution-only basis."""
        decisions = [
            {
                "topic": "t1", "doc_id": "d1", "conflict_type": "factual", "difficulty": "obvious",
                "gold_excerpt": "e0", "classification": "CREDIBILITY", "outcome": "decisive",
                "correct": True, "rationale": "r",
            },
            {
                "topic": "t2", "doc_id": "d2", "conflict_type": "factual", "difficulty": "obvious",
                "gold_excerpt": "e0", "classification": "CREDIBILITY", "outcome": "decisive",
                "correct": True, "rationale": "r",
            },
        ]
        cr = _condition_result_from_decisions("last_write_wins", decisions, total_docs=3)

        assert cr.accuracy == 1.0  # 2/2 decisive, both correct
        assert cr.end_to_end_accuracy == pytest.approx(2 / 3)  # 2/3 total docs
        assert cr.end_to_end_accuracy < cr.accuracy

        d = cr.to_dict()
        assert d["accuracy"] == 1.0
        assert d["end_to_end_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
        assert d["total_docs"] == 3
        assert d["decisive"] == 2

    def test_equal_when_every_document_was_decisive(self):
        """No undetected documents (total_docs == decisive count): both
        metrics agree - end-to-end accuracy is only ever lower, never higher."""
        decisions = [
            {
                "topic": "t1", "doc_id": "d1", "conflict_type": "factual", "difficulty": "obvious",
                "gold_excerpt": "e0", "classification": "CREDIBILITY", "outcome": "decisive",
                "correct": True, "rationale": "r",
            },
        ]
        cr = _condition_result_from_decisions("last_write_wins", decisions, total_docs=1)
        assert cr.accuracy == cr.end_to_end_accuracy == 1.0

    def test_full_pipeline_end_to_end_accounts_for_undetected_doc(self, tmp_path):
        """Through the real run_comparison() pipeline: a seed whose two
        excerpts are IDENTICAL text never produces a detected contradiction
        (ScopedFakeLLM: equal claims -> ENTAILMENT), so it's invisible to
        resolution-only accuracy but must still lower end-to-end accuracy."""
        undetectable = SeedConflict(
            doc_id="doc-undetectable",
            title="t",
            topic="undetectable_topic",
            question="q?",
            excerpts=(
                Excerpt("e0", "s0", "identical text"),
                Excerpt("e1", "s1", "identical text"),
            ),
            gold_excerpt_id="e0",
            gold_answer="a",
            conflict_type=ConflictType.FACTUAL,
            difficulty=Difficulty.MODERATE,
        )
        seeds = [SEEDS_BY_ID["doc-benchmark"], undetectable]
        report = run_comparison(seeds, backend="fake", threshold=0.0, out_dir=str(tmp_path))

        lww = report["conditions"]["last_write_wins"]
        assert lww["total_docs"] == 2
        assert lww["decisive"] == 1  # only doc-benchmark's conflict was detected
        assert lww["end_to_end_accuracy"] < lww["accuracy"]
        assert lww["end_to_end_accuracy"] == pytest.approx(lww["correct"] / 2, abs=1e-4)


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


def test_judge_llm_is_used_for_detection_and_recorded(tmp_path):
    """Detector judge calls go to ``judge_llm``, not the agent client."""

    class Spy(ScopedFakeLLM):
        judge_model = "spy-judge"

        def __init__(self):
            super().__init__()
            self.judge_prompts = 0

        def _raw_generate(self, prompt, *, system, temperature, model):
            if "Classify the relationship" in prompt:
                self.judge_prompts += 1
            return super()._raw_generate(prompt, system=system, temperature=temperature, model=model)

    agent, judge = Spy(), Spy()
    report = run_comparison(
        _make_seeds(), backend="fake", out_dir=str(tmp_path), llm_client=agent, judge_llm=judge
    )
    # both-orders judging (classify_pair): 2 judge calls per pair
    assert judge.judge_prompts == 4 * 10 * 2 and agent.judge_prompts == 0
    assert report["config"]["judge_llm"] == "scoped_fake:spy-judge"
    assert "Judge LLM:" in (tmp_path / "summary.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# per_type_accuracy() (was silently always {} - decision dicts key the gold
# label as "gold_excerpt", not "gold")
# --------------------------------------------------------------------------- #
class TestPerTypeAccuracy:
    def test_per_type_accuracy_is_not_empty(self, tmp_path):
        report = run_comparison(_make_seeds(), backend="fake", out_dir=str(tmp_path))
        for cond in report["conditions"].values():
            assert cond["per_type_accuracy"], f"expected non-empty per_type_accuracy in {cond}"

    def test_per_type_accuracy_matches_conflict_types_in_decisions(self, tmp_path):
        report = run_comparison(_make_seeds(), backend="fake", out_dir=str(tmp_path))
        lww = report["conditions"]["last_write_wins"]
        types_in_decisions = {d["conflict_type"] for d in lww["decisions"]}
        assert set(lww["per_type_accuracy"]) == types_in_decisions

    def test_per_type_accuracy_values_are_fractions_of_correct(self, tmp_path):
        report = run_comparison(_make_seeds(), backend="fake", out_dir=str(tmp_path))
        lww = report["conditions"]["last_write_wins"]
        by_type = defaultdict(list)
        for d in lww["decisions"]:
            by_type[d["conflict_type"]].append(d["correct"])
        expected = {t: sum(v) / len(v) for t, v in by_type.items()}
        for t, acc in expected.items():
            assert lww["per_type_accuracy"][t] == round(acc, 4)


# --------------------------------------------------------------------------- #
# Checkpoint/resume protocol (eval/checkpoint.py, wired into run_comparison())
#
# _make_seeds() gives 4 docs x 5 agents -> C(5,2)=10 pairs/doc, and both-orders
# judging (memory.detector.classify_pair) doubles that to 20 detector judge
# calls per doc, 80 for a full fresh run. These fake-backend judges use
# cache=None (ScopedFakeLLM never touches the LLM cache), so every count below
# is a real, uncached call - checkpoint savings, not cache savings.
# --------------------------------------------------------------------------- #
class _CountingDetectorJudge(ScopedFakeLLM):
    """Counts detector (not reconciler) judge calls: identifies them by the
    prompt text unique to memory.detector.classify_pair's prompt."""

    def __init__(self):
        super().__init__()
        self.judge_calls = 0

    def _raw_generate(self, prompt, *, system, temperature, model):
        if "Classify the relationship" in prompt:
            self.judge_calls += 1
        return super()._raw_generate(prompt, system=system, temperature=temperature, model=model)


class _CrashingDetectorJudge(_CountingDetectorJudge):
    """Raises once its detector-call counter exceeds ``crash_after`` -
    simulates a process crash partway through a run."""

    def __init__(self, crash_after: int):
        super().__init__()
        self.crash_after = crash_after

    def _raw_generate(self, prompt, *, system, temperature, model):
        if "Classify the relationship" in prompt and self.judge_calls >= self.crash_after:
            self.judge_calls += 1
            raise RuntimeError("simulated crash")
        return super()._raw_generate(prompt, system=system, temperature=temperature, model=model)


class TestCheckpointResume:
    def test_partial_checkpoint_is_detected_and_resumed(self, tmp_path):
        seeds = _make_seeds()
        out = tmp_path / "out"

        # 1. Control: an uninterrupted run, to know the correct final numbers
        #    and how many detector calls doing all 4 docs from scratch costs.
        control_judge = _CountingDetectorJudge()
        control = run_comparison(
            seeds, backend="fake", out_dir=str(out / "control"),
            judge_llm=control_judge, reconcile_llm=ScopedFakeLLM(),
        )
        assert control_judge.judge_calls == 4 * 10 * 2
        assert not (out / "control" / "checkpoint.json").exists()

        # 2. Crash partway through doc 3's detection - after exactly 2 whole
        #    docs (2 * 10 pairs * 2 orders = 40 calls) have been checkpointed.
        run_dir = out / "resumed"
        crashing = _CrashingDetectorJudge(crash_after=40)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_comparison(
                seeds, backend="fake", out_dir=str(run_dir),
                judge_llm=crashing, reconcile_llm=ScopedFakeLLM(),
            )
        ckpt_file = run_dir / "checkpoint.json"
        assert ckpt_file.exists()
        saved = json.loads(ckpt_file.read_text(encoding="utf-8"))
        assert len(saved["completed_docs"]) == 2

        # 3. Resume with a fresh (non-crashing) judge: must complete and match
        #    the control run's results exactly.
        resumed_judge = _CountingDetectorJudge()
        result = run_comparison(
            seeds, backend="fake", out_dir=str(run_dir),
            judge_llm=resumed_judge, reconcile_llm=ScopedFakeLLM(), resume=True,
        )

        assert result["detection"] == control["detection"]
        for cond in result["conditions"]:
            assert result["conditions"][cond]["accuracy"] == control["conditions"][cond]["accuracy"]
            assert result["conditions"][cond]["correct"] == control["conditions"][cond]["correct"]
            assert result["conditions"][cond]["decisive"] == control["conditions"][cond]["decisive"]

        # genuinely skipped work: far fewer detector calls than a full run needs
        assert resumed_judge.judge_calls < control_judge.judge_calls
        assert resumed_judge.judge_calls == 2 * 10 * 2  # only docs 3 and 4 remained

        # 4. Checkpoint cleared on successful completion.
        assert not ckpt_file.exists()

    def test_fresh_flag_reprocesses_a_completed_doc(self, tmp_path):
        seeds = _make_seeds()
        out = tmp_path / "out"

        # Get a checkpoint with 1 genuinely completed doc (crash after doc 1).
        crashing = _CrashingDetectorJudge(crash_after=20)
        with pytest.raises(RuntimeError):
            run_comparison(
                seeds, backend="fake", out_dir=str(out),
                judge_llm=crashing, reconcile_llm=ScopedFakeLLM(),
            )
        ckpt_file = out / "checkpoint.json"
        assert len(json.loads(ckpt_file.read_text(encoding="utf-8"))["completed_docs"]) == 1

        # --fresh (resume=False): must ignore that checkpoint and redo every doc.
        fresh_judge = _CountingDetectorJudge()
        run_comparison(
            seeds, backend="fake", out_dir=str(out),
            judge_llm=fresh_judge, reconcile_llm=ScopedFakeLLM(), resume=False,
        )
        assert fresh_judge.judge_calls == 4 * 10 * 2  # nothing skipped
        assert not ckpt_file.exists()  # cleared on this run's successful completion

    def test_checkpoint_cleared_on_successful_completion(self, tmp_path):
        seeds = _make_seeds()
        out = tmp_path / "out"
        run_comparison(seeds, backend="fake", out_dir=str(out))
        assert not (out / "checkpoint.json").exists()

    def test_no_checkpoint_no_resume_flag_behaves_like_a_fresh_run(self, tmp_path):
        """resume=True (the default) with nothing to resume from is just a normal run."""
        seeds = _make_seeds()
        out = tmp_path / "out"
        judge = _CountingDetectorJudge()
        run_comparison(
            seeds, backend="fake", out_dir=str(out), judge_llm=judge, reconcile_llm=ScopedFakeLLM(), resume=True
        )
        assert judge.judge_calls == 4 * 10 * 2
