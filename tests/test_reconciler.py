"""Offline tests for memory/reconciler.py: classification, deterministic scope check, LLM fallback."""

from __future__ import annotations

import json

import pytest

from memory.reconciler import (
    Classification,
    LLMReconciler,
    Reconciliation,
    _parse_reconciler_response,
    pairwise_scopes_agree,
    resolve_conflict,
)
from memory.store import Conflict, MemoryItem, SourceType, Status, Origin


def _item(
    agent_id: str,
    content: str,
    *,
    source_id: str | None = None,
    evidence_span: str | None = None,
    excerpt_id: str | None = None,
) -> MemoryItem:
    meta: dict = {}
    if source_id:
        meta["source_id"] = source_id
    if excerpt_id:
        meta["excerpt_id"] = excerpt_id
    return MemoryItem(
        agent_id=agent_id,
        topic="test_topic",
        content=content,
        evidence_span=evidence_span,
        metadata=meta,
    )


class FakeReconcilerLLM:
    """Minimal fake for testing: returns a scripted JSON response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []
        self.judge_model = "fake-judge"  # required by LLMReconciler

    def generate(self, prompt, *, system=None, model=None, temperature=0.0):
        self.calls.append(prompt)
        return self.response


# --------------------------------------------------------------------------- #
# Deterministic scope check
# --------------------------------------------------------------------------- #
class TestPairwiseScopesAgree:
    def test_same_source_id_agrees(self):
        a = _item("a", "claim", source_id="doc#e0")
        b = _item("b", "claim", source_id="doc#e0")
        assert pairwise_scopes_agree(a, b) is True

    def test_different_source_ids_differ(self):
        a = _item("a", "claim", source_id="doc#e0")
        b = _item("b", "claim", source_id="doc#e1")
        assert pairwise_scopes_agree(a, b) is False

    def test_empty_scope_returns_false(self):
        a = _item("a", "claim")
        b = _item("b", "claim", source_id="doc#e0")
        assert pairwise_scopes_agree(a, b) is False


# --------------------------------------------------------------------------- #
# Reconciliation parse
# --------------------------------------------------------------------------- #
class TestParseReconcilerResponse:
    def test_json(self):
        r, rat = _parse_reconciler_response('{"classification": "COORDINATION", "rationale": "ok"}')
        assert r is Classification.COORDINATION
        assert rat == "ok"

    def test_keyword_fallback(self):
        r, _ = _parse_reconciler_response("This is clearly a COORDINATION case.")
        assert r is Classification.COORDINATION

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _parse_reconciler_response("no classification here at all")


# --------------------------------------------------------------------------- #
# LLMReconciler classify
# --------------------------------------------------------------------------- #
class TestLLMReconcilerClassify:
    def test_deterministic_path_same_evidence(self):
        """Same evidence -> COORDINATION without LLM call."""
        a = _item("a", "claim A", source_id="doc#e0", evidence_span="span X")
        b = _item("b", "claim B", source_id="doc#e0", evidence_span="span X")
        rec = LLMReconciler(llm=None).classify(Conflict(topic="t", items=[a, b]))
        assert rec.classification is Classification.COORDINATION
        assert "deterministic" in rec.rationale

    def test_llm_path_different_evidence(self):
        """Different evidence -> LLM call; parses scripted response."""
        a = _item("a", "claim A", source_id="doc#e0", evidence_span="span X")
        b = _item("b", "claim B", source_id="doc#e1", evidence_span="span Y")
        fake = FakeReconcilerLLM(json.dumps({"classification": "COORDINATION", "rationale": "both true"}))
        rec = LLMReconciler(llm=fake).classify(Conflict(topic="t", items=[a, b]))
        assert rec.classification is Classification.COORDINATION
        assert len(fake.calls) == 1

    def test_llm_garbage_falls_back_to_credibility(self):
        """Unparseable LLM response -> CREDIBILITY (decisive default)."""
        a = _item("a", "A", source_id="d#e0", evidence_span="X")
        b = _item("b", "B", source_id="d#e1", evidence_span="Y")
        fake = FakeReconcilerLLM("this is not json at all")
        rec = LLMReconciler(llm=fake).classify(Conflict(topic="t", items=[a, b]))
        assert rec.classification is Classification.CREDIBILITY


# --------------------------------------------------------------------------- #
# resolve_conflict helper
# --------------------------------------------------------------------------- #
class TestResolveConflict:
    def _fake_resolver(self, winner_id):
        """Minimal resolver that always picks winner_id."""
        class FakeResolver:
            name = "fake"
            def resolve(self, conflict):
                winner = [i for i in conflict.items if i.id == winner_id][0]
                losers = [i for i in conflict.items if i.id != winner_id]
                from baselines.base import Resolution
                return Resolution.single_winner(
                    topic=conflict.topic,
                    strategy=self.name,
                    winner_id=winner.id,
                    superseded_ids=[i.id for i in losers],
                )
        return FakeResolver()

    def test_coordination_returns_none_resolution(self):
        """Coordination -> (reconciliation, None)."""
        a = _item("a", "A", source_id="d#e0", evidence_span="X")
        b = _item("b", "B", source_id="d#e0", evidence_span="X")
        rec, res = resolve_conflict(Conflict(topic="t", items=[a, b]), resolver=None)
        assert rec.is_coordination()
        assert res is None

    def test_credibility_returns_resolution(self):
        """CREDIBILITY -> (reconciliation, resolution)."""
        a = _item("a", "A", source_id="d#e0", evidence_span="X")
        b = _item("b", "B", source_id="d#e1", evidence_span="Y")
        fake_llm = FakeReconcilerLLM(json.dumps({"classification": "CREDIBILITY", "rationale": "cred"}))
        resolver = self._fake_resolver(a.id)
        rec, res = resolve_conflict(
            Conflict(topic="t", items=[a, b]),
            resolver=resolver,
            reconciler=LLMReconciler(llm=fake_llm),
        )
        assert rec.is_credibility()
        assert res is not None
        assert res.winner_id == a.id
