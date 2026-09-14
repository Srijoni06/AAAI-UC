"""Offline tests for reliability/peer_memory.py and reliability/resolver.py."""

from __future__ import annotations

import pytest

from baselines.base import Resolution
from memory.store import Conflict, MemoryItem, Origin, SourceType, Status
from reliability.peer_memory import PeerMemory, _pair_key
from reliability.resolver import ReliabilityResolver


def _item(
    agent_id: str,
    content: str,
    *,
    source_id: str | None = None,
    evidence_span: str | None = None,
    status: Status = Status.PROPOSED,
    timestamp: float = 0.0,
) -> MemoryItem:
    meta: dict = {}
    if source_id:
        meta["source_id"] = source_id
    return MemoryItem(
        agent_id=agent_id,
        topic="test",
        content=content,
        evidence_span=evidence_span,
        status=status,
        timestamp=timestamp,
        metadata=meta,
    )


# --------------------------------------------------------------------------- #
# PeerMemory
# --------------------------------------------------------------------------- #
class TestPeerMemory:
    def test_initial_competence(self):
        pm = PeerMemory()
        assert pm.get_competence("agent_A") == 0.5

    def test_initial_correlation(self):
        pm = PeerMemory()
        assert pm.get_correlation("agent_A", "agent_B") == 0.0

    def test_self_correlation_is_zero(self):
        pm = PeerMemory()
        assert pm.get_correlation("agent_A", "agent_A") == 0.0

    def test_competence_updates_on_correct(self):
        pm = PeerMemory()
        items = [_item("a", "A", source_id="d#e0", status=Status.CONFIRMED)]
        pm.update(items, correct=True)
        assert pm.get_competence("a") > 0.5

    def test_competence_updates_on_incorrect(self):
        """When resolver is wrong, superseded agent gets boosted (was actually right)."""
        pm = PeerMemory()
        items = [_item("a", "A", source_id="d#e0", status=Status.SUPERSEDED)]
        pm.update(items, correct=False)
        assert pm.get_competence("a") > 0.5  # boosted: they were right

    def test_competence_penalizes_confirmed_when_incorrect(self):
        """When resolver is wrong, confirmed agent gets penalised (was actually wrong)."""
        pm = PeerMemory()
        items = [_item("a", "A", source_id="d#e0", status=Status.CONFIRMED)]
        pm.update(items, correct=False)
        assert pm.get_competence("a") < 0.5  # penalised: they were wrong

    def test_competence_bounded(self):
        pm = PeerMemory()
        items = [_item("a", "A", status=Status.CONFIRMED)]
        for _ in range(100):
            pm.update(items, correct=True)
        assert pm.get_competence("a") <= 0.95

    def test_correlation_agreement_positive(self):
        """Agents on the same answer key → positive correlation."""
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0", status=Status.CONFIRMED),
            _item("b", "A", source_id="d#e0", status=Status.CONFIRMED),
            _item("c", "B", source_id="d#e1", status=Status.SUPERSEDED),
        ]
        pm.update(items, correct=True)
        assert pm.get_correlation("a", "b") > 0  # agreed
        assert pm.get_correlation("a", "c") < 0  # disagreed

    def test_correlation_bounded(self):
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0"),
            _item("b", "A", source_id="d#e0"),
        ]
        for _ in range(100):
            pm.update(items, correct=True)
        assert abs(pm.get_correlation("a", "b")) <= 0.95

    def test_pair_key_is_canonical(self):
        assert _pair_key("a", "b") == _pair_key("b", "a")

    def test_n_decisions_increments(self):
        pm = PeerMemory()
        assert pm.n_decisions == 0
        pm.update([_item("a", "A")], correct=True)
        assert pm.n_decisions == 1


# --------------------------------------------------------------------------- #
# ReliabilityResolver
# --------------------------------------------------------------------------- #
class TestReliabilityResolver:
    def test_fresh_resolver_picks_by_provenance(self):
        """Without any history, competence is uniform (0.5) → base scores
        determine the winner (same as static confidence)."""
        pm = PeerMemory()
        rr = ReliabilityResolver(peer_memory=pm)
        items = [
            _item("a", "A", source_id="d#e0", timestamp=1.0),
            _item("b", "B", source_id="d#e1", timestamp=2.0),
        ]
        res = rr.resolve(Conflict(topic="t", items=items))
        assert res.winner_id is not None
        assert len(res.confirmed_ids) == 1

    def test_competence_influences_winner(self):
        """Agent with higher competence should win over equal-provenance items."""
        pm = PeerMemory()
        # Manually boost agent a's competence
        pm.competence["a"] = 0.9
        pm.competence["b"] = 0.1
        rr = ReliabilityResolver(peer_memory=pm, provenance_weight=0.1, competence_weight=3.0)
        items = [
            _item("a", "A", source_id="d#e0", timestamp=1.0),
            _item("b", "B", source_id="d#e1", timestamp=2.0),
        ]
        res = rr.resolve(Conflict(topic="t", items=items))
        winner = next(it for it in items if it.id == res.winner_id)
        assert winner.agent_id == "a"

    def test_correlation_discounts_correlated_group(self):
        """Three correlated agents with high mutual correlation should have
        their combined score discounted vs two independent agents."""
        pm = PeerMemory()
        # Set high correlation within the correlated group
        pm.correlation[_pair_key("a1", "a2")] = 0.8
        pm.correlation[_pair_key("a1", "a3")] = 0.8
        pm.correlation[_pair_key("a2", "a3")] = 0.8
        # Independent agents have no correlation
        rr = ReliabilityResolver(
            peer_memory=pm,
            provenance_weight=1.0,
            competence_weight=0.0,  # uniform competence
            correlation_discount_weight=1.0,
        )
        items = [
            # Correlated group: 3 agents on answer A
            _item("a1", "A", source_id="d#e0"),
            _item("a2", "A", source_id="d#e0"),
            _item("a3", "A", source_id="d#e0"),
            # Independent agents: 2 on answer B
            _item("b1", "B", source_id="d#e1"),
            _item("b2", "B", source_id="d#e1"),
        ]
        res = rr.resolve(Conflict(topic="t", items=items))
        # The discount should make the correlated group's total lower
        # than what raw vote count would suggest
        assert res.winner_id is not None
        # With high discount, the 2 independent agents might win
        winner = next(it for it in items if it.id == res.winner_id)
        assert winner.agent_id in ("b1", "b2")

    def test_update_memory_updates_competence(self):
        """update_memory should change competence values."""
        pm = PeerMemory()
        rr = ReliabilityResolver(peer_memory=pm)
        items = [_item("a", "A", status=Status.CONFIRMED)]
        resolution = Resolution.single_winner(
            topic="t", strategy="test", winner_id=items[0].id, superseded_ids=[]
        )
        rr.update_memory(items, resolution, correct=True)
        assert pm.get_competence("a") != 0.5

    def test_resolution_shape_compatible(self):
        """Output is a valid Resolution with single winner."""
        rr = ReliabilityResolver()
        items = [
            _item("a", "A", source_id="d#e0", timestamp=1.0),
            _item("b", "B", source_id="d#e1", timestamp=2.0),
        ]
        res = rr.resolve(Conflict(topic="t", items=items))
        assert isinstance(res, Resolution)
        assert res.is_single_winner
        assert res.strategy == "reliability_aware"

    def test_empty_conflict(self):
        rr = ReliabilityResolver()
        res = rr.resolve(Conflict(topic="t", items=[]))
        assert not res.is_single_winner  # all_contested
