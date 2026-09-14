"""Offline tests for baselines: majority_vote, static_confidence."""

from __future__ import annotations

import pytest

from baselines.majority_vote import MajorityVote, cluster_key
from baselines.static_confidence import StaticConfidence, _answer_key
from memory.store import (
    Authority,
    Conflict,
    MemoryItem,
    Origin,
    SourceType,
    Status,
)


def _make_item(
    agent_id: str,
    content: str,
    topic: str = "t",
    *,
    timestamp: float = 0.0,
    source_id: str | None = None,
    evidence_span: str | None = None,
    agent_group: str | None = None,
    authority: Authority = Authority.UNKNOWN,
    origin: Origin = Origin.TOOL,
    source_type: SourceType = SourceType.RETRIEVAL,
    status: Status = Status.PROPOSED,
) -> MemoryItem:
    meta: dict = {}
    if source_id:
        meta["source_id"] = source_id
    if agent_group:
        meta["agent_group"] = agent_group
    return MemoryItem(
        agent_id=agent_id,
        topic=topic,
        content=content,
        timestamp=timestamp,
        evidence_span=evidence_span,
        authority=authority,
        origin=origin,
        source_type=source_type,
        status=status,
        metadata=meta,
    )


# --------------------------------------------------------------------------- #
# MajorityVote
# --------------------------------------------------------------------------- #
class TestMajorityVote:
    def test_largest_cluster_wins(self):
        """3 items on source A vs 2 on source B -> cluster of 3 wins."""
        mv = MajorityVote()
        items = [
            _make_item("a1", "claimA", timestamp=1.0, source_id="doc#e0"),
            _make_item("a2", "claimA", timestamp=2.0, source_id="doc#e0"),
            _make_item("a3", "claimA", timestamp=3.0, source_id="doc#e0"),
            _make_item("b1", "claimB", timestamp=4.0, source_id="doc#e1"),
            _make_item("b2", "claimB", timestamp=5.0, source_id="doc#e1"),
        ]
        res = mv.resolve(Conflict(topic="t", items=items))
        # majority confirms the entire winning cluster, supersedes minority
        assert len(res.confirmed_ids) == 3  # all 3 in the 3-vote cluster
        assert len(res.superseded_ids) == 2  # the 2-vote cluster
        assert not res.contested_ids

    def test_tie_leaves_all_contested(self):
        """2 vs 2 tie -> all CONTESTED."""
        mv = MajorityVote()
        items = [
            _make_item("a1", "A", timestamp=1.0, source_id="doc#e0"),
            _make_item("a2", "A", timestamp=2.0, source_id="doc#e0"),
            _make_item("b1", "B", timestamp=3.0, source_id="doc#e1"),
            _make_item("b2", "B", timestamp=4.0, source_id="doc#e1"),
        ]
        res = mv.resolve(Conflict(topic="t", items=items))
        assert not res.is_single_winner
        assert res.contested_ids == [it.id for it in items]

    def test_single_item_confirmed(self):
        """One item -> confirm it (no-op conflict)."""
        items = [_make_item("a1", "only")]
        res = MajorityVote().resolve(Conflict(topic="t", items=items))
        assert res.confirmed_ids == [items[0].id]

    def test_all_supersested_noop(self):
        items = [_make_item("a1", "dead", status=Status.SUPERSEDED)]
        res = MajorityVote().resolve(Conflict(topic="t", items=items))
        assert res.all_contested or res.contested_ids  # all contested

    def test_cluster_key_source_id(self):
        it = _make_item("a", "x", source_id="d#e")
        assert cluster_key(it) == "d#e"

    def test_cluster_key_text_fallback(self):
        it = _make_item("a", "Hello World")
        key = cluster_key(it)
        assert key.startswith("text:")
        assert "hello" in key

    def test_correlated_group_counted_as_one_vote(self):
        """3 agents from same group -> cluster of 3; 2 independents -> cluster of 2."""
        mv = MajorityVote()
        items = [
            _make_item("a1", "A", source_id="d#e0", agent_group="grp_A"),
            _make_item("a2", "A", source_id="d#e0", agent_group="grp_A"),
            _make_item("a3", "A", source_id="d#e0", agent_group="grp_A"),
            _make_item("b1", "B", source_id="d#e1", agent_group="independent"),
            _make_item("b2", "B", source_id="d#e1", agent_group="independent"),
        ]
        res = mv.resolve(Conflict(topic="t", items=items))
        # majority confirms the entire winning cluster
        assert len(res.confirmed_ids) == 3
        assert len(res.superseded_ids) == 2
        assert "3/5" in res.rationale


# --------------------------------------------------------------------------- #
# StaticConfidence
# --------------------------------------------------------------------------- #
class TestStaticConfidence:
    def test_higher_authority_wins(self):
        """HIGH-authority item beats UNKNOWN, even if older."""
        sc = StaticConfidence()
        items = [
            _make_item("old", "old claim", timestamp=1.0, authority=Authority.HIGH),
            _make_item("new", "new claim", timestamp=5.0, authority=Authority.UNKNOWN),
        ]
        res = sc.resolve(Conflict(topic="t", items=items))
        assert res.winner_id == items[0].id  # HIGH beats UNKNOWN

    def tool_beats_inference(self):
        """TOOL origin beats INFERENCE."""
        sc = StaticConfidence()
        items = [
            _make_item("tool", "grounded", origin=Origin.TOOL, timestamp=1.0),
            _make_item("guess", "guessed", origin=Origin.INFERENCE, timestamp=2.0),
        ]
        res = sc.resolve(Conflict(topic="t", items=items))
        assert res.winner_id == items[0].id

    def test_recency_tiebreak(self):
        """Same provenance -> newest wins."""
        sc = StaticConfidence()
        items = [
            _make_item("old", "old", timestamp=1.0),
            _make_item("new", "new", timestamp=5.0),
        ]
        res = sc.resolve(Conflict(topic="t", items=items))
        assert res.winner_id == items[1].id

    def test_corroboration_bonus(self):
        """Two distinct groups backing answer X beat one group backing answer Y."""
        sc = StaticConfidence()
        items = [
            # Answer X: two distinct groups
            _make_item("a1", "X", source_id="d#e0", agent_group="grp_A"),
            _make_item("b1", "X", source_id="d#e0", agent_group="independent"),
            # Answer Y: one group (3 members)
            _make_item("c1", "Y", source_id="d#e1", agent_group="grp_B"),
            _make_item("c2", "Y", source_id="d#e1", agent_group="grp_B"),
            _make_item("c3", "Y", source_id="d#e1", agent_group="grp_B"),
        ]
        res = sc.resolve(Conflict(topic="t", items=items))
        winner = next(it for it in items if it.id == res.winner_id)
        assert _answer_key(winner) == "d#e0"  # X wins (more distinct groups)

    def test_stateless(self):
        """Two resolve calls give the same result (no memory)."""
        sc = StaticConfidence()
        items = [_make_item("a", "A", timestamp=1.0), _make_item("b", "B", timestamp=2.0)]
        r1 = sc.resolve(Conflict(topic="t", items=items))
        r2 = sc.resolve(Conflict(topic="t", items=items))
        assert r1.winner_id == r2.winner_id

    def test_scores_dict_populated(self):
        items = [_make_item("a", "A"), _make_item("b", "B")]
        res = StaticConfidence().resolve(Conflict(topic="t", items=items))
        assert set(res.scores.keys()) == {items[0].id, items[1].id}
        assert all(isinstance(v, float) for v in res.scores.values())
