"""Offline tests for baselines: majority_vote, static_confidence."""

from __future__ import annotations

import pytest

from baselines.majority_vote import MajorityVote, cluster_key
from baselines.static_confidence import (
    CORROBORATION_BONUS,
    INDEPENDENT,
    RECENCY_BONUS,
    StaticConfidence,
    _answer_key,
    _support_group,
)
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

    # ------------------------------------------------------------------- #
    # Regression: independent agents must corroborate EACH OTHER as distinct
    # groups, not collapse into one shared "independent" label.
    #
    # The real bug: _support_group() returned the literal string
    # "independent" for every agent with no real correlated-group membership,
    # instead of falling back to that agent's own id. Two genuinely
    # independent agents who agreed therefore looked like ONE group backing
    # its own answer (n_other_groups=0), so their corroboration bonus was
    # always 0.0 - identical to a single unsupported claim, and
    # indistinguishable from correlated agents (who are SUPPOSED to score 0.0
    # for agreeing with their own group).
    # ------------------------------------------------------------------- #
    def test_support_group_keys_independents_by_agent_id(self):
        """Unit-level: two independents get DIFFERENT support groups; two
        correlated-group members still get the SAME one (unchanged, by design)."""
        b1 = _make_item("b1", "X", agent_group=INDEPENDENT)
        b2 = _make_item("b2", "X", agent_group=INDEPENDENT)
        a1 = _make_item("a1", "Y", agent_group="grp_A")
        a2 = _make_item("a2", "Y", agent_group="grp_A")

        assert _support_group(b1) != _support_group(b2)
        assert _support_group(b1) == "b1"
        assert _support_group(b2) == "b2"
        assert _support_group(a1) == _support_group(a2) == "grp_A"

    def test_independent_agents_corroborate_each_other(self):
        """Two independent agents agreeing get a nonzero corroboration bonus -
        distinguishable from a single unsupported claim (0.0) and computed the
        same way correlated-group agents would earn it from an OUTSIDE group
        (not from each other, which stays 0.0, by design)."""
        sc = StaticConfidence()
        items = [
            # Two independents agree on X: 2 distinct groups -> nonzero bonus.
            _make_item("b1", "X", source_id="d#e0", agent_group=INDEPENDENT),
            _make_item("b2", "X", source_id="d#e0", agent_group=INDEPENDENT),
            # A lone, unsupported claim on Y: no other group -> 0.0.
            _make_item("c1", "Y", source_id="d#e1", agent_group=INDEPENDENT),
            # grp_A members agreeing with EACH OTHER on Z: still 0.0, by design.
            _make_item("a1", "Z", source_id="d#e2", agent_group="grp_A"),
            _make_item("a2", "Z", source_id="d#e2", agent_group="grp_A"),
        ]
        res = sc.resolve(Conflict(topic="t", items=items))

        # All 5 items share one conflict, so they share one recency ranking
        # (RECENCY_BONUS per rank, stable-sorted by timestamp - all 0.0 here,
        # so ranked by list position). Subtract that out to isolate the
        # origin+source_type+authority+corroboration terms being compared.
        ordered = sorted(items, key=lambda it: it.timestamp, reverse=True)
        rank_of = {it.id: r for r, it in enumerate(ordered)}
        n = len(items)

        def without_recency(it) -> float:
            return res.scores[it.id] - RECENCY_BONUS * (n - 1 - rank_of[it.id])

        b1_flat, b2_flat = without_recency(items[0]), without_recency(items[1])
        c1_flat = without_recency(items[2])
        a1_flat, a2_flat = without_recency(items[3]), without_recency(items[4])

        assert b1_flat - c1_flat == pytest.approx(CORROBORATION_BONUS)
        assert b2_flat - c1_flat == pytest.approx(CORROBORATION_BONUS)
        assert a1_flat == pytest.approx(c1_flat)  # grp_A-with-itself: no bonus, ~= lone claim
        assert a2_flat == pytest.approx(c1_flat)

    # ------------------------------------------------------------------- #
    # Regression: resolve() must confirm the WHOLE winning cluster, not just
    # one representative item - the same single-winner bug
    # ReliabilityResolver.resolve() had before it was fixed earlier tonight.
    # best_cluster can legitimately have 2+ members (two items tied for the
    # top score, backing the same answer); collapsing to
    # Resolution.single_winner(winners[0]) wrongly superseded the other tied
    # member(s). A NATURAL tie needs RECENCY_BONUS zeroed out (it strictly
    # separates every item by write-order rank otherwise - see
    # static_confidence.py's own module docstring / the investigation that
    # found this), so this test disables it via monkeypatch to isolate the
    # provenance+corroboration scoring the bug actually lives in.
    # ------------------------------------------------------------------- #
    def test_resolve_confirms_entire_winning_cluster_not_one_item(self, monkeypatch):
        """2 items tied for the top score, same answer: both must be
        confirmed, only the weaker, different-answer item superseded."""
        import baselines.static_confidence as static_confidence_module

        monkeypatch.setattr(static_confidence_module, "RECENCY_BONUS", 0.0)
        sc = StaticConfidence()

        tied_1 = _make_item(
            "agent_A", "X", source_id="d#e0", agent_group=INDEPENDENT,
            origin=Origin.TOOL, source_type=SourceType.RETRIEVAL, authority=Authority.MEDIUM,
        )
        tied_2 = _make_item(
            "agent_B", "X", source_id="d#e0", agent_group=INDEPENDENT,
            origin=Origin.TOOL, source_type=SourceType.RETRIEVAL, authority=Authority.MEDIUM,
        )
        weaker = _make_item(
            "agent_C", "Y", source_id="d#e1", agent_group=INDEPENDENT,
            origin=Origin.TOOL, source_type=SourceType.RETRIEVAL, authority=Authority.LOW,
        )
        res = sc.resolve(Conflict(topic="t", items=[tied_1, tied_2, weaker]))

        assert res.scores[tied_1.id] == pytest.approx(res.scores[tied_2.id])  # genuinely tied
        assert set(res.confirmed_ids) == {tied_1.id, tied_2.id}
        assert set(res.superseded_ids) == {weaker.id}
        assert not res.is_single_winner  # 2 confirmed, not 1 - this is the point
        assert res.winner_id is None

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
