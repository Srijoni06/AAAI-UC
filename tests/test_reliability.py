"""Offline tests for reliability/peer_memory.py and reliability/resolver.py."""

from __future__ import annotations

import pytest

from baselines.base import ItemOutcome, Outcome, Resolution, apply_resolution
from memory.store import Conflict, MemoryItem, Origin, SourceType, SqliteMemoryStore, Status
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
        items = [_item("a", "A", source_id="d#e0")]
        resolution = Resolution.single_winner(topic="t", strategy="test", winner_id=items[0].id, superseded_ids=[])
        pm.update(items, resolution, correct=True)
        assert pm.get_competence("a") > 0.5

    def test_competence_updates_on_incorrect(self):
        """When resolver is wrong, superseded agent gets boosted (was actually right)."""
        pm = PeerMemory()
        items = [_item("a", "A", source_id="d#e0")]
        resolution = Resolution.single_winner(
            topic="t", strategy="test", winner_id="some-other-item", superseded_ids=[items[0].id]
        )
        pm.update(items, resolution, correct=False)
        assert pm.get_competence("a") > 0.5  # boosted: they were right

    def test_competence_penalizes_confirmed_when_incorrect(self):
        """When resolver is wrong, confirmed agent gets penalised (was actually wrong)."""
        pm = PeerMemory()
        items = [_item("a", "A", source_id="d#e0")]
        resolution = Resolution.single_winner(topic="t", strategy="test", winner_id=items[0].id, superseded_ids=[])
        pm.update(items, resolution, correct=False)
        assert pm.get_competence("a") < 0.5  # penalised: they were wrong

    def test_competence_ignores_contested_items(self):
        """An item the resolution neither confirms nor supersedes gets no update."""
        pm = PeerMemory()
        items = [_item("a", "A", source_id="d#e0")]
        resolution = Resolution.all_contested(topic="t", strategy="test", item_ids=[items[0].id])
        pm.update(items, resolution, correct=True)
        assert pm.get_competence("a") == 0.5  # untouched

    def test_competence_bounded(self):
        pm = PeerMemory()
        items = [_item("a", "A")]
        resolution = Resolution.single_winner(topic="t", strategy="test", winner_id=items[0].id, superseded_ids=[])
        for _ in range(100):
            pm.update(items, resolution, correct=True)
        assert pm.get_competence("a") <= 0.95

    # ----------------------------------------------------------------- #
    # Regression: correlation must track *correlated error*, not agreement
    # alone. The real bug: `agreed = a_key == b_key` fed the correlation EMA
    # with no reference to `correct` at all - two agents correctly right
    # together and two agents incorrectly wrong together produced identical
    # correlation trajectories (confirmed empirically: both climbed to
    # 0.6404 after 8 rounds). That directly undermines the project's central
    # claim, since it can't distinguish shared bias from valid corroboration.
    # ----------------------------------------------------------------- #
    def test_correlation_unchanged_on_correct_agreement(self):
        """Agents who agree on the CORRECT answer: not evidence of shared
        bias either way - correlation must be left unchanged, not pushed up."""
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0"),
            _item("b", "A", source_id="d#e0"),
        ]
        resolution = Resolution.coexist(topic="t", strategy="test", item_ids=[it.id for it in items])
        pm.update(items, resolution, correct=True)
        assert pm.get_correlation("a", "b") == 0.0  # untouched, not pushed positive

    def test_correlation_increases_on_incorrect_agreement(self):
        """Agents who agree on an INCORRECT answer: this is the real
        shared-bias signal - correlation must rise."""
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0"),
            _item("b", "A", source_id="d#e0"),
        ]
        resolution = Resolution.coexist(topic="t", strategy="test", item_ids=[it.id for it in items])
        pm.update(items, resolution, correct=False)
        assert pm.get_correlation("a", "b") > 0

    def test_correlation_disagreement_negative_regardless_of_correctness(self):
        """Disagreement is still evidence of independence either way."""
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0"),
            _item("c", "B", source_id="d#e1"),
        ]
        resolution = Resolution(
            topic="t",
            strategy="test",
            outcomes=[
                ItemOutcome(items[0].id, Outcome.CONFIRMED),
                ItemOutcome(items[1].id, Outcome.SUPERSEDED),
            ],
        )
        pm.update(items, resolution, correct=True)
        assert pm.get_correlation("a", "c") < 0

    def test_correlation_bounded(self):
        """Repeated INCORRECT agreement (the actual shared-bias case) must
        still hit the upper clamp - correct agreement no longer moves
        correlation at all, so it can't be used to test the bound."""
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0"),
            _item("b", "A", source_id="d#e0"),
        ]
        resolution = Resolution.coexist(topic="t", strategy="test", item_ids=[it.id for it in items])
        for _ in range(100):
            pm.update(items, resolution, correct=False)
        assert abs(pm.get_correlation("a", "b")) <= 0.95

    # ------------------------------------------------------------------- #
    # Regression: correlation must be scoped to THIS PAIR's own answer
    # cluster, not the topic-level resolver decision. Before this fix, every
    # pair in a conflict was scored against the SAME single `correct` flag
    # (whether the resolver's *confirmed* cluster matched gold) - so a pair
    # that agreed on a *different*, actually-correct cluster could get wrongly
    # flagged as shared bias, and a pair that agreed on a *different*,
    # actually-wrong cluster could wrongly dodge being flagged at all.
    # ------------------------------------------------------------------- #
    def test_cluster_correct_protects_pair_on_a_different_correct_cluster(self):
        """b/e agree on the GOLD-correct answer, but the resolver's own pick
        this round was a different (wrong) cluster (a/c/d) - correct=False.
        With per-cluster info, b/e must NOT be treated as shared bias."""
        pm = PeerMemory()
        items = [
            _item("a", "wrong answer", source_id="d#eAnchor"),
            _item("c", "wrong answer", source_id="d#eAnchor"),
            _item("d", "wrong answer", source_id="d#eAnchor"),
            _item("b", "right answer", source_id="d#eOther"),
            _item("e", "right answer", source_id="d#eOther"),
        ]
        resolution = Resolution(
            topic="t",
            strategy="test",
            outcomes=[
                ItemOutcome(items[0].id, Outcome.CONFIRMED),
                ItemOutcome(items[1].id, Outcome.CONFIRMED),
                ItemOutcome(items[2].id, Outcome.CONFIRMED),
                ItemOutcome(items[3].id, Outcome.SUPERSEDED),
                ItemOutcome(items[4].id, Outcome.SUPERSEDED),
            ],
        )
        cluster_correct = {"d#eAnchor": False, "d#eOther": True}
        pm.update(items, resolution, correct=False, cluster_correct=cluster_correct)

        assert pm.get_correlation("b", "e") == 0.0  # untouched - their cluster was right
        assert pm.get_correlation("a", "c") > 0.0   # correctly flagged - their cluster was wrong
        assert pm.get_correlation("a", "d") > 0.0
        assert pm.get_correlation("c", "d") > 0.0

    def test_cluster_correct_still_flags_wrong_cluster_when_topic_was_correct(self):
        """Symmetric case: a/c agree on a genuinely WRONG answer, but the
        resolver's overall pick this round (b/e's cluster) was correct -
        correct=True. a/c's agreement must still be flagged as shared bias."""
        pm = PeerMemory()
        items = [
            _item("a", "wrong answer", source_id="d#eAnchor"),
            _item("c", "wrong answer", source_id="d#eAnchor"),
            _item("b", "right answer", source_id="d#eOther"),
            _item("e", "right answer", source_id="d#eOther"),
        ]
        resolution = Resolution(
            topic="t",
            strategy="test",
            outcomes=[
                ItemOutcome(items[0].id, Outcome.SUPERSEDED),
                ItemOutcome(items[1].id, Outcome.SUPERSEDED),
                ItemOutcome(items[2].id, Outcome.CONFIRMED),
                ItemOutcome(items[3].id, Outcome.CONFIRMED),
            ],
        )
        cluster_correct = {"d#eAnchor": False, "d#eOther": True}
        pm.update(items, resolution, correct=True, cluster_correct=cluster_correct)

        assert pm.get_correlation("a", "c") > 0.0   # still flagged despite topic-level correct=True
        assert pm.get_correlation("b", "e") == 0.0  # untouched - genuinely valid corroboration

    def test_without_cluster_correct_falls_back_to_topic_level(self):
        """No cluster_correct given -> old topic-level-only behaviour, for
        callers that don't have per-cluster info (backward compatible)."""
        pm = PeerMemory()
        items = [
            _item("a", "A", source_id="d#e0"),
            _item("b", "A", source_id="d#e0"),
        ]
        resolution = Resolution.coexist(topic="t", strategy="test", item_ids=[it.id for it in items])
        pm.update(items, resolution, correct=False)  # no cluster_correct
        assert pm.get_correlation("a", "b") > 0.0  # falls back to the single `correct` flag

    def test_pair_key_is_canonical(self):
        assert _pair_key("a", "b") == _pair_key("b", "a")

    def test_n_decisions_increments(self):
        pm = PeerMemory()
        assert pm.n_decisions == 0
        item = _item("a", "A")
        resolution = Resolution.single_winner(topic="t", strategy="test", winner_id=item.id, superseded_ids=[])
        pm.update([item], resolution, correct=True)
        assert pm.n_decisions == 1

    # ----------------------------------------------------------------- #
    # Regression: PeerMemory.update() must derive winner/loser from the
    # Resolution's own confirmed_ids/superseded_ids, never from item.status.
    #
    # The real bug: eval/run_comparison.py captured `live_items` from the
    # store BEFORE calling apply_resolution(), then passed that same
    # pre-resolution snapshot to update_memory() afterwards.
    # apply_resolution() genuinely writes the new status into the store, but
    # it returns a *new* MemoryItem built from a fresh SELECT - it never
    # mutates the caller's existing objects. So every item in the stale
    # snapshot still read status=PROPOSED, which the old status-based
    # branch (`if it.status in (PROPOSED, CONFIRMED): confirmed else
    # superseded`) classified as "confirmed" - meaning winner and loser were
    # indistinguishable, and every agent in a conflict got the exact same EMA
    # update. Symptom observed in practice: every agent's competence pinned
    # to the same flat value across many calls (e.g. 0.49/0.49/0.49/0.49,
    # later 0.65/0.65/0.65/0.65).
    # ----------------------------------------------------------------- #
    def test_update_derives_winner_loser_from_resolution_not_stale_item_status(self, tmp_path):
        store = SqliteMemoryStore(tmp_path / "mem.db")
        good = _item("agent_good", "the right answer", source_id="doc#e0")
        bad = _item("agent_bad", "the wrong answer", source_id="doc#e1")
        store.add(good)
        store.add(bad)

        # Exactly the buggy call site's pattern: snapshot BEFORE resolving.
        pre_resolution_items = store.list(topic="test")
        assert {it.status for it in pre_resolution_items} == {Status.PROPOSED}

        resolution = Resolution.single_winner(
            topic="test", strategy="test", winner_id=good.id, superseded_ids=[bad.id]
        )
        apply_resolution(store, resolution)  # writes CONFIRMED/SUPERSEDED into the store

        # The pre-resolution objects are untouched (this is what made the bug
        # possible) - proving the fix does not depend on them being fresh.
        assert {it.status for it in pre_resolution_items} == {Status.PROPOSED}

        pm = PeerMemory()
        pm.update(pre_resolution_items, resolution, correct=True)

        good_competence = pm.get_competence("agent_good")
        bad_competence = pm.get_competence("agent_bad")
        assert good_competence > 0.5
        assert bad_competence < 0.5
        assert good_competence != bad_competence  # the bug made these identical

    def test_competence_evolves_correctly_across_multiple_sequential_conflicts(self, tmp_path):
        """A controlled, known-correct/known-incorrect pattern across several
        independent conflicts: agent_reliable always genuinely right and always
        confirmed, agent_flaky always genuinely wrong and always superseded.
        Competence must actually move each round (not get stuck), and the two
        agents must end up clearly - and correctly - separated."""
        store = SqliteMemoryStore(tmp_path / "mem.db")
        pm = PeerMemory()
        rr = ReliabilityResolver(peer_memory=pm)

        reliable_history: list[float] = []
        flaky_history: list[float] = []

        for i in range(6):
            topic = f"topic-{i}"
            reliable = _item("agent_reliable", f"right answer {i}", source_id=f"doc#e{2*i}")
            reliable.topic = topic
            flaky = _item("agent_flaky", f"wrong answer {i}", source_id=f"doc#e{2*i+1}")
            flaky.topic = topic
            store.add(reliable)
            store.add(flaky)

            live_items = store.list(topic=topic)
            resolution = Resolution.single_winner(
                topic=topic, strategy="test", winner_id=reliable.id, superseded_ids=[flaky.id]
            )
            apply_resolution(store, resolution)
            # correct=True every round: agent_reliable really is right every time.
            rr.update_memory(live_items, resolution, correct=True)

            reliable_history.append(pm.get_competence("agent_reliable"))
            flaky_history.append(pm.get_competence("agent_flaky"))

        # Genuinely evolving, not flat: each round moves the estimate.
        assert len(set(reliable_history)) == len(reliable_history)
        assert len(set(flaky_history)) == len(flaky_history)
        # Monotonic in the right direction under this EMA with a constant signal.
        assert reliable_history == sorted(reliable_history)
        assert flaky_history == sorted(flaky_history, reverse=True)
        # Correctly reflects who was actually right.
        assert reliable_history[-1] > 0.5 > flaky_history[-1]
        assert reliable_history[-1] != flaky_history[-1]


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
        # The discount should make the correlated group's total lower than
        # what raw vote count would suggest, so the 2 independent agents win -
        # and BOTH of them get confirmed (clustered confirmation), not one
        # arbitrarily picked winner with the other marked as a loser.
        by_agent = {it.agent_id: it.id for it in items}
        assert set(res.confirmed_ids) == {by_agent["b1"], by_agent["b2"]}
        assert set(res.superseded_ids) == {by_agent["a1"], by_agent["a2"], by_agent["a3"]}
        assert res.winner_id is None  # more than one confirmed -> no single winner_id

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

    # ----------------------------------------------------------------- #
    # Regression: resolve() must CONFIRM the whole winning answer cluster,
    # not pick one item out of it and supersede everyone else.
    #
    # The real bug: Step 4 grouped and scored claims by answer cluster
    # (correctly), but then picked a single `winner_item` from the winning
    # cluster and marked *everything else in the entire conflict* -
    # including other members of that same winning cluster - as SUPERSEDED.
    # Two agents who independently agreed on the correct answer would have
    # one of them wrongly treated as a loser: via the (already-fixed)
    # PeerMemory update, that agent's competence would be pushed toward 0
    # as if they'd been wrong, even though they agreed with the winning,
    # correct answer.
    # ----------------------------------------------------------------- #
    def test_resolve_confirms_entire_winning_cluster_not_one_item(self):
        """2 agents agree on the correct answer, 1 disagrees (wrong): both
        agreeing agents must be confirmed, only the dissenter superseded."""
        rr = ReliabilityResolver()
        correct_1 = _item("agent_A", "OntoNotes 5.0 is the benchmark.", source_id="doc#results")
        correct_2 = _item("agent_B", "The benchmark used is OntoNotes 5.0.", source_id="doc#results")
        wrong = _item("agent_C", "CoNLL-2003 is the benchmark.", source_id="doc#intro")

        res = rr.resolve(Conflict(topic="t", items=[correct_1, correct_2, wrong]))

        assert set(res.confirmed_ids) == {correct_1.id, correct_2.id}
        assert set(res.superseded_ids) == {wrong.id}
        assert not res.is_single_winner  # 2 confirmed, not 1 - this is the point
        assert res.winner_id is None
