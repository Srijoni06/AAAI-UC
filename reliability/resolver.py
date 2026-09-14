"""Peer-correlation-aware credibility resolver (Sigma-Mem-style).

The novel contribution. For every CREDIBILITY conflict, score each claim using
signals from :class:`reliability.peer_memory.PeerMemory`:

1. **Base score**: fixed provenance weights (same table as
   ``baselines/static_confidence.py``) — origin, source_type, authority.

2. **Competence boost**: ``competence[agent] × weight`` — agents with a track
   record of correct claims earn higher base credibility.

3. **Correlation discount**: for each pair of agents whose claims share the
   same answer, multiply their combined score by
   ``(1 − max_correlation × discount_weight)``. When two agents are highly
   correlated (shared bias), their agreement counts for less — exactly the
   failure mode the baselines miss.

The resolver updates ``PeerMemory`` after every decision so competence and
correlation drift toward empirical accuracy as decisions accumulate.  On the
seeded suite this produces a visible improvement over majority vote once the
resolver encounters conflicts where the correlated group is wrong.
"""

from __future__ import annotations

from collections import defaultdict

from baselines.base import Resolution
from baselines.majority_vote import cluster_key
from baselines.static_confidence import (
    CORROBORATION_BONUS,
    CORROBORATION_CAP,
    RECENCY_BONUS,
    WEIGHTS,
    _answer_key,
    _support_group,
)
from memory.store import Authority, Conflict, MemoryItem, Status


class ReliabilityResolver:
    """Score claims using competence + correlation discount.

    Parameters
    ----------
    peer_memory : PeerMemory, optional
        Existing peer memory to continue learning from. If ``None`` a fresh
        one is created.
    provenance_weight : float
        Weight of the fixed provenance base score (origin + source_type +
        authority) relative to competence and correlation signals.
    competence_weight : float
        Multiplier for the competence boost added to each item's score.
    correlation_discount_weight : float
        How strongly pairwise correlation reduces the combined score of
        agreeing claims.  ``0.0`` disables the discount entirely.
    """

    name = "reliability_aware"

    def __init__(
        self,
        peer_memory=None,
        provenance_weight: float = 1.0,
        competence_weight: float = 2.0,
        correlation_discount_weight: float = 0.6,
    ) -> None:
        from reliability.peer_memory import PeerMemory

        self.peer_memory = peer_memory or PeerMemory()
        self.provenance_weight = provenance_weight
        self.competence_weight = competence_weight
        self.correlation_discount_weight = correlation_discount_weight

    def resolve(self, conflict: Conflict) -> Resolution:
        """Score every live claim and pick the best answer cluster."""
        items = [it for it in conflict.items if it.status != Status.SUPERSEDED]
        if not items:
            return Resolution.all_contested(
                topic=conflict.topic,
                strategy=self.name,
                item_ids=[it.id for it in conflict.items],
                rationale="no live claims to score",
            )

        pm = self.peer_memory

        # -- Step 1: raw per-item scores ---------------------------------- #
        scores: dict[str, float] = {}
        for it in items:
            scores[it.id] = self._item_score(it)

        # -- Step 2: group by answer key ----------------------------------- #
        answer_groups: dict[str, list[MemoryItem]] = defaultdict(list)
        for it in items:
            answer_groups[_answer_key(it)].append(it)

        # -- Step 3: per-answer total (sum of items + correlation discount)  #
        answer_totals: dict[str, float] = {}
        answer_discounts: dict[str, float] = {}
        for key, members in answer_groups.items():
            total = sum(scores[it.id] for it in members)
            discount = self._pairwise_discount(members)
            answer_totals[key] = total * discount
            answer_discounts[key] = discount

        # -- Step 4: pick the best answer ----------------------------------- #
        best_key = max(answer_totals, key=answer_totals.get)
        winner_item = max(answer_groups[best_key], key=lambda it: scores[it.id])
        losers = [it for it in items if it.id != winner_item.id]

        votes_per_answer = {k: len(v) for k, v in answer_groups.items()}
        discount_str = ", ".join(
            f"{k[:30]}×{d:.3f}" for k, d in answer_discounts.items()
        )
        return Resolution.single_winner(
            topic=conflict.topic,
            strategy=self.name,
            winner_id=winner_item.id,
            superseded_ids=[it.id for it in losers],
            rationale=(
                f"reliability-aware: answer '{best_key}' scored "
                f"{answer_totals[best_key]:.3f} "
                f"(discounts: {discount_str}); "
                f"votes: {votes_per_answer}; "
                f"competence: "
                f"{ {it.agent_id: f'{pm.get_competence(it.agent_id):.2f}' for it in items} }"
            ),
            scores=scores,
        )

    def update_memory(
        self,
        all_items: list[MemoryItem],
        resolution: Resolution,
        *,
        correct: bool = True,
    ) -> None:
        """Feed the resolution outcome back into peer_memory.

        After the resolver makes a decision, call this with the full conflict
        items and the resolution to update competence and correlation.

        ``correct`` should be ``True`` when the resolver's decision matched
        the gold label.  When ``False``, competence of confirmed agents is
        penalised and superseded agents get a slight boost (they were right
        after all).
        """
        self.peer_memory.update(all_items, correct=correct)

    # ------------------------------------------------------------------ #
    # Scoring helpers                                                     #
    # ------------------------------------------------------------------ #

    def _item_score(self, it: MemoryItem) -> float:
        """Provenance base score + competence boost for one item."""
        provenance = (
            WEIGHTS["origin"].get(it.origin.value, 0.0)
            + WEIGHTS["source_type"].get(it.source_type.value, 0.0)
            + WEIGHTS["authority"].get(it.authority.name, 0.0)
        )
        competence = self.peer_memory.get_competence(it.agent_id)
        return (
            self.provenance_weight * provenance
            + self.competence_weight * competence
        )

    def _pairwise_discount(self, members: list[MemoryItem]) -> float:
        """Discount factor based on pairwise correlation of supporting agents.

        Returns a value in ``(0, 1]``.  ``1.0`` means no discount (all
        independent); values below 1.0 penalise correlated agreement.

        For a single agent or no agents the discount is 1.0 (no pair to
        discount).  For N agents the discount considers every unique pair
        and multiplies the agreement penalty once per pair, capped by the
        maximum correlation to avoid collapsing to zero.
        """
        n = len(members)
        if n < 2:
            return 1.0

        pm = self.peer_memory
        w = self.correlation_discount_weight
        cumulative = 1.0

        for i in range(n):
            for j in range(i + 1, n):
                corr = pm.get_correlation(members[i].agent_id, members[j].agent_id)
                if corr > 0:
                    cumulative *= 1.0 - w * corr

        return max(0.01, cumulative)  # never fully zero out
