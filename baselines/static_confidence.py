"""Static-confidence resolver - the primary comparison baseline.

LatticeMind-style: a **fixed weight table** over evidence-type signals decides
which claim wins a conflict. No learning of any kind: an agent that has been
wrong fifteen times in a row still earns exactly the same channel weights on
write sixteen, and the resolver has no way to notice that its agreement signal
is inflated by correlated bias. Our contribution (``reliability/``) replaces
this fixed table with an online reliability signal.

Score for one claim:

  score = w_origin[origin]                # externally grounded vs model guess
        + w_source_type[source_type]      # retrieval / tool / user / model
        + w_authority[authority]          # channel credibility tier
        + corroboration_bonus             # per *distinct additional group*
                                          # backing the same answer, capped
        + recency_bonus                   # tiny tie-break, newest claim first

The corroboration bonus counts distinct **agent groups** (from
``metadata['agent_group']``, the correlated-group label the orchestrator
writes), not raw votes: two members of one correlated group backing a claim add
one bonus, not two. This is the *static* approximation of peer-correlation
awareness -- it relies on the group label being handed to it, and its bonus size
is a fixed constant. It cannot (a) estimate correlation from outcomes, (b) learn
that a particular group or agent is unreliable, or (c) update any weight over
time. Those gaps are exactly what the reliability engine adds.

On ties the newest claim wins (deterministic, decisiveness-first: this baseline
always crowns a winner).
"""

from __future__ import annotations

from baselines.base import Resolution
from memory.store import Authority, Conflict, MemoryItem, Origin, SourceType, Status

INDEPENDENT = "independent"  # orchestrator's label for group-less agents

WEIGHTS: dict[str, dict[str, float]] = {
    "origin": {Origin.TOOL.value: 2.0, Origin.INFERENCE.value: 0.5},
    "source_type": {
        SourceType.RETRIEVAL.value: 1.5,
        SourceType.TOOL.value: 1.5,
        SourceType.USER.value: 2.5,
        SourceType.MODEL.value: 0.5,
        SourceType.UNKNOWN.value: 0.0,
    },
    # Fixed channel tiers; nothing here ever updates from outcomes.
    "authority": {
        Authority.AUTHORITATIVE.name: 3.0,
        Authority.HIGH.name: 2.0,
        Authority.MEDIUM.name: 1.0,
        Authority.LOW.name: 0.25,
        Authority.UNKNOWN.name: 0.0,
    },
}

CORROBORATION_BONUS = 0.75  # per distinct additional group, capped
CORROBORATION_CAP = 1.5
RECENCY_BONUS = 0.05  # per recency rank (newest gets most), capped at 10 ranks


def weight_table() -> dict[str, dict[str, float]]:
    """The fixed table, for eval logging / README display."""
    import copy

    return copy.deepcopy(WEIGHTS)


def _support_group(item: MemoryItem) -> str:
    """Which group's backing this claim represents (falls back to the agent)."""
    grp = item.metadata.get("agent_group")
    return str(grp) if grp else item.agent_id


def _answer_key(item: MemoryItem) -> str:
    """Which 'answer' a claim backs: its evidence excerpt if known, else text."""
    src = item.metadata.get("source_id")
    if src:
        return str(src)
    if item.evidence_span:
        return f"evidence:{item.evidence_span.strip()}"
    return f"text:{item.content.strip().lower()}"


class StaticConfidence:
    """Fixed provenance weights + capped group-corroboration. Stateless."""

    name = "static_confidence"

    def resolve(self, conflict: Conflict) -> Resolution:
        items = [it for it in conflict.items if it.status != Status.SUPERSEDED]
        if not items:
            return Resolution.all_contested(
                topic=conflict.topic,
                strategy=self.name,
                item_ids=[it.id for it in conflict.items],
                rationale="no live claims to score",
            )

        # --- corroboration: distinct groups per answer key ------------------ #
        groups_per_answer: dict[str, set[str]] = {}
        for it in items:
            groups_per_answer.setdefault(_answer_key(it), set()).add(
                _support_group(it)
            )

        scores: dict[str, float] = {}
        for it in items:
            key = _answer_key(it)
            n_other_groups = len(groups_per_answer[key] - {_support_group(it)})
            bonus = min(CORROBORATION_CAP, CORROBORATION_BONUS * n_other_groups)
            scores[it.id] = (
                WEIGHTS["origin"].get(it.origin.value, 0.0)
                + WEIGHTS["source_type"].get(it.source_type.value, 0.0)
                + WEIGHTS["authority"].get(it.authority.name, 0.0)
                + bonus
            )

        # Recency tie-break: newest claim gets the largest bonus. Deliberately
        # tiny -- it only separates claims the fixed table scores equally.
        n = len(items)
        ordered = sorted(items, key=lambda it: it.timestamp, reverse=True)
        for rank, it in enumerate(ordered):
            scores[it.id] += RECENCY_BONUS * (n - 1 - rank)

        best_score = max(scores.values())
        # Among all items at the top score, pick the one from the cluster
        # whose answer contains the most recent item (recency tie-break).
        top_items = [it for it in items if abs(scores[it.id] - best_score) < 1e-9]
        # Group top items by answer key; pick the cluster whose newest member
        # is most recent overall.
        top_clusters: dict[str, list[MemoryItem]] = {}
        for it in top_items:
            top_clusters.setdefault(_answer_key(it), []).append(it)
        best_cluster = max(
            top_clusters.values(),
            key=lambda members: max(m.timestamp for m in members),
        )
        winners = best_cluster
        losers = [it for it in items if it.id not in {w.id for w in winners}]

        return Resolution.single_winner(
            topic=conflict.topic,
            strategy=self.name,
            winner_id=winners[0].id,
            superseded_ids=[it.id for it in losers],
            rationale=(
                f"fixed weight table: top score {best_score:.2f} "
                f"({winners[0].agent_id}); corroboration groups per answer: "
                f"{ {k: len(v) for k, v in groups_per_answer.items()} }"
            ),
            scores=scores,
        )


def resolve(conflict: Conflict) -> Resolution:
    return StaticConfidence().resolve(conflict)
