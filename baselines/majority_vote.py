"""Majority-vote resolver: the largest meaning-cluster wins.

A strong non-learning baseline. For every conflict it groups the live claims by
their *meaning* (the excerpt each claim was grounded in when provenance is
available, falling back to normalized claim text), and confirms the biggest
group while superseding the rest.

The point of this baseline in the paper: majority vote treats agreement as
independent confirmation. It has no notion of *who* agreed -- three members of
a correlated group voting together carry exactly the same weight as three
independent agents. Its signature failure is confirming a wrong claim because
a correlated group outnumbers a correct independent agent, which is exactly the
failure mode the reliability layer in ``reliability/`` targets.
"""

from __future__ import annotations

import re
from collections import defaultdict

from baselines.base import Resolution
from memory.store import Conflict, MemoryItem, Status


def cluster_key(item: MemoryItem) -> str:
    """The equivalence class a claim votes with.

    Claims are votes for the passage they were grounded in, so claims derived
    from the same excerpt are the same 'answer' even if their sentences differ
    (e.g. paraphrases from agents with different prompts).

    Provenance path (preferred when present): ``metadata['source_id']``
    (``"<doc_id>#<excerpt_id>"``), falling back to ``evidence_span``.

    Text path (fallback when provenance is missing): normalized claim text, so
    paraphrase-free duplicates of the same answer cluster together.
    """
    src = item.metadata.get("source_id")
    if src:
        return str(src)
    if item.evidence_span:
        return f"evidence:{item.evidence_span.strip()}"
    text = re.sub(r"\s+", " ", item.content.strip().lower())
    text = re.sub(r"[^a-z0-9 ]", "", text).strip()
    return f"text:{text}"


class MajorityVote:
    """Cluster claims by meaning; the largest cluster wins.

    ``name_suffix`` lets evaluation label correlated vs independent variants
    without duplicating the logic. Ties (no unique largest cluster) leave every
    claim live and flagged CONTESTED -- the baseline refuses to guess, which is
    the honest behaviour our reliability engine should improve on by breaking
    ties with evidence rather than with write order.
    """

    name = "majority_vote"

    def __init__(self, name_suffix: str | None = None) -> None:
        self.name = MajorityVote.name + (f"_{name_suffix}" if name_suffix else "")

    def resolve(self, conflict: Conflict) -> Resolution:
        items = [it for it in conflict.items if it.status != Status.SUPERSEDED]
        if not items:
            return Resolution.all_contested(
                topic=conflict.topic,
                strategy=self.name,
                item_ids=[it.id for it in conflict.items],
                rationale="no live claims to vote",
            )

        clusters: dict[str, list[MemoryItem]] = defaultdict(list)
        for it in items:
            clusters[cluster_key(it)].append(it)

        best = max(clusters.values(), key=len)
        if len(best) < 2 and len(clusters) > 1:
            # All singleton clusters: no majority exists.
            return Resolution.all_contested(
                topic=conflict.topic,
                strategy=self.name,
                item_ids=[it.id for it in items],
                rationale=(
                    f"no majority: {len(items)} claim(s) across "
                    f"{len(clusters)} distinct answer(s), all singletons"
                ),
                scores={it.id: 1.0 for it in items},
            )

        runner_up = sorted((len(v) for v in clusters.values()), reverse=True)
        runner_up = runner_up[1] if len(runner_up) > 1 else 0
        tie = len(best) == runner_up

        if tie:
            return Resolution.all_contested(
                topic=conflict.topic,
                strategy=self.name,
                item_ids=[it.id for it in items],
                rationale=(
                    f"tie: {len(best)}-way cluster tie between "
                    f"{len(clusters)} distinct answer(s)"
                ),
                scores={it.id: 1.0 for it in items},
            )

        from baselines.base import ItemOutcome, Outcome
        votes = {key: len(members) for key, members in clusters.items()}
        winner_ids = {m.id for m in best}
        outcomes = [
            ItemOutcome(it.id, Outcome.CONFIRMED, f"in winning cluster ({len(best)} votes)")
            if it.id in winner_ids
            else ItemOutcome(it.id, Outcome.SUPERSEDED, "minority cluster")
            for it in items
        ]
        return Resolution(
            topic=conflict.topic,
            strategy=self.name,
            outcomes=outcomes,
            rationale=(
                f"largest meaning cluster ({len(best)}/{len(items)} votes) wins; "
                f"votes per answer: {votes}"
            ),
            scores={it.id: float(votes[cluster_key(it)]) for it in items},
        )


def resolve(conflict: Conflict) -> Resolution:
    return MajorityVote().resolve(conflict)
