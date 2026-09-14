"""Online per-agent reliability + pairwise correlation (Sigma-Mem-style).

Tracks two signals that the resolver uses to weight claims:

* ``competence[agent]``: running estimate of how often this agent's claims
  survive reconciliation, updated from resolution outcomes (post-decision
  correctness feedback). Starts at 0.5 (uninformative prior) and drifts
  toward the empirical confirmation rate as decisions accumulate.

* ``correlation[(a, b)]``: whether two agents tend to agree or disagree on
  the same topic. High positive correlation means their agreement is weak
  corroboration (shared bias / correlated error), not two independent
  confirmations. Updated from the agreement/disagreement pattern of every
  resolved conflict.

Both are consumed by :class:`reliability.resolver.ReliabilityResolver` as the
credibility-weighting function, replacing the static evidence-type table in
``baselines/static_confidence.py``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


def _pair_key(a: str, b: str) -> frozenset[str]:
    """Canonical unordered pair key for the correlation matrix."""
    return frozenset([a, b])


@dataclass
class PeerMemory:
    """Per-agent competence + pairwise correlation, updated online.

    Competence starts at ``DEFAULT_COMPETENCE`` (uninformative) and moves
    toward the empirical confirmation rate via exponential moving average.

    Correlation starts at 0 and tracks whether agents tend to agree or
    disagree across resolved conflicts. Updated using the agreement/
    disagreement pattern of all items in a conflict, not just winners
    vs losers.
    """

    DEFAULT_COMPETENCE: float = 0.5
    COMPETENCE_EMA_ALPHA: float = 0.15
    CORRELATION_EMA_ALPHA: float = 0.12
    MIN_COMPETENCE: float = 0.05
    MAX_COMPETENCE: float = 0.95
    MAX_CORRELATION: float = 0.95

    competence: dict[str, float] = field(default_factory=dict)
    correlation: dict[frozenset[str], float] = field(default_factory=dict)
    n_decisions: int = 0

    # -- read ----------------------------------------------------------- #

    def get_competence(self, agent_id: str) -> float:
        """Return the agent's current competence estimate."""
        return self.competence.get(agent_id, self.DEFAULT_COMPETENCE)

    def get_correlation(self, a: str, b: str) -> float:
        """Return the pairwise correlation between two agents."""
        if a == b:
            return 0.0
        return self.correlation.get(_pair_key(a, b), 0.0)

    # -- update --------------------------------------------------------- #

    def update(self, items: Iterable, *, correct: bool = True) -> None:
        """Update competence and correlation from a resolution decision.

        ``items``: the full list of MemoryItem objects that were in the
        conflict (not just the winner or loser — the full topic group).

        ``correct``: whether the resolver's decision was right (matched
        gold label). When False (incorrect), competence of confirmed agents
        is penalized.

        Competence update: exponential moving average toward the observed
        outcome (1.0 for confirmed agents, 0.0 for superseded agents).

        Correlation update: agents whose claims share the same answer key
        (i.e. they agreed on the same excerpt/source) are "in agreement"
        on this topic; agents on different answer keys are "in disagreement".
        Agreement within a topic pushes correlation positive; disagreement
        pushes it negative.
        """
        items = list(items)
        if not items:
            return

        # -- Competence update ------------------------------------------ #
        from baselines.majority_vote import cluster_key as _ck

        confirmed_ids: set[str] = set()
        superseded_ids: set[str] = set()
        # We don't have the Resolution here — extract from items directly.
        # Items still PROPOSED or CONFIRMED were "not superseded" in the
        # most recent decision; items that were SUPERSEDED were losers.
        for it in items:
            from memory.store import Status
            if it.status in (Status.PROPOSED, Status.CONFIRMED):
                confirmed_ids.add(it.id)
            else:
                superseded_ids.add(it.id)

        alpha_c = self.COMPETENCE_EMA_ALPHA
        for it in items:
            old = self.get_competence(it.agent_id)
            if it.id in confirmed_ids:
                target = 1.0 if correct else 0.0
            else:
                target = 0.0 if correct else 1.0
            new = old + alpha_c * (target - old)
            self.competence[it.agent_id] = max(
                self.MIN_COMPETENCE, min(self.MAX_COMPETENCE, new)
            )

        # -- Correlation update ----------------------------------------- #
        # Group items by answer key (which "side" they support).
        answer_groups: dict[str, list[str]] = defaultdict(list)  # key -> [agent_ids]
        for it in items:
            answer_groups[_ck(it)].append(it.agent_id)

        alpha_r = self.CORRELATION_EMA_ALPHA
        n_agents = len(items)
        if n_agents < 2:
            self.n_decisions += 1
            return

        for a_idx in range(n_agents):
            for b_idx in range(a_idx + 1, n_agents):
                a_id = items[a_idx].agent_id
                b_id = items[b_idx].agent_id
                if a_id == b_id:
                    continue

                # Did they agree (same answer key) or disagree?
                a_key = _ck(items[a_idx])
                b_key = _ck(items[b_idx])
                agreed = a_key == b_key

                old_corr = self.get_correlation(a_id, b_id)
                target = 1.0 if agreed else -1.0
                new_corr = old_corr + alpha_r * (target - old_corr)
                new_corr = max(-self.MAX_CORRELATION, min(self.MAX_CORRELATION, new_corr))
                self.correlation[_pair_key(a_id, b_id)] = new_corr

        self.n_decisions += 1
