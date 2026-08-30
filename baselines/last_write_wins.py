"""Last-write-wins resolver: the newest claim on the topic wins.

The simplest possible strategy - no learning, no content inspection, no
provenance. Present so we have an end-to-end pipeline from day one.
"""

from __future__ import annotations

from baselines.base import Resolution
from memory.store import Conflict


class LastWriteWins:
    name = "last_write_wins"

    def resolve(self, conflict: Conflict) -> Resolution:
        ordered = sorted(conflict.items, key=lambda it: it.timestamp)
        winner = ordered[-1]
        losers = ordered[:-1]
        return Resolution.single_winner(
            topic=conflict.topic,
            strategy=self.name,
            winner_id=winner.id,
            superseded_ids=[it.id for it in losers],
            rationale=(
                f"most recent write ({winner.agent_id} @ {winner.timestamp:.3f}) "
                f"supersedes {len(losers)} earlier claim(s)"
            ),
            scores={it.id: it.timestamp for it in conflict.items},
        )


def resolve(conflict: Conflict) -> Resolution:
    return LastWriteWins().resolve(conflict)
