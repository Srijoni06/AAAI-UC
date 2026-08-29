"""Shared resolver contract.

Every resolver (baselines + our reliability-aware resolver) takes a
``memory.store.Conflict`` and returns a ``Resolution``. ``apply_resolution``
writes the decision back into the store: winner -> CONFIRMED, losers ->
SUPERSEDED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memory.store import Conflict, MemoryStore, Status


@dataclass
class Resolution:
    topic: str
    strategy: str
    winner_id: str
    superseded_ids: list[str]
    rationale: str
    scores: dict[str, float] = field(default_factory=dict)


class Resolver(Protocol):
    name: str

    def resolve(self, conflict: Conflict) -> Resolution: ...


def apply_resolution(store: MemoryStore, resolution: Resolution) -> None:
    store.update_status(resolution.winner_id, Status.CONFIRMED)
    for item_id in resolution.superseded_ids:
        store.update_status(item_id, Status.SUPERSEDED)
