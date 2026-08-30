"""Shared resolver contract.

Every resolver (baselines + our reliability-aware resolver) takes a
``memory.store.Conflict`` and returns a :class:`Resolution`.

Unlike the milestone-1 version, a ``Resolution`` is **not** forced to crown a
single winner. It assigns an :class:`Outcome` to every item in the conflict, so a
resolver can legitimately say:

* ``CONFIRMED``  - this claim survives (possibly alongside other CONFIRMED items
  that validly coexist -- the "coordination" case);
* ``CONTESTED``  - still unresolved, keep it live and flagged;
* ``SUPERSEDED`` - this claim lost, kept only for the audit trail.

``apply_resolution`` writes the decision back into the store, routing every
outcome through ``store.update_status`` so the lifecycle state machine still
applies. :meth:`Resolution.single_winner` reproduces the old one-winner shape,
and ``winner_id`` / ``superseded_ids`` stay available as derived views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from memory.store import Conflict, MemoryStore, Status


class Outcome(str, Enum):
    """Per-item verdict a resolver can assign within a conflict."""

    CONFIRMED = "CONFIRMED"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class ItemOutcome:
    """The verdict for one item in a resolved conflict."""

    item_id: str
    outcome: Outcome
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            self.outcome = Outcome(self.outcome)


_OUTCOME_TO_STATUS: dict[Outcome, Status] = {
    Outcome.CONFIRMED: Status.CONFIRMED,
    Outcome.CONTESTED: Status.CONTESTED,
    Outcome.SUPERSEDED: Status.SUPERSEDED,
}


@dataclass
class Resolution:
    """Outcome of a resolver deciding a single :class:`Conflict`.

    ``outcomes`` carries one :class:`ItemOutcome` per item in the conflict. There
    is no requirement that exactly one item be ``CONFIRMED``.
    """

    topic: str
    strategy: str
    outcomes: list[ItemOutcome]
    rationale: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for io in self.outcomes:
            if not isinstance(io, ItemOutcome):
                raise TypeError(f"outcomes must be ItemOutcome, got {io!r}")
            if io.item_id in seen:
                raise ValueError(f"item {io.item_id!r} appears twice in a Resolution")
            seen.add(io.item_id)

    # -- constructors -------------------------------------------------- #
    @classmethod
    def single_winner(
        cls,
        topic: str,
        strategy: str,
        winner_id: str,
        superseded_ids: list[str],
        *,
        rationale: str = "",
        scores: Optional[dict[str, float]] = None,
    ) -> "Resolution":
        """The classic one-winner shape: winner CONFIRMED, the rest SUPERSEDED."""
        outcomes = [ItemOutcome(winner_id, Outcome.CONFIRMED, "selected winner")]
        outcomes += [
            ItemOutcome(i, Outcome.SUPERSEDED, "lost to winner") for i in superseded_ids
        ]
        return cls(topic, strategy, outcomes, rationale, scores or {})

    @classmethod
    def coexist(
        cls,
        topic: str,
        strategy: str,
        item_ids: list[str],
        *,
        rationale: str = "",
        scores: Optional[dict[str, float]] = None,
    ) -> "Resolution":
        """Every claim validly coexists -- all CONFIRMED (the coordination case)."""
        return cls(
            topic,
            strategy,
            [ItemOutcome(i, Outcome.CONFIRMED, "coexists") for i in item_ids],
            rationale,
            scores or {},
        )

    @classmethod
    def all_contested(
        cls,
        topic: str,
        strategy: str,
        item_ids: list[str],
        *,
        rationale: str = "",
        scores: Optional[dict[str, float]] = None,
    ) -> "Resolution":
        """No verdict reached -- keep every claim live and flagged CONTESTED."""
        return cls(
            topic,
            strategy,
            [ItemOutcome(i, Outcome.CONTESTED, "unresolved") for i in item_ids],
            rationale,
            scores or {},
        )

    # -- views ------------------------------------------------------- #
    def ids_with(self, outcome: Outcome) -> list[str]:
        return [io.item_id for io in self.outcomes if io.outcome is outcome]

    @property
    def confirmed_ids(self) -> list[str]:
        return self.ids_with(Outcome.CONFIRMED)

    @property
    def contested_ids(self) -> list[str]:
        return self.ids_with(Outcome.CONTESTED)

    @property
    def superseded_ids(self) -> list[str]:
        return self.ids_with(Outcome.SUPERSEDED)

    @property
    def winner_id(self) -> Optional[str]:
        """Back-compat: the sole CONFIRMED id, or ``None`` if this is not a
        single-winner resolution."""
        confirmed = self.confirmed_ids
        return confirmed[0] if len(confirmed) == 1 else None

    @property
    def is_single_winner(self) -> bool:
        return len(self.confirmed_ids) == 1 and not self.contested_ids


class Resolver(Protocol):
    name: str

    def resolve(self, conflict: Conflict) -> Resolution: ...


def apply_resolution(store: MemoryStore, resolution: Resolution) -> None:
    """Apply every :class:`ItemOutcome` to the store via ``update_status`` (so an
    illegal move, e.g. re-opening a SUPERSEDED item, still raises)."""
    for io in resolution.outcomes:
        store.update_status(io.item_id, _OUTCOME_TO_STATUS[io.outcome])
