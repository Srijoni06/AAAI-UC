"""Shared multi-agent memory store.

Milestone 1 scope: the memory-item schema plus a small add / read / list /
list-conflicts API, backed by a JSON file. The backend sits behind an abstract
base class so a SQLite implementation can be dropped in later without touching
callers.

Conflict detection here is deliberately naive (same ``topic`` key, differing
normalized ``content``). Embedding-similarity clustering + LLM-judge
verification will replace it in ``memory/detector.py``.
"""

from __future__ import annotations

import abc
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional


class Status(str, Enum):
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MemoryItem:
    """One claim written into shared memory by one agent."""

    agent_id: str
    topic: str  # shared key grouping claims about the same question/subject
    content: str  # the claim / finding text
    id: str = field(default_factory=_new_id)
    embedding: Optional[list[float]] = None
    timestamp: float = field(default_factory=time.time)
    status: Status = Status.PROPOSED
    source_doc_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "topic": self.topic,
            "content": self.content,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "source_doc_id": self.source_doc_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        return cls(
            agent_id=d["agent_id"],
            topic=d["topic"],
            content=d["content"],
            id=d["id"],
            embedding=d.get("embedding"),
            timestamp=d["timestamp"],
            status=Status(d["status"]),
            source_doc_id=d.get("source_doc_id"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Conflict:
    """A group of >= 2 live memory items about the same topic that disagree."""

    topic: str
    items: list[MemoryItem]

    @property
    def agent_ids(self) -> list[str]:
        return [it.agent_id for it in self.items]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".!?")


class MemoryStore(abc.ABC):
    """Backend-agnostic interface for the shared memory store."""

    @abc.abstractmethod
    def add(self, item: MemoryItem) -> MemoryItem: ...

    @abc.abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]: ...

    @abc.abstractmethod
    def list(
        self,
        *,
        topic: Optional[str] = None,
        status: Optional[Status] = None,
    ) -> list[MemoryItem]: ...

    @abc.abstractmethod
    def update_status(self, item_id: str, status: Status) -> MemoryItem: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    def list_conflicts(self) -> list[Conflict]:
        """Naive conflict scan: group live items by topic, flag disagreement.

        "Live" means not SUPERSEDED. A topic is in conflict when its live items
        carry more than one distinct normalized content string.
        """
        live = [it for it in self.list() if it.status != Status.SUPERSEDED]
        by_topic: dict[str, list[MemoryItem]] = {}
        for it in live:
            by_topic.setdefault(it.topic, []).append(it)

        conflicts: list[Conflict] = []
        for topic, items in by_topic.items():
            if len({_normalize(it.content) for it in items}) > 1:
                conflicts.append(
                    Conflict(topic=topic, items=sorted(items, key=lambda x: x.timestamp))
                )
        return conflicts


class JsonMemoryStore(MemoryStore):
    """JSON-file-backed store. One file, list of serialized items."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._items: dict[str, MemoryItem] = {}
        if self.path.exists():
            self._load()
        else:
            self._flush()

    # --- persistence ---------------------------------------------------------
    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = {d["id"]: MemoryItem.from_dict(d) for d in raw}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [it.to_dict() for it in self._sorted()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _sorted(self) -> list[MemoryItem]:
        return sorted(self._items.values(), key=lambda it: it.timestamp)

    # --- API ---------------------------------------------------------------
    def add(self, item: MemoryItem) -> MemoryItem:
        self._items[item.id] = item
        self._flush()
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def list(
        self,
        *,
        topic: Optional[str] = None,
        status: Optional[Status] = None,
    ) -> list[MemoryItem]:
        out: Iterable[MemoryItem] = self._sorted()
        if topic is not None:
            out = [it for it in out if it.topic == topic]
        if status is not None:
            out = [it for it in out if it.status == status]
        return list(out)

    def update_status(self, item_id: str, status: Status) -> MemoryItem:
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no memory item with id {item_id!r}")
        item.status = status
        self._flush()
        return item

    def clear(self) -> None:
        self._items.clear()
        self._flush()


def open_store(path: str | Path, backend: str = "json") -> MemoryStore:
    """Factory so callers stay backend-agnostic."""
    if backend == "json":
        return JsonMemoryStore(path)
    raise ValueError(f"unknown memory backend: {backend!r}")
