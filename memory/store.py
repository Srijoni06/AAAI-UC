"""Shared multi-agent memory store.

Each item written by an agent carries a conflict-aware ``status`` (PROPOSED /
CONFIRMED / CONTESTED / SUPERSEDED) so downstream components can reason about
disagreement instead of silently overwriting.

What this module guarantees:

1. **Only valid status transitions happen.** The lifecycle is a small state
   machine (``_ALLOWED_TRANSITIONS``): PROPOSED is an entry state only,
   SUPERSEDED is terminal. An illegal move -- e.g. SUPERSEDED -> PROPOSED --
   raises :class:`InvalidTransitionError` rather than mutating the item.
2. **Concurrent writers are safe.** :class:`SqliteMemoryStore` runs the database
   in WAL mode with a busy timeout, so several agents (threads or processes) can
   write at once without "database is locked" errors or lost updates. It is the
   default backend from :func:`open_store`.
3. **Rich per-item provenance.** Every item records where the claim came from
   (``source_type``, ``source_doc_id``), how far the channel should be trusted
   (``authority``), whether it is externally grounded or the model's own guess
   (``origin``), the exact supporting text (``evidence_span``), and a ``version``
   counter alongside the existing ``timestamp``.

Conflict detection here is still the naive same-``topic`` scan; embedding
clustering + LLM-judge verification will replace it in ``memory/detector.py``.
"""

from __future__ import annotations

import abc
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Iterable, Optional


class Status(str, Enum):
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"


class SourceType(str, Enum):
    """Where the raw material for a claim came from."""

    RETRIEVAL = "retrieval"  # pulled from a document / corpus / knowledge base
    TOOL = "tool"            # returned by a tool or API call
    USER = "user"            # asserted directly by a human
    MODEL = "model"          # produced by an LLM with no external grounding
    UNKNOWN = "unknown"


class Authority(IntEnum):
    """Credibility level of the source channel. Ordered: higher beats lower."""

    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    AUTHORITATIVE = 4


class Origin(str, Enum):
    """The one bit that matters most for corroboration reasoning."""

    TOOL = "tool"            # externally grounded: a tool result or retrieval
    INFERENCE = "inference"  # the model's own inference / synthesis


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _coerce_str_enum(enum_cls, value):
    """Accept an enum member, its ``.value`` ('tool'), or its name ('TOOL')."""
    if isinstance(value, enum_cls):
        return value
    s = str(value)
    try:
        return enum_cls(s.lower())
    except ValueError:
        pass
    if s.upper() in enum_cls.__members__:
        return enum_cls[s.upper()]
    raise ValueError(f"unknown {enum_cls.__name__}: {value!r}")


def _coerce_authority(value) -> "Authority":
    if isinstance(value, Authority):
        return value
    if isinstance(value, bool):
        raise TypeError("authority must not be a bool")
    if isinstance(value, int):
        return Authority(value)
    s = str(value).strip().upper()
    if s in Authority.__members__:
        return Authority[s]
    if s.isdigit():
        return Authority(int(s))
    raise ValueError(f"unknown authority: {value!r}")


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

    # --- richer provenance / credibility metadata --------------------------
    source_type: SourceType = SourceType.UNKNOWN
    authority: Authority = Authority.UNKNOWN
    origin: Origin = Origin.INFERENCE
    evidence_span: Optional[str] = None  # verbatim text supporting the claim
    version: int = 1

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, Status):
            self.status = Status(self.status)
        if not isinstance(self.source_type, SourceType):
            self.source_type = _coerce_str_enum(SourceType, self.source_type)
        if not isinstance(self.authority, Authority):
            self.authority = _coerce_authority(self.authority)
        if not isinstance(self.origin, Origin):
            self.origin = _coerce_str_enum(Origin, self.origin)
        self.version = int(self.version)

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
            "source_type": self.source_type.value,
            "authority": self.authority.name,
            "origin": self.origin.value,
            "evidence_span": self.evidence_span,
            "version": self.version,
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
            source_type=d.get("source_type", SourceType.UNKNOWN.value),
            authority=d.get("authority", Authority.UNKNOWN.name),
            origin=d.get("origin", Origin.INFERENCE.value),
            evidence_span=d.get("evidence_span"),
            version=d.get("version", 1),
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


# --------------------------------------------------------------------------- #
# Status lifecycle
# --------------------------------------------------------------------------- #
class InvalidTransitionError(RuntimeError):
    """Raised when a status change would violate the memory-item lifecycle."""

    def __init__(self, current: Status, requested: Status) -> None:
        super().__init__(
            f"invalid status transition: {current.value} -> {requested.value}"
        )
        self.current = current
        self.requested = requested


# PROPOSED is an entry state only (nothing transitions *back* to it).
# SUPERSEDED is terminal.
_ALLOWED_TRANSITIONS: dict[Status, set[Status]] = {
    Status.PROPOSED: {Status.CONFIRMED, Status.CONTESTED, Status.SUPERSEDED},
    Status.CONTESTED: {Status.CONFIRMED, Status.SUPERSEDED},
    Status.CONFIRMED: {Status.CONTESTED, Status.SUPERSEDED},
    Status.SUPERSEDED: set(),
}


def can_transition(current: Status, requested: Status) -> bool:
    """True if ``current -> requested`` is legal. A no-op is always allowed."""
    if current == requested:
        return True
    return requested in _ALLOWED_TRANSITIONS.get(current, set())


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

    # -- shared helpers -------------------------------------------------- #
    @staticmethod
    def _guard_transition(current: Status, requested: Status) -> None:
        if not can_transition(current, requested):
            raise InvalidTransitionError(current, requested)

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
    """JSON-file-backed store. One file, list of serialized items.

    Kept for small local fixtures; it rewrites the whole file on every mutation
    and gives no concurrency guarantees. Use :class:`SqliteMemoryStore` when more
    than one writer is involved.
    """

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
        if item.id in self._items:
            raise KeyError(f"duplicate memory item id {item.id!r}")
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
        if not isinstance(status, Status):
            status = Status(status)
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no memory item with id {item_id!r}")
        if item.status != status:
            self._guard_transition(item.status, status)
            item.status = status
            self._flush()
        return item

    def clear(self) -> None:
        self._items.clear()
        self._flush()


class SqliteMemoryStore(MemoryStore):
    """SQLite-backed store in WAL mode -- the default persistent backend.

    WAL lets many readers run concurrently with a single writer, and
    ``busy_timeout`` makes concurrent writers queue instead of failing with
    "database is locked". Every operation uses its own short-lived connection, so
    one instance is safe to share across threads and multiple instances (threads
    or processes) may point at the same file.
    """

    _COLS = (
        "id", "agent_id", "topic", "content", "embedding", "timestamp", "status",
        "source_doc_id", "source_type", "authority", "origin", "evidence_span",
        "version", "metadata",
    )

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_items (
        id            TEXT PRIMARY KEY,
        agent_id      TEXT NOT NULL,
        topic         TEXT NOT NULL,
        content       TEXT NOT NULL,
        embedding     TEXT,
        timestamp     REAL NOT NULL,
        status        TEXT NOT NULL,
        source_doc_id TEXT,
        source_type   TEXT NOT NULL,
        authority     TEXT NOT NULL,
        origin        TEXT NOT NULL,
        evidence_span TEXT,
        version       INTEGER NOT NULL DEFAULT 1,
        metadata      TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_items_topic  ON memory_items(topic);
    CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(self._SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # --- connection management -------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # journal_mode=WAL is persisted in the db header; re-asserting is cheap.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            with conn:  # commit on success, rollback on exception
                yield conn
        finally:
            conn.close()

    # --- (de)serialization ---------------------------------------------- #
    @staticmethod
    def _row_from_item(item: MemoryItem) -> dict:
        d = item.to_dict()
        d["embedding"] = json.dumps(d["embedding"]) if d["embedding"] is not None else None
        d["metadata"] = json.dumps(d["metadata"])
        return d

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> MemoryItem:
        d = dict(row)
        d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
        return MemoryItem.from_dict(d)

    # --- API ----------------------------------------------------------- #
    def add(self, item: MemoryItem) -> MemoryItem:
        row = self._row_from_item(item)
        cols = ", ".join(self._COLS)
        placeholders = ", ".join("?" for _ in self._COLS)
        try:
            with self._tx() as conn:
                conn.execute(
                    f"INSERT INTO memory_items ({cols}) VALUES ({placeholders})",
                    [row[c] for c in self._COLS],
                )
        except sqlite3.IntegrityError as exc:
            raise KeyError(f"duplicate memory item id {item.id!r}") from exc
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def list(
        self,
        *,
        topic: Optional[str] = None,
        status: Optional[Status] = None,
    ) -> list[MemoryItem]:
        query = "SELECT * FROM memory_items"
        clauses: list[str] = []
        params: list = []
        if topic is not None:
            clauses.append("topic = ?")
            params.append(topic)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value if isinstance(status, Status) else str(status))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp"
        with self._tx() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._item_from_row(r) for r in rows]

    def update_status(self, item_id: str, status: Status) -> MemoryItem:
        if not isinstance(status, Status):
            status = Status(status)
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no memory item with id {item_id!r}")
            current = Status(row["status"])
            if current != status:
                self._guard_transition(current, status)
                conn.execute(
                    "UPDATE memory_items SET status = ? WHERE id = ?",
                    (status.value, item_id),
                )
                row = conn.execute(
                    "SELECT * FROM memory_items WHERE id = ?", (item_id,)
                ).fetchone()
        return self._item_from_row(row)

    def clear(self) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM memory_items")


def open_store(path: str | Path, backend: str = "sqlite") -> MemoryStore:
    """Factory so callers stay backend-agnostic. Defaults to WAL-mode SQLite."""
    if backend == "sqlite":
        return SqliteMemoryStore(path)
    if backend == "json":
        return JsonMemoryStore(path)
    raise ValueError(f"unknown memory backend: {backend!r}")
