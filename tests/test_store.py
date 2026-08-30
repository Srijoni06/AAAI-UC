"""Tests for the shared memory store: schema, status lifecycle, SQLite/WAL
persistence, concurrent writes, and the multi-outcome Resolution contract.

This is the part that has to be *correct*, not just runnable. No LLM calls here.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from baselines.base import ItemOutcome, Outcome, Resolution, apply_resolution
from baselines.last_write_wins import LastWriteWins
from baselines.last_write_wins import resolve as lww_resolve
from memory.store import (
    Authority,
    Conflict,
    InvalidTransitionError,
    JsonMemoryStore,
    MemoryItem,
    Origin,
    SourceType,
    SqliteMemoryStore,
    Status,
    _normalize,
    can_transition,
    open_store,
)


@pytest.fixture
def store(tmp_path) -> SqliteMemoryStore:
    return SqliteMemoryStore(tmp_path / "mem.db")


def _item(agent_id="a", topic="t", content="hello", **kw) -> MemoryItem:
    return MemoryItem(agent_id=agent_id, topic=topic, content=content, **kw)


def _conflict(store, topic="k"):
    """Two disagreeing claims on one topic, deterministic timestamps."""
    a = store.add(_item("a", topic, "claim one", timestamp=1.0))
    b = store.add(_item("b", topic, "claim two", timestamp=2.0))
    return store.list_conflicts()[0], a, b


# --------------------------------------------------------------------------- #
# schema / add / list
# --------------------------------------------------------------------------- #
def test_add_get_roundtrip(store):
    item = store.add(_item(content="hello"))
    got = store.get(item.id)
    assert got is not None
    assert got.content == "hello"
    assert got.status is Status.PROPOSED
    # rich-metadata defaults
    assert got.source_type is SourceType.UNKNOWN
    assert got.authority is Authority.UNKNOWN
    assert got.origin is Origin.INFERENCE
    assert got.evidence_span is None
    assert got.version == 1


def test_rich_metadata_roundtrips(store):
    item = store.add(
        _item(
            content="ROUGE-L rose 4.8 points",
            source_doc_id="glxnet",
            source_type="retrieval",
            authority="HIGH",
            origin="tool",
            evidence_span="ROUGE-L improved by 4.8 points over the baseline",
            version=3,
            embedding=[0.1, 0.2, 0.3],
        )
    )
    got = store.get(item.id)
    assert got.source_type is SourceType.RETRIEVAL
    assert got.authority is Authority.HIGH
    assert got.origin is Origin.TOOL
    assert got.evidence_span == "ROUGE-L improved by 4.8 points over the baseline"
    assert got.version == 3
    assert got.embedding == [0.1, 0.2, 0.3]


def test_rich_metadata_rejects_unknown_values():
    with pytest.raises(ValueError):
        _item(source_type="telepathy")
    with pytest.raises(ValueError):
        _item(authority="ABSOLUTE")


def test_authority_is_ordered():
    assert Authority.AUTHORITATIVE > Authority.HIGH > Authority.LOW > Authority.UNKNOWN


def test_persistence_reload(tmp_path):
    path = tmp_path / "mem.db"
    s1 = SqliteMemoryStore(path)
    s1.add(_item(content="x", authority="MEDIUM"))
    s2 = SqliteMemoryStore(path)
    items = s2.list()
    assert len(items) == 1
    assert items[0].authority is Authority.MEDIUM


def test_list_filters(store):
    store.add(_item("a", "t1", "x"))
    store.add(_item("b", "t2", "y"))
    assert len(store.list(topic="t1")) == 1
    assert len(store.list(status=Status.PROPOSED)) == 2
    assert store.list(status=Status.CONFIRMED) == []


def test_list_is_ordered_by_timestamp(store):
    store.add(_item("a", "t", "second", timestamp=100.0))
    store.add(_item("b", "t", "first", timestamp=50.0))
    assert [it.content for it in store.list()] == ["first", "second"]


# --------------------------------------------------------------------------- #
# list_conflicts
# --------------------------------------------------------------------------- #
def test_list_conflicts_flags_disagreement(store):
    store.add(_item("a", "benchmark", "It is CoNLL-2003."))
    store.add(_item("b", "benchmark", "It is OntoNotes 5.0"))
    conflicts = store.list_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].topic == "benchmark"
    assert set(conflicts[0].agent_ids) == {"a", "b"}


def test_no_conflict_when_agents_agree(store):
    store.add(_item("a", "benchmark", "It is CoNLL-2003."))
    store.add(_item("b", "benchmark", "it is conll-2003"))
    assert store.list_conflicts() == []


def test_superseded_items_excluded_from_conflicts(store):
    a = store.add(_item("a", "k", "old"))
    store.add(_item("b", "k", "new"))
    assert len(store.list_conflicts()) == 1
    store.update_status(a.id, Status.SUPERSEDED)
    assert store.list_conflicts() == []


# --------------------------------------------------------------------------- #
# status lifecycle: valid transitions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "start,end",
    [
        (Status.PROPOSED, Status.CONFIRMED),
        (Status.PROPOSED, Status.CONTESTED),
        (Status.PROPOSED, Status.SUPERSEDED),
        (Status.CONTESTED, Status.CONFIRMED),
        (Status.CONTESTED, Status.SUPERSEDED),
        (Status.CONFIRMED, Status.CONTESTED),
        (Status.CONFIRMED, Status.SUPERSEDED),
    ],
)
def test_valid_transitions_are_allowed(store, start, end):
    it = store.add(_item(content="claim", status=start))
    updated = store.update_status(it.id, end)
    assert updated.status is end
    assert store.get(it.id).status is end
    assert can_transition(start, end)


def test_same_status_is_a_noop(store):
    it = store.add(_item(content="claim", status=Status.CONFIRMED))
    assert store.update_status(it.id, Status.CONFIRMED).status is Status.CONFIRMED


# --------------------------------------------------------------------------- #
# status lifecycle: invalid transitions are blocked (not silently applied)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "start,end",
    [
        (Status.SUPERSEDED, Status.PROPOSED),
        (Status.SUPERSEDED, Status.CONFIRMED),
        (Status.SUPERSEDED, Status.CONTESTED),
        (Status.CONFIRMED, Status.PROPOSED),
        (Status.CONTESTED, Status.PROPOSED),
    ],
)
def test_invalid_transitions_raise_and_do_not_mutate(store, start, end):
    it = store.add(_item(content="claim", status=start))
    with pytest.raises(InvalidTransitionError) as exc:
        store.update_status(it.id, end)
    assert exc.value.current is start
    assert exc.value.requested is end
    assert store.get(it.id).status is start  # unchanged on disk


def test_superseded_going_back_to_proposed_is_blocked(store):
    it = store.add(_item(content="claim"))
    store.update_status(it.id, Status.SUPERSEDED)
    with pytest.raises(InvalidTransitionError):
        store.update_status(it.id, Status.PROPOSED)
    assert store.get(it.id).status is Status.SUPERSEDED


def test_invalid_transition_blocked_in_json_backend_too(tmp_path):
    store = JsonMemoryStore(tmp_path / "mem.json")
    it = store.add(_item(content="claim"))
    store.update_status(it.id, Status.SUPERSEDED)
    with pytest.raises(InvalidTransitionError):
        store.update_status(it.id, Status.CONFIRMED)


def test_can_transition_table():
    assert can_transition(Status.PROPOSED, Status.CONFIRMED)
    assert can_transition(Status.CONFIRMED, Status.CONFIRMED)  # no-op
    assert not can_transition(Status.SUPERSEDED, Status.CONFIRMED)
    assert not can_transition(Status.CONFIRMED, Status.PROPOSED)


def test_update_status_missing_id_raises(store):
    with pytest.raises(KeyError):
        store.update_status("nope", Status.CONFIRMED)


# --------------------------------------------------------------------------- #
# multi-outcome Resolution contract
# --------------------------------------------------------------------------- #
def test_resolution_single_winner_shape(store):
    conflict, a, b = _conflict(store)
    res = Resolution.single_winner("k", "manual", b.id, [a.id])
    assert res.confirmed_ids == [b.id]
    assert res.superseded_ids == [a.id]
    assert res.contested_ids == []
    assert res.winner_id == b.id  # back-compat view
    assert res.is_single_winner


def test_resolution_coexist_confirms_all(store):
    conflict, a, b = _conflict(store)
    res = Resolution.coexist("k", "reconciler", [a.id, b.id])
    assert set(res.confirmed_ids) == {a.id, b.id}
    assert res.winner_id is None  # >1 CONFIRMED -> no single winner
    assert not res.is_single_winner


def test_resolution_all_contested(store):
    conflict, a, b = _conflict(store)
    res = Resolution.all_contested("k", "reconciler", [a.id, b.id])
    assert set(res.contested_ids) == {a.id, b.id}
    assert res.confirmed_ids == []


def test_resolution_rejects_duplicate_items():
    with pytest.raises(ValueError):
        Resolution(
            topic="k",
            strategy="broken",
            outcomes=[
                ItemOutcome("x", Outcome.CONFIRMED),
                ItemOutcome("x", Outcome.SUPERSEDED),
            ],
        )


def test_resolution_rejects_non_outcome():
    with pytest.raises((TypeError, ValueError)):
        Resolution(topic="k", strategy="broken", outcomes=[ItemOutcome("x", "WINNER")])


# --------------------------------------------------------------------------- #
# apply_resolution
# --------------------------------------------------------------------------- #
def test_apply_resolution_single_winner(store):
    conflict, a, b = _conflict(store)
    apply_resolution(store, lww_resolve(conflict))
    assert store.get(b.id).status is Status.CONFIRMED
    assert store.get(a.id).status is Status.SUPERSEDED
    assert store.list_conflicts() == []


def test_apply_resolution_coexist(store):
    conflict, a, b = _conflict(store)
    apply_resolution(store, Resolution.coexist("k", "reconciler", [a.id, b.id]))
    assert store.get(a.id).status is Status.CONFIRMED
    assert store.get(b.id).status is Status.CONFIRMED


def test_apply_resolution_all_contested(store):
    conflict, a, b = _conflict(store)
    apply_resolution(store, Resolution.all_contested("k", "reconciler", [a.id, b.id]))
    assert store.get(a.id).status is Status.CONTESTED
    assert store.get(b.id).status is Status.CONTESTED


def test_apply_resolution_mixed_outcomes(store):
    keep1 = store.add(_item("a", "k", "keep-1", timestamp=1.0))
    keep2 = store.add(_item("b", "k", "keep-2", timestamp=2.0))
    drop = store.add(_item("c", "k", "drop", timestamp=3.0))
    res = Resolution(
        topic="k",
        strategy="reconciler",
        outcomes=[
            ItemOutcome(keep1.id, Outcome.CONFIRMED),
            ItemOutcome(keep2.id, Outcome.CONFIRMED),
            ItemOutcome(drop.id, Outcome.SUPERSEDED),
        ],
    )
    apply_resolution(store, res)
    assert store.get(keep1.id).status is Status.CONFIRMED
    assert store.get(keep2.id).status is Status.CONFIRMED
    assert store.get(drop.id).status is Status.SUPERSEDED


def test_apply_resolution_illegal_outcome_raises(store):
    it = store.add(_item(content="x"))
    store.update_status(it.id, Status.SUPERSEDED)
    res = Resolution(
        topic="k", strategy="broken", outcomes=[ItemOutcome(it.id, Outcome.CONFIRMED)]
    )
    with pytest.raises(InvalidTransitionError):
        apply_resolution(store, res)


# --------------------------------------------------------------------------- #
# last_write_wins still works and maps onto the new contract
# --------------------------------------------------------------------------- #
def test_last_write_wins_resolution(store):
    first = store.add(_item("a", "k", "claim one", timestamp=1.0))
    second = store.add(_item("b", "k", "claim two", timestamp=2.0))

    conflict = store.list_conflicts()[0]
    resolution = LastWriteWins().resolve(conflict)
    assert resolution.strategy == "last_write_wins"
    assert resolution.winner_id == second.id
    assert resolution.superseded_ids == [first.id]
    assert resolution.scores[second.id] == 2.0

    apply_resolution(store, resolution)
    assert store.get(second.id).status is Status.CONFIRMED
    assert store.get(first.id).status is Status.SUPERSEDED


# --------------------------------------------------------------------------- #
# SQLite backend: WAL + concurrency
# --------------------------------------------------------------------------- #
def test_sqlite_database_is_in_wal_mode(tmp_path):
    path = tmp_path / "mem.db"
    SqliteMemoryStore(path)
    conn = sqlite3.connect(path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_sqlite_duplicate_id_raises(store):
    it = _item(content="x")
    store.add(it)
    with pytest.raises(KeyError):
        store.add(it)


def test_open_store_defaults_to_sqlite(tmp_path):
    assert isinstance(open_store(tmp_path / "m.db"), SqliteMemoryStore)
    assert isinstance(open_store(tmp_path / "m.json", backend="json"), JsonMemoryStore)
    with pytest.raises(ValueError):
        open_store(tmp_path / "x", backend="bogus")


def test_sqlite_concurrent_writes_do_not_lose_rows(tmp_path):
    path = tmp_path / "mem.db"
    SqliteMemoryStore(path)  # initialise schema once, up front

    n_writers, per_writer = 8, 25

    def worker(k: int) -> None:
        s = SqliteMemoryStore(path)  # a fresh backend per thread = independent agent
        for i in range(per_writer):
            s.add(_item(f"agent_{k}", f"t{k}", f"claim {k}-{i}"))

    with ThreadPoolExecutor(max_workers=n_writers) as ex:
        for fut in [ex.submit(worker, k) for k in range(n_writers)]:
            fut.result()  # re-raise anything a worker hit

    items = SqliteMemoryStore(path).list()
    assert len(items) == n_writers * per_writer
    assert len({it.id for it in items}) == n_writers * per_writer


def test_sqlite_reads_are_safe_during_writes(tmp_path):
    path = tmp_path / "mem.db"
    SqliteMemoryStore(path)

    errors: list[BaseException] = []
    done = threading.Event()

    def writer() -> None:
        try:
            s = SqliteMemoryStore(path)
            for i in range(100):
                s.add(_item("w", "x", f"c{i}"))
        except BaseException as exc:  # noqa: BLE001 - test wants any failure surfaced
            errors.append(exc)
        finally:
            done.set()

    def reader() -> None:
        try:
            s = SqliteMemoryStore(path)
            while not done.is_set():
                s.list(topic="x")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    tw, tr = threading.Thread(target=writer), threading.Thread(target=reader)
    tr.start()
    tw.start()
    tw.join()
    tr.join()

    assert errors == []
    assert len(SqliteMemoryStore(path).list()) == 100


def test_sqlite_concurrent_status_updates_are_serialised(tmp_path):
    path = tmp_path / "mem.db"
    s = SqliteMemoryStore(path)
    items = [s.add(_item(f"a{i}", "t", "claim")) for i in range(30)]

    errors: list[BaseException] = []

    def confirm(item_id: str) -> None:
        try:
            SqliteMemoryStore(path).update_status(item_id, Status.CONFIRMED)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in [ex.submit(confirm, it.id) for it in items]:
            fut.result()

    assert errors == []
    assert all(it.status is Status.CONFIRMED for it in SqliteMemoryStore(path).list())


# --------------------------------------------------------------------------- #
# JSON backend (legacy, kept for small fixtures)
# --------------------------------------------------------------------------- #
def test_json_backend_roundtrip(tmp_path):
    path = tmp_path / "mem.json"
    s1 = JsonMemoryStore(path)
    it = s1.add(_item(content="persisted", source_type="retrieval", version=2))
    s1.update_status(it.id, Status.CONFIRMED)

    s2 = JsonMemoryStore(path)
    got = s2.get(it.id)
    assert got is not None
    assert got.status is Status.CONFIRMED
    assert got.source_type is SourceType.RETRIEVAL
    assert got.version == 2


def test_json_backend_rejects_duplicate_id(tmp_path):
    store = JsonMemoryStore(tmp_path / "mem.json")
    it = _item(content="x")
    store.add(it)
    with pytest.raises(KeyError):
        store.add(it)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def test_normalize():
    assert _normalize("  The   Gain was 4.8! ") == "the gain was 4.8"


def test_conflict_dataclass_agent_ids():
    c = Conflict(topic="t", items=[_item("a"), _item("b")])
    assert c.agent_ids == ["a", "b"]
