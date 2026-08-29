"""Tests for the shared memory store and its naive conflict scan."""

from __future__ import annotations

import time

from baselines.base import apply_resolution
from baselines.last_write_wins import resolve as lww_resolve
from memory.store import JsonMemoryStore, MemoryItem, Status


def _store(tmp_path) -> JsonMemoryStore:
    return JsonMemoryStore(tmp_path / "mem.json")


def test_add_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    item = store.add(MemoryItem(agent_id="a", topic="t", content="hello"))
    got = store.get(item.id)
    assert got is not None
    assert got.content == "hello"
    assert got.status is Status.PROPOSED


def test_persistence_reload(tmp_path):
    path = tmp_path / "mem.json"
    s1 = JsonMemoryStore(path)
    s1.add(MemoryItem(agent_id="a", topic="t", content="x"))
    s2 = JsonMemoryStore(path)
    assert len(s2.list()) == 1


def test_list_filters(tmp_path):
    store = _store(tmp_path)
    store.add(MemoryItem(agent_id="a", topic="t1", content="x"))
    store.add(MemoryItem(agent_id="b", topic="t2", content="y"))
    assert len(store.list(topic="t1")) == 1
    assert len(store.list(status=Status.PROPOSED)) == 2


def test_list_conflicts_flags_disagreement(tmp_path):
    store = _store(tmp_path)
    store.add(MemoryItem(agent_id="a", topic="benchmark", content="It is CoNLL-2003."))
    store.add(MemoryItem(agent_id="b", topic="benchmark", content="It is OntoNotes 5.0"))
    conflicts = store.list_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].topic == "benchmark"
    assert set(conflicts[0].agent_ids) == {"a", "b"}


def test_no_conflict_when_agents_agree(tmp_path):
    store = _store(tmp_path)
    store.add(MemoryItem(agent_id="a", topic="benchmark", content="It is CoNLL-2003."))
    store.add(MemoryItem(agent_id="b", topic="benchmark", content="it is conll-2003"))
    assert store.list_conflicts() == []


def test_superseded_items_excluded_from_conflicts(tmp_path):
    store = _store(tmp_path)
    a = store.add(MemoryItem(agent_id="a", topic="k", content="old"))
    store.add(MemoryItem(agent_id="b", topic="k", content="new"))
    assert len(store.list_conflicts()) == 1
    store.update_status(a.id, Status.SUPERSEDED)
    assert store.list_conflicts() == []


def test_last_write_wins_resolution(tmp_path):
    store = _store(tmp_path)
    first = store.add(MemoryItem(agent_id="a", topic="k", content="claim one"))
    time.sleep(0.01)
    second = store.add(MemoryItem(agent_id="b", topic="k", content="claim two"))

    conflict = store.list_conflicts()[0]
    resolution = lww_resolve(conflict)
    assert resolution.winner_id == second.id
    assert resolution.superseded_ids == [first.id]

    apply_resolution(store, resolution)
    assert store.get(second.id).status is Status.CONFIRMED
    assert store.get(first.id).status is Status.SUPERSEDED
    assert store.list_conflicts() == []
