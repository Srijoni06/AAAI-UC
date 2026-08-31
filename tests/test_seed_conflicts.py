"""Structural tests for the seeded contradiction suite.

No LLM calls: these check that every seed is well-formed and that the suite has
the spread of conflict types / difficulty the evaluation depends on.
"""

from __future__ import annotations

import pytest

from domain.seed_conflicts import (
    COEXIST,
    SEED_CONFLICTS,
    SEEDS_BY_ID,
    ConflictType,
    Difficulty,
    Excerpt,
    SeedConflict,
    by_difficulty,
    by_type,
    get,
    type_counts,
)


def test_suite_size_in_expected_range():
    assert 15 <= len(SEED_CONFLICTS) <= 20


def test_every_seed_has_at_least_two_excerpts():
    for s in SEED_CONFLICTS:
        assert len(s.excerpts) >= 2, s.doc_id


def test_excerpt_ids_unique_within_seed():
    for s in SEED_CONFLICTS:
        ids = [e.excerpt_id for e in s.excerpts]
        assert len(ids) == len(set(ids)), s.doc_id


def test_doc_ids_unique():
    ids = [s.doc_id for s in SEED_CONFLICTS]
    assert len(ids) == len(set(ids))


def test_topics_unique():
    # the store groups claims by topic, so two seeds sharing a topic would merge
    topics = [s.topic for s in SEED_CONFLICTS]
    assert len(topics) == len(set(topics))


def test_questions_and_answers_nonempty():
    for s in SEED_CONFLICTS:
        assert s.question.strip()
        assert s.gold_answer.strip()
        assert s.notes.strip()


def test_gold_excerpt_id_is_valid_or_coexist():
    for s in SEED_CONFLICTS:
        assert s.gold_excerpt_id == COEXIST or s.gold_excerpt_id in s.excerpt_ids, s.doc_id


def test_excerpt_text_nonempty_and_distinct():
    for s in SEED_CONFLICTS:
        texts = [e.text.strip() for e in s.excerpts]
        assert all(texts), s.doc_id
        # the whole point is that excerpts describe the same fact differently
        assert len(set(texts)) == len(texts), s.doc_id


def test_all_four_conflict_types_present():
    present = {s.conflict_type for s in SEED_CONFLICTS}
    assert present == set(ConflictType)


def test_every_conflict_type_has_multiple_examples():
    counts = type_counts()
    for t in ConflictType:
        assert counts[t.value] >= 2, (t, counts)


def test_difficulty_spread():
    present = {s.difficulty for s in SEED_CONFLICTS}
    assert Difficulty.OBVIOUS in present
    assert Difficulty.SUBTLE in present
    assert Difficulty.MODERATE in present


def test_at_least_one_coexist_seed():
    assert any(s.gold_excerpt_id == COEXIST for s in SEED_CONFLICTS)


def test_original_milestone1_seeds_retained():
    for doc_id, gold in [
        ("doc-benchmark", "results"),
        ("doc-gain", "discussion"),
        ("doc-sota", "erratum"),
    ]:
        s = get(doc_id)
        assert s.gold_excerpt_id == gold


def test_seeds_by_id_matches_list():
    assert set(SEEDS_BY_ID) == {s.doc_id for s in SEED_CONFLICTS}
    assert len(SEEDS_BY_ID) == len(SEED_CONFLICTS)


def test_helpers_partition_the_suite():
    from_type = sum(len(by_type(t)) for t in ConflictType)
    from_diff = sum(len(by_difficulty(d)) for d in Difficulty)
    assert from_type == len(SEED_CONFLICTS)
    assert from_diff == len(SEED_CONFLICTS)


def test_seedconflict_excerpt_lookup():
    s = get("doc-benchmark")
    assert isinstance(s.excerpt("results"), Excerpt)
    with pytest.raises(KeyError):
        s.excerpt("does-not-exist")


def test_frozen_dataclasses():
    s = SEED_CONFLICTS[0]
    with pytest.raises(Exception):
        s.doc_id = "mutated"  # type: ignore[misc]
    with pytest.raises(Exception):
        s.excerpts[0].text = "mutated"  # type: ignore[misc]
