"""Tests for the 5-agent orchestration loop.

No network: a fake LLM returns a deterministic claim per (excerpt, prompt) so we
can assert grouping, excerpt assignment, correlated-agent agreement, and that
real provenance is written for every claim.
"""

from __future__ import annotations

import pytest

from agents.orchestrator import (
    DEFAULT_ROSTER,
    INDEPENDENT,
    AgentSpec,
    agent_groups,
    assign_excerpts,
    correlated_groups,
    independent_agents,
    run,
)
from common.llm import LLMClient
from domain.seed_conflicts import SEED_CONFLICTS, get
from memory.store import Origin, SourceType, SqliteMemoryStore, Status


class FakeLLM(LLMClient):
    """Deterministic, offline. Claim depends only on the excerpt text so that
    agents reading the same excerpt (the correlated group) produce identical
    claims and agents on other excerpts differ."""

    backend = "fake"
    agent_model = "fake-agent"
    judge_model = "fake-judge"

    def __init__(self):
        super().__init__(cache=None)
        self.calls: list[dict] = []

    def _raw_generate(self, prompt, *, system, temperature, model):
        # prompt embeds the excerpt between "Excerpt:\n" and "\n\nQuestion:"
        excerpt = prompt.split("Excerpt:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        self.calls.append({"system": system, "excerpt": excerpt, "model": model})
        return f"CLAIM<{excerpt[:40]}>"


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def store(tmp_path):
    return SqliteMemoryStore(tmp_path / "mem.db")


@pytest.fixture
def two_seeds():
    return [get("doc-benchmark"), get("doc-cost")]


# --------------------------------------------------------------------------- #
# roster / grouping
# --------------------------------------------------------------------------- #
def test_default_roster_is_five_agents():
    assert [s.agent_id for s in DEFAULT_ROSTER] == [
        "agent_A", "agent_B", "agent_C", "agent_D", "agent_E"
    ]


def test_default_grouping_A_C_D_correlated_B_E_independent():
    corr = correlated_groups()
    assert corr == {"grp_A": ["agent_A", "agent_C", "agent_D"]}
    assert independent_agents() == ["agent_B", "agent_E"]


def test_agent_groups_treats_independents_as_singletons():
    groups = agent_groups()
    assert groups["grp_A"] == ["agent_A", "agent_C", "agent_D"]
    assert groups["agent_B"] == ["agent_B"]
    assert groups["agent_E"] == ["agent_E"]


def test_grouping_is_configurable():
    # three independent agents vs a two-member correlated group
    roster = [
        AgentSpec("i1", "p", group=INDEPENDENT),
        AgentSpec("i2", "p", group=INDEPENDENT),
        AgentSpec("i3", "p", group=INDEPENDENT),
        AgentSpec("c1", "p", group="grp_x"),
        AgentSpec("c2", "p", group="grp_x"),
    ]
    assert correlated_groups(roster) == {"grp_x": ["c1", "c2"]}
    assert independent_agents(roster) == ["i1", "i2", "i3"]


# --------------------------------------------------------------------------- #
# excerpt assignment
# --------------------------------------------------------------------------- #
def test_correlated_group_all_read_the_anchor_excerpt():
    seed = get("doc-benchmark")  # excerpts: intro, results
    mapping = assign_excerpts(seed)
    assert mapping["agent_A"].excerpt_id == "intro"
    assert mapping["agent_C"].excerpt_id == "intro"
    assert mapping["agent_D"].excerpt_id == "intro"
    # independents land on the *other* excerpt, so they can contradict the group
    assert mapping["agent_B"].excerpt_id == "results"
    assert mapping["agent_E"].excerpt_id == "results"


def test_anchor_excerpt_index_is_configurable():
    seed = get("doc-benchmark")
    mapping = assign_excerpts(seed, anchor_excerpt_index=1)
    assert mapping["agent_A"].excerpt_id == "results"
    assert mapping["agent_B"].excerpt_id == "intro"


def test_independents_spread_over_multiple_non_anchor_excerpts():
    seed = get("doc-benchmark")
    roster = [
        AgentSpec("anchor", "p", group="g"),
        AgentSpec("m", "p", group="g"),
        AgentSpec("i1", "p", group=INDEPENDENT),
        AgentSpec("i2", "p", group=INDEPENDENT),
    ]
    # only 2 excerpts -> both independents share the single non-anchor excerpt
    mapping = assign_excerpts(seed, roster)
    assert mapping["anchor"].excerpt_id == "intro"
    assert mapping["i1"].excerpt_id == "results"
    assert mapping["i2"].excerpt_id == "results"


# --------------------------------------------------------------------------- #
# run(): writes + provenance
# --------------------------------------------------------------------------- #
def test_run_writes_one_item_per_agent_per_seed(store, llm, two_seeds):
    writes = run(store, seeds=two_seeds, llm=llm)
    assert len(writes) == 2 * 5
    items = store.list()
    assert len(items) == 10
    assert {it.agent_id for it in items} == {
        "agent_A", "agent_B", "agent_C", "agent_D", "agent_E"
    }


def test_correlated_agents_produce_identical_claims(store, llm, two_seeds):
    writes = run(store, seeds=two_seeds, llm=llm)
    for seed in two_seeds:
        per_agent = {w.agent_id: w.claim for w in writes if w.seed is seed}
        assert per_agent["agent_A"] == per_agent["agent_C"] == per_agent["agent_D"]
        # an independent agent read the other excerpt -> different claim
        assert per_agent["agent_B"] != per_agent["agent_A"]


def test_every_write_has_real_provenance(store, llm, two_seeds):
    run(store, seeds=two_seeds, llm=llm)
    for it in store.list():
        seed = get(it.source_doc_id)
        # not schema defaults
        assert it.source_type is SourceType.RETRIEVAL
        assert it.origin is Origin.TOOL
        assert it.evidence_span is not None
        # evidence_span is the verbatim slice of a real excerpt of that doc
        assert it.evidence_span in {e.text for e in seed.excerpts}
        exc_id = it.metadata["excerpt_id"]
        assert it.evidence_span == seed.excerpt(exc_id).text
        assert it.metadata["source_id"] == f"{it.source_doc_id}#{exc_id}"
        assert it.metadata["section"] == seed.excerpt(exc_id).section
        assert it.metadata["question"] == seed.question
        assert it.metadata["agent_group"] in {"grp_A", INDEPENDENT}


def test_provenance_group_tag_matches_roster(store, llm, two_seeds):
    run(store, seeds=two_seeds, llm=llm)
    for it in store.list():
        expected = "grp_A" if it.agent_id in {"agent_A", "agent_C", "agent_D"} else INDEPENDENT
        assert it.metadata["agent_group"] == expected


def test_correlated_group_shares_evidence_span(store, llm, two_seeds):
    run(store, seeds=two_seeds, llm=llm)
    for seed in two_seeds:
        spans = {
            it.evidence_span
            for it in store.list(topic=seed.topic)
            if it.agent_id in {"agent_A", "agent_C", "agent_D"}
        }
        assert len(spans) == 1  # all three read the exact same slice


def test_run_is_configurable_roster(store, llm):
    roster = [
        AgentSpec("solo_1", "p", group=INDEPENDENT),
        AgentSpec("solo_2", "p", group=INDEPENDENT),
    ]
    writes = run(store, seeds=[get("doc-benchmark")], llm=llm, roster=roster)
    assert {w.agent_id for w in writes} == {"solo_1", "solo_2"}
    # different excerpts -> different claims, no correlated group
    assert writes[0].claim != writes[1].claim
    assert correlated_groups(roster) == {}


def test_items_are_proposed_on_write(store, llm, two_seeds):
    run(store, seeds=two_seeds, llm=llm)
    assert all(it.status is Status.PROPOSED for it in store.list())


def test_correct_flag_tracks_gold_excerpt(store, llm):
    # doc-benchmark gold = "results"; correlated group reads "intro" -> off-gold
    writes = run(store, seeds=[get("doc-benchmark")], llm=llm)
    by_agent = {w.agent_id: w for w in writes}
    assert by_agent["agent_A"].correct is False
    assert by_agent["agent_B"].correct is True


def test_coexist_seed_leaves_correct_flag_none(store, llm):
    writes = run(store, seeds=[get("doc-languages")], llm=llm)
    assert all(w.correct is None for w in writes)


def test_full_suite_runs_offline(store, llm):
    writes = run(store, seeds=SEED_CONFLICTS, llm=llm)
    assert len(writes) == len(SEED_CONFLICTS) * 5
    assert len(store.list()) == len(SEED_CONFLICTS) * 5
