"""Tests for memory/detector.py: two-stage contradiction detection."""

from __future__ import annotations

import pytest

from common.llm import LLMClient
from memory.detector import (
    ALL_RELATIONSHIPS,
    CandidatePair,
    ConflictDetector,
    ConflictPair,
    Relationship,
    SentenceEmbedder,
    cosine_similarity,
)
from memory.store import (
    Conflict,
    JsonMemoryStore,
    MemoryItem,
    SqliteMemoryStore,
    Status,
)


class FakeJudgeLLM(LLMClient):
    """Offline test LLM that classifies based on rule-based keyword triggers."""

    backend = "test"
    agent_model = "test-agent"
    judge_model = "test-judge"

    def __init__(self, responses: dict[tuple[str, str], str] | None = None):
        super().__init__(cache=None)
        self.responses = responses or {}
        self.call_history: list[str] = []

    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str:
        self.call_history.append(prompt)
        for (sub1, sub2), rel in self.responses.items():
            if sub1 in prompt and sub2 in prompt:
                return (
                    f'{{"relationship": "{rel}", "rationale": "Matched test rule {sub1}/{sub2}"}}'
                )

        # Fallback heuristic for common test scenarios
        prompt_lower = prompt.lower()
        if "under 24 gpu-hours" in prompt_lower and "11,500 a100-hours" in prompt_lower:
            return '{"relationship": "CONTRADICTION", "rationale": "24 GPU-hours conflicts with 11,500 A100-hours."}'
        if "under 24 gpu-hours" in prompt_lower and "under 24 gpu-hours of compute" in prompt_lower:
            return '{"relationship": "ENTAILMENT", "rationale": "Both state compute was under 24 GPU-hours."}'
        if "english, german" in prompt_lower and "french and italian" in prompt_lower:
            return '{"relationship": "NEUTRAL", "rationale": "Different language evaluations that can coexist."}'

        return '{"relationship": "NEUTRAL", "rationale": "No conflict identified."}'


def make_item(
    topic: str,
    content: str,
    agent_id: str = "agent_A",
    item_id: str | None = None,
    embedding: list[float] | None = None,
    status: Status = Status.PROPOSED,
) -> MemoryItem:
    item = MemoryItem(
        agent_id=agent_id,
        topic=topic,
        content=content,
        embedding=embedding,
        status=status,
    )
    if item_id is not None:
        item.id = item_id
    return item


# --------------------------------------------------------------------------- #
# Math & Candidate Filtering Tests
# --------------------------------------------------------------------------- #
def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_stage1_candidate_filtering():
    def mock_embed(texts: list[str]) -> list[list[float]]:
        # Map specific phrases to fixed vectors
        res = []
        for t in texts:
            if "24" in t:
                res.append([1.0, 0.1])
            elif "11,500" in t:
                res.append([0.9, 0.2])  # high similarity to "24" because both discuss compute
            else:
                res.append([0.0, 1.0])
        return res

    embedder = SentenceEmbedder(embed_fn=mock_embed)
    detector = ConflictDetector(embedder=embedder, similarity_threshold=0.5)

    item1 = make_item("compute", "Pretraining took under 24 GPU-hours.", agent_id="agent_A", item_id="1")
    item2 = make_item("compute", "Pretraining took 11,500 A100-hours.", agent_id="agent_B", item_id="2")
    item3 = make_item("compute", "Completely unrelated text.", agent_id="agent_C", item_id="3")
    item_diff_topic = make_item("dataset", "Pretraining took under 24 GPU-hours.", agent_id="agent_D", item_id="4")

    candidates = detector.find_candidates([item1, item2, item3, item_diff_topic])

    # Only item1 and item2 share topic and exceed similarity threshold
    assert len(candidates) == 1
    c = candidates[0]
    assert c.topic == "compute"
    assert {c.item_a.id, c.item_b.id} == {"1", "2"}
    assert c.similarity > 0.8


def test_stage1_preserves_existing_embeddings():
    embed_calls = 0

    def mock_embed(texts: list[str]) -> list[list[float]]:
        nonlocal embed_calls
        embed_calls += len(texts)
        return [[1.0, 0.0] for _ in texts]

    embedder = SentenceEmbedder(embed_fn=mock_embed)
    detector = ConflictDetector(embedder=embedder, similarity_threshold=0.5)

    # Item with pre-existing embedding
    item1 = make_item("topic_x", "Claim 1", item_id="1", embedding=[1.0, 0.0])
    item2 = make_item("topic_x", "Claim 2", item_id="2")

    detector.find_candidates([item1, item2])
    # Only item2 had to be embedded
    assert embed_calls == 1
    assert item1.embedding == [1.0, 0.0]
    assert item2.embedding == [1.0, 0.0]


def test_stage1_ignores_superseded():
    embedder = SentenceEmbedder(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    detector = ConflictDetector(embedder=embedder, similarity_threshold=0.5)

    item1 = make_item("topic_x", "Claim 1", item_id="1", status=Status.SUPERSEDED)
    item2 = make_item("topic_x", "Claim 2", item_id="2", status=Status.CONFIRMED)

    candidates = detector.find_candidates([item1, item2])
    assert candidates == []


# --------------------------------------------------------------------------- #
# Stage 2: Paraphrase Rejection & Genuine Contradiction Detection
# --------------------------------------------------------------------------- #
def test_paraphrases_are_not_flagged_as_conflicts():
    """Paraphrases (same fact, different wording) must be classified as ENTAILMENT

    and omitted from default list_conflicts / detect output.
    """
    embedder = SentenceEmbedder(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    fake_llm = FakeJudgeLLM()
    detector = ConflictDetector(llm=fake_llm, embedder=embedder, similarity_threshold=0.5)

    # Two paraphrased statements of the exact same fact
    item_a = make_item(
        "compute",
        "Pretraining the released model took under 24 GPU-hours.",
        agent_id="agent_A",
        item_id="1",
    )
    item_c = make_item(
        "compute",
        "The released model's pretraining required under 24 GPU-hours of compute.",
        agent_id="agent_C",
        item_id="2",
    )

    # Default detection: returns only confirmed CONTRADICTIONS
    conflicts = detector.detect([item_a, item_c])
    assert conflicts == [], "Paraphrases must NOT be flagged as conflicts!"

    # But debugging / inspecting ENTAILMENT reveals the pair
    entailments = detector.detect([item_a, item_c], relationships={Relationship.ENTAILMENT})
    assert len(entailments) == 1
    assert entailments[0].relationship == Relationship.ENTAILMENT
    assert "24 GPU-hours" in entailments[0].rationale


def test_genuine_contradiction_is_flagged_as_conflict():
    """Conflicting claims must be classified as CONTRADICTION and returned as a ConflictPair."""
    embedder = SentenceEmbedder(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    fake_llm = FakeJudgeLLM()
    detector = ConflictDetector(llm=fake_llm, embedder=embedder, similarity_threshold=0.5)

    item_a = make_item(
        "compute",
        "Pretraining the released model took under 24 GPU-hours.",
        agent_id="agent_A",
        item_id="1",
    )
    item_b = make_item(
        "compute",
        "Pretraining used approximately 11,500 A100-hours.",
        agent_id="agent_B",
        item_id="2",
    )

    conflicts = detector.detect([item_a, item_b])
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert isinstance(conflict, Conflict)
    assert isinstance(conflict, ConflictPair)
    assert conflict.topic == "compute"
    assert conflict.relationship == Relationship.CONTRADICTION
    assert {it.id for it in conflict.items} == {"1", "2"}
    assert conflict.agent_ids == ["agent_A", "agent_B"]
    assert "conflicts" in conflict.rationale.lower() or "11,500" in conflict.rationale


def test_neutral_claims_are_not_flagged_as_conflicts():
    """Non-contradictory claims on the same topic must be classified as NEUTRAL

    and omitted from default conflict output.
    """
    embedder = SentenceEmbedder(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    fake_llm = FakeJudgeLLM()
    detector = ConflictDetector(llm=fake_llm, embedder=embedder, similarity_threshold=0.5)

    item_b = make_item(
        "languages",
        "The model was evaluated on English, German, and Spanish benchmarks.",
        agent_id="agent_B",
        item_id="1",
    )
    item_e = make_item(
        "languages",
        "The model was evaluated on French and Italian benchmarks.",
        agent_id="agent_E",
        item_id="2",
    )

    conflicts = detector.detect([item_b, item_e])
    assert conflicts == [], "Neutral / compatible statements must NOT be flagged as conflicts!"

    # With debug inspection:
    neutrals = detector.detect([item_b, item_e], relationships={Relationship.NEUTRAL})
    assert len(neutrals) == 1
    assert neutrals[0].relationship == Relationship.NEUTRAL


# --------------------------------------------------------------------------- #
# Store Integration & Parsing Tests
# --------------------------------------------------------------------------- #
def test_store_list_conflicts_delegation(tmp_path):
    store = SqliteMemoryStore(tmp_path / "test.db")
    embedder = SentenceEmbedder(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    fake_llm = FakeJudgeLLM()
    detector = ConflictDetector(llm=fake_llm, embedder=embedder, similarity_threshold=0.5)

    item_a = make_item("compute", "Pretraining the released model took under 24 GPU-hours.", agent_id="agent_A")
    item_c = make_item("compute", "The released model's pretraining required under 24 GPU-hours of compute.", agent_id="agent_C")
    item_b = make_item("compute", "Pretraining used approximately 11,500 A100-hours.", agent_id="agent_B")

    store.add(item_a)
    store.add(item_c)
    store.add(item_b)

    # 1. Naive scan (no detector): flags topic because distinct text strings exist
    naive_conflicts = store.list_conflicts()
    assert len(naive_conflicts) == 1
    assert len(naive_conflicts[0].items) == 3

    # 2. Real detector passed directly:
    # A vs C is ENTAILMENT (ignored)
    # A vs B is CONTRADICTION (flagged)
    # C vs B is CONTRADICTION (flagged)
    real_conflicts = store.list_conflicts(detector=detector)
    assert len(real_conflicts) == 2
    for c in real_conflicts:
        assert isinstance(c, ConflictPair)
        assert c.relationship == Relationship.CONTRADICTION

    # 3. Store property detector:
    store.detector = detector
    assert len(store.list_conflicts()) == 2

    # 4. Debugging view: see ALL relationships
    all_pairs = store.list_conflicts(relationships=ALL_RELATIONSHIPS)
    assert len(all_pairs) == 3
    rels = {p.relationship for p in all_pairs}
    assert rels == {Relationship.ENTAILMENT, Relationship.CONTRADICTION}


def test_judge_response_parsing_variants():
    from memory.detector import _parse_judge_response

    # Pure JSON
    rel, rat = _parse_judge_response('{"relationship": "CONTRADICTION", "rationale": "Direct clash."}')
    assert rel == Relationship.CONTRADICTION
    assert rat == "Direct clash."

    # Markdown code block
    rel, rat = _parse_judge_response('```json\n{"relationship": "ENTAILMENT", "rationale": "Paraphrase."}\n```')
    assert rel == Relationship.ENTAILMENT
    assert rat == "Paraphrase."

    # Plain text with keyword
    rel, rat = _parse_judge_response("Based on analysis, this is clearly a CONTRADICTION between two dates.")
    assert rel == Relationship.CONTRADICTION


# --------------------------------------------------------------------------- #
# Pair ordering, question-in-prompt, and Stage-1 threshold semantics
# --------------------------------------------------------------------------- #
class CapturingJudge(LLMClient):
    backend = "test"
    agent_model = "test-agent"
    judge_model = "test-judge"

    def __init__(self):
        super().__init__(cache=None)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def _raw_generate(self, prompt, *, system, temperature, model):
        self.prompts.append(prompt)
        self.systems.append(system)
        return '{"relationship": "CONTRADICTION", "rationale": "test"}'


def _same_vec_embedder() -> SentenceEmbedder:
    return SentenceEmbedder(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])


def test_pair_order_is_independent_of_item_ids():
    # ids are random uuids in real runs; order must not follow them
    a = make_item("t", "claim one", agent_id="agent_A", item_id="zzz")
    b = make_item("t", "claim two", agent_id="agent_B", item_id="aaa")
    for x, y in ((a, b), (b, a)):
        pair = CandidatePair(x, y, 1.0, "t")
        assert pair.item_a is a and pair.item_b is b


def test_pair_order_same_agent_falls_back_to_content_then_id():
    first = make_item("t", "alpha", agent_id="agent_A", item_id="2")
    second = make_item("t", "beta", agent_id="agent_A", item_id="1")
    pair = CandidatePair(second, first, 1.0, "t")
    assert pair.item_a is first and pair.item_b is second


def test_judge_prompt_includes_question_when_present():
    llm = CapturingJudge()
    detector = ConflictDetector(llm=llm, embedder=_same_vec_embedder(), similarity_threshold=-1.0)
    a = make_item("supervision", "The method does not require human-labeled training data.", agent_id="agent_A")
    b = make_item("supervision", "Yes.", agent_id="agent_E")
    for it in (a, b):
        it.metadata["question"] = "Does the method require human-labeled training data?"

    detector.detect([a, b], relationships=ALL_RELATIONSHIPS)

    assert len(llm.prompts) == 1
    assert "Question: Does the method require human-labeled training data?" in llm.prompts[0]
    assert '"Yes."' in llm.prompts[0]
    assert "Question" in llm.systems[0]  # system prompt explains how to read it


def test_judge_prompt_omits_question_line_when_absent():
    llm = CapturingJudge()
    detector = ConflictDetector(llm=llm, embedder=_same_vec_embedder(), similarity_threshold=-1.0)
    a = make_item("t", "claim one", agent_id="agent_A")
    b = make_item("t", "claim two", agent_id="agent_B")

    detector.detect([a, b], relationships=ALL_RELATIONSHIPS)

    assert "Question:" not in llm.prompts[0]


def test_negative_similarity_pair_dropped_at_zero_threshold_kept_at_minus_one():
    # cosine([1, 0], [-0.1, 1]) is about -0.1: real "Yes." vs sentence pairs land here
    embed = lambda texts: [[1.0, 0.0] if t == "left" else [-0.1, 1.0] for t in texts]
    detector = ConflictDetector(embedder=SentenceEmbedder(embed_fn=embed))
    a = make_item("t", "left", agent_id="agent_A")
    b = make_item("t", "right", agent_id="agent_B")

    assert detector.find_candidates([a, b], threshold=0.0) == []
    kept = detector.find_candidates([a, b], threshold=-1.0)
    assert len(kept) == 1 and kept[0].similarity < 0.0
