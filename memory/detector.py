"""Two-stage contradiction detection for shared multi-agent memory.

Replaces the naive same-topic scan with:
  Stage 1: Semantic similarity filtering using sentence-transformers. Items on
           the same topic are embedded and compared pairwise via cosine similarity.
           Pairs exceeding a similarity threshold (default 0.5) become candidate conflicts.
  Stage 2: LLM judge classification. Each candidate pair is evaluated by the judge
           LLM and classified into exactly one of:
             - ENTAILMENT    (agree, different wording / paraphrases)
             - CONTRADICTION (genuinely disagree / conflicting facts)
             - NEUTRAL       (same topic, not comparable / distinct aspects)

By default, ``list_conflicts()`` returns only confirmed CONTRADICTION pairs.
ENTAILMENT and NEUTRAL pairs can be included for debugging.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence

from common.llm import LLMClient, make_llm
from memory.store import Conflict, MemoryItem, MemoryStore, Status


class Relationship(str, Enum):
    """Logical relationship between two claims on the same topic."""

    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


DEFAULT_RELATIONSHIPS = {Relationship.CONTRADICTION}
ALL_RELATIONSHIPS = {
    Relationship.ENTAILMENT,
    Relationship.CONTRADICTION,
    Relationship.NEUTRAL,
}

DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    sim = dot / (norm_u * norm_v)
    return max(-1.0, min(1.0, sim))


class SentenceEmbedder:
    """Wrapper around sentence-transformers embedding models.

    Supports lazy loading and a custom ``embed_fn`` callable for fast, offline unit testing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self._embed_fn = embed_fn
        self.device = device
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


@dataclass
class CandidatePair:
    """A pair of memory items on the same topic that passed Stage 1 similarity filtering."""

    item_a: MemoryItem
    item_b: MemoryItem
    similarity: float
    topic: str

    def __post_init__(self) -> None:
        # Canonical order by id to guarantee deterministic caching and comparisons
        if self.item_a.id > self.item_b.id:
            self.item_a, self.item_b = self.item_b, self.item_a


@dataclass
class ConflictPair(Conflict):
    """A classified pairwise relationship between two memory items on the same topic.

    Subclasses :class:`memory.store.Conflict` so it seamlessly interoperates with
    resolvers expecting a ``Conflict`` object with ``items=[item_a, item_b]``.
    """

    relationship: Relationship = Relationship.CONTRADICTION
    similarity: float = 1.0
    rationale: str = ""

    @property
    def item_a(self) -> MemoryItem:
        return self.items[0]

    @property
    def item_b(self) -> MemoryItem:
        return self.items[1]


JUDGE_SYSTEM_PROMPT = """You are an expert logical reasoning judge evaluating pairs of claims in shared multi-agent memory.
Your task is to classify the logical relationship between two claims on the same topic into EXACTLY ONE category:

1. ENTAILMENT: The two claims agree in substance or affirm the same underlying fact, even if phrased with different words, different sentence structures, or varying detail levels (e.g. paraphrases, synonyms, or logically equivalent statements).
2. CONTRADICTION: The two claims make conflicting, mutually incompatible, or opposing factual assertions (e.g. different numbers, opposing outcomes, conflicting attributions, or one asserting what the other refutes).
3. NEUTRAL: Both claims are on the same general topic, but they assert different non-conflicting facts or aspects that can both be true simultaneously without entailing or contradicting each other.

You must respond in JSON format with exactly two keys:
{
  "relationship": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL",
  "rationale": "<brief explanation of why this classification was chosen>"
}
"""


def _parse_judge_response(raw_text: str) -> tuple[Relationship, str]:
    """Parse JSON or text response from the judge LLM."""
    cleaned = raw_text.strip()
    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    elif cleaned.startswith("{") and cleaned.endswith("}"):
        pass
    else:
        # Look for the first JSON-like object
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]

    try:
        data = json.loads(cleaned)
        rel_str = str(data.get("relationship", "")).strip().upper()
        rationale = str(data.get("rationale", "")).strip()
        if rel_str in Relationship.__members__:
            return Relationship[rel_str], rationale
    except Exception:
        pass

    # Fallback to regex keyword extraction if JSON decoding failed
    upper = raw_text.upper()
    rationale = raw_text.strip()
    if "CONTRADICTION" in upper:
        return Relationship.CONTRADICTION, rationale
    if "ENTAILMENT" in upper:
        return Relationship.ENTAILMENT, rationale
    if "NEUTRAL" in upper:
        return Relationship.NEUTRAL, rationale

    return Relationship.NEUTRAL, f"Unable to parse judge response: {raw_text[:100]}"


class ConflictDetector:
    """Two-stage contradiction detector using embeddings and an LLM judge."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        embedder: Optional[SentenceEmbedder] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._llm = llm
        self.embedder = embedder or SentenceEmbedder(model_name=model_name)
        self.similarity_threshold = similarity_threshold

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = make_llm()
        return self._llm

    def _ensure_embeddings(self, items: list[MemoryItem]) -> None:
        """Compute and store embeddings for any items missing one."""
        missing = [it for it in items if it.embedding is None]
        if not missing:
            return
        texts = [it.content for it in missing]
        embeddings = self.embedder.embed(texts)
        for it, emb in zip(missing, embeddings):
            it.embedding = emb

    def find_candidates(
        self,
        items: Iterable[MemoryItem],
        *,
        threshold: Optional[float] = None,
    ) -> list[CandidatePair]:
        """Stage 1: Find pairs of items on the same topic whose cosine similarity >= threshold."""
        thresh = self.similarity_threshold if threshold is None else threshold
        live = [it for it in items if it.status != Status.SUPERSEDED]
        if len(live) < 2:
            return []

        self._ensure_embeddings(live)

        by_topic: dict[str, list[MemoryItem]] = {}
        for it in live:
            by_topic.setdefault(it.topic, []).append(it)

        candidates: list[CandidatePair] = []
        for topic, topic_items in by_topic.items():
            n = len(topic_items)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    it_a = topic_items[i]
                    it_b = topic_items[j]
                    if it_a.id == it_b.id:
                        continue
                    sim = cosine_similarity(it_a.embedding or [], it_b.embedding or [])
                    if sim >= thresh:
                        candidates.append(
                            CandidatePair(
                                item_a=it_a,
                                item_b=it_b,
                                similarity=sim,
                                topic=topic,
                            )
                        )
        return candidates

    def classify_pair(self, candidate: CandidatePair) -> ConflictPair:
        """Stage 2: Classify the relationship of a candidate pair using the judge LLM."""
        prompt = (
            f"Topic: {candidate.topic}\n\n"
            f"Claim 1 (by {candidate.item_a.agent_id}):\n\"{candidate.item_a.content}\"\n\n"
            f"Claim 2 (by {candidate.item_b.agent_id}):\n\"{candidate.item_b.content}\"\n\n"
            f"Classify the relationship between Claim 1 and Claim 2 as ENTAILMENT, CONTRADICTION, or NEUTRAL:"
        )

        response = self.llm.generate(
            prompt,
            system=JUDGE_SYSTEM_PROMPT,
            model=self.llm.judge_model,
            temperature=0.0,
        )

        rel, rationale = _parse_judge_response(response)
        return ConflictPair(
            topic=candidate.topic,
            items=[candidate.item_a, candidate.item_b],
            relationship=rel,
            similarity=candidate.similarity,
            rationale=rationale,
        )

    def detect(
        self,
        items: Iterable[MemoryItem],
        *,
        relationships: Optional[set[Relationship]] = None,
        threshold: Optional[float] = None,
    ) -> list[ConflictPair]:
        """Run Stage 1 and Stage 2 across items, returning classified pairs.

        By default, returns only confirmed CONTRADICTION pairs.
        Pass a custom ``relationships`` set (e.g. ALL_RELATIONSHIPS) to inspect ENTAILMENT/NEUTRAL.
        """
        target_rels = DEFAULT_RELATIONSHIPS if relationships is None else relationships
        candidates = self.find_candidates(items, threshold=threshold)
        classified: list[ConflictPair] = []
        for cand in candidates:
            pair = self.classify_pair(cand)
            if pair.relationship in target_rels:
                classified.append(pair)
        return classified

    def list_conflicts(
        self,
        store: MemoryStore,
        *,
        relationships: Optional[set[Relationship]] = None,
        threshold: Optional[float] = None,
    ) -> list[ConflictPair]:
        """Run two-stage contradiction detection over all live items in a store."""
        live_items = [it for it in store.list() if it.status != Status.SUPERSEDED]
        return self.detect(
            live_items,
            relationships=relationships,
            threshold=threshold,
        )
