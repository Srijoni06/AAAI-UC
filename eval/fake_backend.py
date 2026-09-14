"""Deterministic offline LLM backends + embedder for tests and the eval harness.

Lets the whole pipeline (orchestrate -> detect -> classify -> resolve -> score)
run end-to-end with no network: no Ollama, no Gemini, no embedding-model
download.

* :class:`ScopedFakeLLM` -- agent stand-in whose claim depends only on the
  excerpt it read, so agents on the same excerpt agree and agents on different
  excerpts disagree deterministically (the correlated-group behaviour without
  an LLM).
* :class:`RuleJudgeLLM` -- judge stand-in: equal claims -> ENTAILMENT
  (paraphrase), different claims -> CONTRADICTION.
* :class:`ScriptedReconcilerLLM` -- reconciler stand-in driven by a per-topic
  classification table, defaulting to CREDIBILITY.
* :class:`FakeEmbedder` -- hash-based deterministic embeddings: identical texts
  -> identical vectors, different texts -> (near-)orthogonal vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Sequence

from common.cache import LLMCache
from common.llm import LLMClient
from memory.detector import Relationship


class ScopedFakeLLM(LLMClient):
    """Agent stand-in: claim depends only on the excerpt text.

    Reads the excerpt between ``"Excerpt:\\n"`` and ``"\\n\\nQuestion:"`` in
    the prompt and returns ``"CLAIM<{first 40 chars}>"``.  Agents that read
    the same excerpt therefore produce identical claims; agents on different
    excerpts disagree.  This is the core deterministic scaffolding the eval
    harness relies on.
    """

    backend = "scoped_fake"
    agent_model = "scoped-fake-agent"
    judge_model = "scoped-fake-judge"

    def __init__(self) -> None:
        super().__init__(cache=None)
        self.calls: list[dict] = []

    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str:
        self.calls.append(
            {"prompt": prompt[:200], "system": system[:120], "model": model}
        )
        # Agent call: "Excerpt:\n<text>\n\nQuestion:"
        m = re.search(r"Excerpt:\n(.+?)\n\nQuestion:", prompt, re.DOTALL)
        if m:
            excerpt = m.group(1).strip()
            return f"CLAIM<{excerpt[:40]}>"

        # Judge call: extract the two quoted claims
        if "Classify the relationship" in prompt or "classify as ENTAILMENT" in prompt.lower():
            claims = re.findall(r':\n"([^"]+)"', prompt)
            if len(claims) == 2:
                a, b = (c.strip() for c in claims)
                if a.lower() == b.lower():
                    return json.dumps(
                        {"relationship": "ENTAILMENT", "rationale": "identical claims"}
                    )
                return json.dumps(
                    {"relationship": "CONTRADICTION", "rationale": "different claims"}
                )

        # Reconciler call: "Topic: <topic>" -> CREDIBILITY unless scripted
        m2 = re.search(r"Topic:\s*(\S+)", prompt)
        if m2:
            topic = m2.group(1)
            if topic == "language_coverage_claims":
                return json.dumps(
                    {"classification": "COORDINATION", "rationale": "both true with different scopes"}
                )
            return json.dumps(
                {"classification": "CREDIBILITY", "rationale": "mutually exclusive claims"}
            )

        return json.dumps(
            {"relationship": "NEUTRAL", "rationale": "unrecognised prompt"}
        )


class RuleJudgeLLM(LLMClient):
    """Rule-based judge: equal claims -> ENTAILMENT, different -> CONTRADICTION."""

    backend = "rule_judge"
    agent_model = "rule-judge-agent"
    judge_model = "rule-judge"

    def __init__(self) -> None:
        super().__init__(cache=None)
        self.calls: list[str] = []

    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str:
        self.calls.append(prompt[:200])
        claims = re.findall(r':\n"([^"]+)"', prompt)
        if len(claims) == 2:
            a, b = (c.strip().lower() for c in claims)
            if a == b:
                return json.dumps(
                    {"relationship": "ENTAILMENT", "rationale": "identical claims"}
                )
            return json.dumps(
                {"relationship": "CONTRADICTION", "rationale": "different claims"}
            )
        return json.dumps(
            {"relationship": "NEUTRAL", "rationale": "could not parse claims"}
        )


class ScriptedReconcilerLLM(LLMClient):
    """Reconciler driven by a per-topic classification table."""

    backend = "scripted_reconciler"
    agent_model = "reconciler-agent"
    judge_model = "reconciler-judge"

    def __init__(
        self,
        classifications: dict[str, str] | None = None,
        default: str = "CREDIBILITY",
    ) -> None:
        super().__init__(cache=None)
        self.classifications = classifications or {}
        self.default = default
        self.calls: list[str] = []

    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str:
        self.calls.append(prompt[:200])
        m = re.search(r"Topic:\s*(\S+)", prompt)
        topic = m.group(1) if m else ""
        cls = self.classifications.get(topic, self.default)
        return json.dumps({"classification": cls, "rationale": f"scripted: {topic}"})


def _hash_vec(text: str, dim: int = 64) -> list[float]:
    """Deterministic hash-based embedding: identical text -> identical vector."""
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class FakeEmbedder:
    """Deterministic hash-based embedder for offline testing."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, self.dim) for t in texts]
