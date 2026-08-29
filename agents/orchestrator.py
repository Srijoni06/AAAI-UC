"""Minimal multi-agent orchestration loop for milestone 1.

Two summarizer agents each read a *different* excerpt of the *same* source
document and answer the same question in one sentence. Their answers go into the
shared memory store under a common ``topic`` key. The hardcoded documents are
built so the two excerpts genuinely support different answers - the
contradiction is in the source material, not model noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.llm import LLMClient, make_llm
from memory.store import MemoryItem, MemoryStore, Status

_SYSTEM_INSTRUCTION = (
    "You are a research-literature summarization agent. You are given a single "
    "excerpt from a paper and one question. Answer in ONE declarative sentence, "
    "stating only what THIS excerpt supports. Do not hedge, do not mention "
    "alternative possibilities, do not cite anything outside the excerpt."
)


@dataclass
class Document:
    doc_id: str
    title: str
    topic: str  # grouping key in shared memory
    question: str
    excerpt_a: str  # goes to agent A
    excerpt_b: str  # goes to agent B


# --- hardcoded testbed (3 documents, each seeds one genuine contradiction) -----
DOCUMENTS: list[Document] = [
    Document(
        doc_id="doc-benchmark",
        title="Neural Entity Tagger: Setup and Results",
        topic="primary_benchmark",
        question="What is the primary benchmark dataset used to evaluate the method?",
        excerpt_a=(
            "Section 1 (Introduction). In this work we primarily evaluate our "
            "tagger on the CoNLL-2003 English named-entity benchmark, the "
            "standard testbed for this task, and use it for all ablations."
        ),
        excerpt_b=(
            "Section 5 (Main Results). Our headline numbers are reported on "
            "OntoNotes 5.0, which we treat as the primary benchmark for this "
            "paper because of its broader entity inventory; CoNLL-2003 is used "
            "only as a small secondary check."
        ),
    ),
    Document(
        doc_id="doc-gain",
        title="Adaptive Reranking for Retrieval QA",
        topic="improvement_over_baseline",
        question="How large is the method's improvement over the baseline?",
        excerpt_a=(
            "Abstract. Our adaptive reranker delivers a substantial +4.2 EM "
            "improvement over the strong DPR baseline on the target task."
        ),
        excerpt_b=(
            "Section 6 (Discussion). After controlling for the retriever budget, "
            "the adaptive reranker yields only a modest +0.5 EM gain over the "
            "DPR baseline, within one standard deviation across seeds."
        ),
    ),
    Document(
        doc_id="doc-sota",
        title="Long-Context Compression with LSH Attention (v2 with erratum)",
        topic="beats_prior_sota",
        question="Does the method outperform the prior state of the art?",
        excerpt_a=(
            "Section 4 (original). On the long-document suite our method sets a "
            "new state of the art, surpassing the previous best system by 1.8 "
            "ROUGE-L."
        ),
        excerpt_b=(
            "Erratum (added in v2). A post-publication audit found test-set "
            "contamination in the long-document suite. With the cleaned split, "
            "the method does NOT exceed the prior state of the art and trails "
            "it by 0.3 ROUGE-L."
        ),
    ),
]


class SummarizerAgent:
    def __init__(self, agent_id: str, llm: LLMClient, model: str | None = None):
        self.agent_id = agent_id
        self.llm = llm
        self.model = model or llm.agent_model

    def summarize(self, excerpt: str, question: str) -> str:
        prompt = f"Excerpt:\n{excerpt}\n\nQuestion: {question}\nOne-sentence answer:"
        return self.llm.generate(
            prompt,
            system=_SYSTEM_INSTRUCTION,
            temperature=0.2,
            model=self.model,
        )


def run(
    store: MemoryStore,
    documents: list[Document] | None = None,
    llm: LLMClient | None = None,
) -> list[MemoryItem]:
    """Run both agents over every document and write their claims to ``store``."""
    documents = documents if documents is not None else DOCUMENTS
    llm = llm or make_llm()

    agent_a = SummarizerAgent("agent_A", llm)
    agent_b = SummarizerAgent("agent_B", llm)

    written: list[MemoryItem] = []
    for doc in documents:
        for agent, excerpt, which in (
            (agent_a, doc.excerpt_a, "excerpt_a"),
            (agent_b, doc.excerpt_b, "excerpt_b"),
        ):
            claim = agent.summarize(excerpt, doc.question)
            item = store.add(
                MemoryItem(
                    agent_id=agent.agent_id,
                    topic=doc.topic,
                    content=claim,
                    status=Status.PROPOSED,
                    source_doc_id=doc.doc_id,
                    metadata={"question": doc.question, "excerpt": which},
                )
            )
            written.append(item)
    return written
