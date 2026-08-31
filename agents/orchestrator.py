"""Multi-agent orchestration loop over the seeded contradiction suite.

Five summarizer agents each read one excerpt of a source document and answer the
same question in one sentence. Their answers go into the shared memory store
under a common ``topic`` key, with real provenance attached (which document,
which excerpt, the verbatim slice they read).

The five agents are deliberately **not** all independent:

* ``agent_A`` - independent, its own system prompt. Anchor of group ``grp_A``.
* ``agent_B`` - independent, a different ("skeptical") system prompt.
* ``agent_C`` - in ``grp_A``: same model as ``agent_A`` and a near-identical
  system prompt, and it reads the *same excerpt* as ``agent_A``.
* ``agent_D`` - in ``grp_A``: same setup as ``agent_C``.
* ``agent_E`` - independent, a third ("terse extractor") system prompt.

So ``agent_C`` / ``agent_D`` are shared-bias copies of ``agent_A``: when
``agent_A`` misreads an excerpt they tend to misread it the same way. Three
agents in ``grp_A`` agreeing is therefore *not* the same evidence as three
independent agents agreeing - which is exactly the failure mode the
reliability layer later has to catch.

Which agents are grouped is configurable: pass a custom ``roster`` (a list of
:class:`AgentSpec`) to :func:`run`. :func:`agent_groups` reports the grouping so
evaluation code can compare "N independent agree" against "N correlated agree".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from common.llm import LLMClient, make_llm
from domain.seed_conflicts import SEED_CONFLICTS, Excerpt, SeedConflict
from memory.store import MemoryItem, MemoryStore, Origin, SourceType, Status

# Back-compat alias: milestone 1 code / demo imported ``DOCUMENTS`` from here.
DOCUMENTS = SEED_CONFLICTS

INDEPENDENT = "independent"  # group name meaning "no shared bias with anyone"

# --- system prompts ------------------------------------------------------- #
# grp_A's prompt. agent_C / agent_D get trivially reworded variants of *this*
# one (see _CORRELATED_VARIANTS) so they share agent_A's framing and blind spots.
_PROMPT_ANCHOR = (
    "You are a research-literature summarization agent. You are given a single "
    "excerpt from a paper and one question. Answer in ONE declarative sentence, "
    "stating only what THIS excerpt supports. Do not hedge, do not mention "
    "alternative possibilities, do not cite anything outside the excerpt."
)

# Reworded but not behaviourally different - the point is shared bias, not
# genuine diversity.
_CORRELATED_VARIANTS = {
    "agent_C": (
        "You are a research-paper summarization agent. You receive one excerpt "
        "from a paper and one question. Respond with ONE declarative sentence "
        "that states only what THIS excerpt supports. Do not hedge, do not raise "
        "alternative possibilities, and do not use anything outside the excerpt."
    ),
    "agent_D": (
        "You summarize research papers. Given a single paper excerpt and a "
        "question, reply in ONE declarative sentence saying only what THIS "
        "excerpt supports. No hedging, no alternatives, nothing from outside "
        "the excerpt."
    ),
}

_PROMPT_SKEPTIC = (
    "You are a careful evidence analyst. Read the excerpt and answer the "
    "question in one sentence. If the excerpt qualifies, limits, or corrects a "
    "claim, your answer MUST reflect that qualification rather than the "
    "headline version."
)

_PROMPT_TERSE = (
    "Extract the single factual answer to the question from the excerpt. "
    "Reply with one short sentence and no preamble."
)


@dataclass(frozen=True)
class AgentSpec:
    """Configuration for one summarizer agent.

    ``group`` is the shared-bias group. Agents that share a ``group`` name
    (other than :data:`INDEPENDENT`) use similar prompts and read the same
    excerpt, so their agreement is correlated. ``INDEPENDENT`` agents share a
    group name only nominally - each is treated as its own group of one.
    """

    agent_id: str
    system: str
    group: str = INDEPENDENT
    model: str | None = None  # None -> llm.agent_model


DEFAULT_ROSTER: list[AgentSpec] = [
    AgentSpec("agent_A", _PROMPT_ANCHOR, group="grp_A"),
    AgentSpec("agent_B", _PROMPT_SKEPTIC, group=INDEPENDENT),
    AgentSpec("agent_C", _CORRELATED_VARIANTS["agent_C"], group="grp_A"),
    AgentSpec("agent_D", _CORRELATED_VARIANTS["agent_D"], group="grp_A"),
    AgentSpec("agent_E", _PROMPT_TERSE, group=INDEPENDENT),
]


# --------------------------------------------------------------------------- #
# grouping helpers (used by demo + evaluation)
# --------------------------------------------------------------------------- #
def agent_groups(roster: Iterable[AgentSpec] | None = None) -> dict[str, list[str]]:
    """Map every group name to the agent ids in it, preserving roster order.

    ``INDEPENDENT`` agents are each returned as their own singleton group keyed
    by their agent id, so callers can treat the result uniformly.
    """
    roster = list(roster if roster is not None else DEFAULT_ROSTER)
    groups: dict[str, list[str]] = {}
    for spec in roster:
        key = spec.agent_id if spec.group == INDEPENDENT else spec.group
        groups.setdefault(key, []).append(spec.agent_id)
    return groups


def correlated_groups(
    roster: Iterable[AgentSpec] | None = None,
) -> dict[str, list[str]]:
    """Only the shared-bias groups with more than one member."""
    return {g: ids for g, ids in agent_groups(roster).items() if len(ids) > 1}


def independent_agents(roster: Iterable[AgentSpec] | None = None) -> list[str]:
    roster = list(roster if roster is not None else DEFAULT_ROSTER)
    return [s.agent_id for s in roster if s.group == INDEPENDENT]


def _group_sizes(roster: list[AgentSpec]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for spec in roster:
        key = spec.agent_id if spec.group == INDEPENDENT else spec.group
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def assign_excerpts(
    seed: SeedConflict,
    roster: list[AgentSpec] | None = None,
    *,
    anchor_excerpt_index: int = 0,
) -> dict[str, Excerpt]:
    """Decide which excerpt of ``seed`` each agent reads.

    * Every correlated group reads the same excerpt - ``excerpts[anchor_excerpt
      _index]`` - so the group's agreement carries no independent evidence.
    * When a correlated group is present, independent agents are spread
      round-robin over the *other* excerpts, so they can land on the excerpt
      that actually contradicts the group.
    * With no correlated group, independents are spread over *all* excerpts.
    * With a single-excerpt seed everyone reads it.
    """
    roster = roster if roster is not None else DEFAULT_ROSTER
    excerpts = seed.excerpts
    n = len(excerpts)
    anchor_idx = anchor_excerpt_index % n
    anchor = excerpts[anchor_idx]

    has_correlated = any(s.group != INDEPENDENT for s in roster)
    if has_correlated:
        pool = [e for i, e in enumerate(excerpts) if i != anchor_idx] or [anchor]
    else:
        pool = list(excerpts)

    out: dict[str, Excerpt] = {}
    indep_seen = 0
    for spec in roster:
        if spec.group != INDEPENDENT:
            out[spec.agent_id] = anchor
        else:
            out[spec.agent_id] = pool[indep_seen % len(pool)]
            indep_seen += 1
    return out


class SummarizerAgent:
    def __init__(self, spec: AgentSpec, llm: LLMClient):
        self.spec = spec
        self.agent_id = spec.agent_id
        self.group = spec.group
        self.llm = llm
        self.model = spec.model or llm.agent_model

    def summarize(self, excerpt_text: str, question: str) -> str:
        prompt = (
            f"Excerpt:\n{excerpt_text}\n\n"
            f"Question: {question}\nOne-sentence answer:"
        )
        return self.llm.generate(
            prompt,
            system=self.spec.system,
            temperature=0.2,
            model=self.model,
        )


@dataclass
class AgentWrite:
    """One agent's claim plus everything that produced it (for demo/eval)."""

    seed: SeedConflict
    agent_id: str
    group: str
    excerpt: Excerpt
    claim: str
    item: MemoryItem
    correct: bool | None = field(default=None)

    def __post_init__(self) -> None:
        gold = self.seed.gold_excerpt_id
        # COEXIST seeds have no single right excerpt; leave ``correct`` as None.
        if gold in self.seed.excerpt_ids:
            self.correct = self.excerpt.excerpt_id == gold


def run(
    store: MemoryStore,
    seeds: list[SeedConflict] | None = None,
    llm: LLMClient | None = None,
    *,
    roster: list[AgentSpec] | None = None,
    anchor_excerpt_index: int = 0,
) -> list[AgentWrite]:
    """Run every agent over every seed and write their claims to ``store``.

    Returns one :class:`AgentWrite` per (seed, agent), in roster order within
    each seed.
    """
    seeds = seeds if seeds is not None else SEED_CONFLICTS
    roster = roster if roster is not None else DEFAULT_ROSTER
    llm = llm or make_llm()

    agents = [SummarizerAgent(spec, llm) for spec in roster]

    writes: list[AgentWrite] = []
    for seed in seeds:
        excerpt_for = assign_excerpts(
            seed, roster, anchor_excerpt_index=anchor_excerpt_index
        )
        for agent in agents:
            excerpt = excerpt_for[agent.agent_id]
            claim = agent.summarize(excerpt.text, seed.question)
            item = store.add(
                MemoryItem(
                    agent_id=agent.agent_id,
                    topic=seed.topic,
                    content=claim,
                    status=Status.PROPOSED,
                    source_doc_id=seed.doc_id,
                    # --- real provenance ------------------------------------
                    source_type=SourceType.RETRIEVAL,  # read from a document
                    origin=Origin.TOOL,                # grounded in retrieved text
                    evidence_span=excerpt.text,        # the exact slice it read
                    metadata={
                        "question": seed.question,
                        "excerpt_id": excerpt.excerpt_id,
                        "section": excerpt.section,
                        # precise pointer to the passage this claim rests on
                        "source_id": f"{seed.doc_id}#{excerpt.excerpt_id}",
                        "agent_group": agent.group,
                    },
                )
            )
            writes.append(
                AgentWrite(
                    seed=seed,
                    agent_id=agent.agent_id,
                    group=agent.group,
                    excerpt=excerpt,
                    claim=claim,
                    item=item,
                )
            )
    return writes
