"""Conflict reconciliation: classify each verified contradiction before resolving.

Animesh's conflict-classification stage (per the work-division table: "decides
whether two agents' claims actually contradict each other, and classifies
conflicts by type"). Detection (``memory/detector.py``) already established
*that* a contradiction exists; this module decides **what kind** of decision the
conflict needs:

* **CREDIBILITY** - exactly one claim can be right; pick a winner and supersede
  the losers (proceed to a resolver).
* **COORDINATION** - the claims only look contradictory; both hold with
  different scopes (e.g. "trained on 90 languages" vs "evaluated on 46").
  Confirm all claims and let them coexist.

The classifier has two layers:

1. ``LLMReconciler`` asks the judge LLM to emit
   ``{"classification": "CREDIBILITY" | "COORDINATION", "rationale": ...}``.
2. ``pairwise_scopes_agree`` is the deterministic, dependency-free fallback:
   two claims contradict only if they also *disagree on their evidence* - when
   both claims' ``evidence_span`` provenance points at the same supporting
   passage, the disagreement is scope confusion, not a factual conflict, and
   the pair classifies as COORDINATION without an LLM call.

``reconcile()`` wires both together: the deterministic check runs first and
takes precedence; the LLM is consulted only for pairs whose evidence genuinely
differs, and an unusable LLM verdict falls back to CREDIBILITY (the safe,
decisive default that keeps the pipeline moving).

The resolvers consuming the verdict: ``baselines/*`` and ``reliability.resolver``
both emit the common ``baselines.base.Resolution`` shape, so ``eval`` can compare
all conditions on the same classified conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from common.llm import LLMClient
from memory.store import Conflict, MemoryItem

INDEPENDENT = "independent"


class Classification(str, Enum):
    """What kind of decision a verified contradiction needs."""

    CREDIBILITY = "CREDIBILITY"    # one claim should defeat the other
    COORDINATION = "COORDINATION"  # the claims validly coexist


RECONCILER_SYSTEM_PROMPT = """You are an expert arbiter for a shared multi-agent memory.
Two agents wrote claims that were flagged as contradictory. Decide what KIND of decision is needed:

1. CREDIBILITY: exactly one claim can be right (different numbers, dates, names,
   mutually exclusive facts). A resolution should pick a winner and discard the loser.
2. COORDINATION: both claims are actually true with different scopes or about
   different stages/aspects (e.g. "trained on 90 languages" vs "evaluated on 46",
   "released in 2023" vs "paper written in 2022"). Both should be kept.

Respond in JSON with exactly two keys:
{
  "classification": "CREDIBILITY" | "COORDINATION",
  "rationale": "<brief explanation>"
}
"""


def _parse_reconciler_response(raw: str) -> tuple[Classification, str]:
    """Parse the judge's JSON (with plain-text keyword fallback)."""
    import json
    import re

    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last > first:
            cleaned = cleaned[first : last + 1]

    try:
        data = json.loads(cleaned)
        cls_str = str(data.get("classification", "")).strip().upper()
        rationale = str(data.get("rationale", "")).strip()
        if cls_str in Classification.__members__:
            return Classification[cls_str], rationale
    except Exception:
        pass

    upper = raw.upper()
    rationale = raw.strip()
    if "COORDINATION" in upper:
        return Classification.COORDINATION, rationale
    if "CREDIBILITY" in upper:
        return Classification.CREDIBILITY, rationale
    raise ValueError(f"cannot parse reconciler response: {raw[:120]!r}")


def _scope(item: MemoryItem) -> str:
    """The evidence scope a claim rests on (its excerpt, when known)."""
    src = item.metadata.get("source_id")
    if src:
        return str(src)
    if item.evidence_span:
        return item.evidence_span.strip()
    return ""


def pairwise_scopes_agree(item_a: MemoryItem, item_b: MemoryItem) -> bool:
    """Deterministic COORDINATION check: do both claims cite the same evidence?

    When both items carry a non-empty evidence scope and the scopes are
    identical, the contradiction is a matter of interpretation or scope, not a
    factual clash between sources - the pair should COORDINATE (coexist)
    rather than go through winner selection.

    Empty scopes (provenance-free items) never count as agreeing, so the
    deterministic layer stays quiet for legacy/unprovenanced data.
    """
    scope_a = _scope(item_a)
    scope_b = _scope(item_b)
    if not scope_a or not scope_b:
        return False
    return scope_a == scope_b


@dataclass
class Reconciliation:
    """The classification verdict for one verified conflict."""

    topic: str
    classification: Classification
    rationale: str = ""
    items: list[MemoryItem] | None = None  # the conflict's items, for downstream

    def is_credibility(self) -> bool:
        return self.classification is Classification.CREDIBILITY

    def is_coordination(self) -> bool:
        return self.classification is Classification.COORDINATION


class LLMReconciler:
    """Classifies verified conflicts via the judge LLM."""

    name = "llm_reconciler"

    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            from common.llm import make_llm

            self._llm = make_llm()
        return self._llm

    def classify(self, conflict: Conflict) -> Reconciliation:
        item_a, item_b = conflict.items[0], conflict.items[1]

        # Layer 1: deterministic scope check takes precedence.
        if pairwise_scopes_agree(item_a, item_b):
            return Reconciliation(
                topic=conflict.topic,
                classification=Classification.COORDINATION,
                rationale=(
                    "deterministic: both claims cite the same evidence_span, "
                    "so the disagreement is scope confusion, not a factual conflict"
                ),
                items=list(conflict.items),
            )

        # Layer 2: LLM judge for pairs with genuinely different evidence.
        prompt = (
            f"Topic: {conflict.topic}\n\n"
            f"Claim 1 (by {item_a.agent_id}):\n\"{item_a.content}\"\n\n"
            f"Claim 2 (by {item_b.agent_id}):\n\"{item_b.content}\"\n\n"
            f"Classify this contradiction as CREDIBILITY or COORDINATION:"
        )
        response = self.llm.generate(
            prompt,
            system=RECONCILER_SYSTEM_PROMPT,
            model=self.llm.judge_model,
            temperature=0.0,
        )
        try:
            cls, rationale = _parse_reconciler_response(response)
        except ValueError as exc:
            cls, rationale = Classification.CREDIBILITY, str(exc)

        return Reconciliation(
            topic=conflict.topic,
            classification=cls,
            rationale=rationale,
            items=list(conflict.items),
        )


def resolve_conflict(
    conflict: Conflict,
    resolver,
    reconciler: LLMReconciler | None = None,
) -> tuple[Reconciliation, object | None]:
    """Classify ``conflict`` and, when CREDIBILITY, run ``resolver`` on it.

    Returns ``(reconciliation, resolution_or_None)``. COORDINATION conflicts
    return no resolution (all items stay live); callers that want to mark them
    can confirm every item themselves. Configurable ``resolver`` keeps this
    orchestrator-agnostic: any baselines.base.Resolver-compatible object works.
    """
    rec = (reconciler or LLMReconciler()).classify(conflict)
    if rec.is_coordination():
        return rec, None
    return rec, resolver.resolve(conflict)
