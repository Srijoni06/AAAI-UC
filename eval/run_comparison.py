"""Comparison harness across resolution conditions.

Runs the same seeded suite under:
  1. null             -- no resolution (control: all claims stay PROPOSED)
  2. last_write_wins  -- newest claim wins
  3. majority_vote    -- largest meaning-cluster wins (correlated = votes at face value)
  4. static_confidence -- fixed provenance weight table, no learning

The reliability-aware condition plugs in automatically when
``reliability.resolver`` exposes a ``Resolver``-compatible class.

Usage (offline, deterministic, no network)::

    python -m eval.run_comparison              # all 20 docs, fake backend
    python -m eval.run_comparison --limit 4    # demo subset
    python -m eval.run_comparison --backend real   # Ollama / Gemini

Outputs ``results/run_comparison.json`` + ``results/summary.md`` and
``results/detection_pairs.json`` (every same-topic pair: claim text, similarity,
judge verdict, TP/FP/FN/TN), and prints the summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from agents.orchestrator import DEFAULT_ROSTER, run as orch_run
from baselines.base import apply_resolution
from baselines.last_write_wins import LastWriteWins
from baselines.majority_vote import MajorityVote
from baselines.static_confidence import StaticConfidence
from domain.seed_conflicts import (
    COEXIST,
    SEED_CONFLICTS,
    SEEDS_BY_ID,
    SeedConflict,
    type_counts,
)
from memory.detector import (
    ALL_RELATIONSHIPS,
    ConflictDetector,
    Relationship,
    cosine_similarity,
    order_pair,
)
from memory.store import Conflict
from memory.reconciler import (
    Classification,
    LLMReconciler,
    Reconciliation,
    resolve_conflict,
)
from memory.store import MemoryItem, SqliteMemoryStore, Status


# ---------------------------------------------------------------------------
# Resolver registry -- plug in reliability.resolver when it lands
# ---------------------------------------------------------------------------

def _build_resolvers() -> dict[str, object]:
    resolvers: dict[str, object] = {
        "last_write_wins": LastWriteWins(),
        "majority_vote": MajorityVote(),
        "static_confidence": StaticConfidence(),
    }
    try:
        from reliability.resolver import ReliabilityResolver  # type: ignore
        resolvers["reliability_aware"] = ReliabilityResolver()
    except (ImportError, AttributeError):
        pass
    return resolvers


# ---------------------------------------------------------------------------
# Snapshot: run agents once, freeze results, replay per condition
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """Frozen agent writes + detection results, replayable across conditions."""

    seeds: list[SeedConflict]
    item_dicts: list[dict]            # MemoryItem.to_dict() snapshots
    pair_specs: list[dict]            # detected CONTRADICTION pair specs
    seed_topic_map: dict[str, str]    # topic -> doc_id
    detection_tp: int = 0
    detection_fp: int = 0
    detection_fn: int = 0
    pair_records: list[dict] = field(default_factory=list)  # every same-topic pair

    @property
    def topics(self) -> list[str]:
        return list(self.seed_topic_map.keys())


def _snapshot_writes(writes, seeds: list[SeedConflict]) -> Snapshot:
    """Capture writes + per-item metadata into a serializable snapshot."""
    item_dicts = [w.item.to_dict() for w in writes]
    topic_map = {s.topic: s.doc_id for s in seeds}
    return Snapshot(seeds=seeds, item_dicts=item_dicts, pair_specs=[], seed_topic_map=topic_map)


def _item_view(it: MemoryItem) -> dict:
    return {
        "item_id": it.id,
        "agent_id": it.agent_id,
        "agent_group": it.metadata.get("agent_group"),
        "excerpt_id": it.metadata.get("excerpt_id"),
        "text": it.content,
    }


def _detect_one_doc(
    seed: SeedConflict, doc_items: list[MemoryItem], detector: ConflictDetector
) -> tuple[list[dict], list[dict]]:
    """Run two-stage detection for exactly one document's items.

    Returns ``(pair_specs, pair_records)`` scoped to this document only - the
    checkpointable unit of Phase B work. A topic's candidate pairs never cross
    into another topic (``find_candidates`` groups by topic internally), so
    detecting one document's items in isolation is identical to detecting them
    as part of a larger batch.

    The judge is called on every Stage-1 candidate regardless of the verdict,
    so asking for all relationships costs nothing extra and lets us record
    *why* a gold contradiction was missed (judged ENTAILMENT/NEUTRAL vs never
    judged).
    """
    judged = detector.detect(doc_items, relationships=ALL_RELATIONSHIPS)
    verdicts = {frozenset([p.item_a.id, p.item_b.id]): p for p in judged}

    specs = [
        {
            "item_a_id": p.item_a.id,
            "item_b_id": p.item_b.id,
            "topic": p.topic,
            "similarity": p.similarity,
            "rationale": p.rationale,
        }
        for p in judged
        if p.relationship == Relationship.CONTRADICTION
    ]

    coexist = seed.gold_excerpt_id == COEXIST
    records: list[dict] = []
    for i in range(len(doc_items)):
        for j in range(i + 1, len(doc_items)):
            a, b = order_pair(doc_items[i], doc_items[j])  # same order the judge saw
            cross = a.metadata.get("excerpt_id") != b.metadata.get("excerpt_id")
            gold_positive = cross and not coexist
            judged_pair = verdicts.get(frozenset([a.id, b.id]))
            verdict = judged_pair.relationship.value if judged_pair else None
            sim = (
                judged_pair.similarity
                if judged_pair
                else cosine_similarity(a.embedding or [], b.embedding or [])
            )
            flagged = verdict == Relationship.CONTRADICTION.value
            if flagged:
                outcome = "TP" if gold_positive else "FP"
            else:
                outcome = "FN" if gold_positive else "TN"
            records.append(
                {
                    "doc_id": seed.doc_id,
                    "topic": seed.topic,
                    "conflict_type": seed.conflict_type.value,
                    "difficulty": seed.difficulty.value,
                    "gold_excerpt_id": seed.gold_excerpt_id,
                    "question": a.metadata.get("question"),
                    "gold_positive": gold_positive,
                    "outcome": outcome,
                    "judged": judged_pair is not None,
                    "verdict": verdict,
                    "rationale": judged_pair.rationale if judged_pair else None,
                    # Both-orders judging (memory.detector._merge_orders): the
                    # individual per-order verdicts that fed the merged one above.
                    "verdict_a_first": judged_pair.verdict_a_first.value
                    if judged_pair and judged_pair.verdict_a_first
                    else None,
                    "rationale_a_first": judged_pair.rationale_a_first if judged_pair else None,
                    "verdict_b_first": judged_pair.verdict_b_first.value
                    if judged_pair and judged_pair.verdict_b_first
                    else None,
                    "rationale_b_first": judged_pair.rationale_b_first if judged_pair else None,
                    "similarity": round(sim, 4),
                    "claim_1": _item_view(a),
                    "claim_2": _item_view(b),
                }
            )
    return specs, records


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def _detection_metrics(pair_records: list[dict]) -> dict:
    """Pair-level detection precision/recall/F1, tallied from ``pair_records``.

    Each record's ``outcome`` (TP/FP/FN/TN) was already decided per-pair, per
    document, in ``_detect_one_doc`` - a pure tally here (rather than
    re-deriving TP/FP/FN by looking pairs back up by item id, as this used to)
    keeps this self-contained and safe to compute after a checkpoint resume,
    where resumed docs' records come from a *different* run whose items have
    different random ids than the current run's freshly-generated ones.
    """
    tp = sum(1 for r in pair_records if r["outcome"] == "TP")
    fp = sum(1 for r in pair_records if r["outcome"] == "FP")
    fn = sum(1 for r in pair_records if r["outcome"] == "FN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Per-condition scoring
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    name: str
    total_conflicts: int = 0
    decisive: int = 0
    correct: int = 0
    incorrect: int = 0
    contested: int = 0
    coordination: int = 0
    decisions: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.decisive if self.decisive > 0 else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.contested / self.total_conflicts if self.total_conflicts > 0 else 0.0

    def per_type_accuracy(self) -> dict[str, float]:
        by_type: dict[str, list[bool]] = defaultdict(list)
        for d in self.decisions:
            # decision dicts key the gold label as "gold_excerpt" (set in
            # _score_condition); this used to check "gold", a key that is
            # never present, so per_type_accuracy() always returned {}.
            if d.get("gold_excerpt") is not None:
                by_type[d["conflict_type"]].append(d["correct"])
        return {t: sum(v) / len(v) for t, v in by_type.items() if v}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_conflicts": self.total_conflicts,
            "decisive": self.decisive,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "contested": self.contested,
            "coordination": self.coordination,
            "accuracy": round(self.accuracy, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "per_type_accuracy": {k: round(v, 4) for k, v in self.per_type_accuracy().items()},
            "decisions": self.decisions,
        }


def _score_one_topic(
    name: str,
    resolver,
    store: SqliteMemoryStore,
    snapshot: Snapshot,
    topic: str,
    reconciler_llm=None,
) -> Optional[dict]:
    """Score exactly one topic's conflict under one condition.

    Returns the decision dict, or ``None`` if the topic has fewer than 2 live
    items (mirrors the original loop's silent skip of such topics: no decision
    is recorded and it does not count toward ``total_conflicts``). This is the
    checkpointable unit of Phase C work - one ``(condition, doc)`` pair.
    """
    all_items = store.list(topic=topic)
    live_items = [it for it in all_items if it.status != Status.SUPERSEDED]
    if len(live_items) < 2:
        return None
    conflict = Conflict(topic=topic, items=live_items)

    seed = SEEDS_BY_ID.get(snapshot.seed_topic_map.get(topic, ""))
    gold = seed.gold_excerpt_id if seed else "?"
    all_excerpt_ids = {it.metadata.get("excerpt_id", "") for it in live_items}

    # Classify: CREDIBILITY vs COORDINATION
    if reconciler_llm is not None:
        llm_rec = LLMReconciler(reconciler_llm).classify(conflict)
    else:
        llm_rec = Reconciliation(
            topic=topic,
            classification=Classification.CREDIBILITY,
            rationale="no reconciler (default CREDIBILITY)",
        )

    decision = {
        "topic": topic,
        "doc_id": snapshot.seed_topic_map.get(topic, ""),
        "conflict_type": seed.conflict_type.value if seed else "?",
        "difficulty": seed.difficulty.value if seed else "?",
        "gold_excerpt": gold,
        "classification": llm_rec.classification.value,
    }

    # --- COORDINATION: all live claims coexist ---
    if llm_rec.is_coordination():
        from baselines.base import Resolution

        resolution = Resolution.coexist(
            topic=topic,
            strategy="coordination",
            item_ids=[it.id for it in live_items],
            rationale=llm_rec.rationale,
        )
        apply_resolution(store, resolution)
        confirmed_excerpts = all_excerpt_ids
        is_correct = True if gold == COEXIST else gold in confirmed_excerpts
        decision["outcome"] = "decisive"
        decision["correct"] = is_correct
        decision["rationale"] = llm_rec.rationale
        return decision

    # --- CREDIBILITY: delegate to resolver ---
    if name == "null":
        decision["outcome"] = "contested"
        decision["correct"] = False
        decision["rationale"] = "null resolver: no resolution applied"
        return decision

    resolution = resolver.resolve(conflict)
    apply_resolution(store, resolution)

    confirmed = [store.get(cid) for cid in resolution.confirmed_ids]
    confirmed = [c for c in confirmed if c is not None]
    confirmed_excerpts = {c.metadata.get("excerpt_id", "") for c in confirmed}

    if confirmed_excerpts:
        is_correct = (
            all_excerpt_ids.issubset(confirmed_excerpts)
            if gold == COEXIST
            else confirmed_excerpts == {gold}
        )
        decision["outcome"] = "decisive"
        decision["correct"] = is_correct
        decision["rationale"] = resolution.rationale
    else:
        is_correct = False
        decision["outcome"] = "contested"
        decision["correct"] = False
        decision["rationale"] = resolution.rationale

    # Update reliability memory if resolver supports it. Re-fetch from the
    # store rather than reusing `live_items` (captured before apply_resolution
    # ran): PeerMemory.update() no longer trusts item.status either way (it
    # reads confirmed/superseded ids straight from `resolution`), but a stale
    # pre-resolution snapshot is wrong to hand to a resolver update on general
    # principle, so this stops doing it regardless of what the callee needs.
    if hasattr(resolver, "update_memory"):
        post_items = store.list(topic=topic)
        resolver.update_memory(post_items, resolution, correct=bool(confirmed_excerpts) and is_correct)

    return decision


def _condition_result_from_decisions(name: str, decisions: list[dict]) -> ConditionResult:
    """Reconstruct a full ``ConditionResult`` purely from its decision dicts.

    Every summary counter (``total_conflicts``, ``decisive``, ``correct``, ...)
    is derivable from ``decision["outcome"]``/``["correct"]``/["classification"]``,
    so a condition's state - whether freshly computed or resumed from a
    checkpoint - is fully described by its decisions list alone. This is what
    makes merging checkpoint-resumed and freshly-scored decisions trivial: just
    concatenate the dicts and rebuild the counters from them.
    """
    rec = ConditionResult(name=name)
    for d in decisions:
        rec.total_conflicts += 1
        if d.get("classification") == "COORDINATION":
            rec.coordination += 1
        if d["outcome"] == "decisive":
            rec.decisive += 1
            if d["correct"]:
                rec.correct += 1
            else:
                rec.incorrect += 1
        else:
            rec.contested += 1
    rec.decisions = list(decisions)
    return rec


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _llm_label(llm, model_attr: str) -> str:
    return f"{getattr(llm, 'backend', '?')}:{getattr(llm, model_attr, '?')}"


def _write_summary(
    detection: dict,
    results: dict[str, ConditionResult],
    out_dir: Path,
    config: dict,
    pair_records: Optional[list[dict]] = None,
) -> str:
    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"**Backend:** {config['backend']}  ")
    lines.append(f"**Documents:** {config['n_docs']}  ")
    lines.append(f"**Agents:** {len(DEFAULT_ROSTER)}  ")
    lines.append(f"**Similarity threshold:** {config['threshold']}  ")
    lines.append(f"**Agent LLM:** {config.get('agent_llm')}  ")
    lines.append(f"**Judge LLM:** {config.get('judge_llm')}  ")
    lines.append("")

    lines.append("## Detection Metrics (pair-level)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in detection.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    missed = [r for r in (pair_records or []) if r["outcome"] == "FN"]
    if pair_records is not None:
        lines.append(f"## Missed contradictions (false negatives): {len(missed)}")
        lines.append("")
        lines.append("All judged pairs: `detection_pairs.json`.")
        lines.append("")
        for r in missed:
            why = f"judged {r['verdict']}" if r["judged"] else "never judged (below Stage-1 threshold)"
            lines.append(
                f"- **{r['doc_id']}** ({r['conflict_type']}/{r['difficulty']}) - {why}, "
                f"sim {r['similarity']}"
            )
            for key in ("claim_1", "claim_2"):
                c = r[key]
                lines.append(f"  - {c['agent_id']} [{c['excerpt_id']}]: \"{c['text']}\"")
        lines.append("")

    lines.append("## Resolution Accuracy")
    lines.append("")
    lines.append("| Condition | Accuracy | Decisive | Correct | Contested | Escalation |")
    lines.append("|-----------|----------|----------|---------|-----------|------------|")
    for name, cr in results.items():
        lines.append(
            f"| {name} | {cr.accuracy:.1%} | {cr.decisive} | {cr.correct} | "
            f"{cr.contested} | {cr.escalation_rate:.1%} |"
        )
    lines.append("")

    lines.append("## Per-Type Accuracy")
    lines.append("")
    all_types = set()
    for cr in results.values():
        all_types.update(cr.per_type_accuracy().keys())
    if all_types:
        header = "| Condition | " + " | ".join(sorted(all_types)) + " |"
        sep = "|-----------|" + "|".join(["----------"] * len(all_types)) + "|"
        lines.append(header)
        lines.append(sep)
        for name, cr in results.items():
            row = f"| {name} |"
            for t in sorted(all_types):
                val = cr.per_type_accuracy().get(t)
                row += f" {val:.1%} |" if val is not None else " -- |"
            lines.append(row)
    lines.append("")

    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_comparison(
    seeds: list[SeedConflict] | None = None,
    *,
    backend: str = "fake",
    threshold: float = -1.0,
    out_dir: str = "results",
    llm_client=None,
    embedder_obj=None,
    reconcile_llm=None,
    judge_llm=None,
    resume: bool = True,
) -> dict:
    """Full comparison pipeline. Returns results dict.

    ``resume``: if a checkpoint from a matching prior run exists in
    ``out_dir`` (see ``eval.checkpoint``), skip detection/resolution for the
    documents and conditions it already completed and pick up from there.
    Pass ``resume=False`` (the CLI's ``--fresh``) to ignore any existing
    checkpoint and reprocess everything. The checkpoint is cleared once this
    function returns successfully, so a later fresh run never mistakes it for
    resumable state.
    """
    from eval import checkpoint as ckpt_mod
    from eval.fake_backend import FakeEmbedder, ScopedFakeLLM

    seeds = seeds or SEED_CONFLICTS
    doc_ids = [s.doc_id for s in seeds]
    seed_by_doc = {s.doc_id: s for s in seeds}
    topic_by_doc = {s.doc_id: s.topic for s in seeds}

    # Build LLM / embedder for the specified backend
    if backend == "fake":
        llm_client = llm_client or ScopedFakeLLM()
        embedder_obj = embedder_obj or FakeEmbedder()
        reconcile_llm = reconcile_llm or ScopedFakeLLM()
        judge_llm = judge_llm or llm_client
    else:
        from common.llm import make_judge_llm, make_llm
        from memory.detector import SentenceEmbedder

        llm_client = llm_client or make_llm()
        embedder_obj = embedder_obj or SentenceEmbedder()
        # The judge role (detector NLI + reconciler) may run on a different backend
        # from the agents (JUDGE_BACKEND); it follows LLM_BACKEND when unset.
        judge_llm = judge_llm or make_judge_llm()
        reconcile_llm = reconcile_llm or judge_llm

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- checkpoint: load if resumable for this exact configuration --------
    ckpt_file = ckpt_mod.checkpoint_path(out_path)
    fingerprint = ckpt_mod.fingerprint(
        backend=backend,
        threshold=threshold,
        agent_llm=_llm_label(llm_client, "agent_model"),
        judge_llm=_llm_label(judge_llm, "judge_model"),
        reconciler_llm=_llm_label(reconcile_llm, "judge_model"),
        doc_ids=sorted(doc_ids),
    )
    ckpt = ckpt_mod.load_checkpoint(ckpt_file, fingerprint) if resume else None
    if ckpt is None:
        ckpt = ckpt_mod.Checkpoint(config_fingerprint=fingerprint)
    elif ckpt.completed_docs:
        print(f"[checkpoint] resuming: {len(ckpt.completed_docs)}/{len(doc_ids)} docs already detected")

    # --- Phase A: run agents once, capture writes (cheap/cached even on resume) ---
    import tempfile as _tmp
    _a_dir = _tmp.mkdtemp(prefix="eval_agent_")
    store_a = SqliteMemoryStore(os.path.join(_a_dir, "agents.db"))
    writes = orch_run(store_a, seeds=seeds, llm=llm_client)
    print(f"[phase A] ran {len(writes)} agent writes over {len(seeds)} seeds")

    snapshot = _snapshot_writes(writes, seeds)
    items_all = [MemoryItem.from_dict(d) for d in snapshot.item_dicts]
    items_by_topic: dict[str, list[MemoryItem]] = defaultdict(list)
    for it in items_all:
        items_by_topic[it.topic].append(it)

    # --- Phase B: detect contradictions, one document at a time -------------
    detector = ConflictDetector(llm=judge_llm, embedder=embedder_obj, similarity_threshold=threshold)
    pending_docs = [d for d in doc_ids if d not in ckpt.completed_docs]
    for doc_id in pending_docs:
        topic = topic_by_doc[doc_id]
        specs, records = _detect_one_doc(seed_by_doc[doc_id], items_by_topic.get(topic, []), detector)
        ckpt.completed_docs.add(doc_id)
        ckpt.pair_specs_by_doc[doc_id] = specs
        ckpt.pair_records_by_doc[doc_id] = records
        ckpt_mod.save_checkpoint(ckpt, ckpt_file)

    # Reassemble the full snapshot (checkpoint-resumed + freshly-detected), in seed order
    snapshot.pair_specs = [s for d in doc_ids for s in ckpt.pair_specs_by_doc.get(d, [])]
    snapshot.pair_records = [r for d in doc_ids for r in ckpt.pair_records_by_doc.get(d, [])]
    det_metrics = _detection_metrics(snapshot.pair_records)
    n_flagged = len(snapshot.pair_specs)
    n_topics = len(set(ps["topic"] for ps in snapshot.pair_specs))
    print(f"[phase B] detected {n_flagged} contradiction pairs across {n_topics} topics")
    print(f"          P={det_metrics['precision']:.3f}  R={det_metrics['recall']:.3f}  F1={det_metrics['f1']:.3f}")

    # A topic with zero detected CONTRADICTION pairs has nothing to resolve -
    # Phase C must not manufacture a conflict for it (matches the pre-refactor
    # behavior of only ever iterating topics present in the pair_specs grouping).
    topics_with_conflicts = {ps["topic"] for ps in snapshot.pair_specs}

    # --- Phase C: resolve under each condition, one document at a time -----
    all_resolvers = _build_resolvers()
    conditions = ["null", "last_write_wins", "majority_vote", "static_confidence"]
    # Auto-add reliability if available
    if "reliability_aware" in all_resolvers:
        conditions.append("reliability_aware")

    tmp_root = _tmp.mkdtemp(prefix="eval_")
    results: dict[str, ConditionResult] = {}
    for cond in conditions:
        t0 = time.perf_counter()
        resolver = None if cond == "null" else all_resolvers[cond]

        store = SqliteMemoryStore(os.path.join(tmp_root, f"cond_{cond}.db"))
        for it in items_all:
            store.add(it)

        done_for_cond = ckpt.completed_conditions.setdefault(cond, set())
        decisions_for_cond = ckpt.decisions_by_condition.setdefault(cond, {})
        for doc_id in (d for d in doc_ids if d not in done_for_cond):
            topic = topic_by_doc[doc_id]
            if topic in topics_with_conflicts:
                decision = _score_one_topic(cond, resolver, store, snapshot, topic, reconciler_llm=reconcile_llm)
                if decision is not None:
                    decisions_for_cond[doc_id] = decision
            done_for_cond.add(doc_id)
            ckpt_mod.save_checkpoint(ckpt, ckpt_file)

        ordered_decisions = [decisions_for_cond[d] for d in doc_ids if d in decisions_for_cond]
        cr = _condition_result_from_decisions(cond, ordered_decisions)
        elapsed = time.perf_counter() - t0
        results[cond] = cr
        print(
            f"[phase C] {cond:25s}  acc={cr.accuracy:.1%}  "
            f"correct={cr.correct}/{cr.decisive}  "
            f"contested={cr.contested}  ({elapsed:.2f}s)"
        )

    # --- Phase D: write results -------------------------------------------
    config = {
        "backend": backend,
        "n_docs": len(seeds),
        "threshold": threshold,
        "agent_llm": _llm_label(llm_client, "agent_model"),
        "judge_llm": _llm_label(judge_llm, "judge_model"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report = {"config": config, "detection": det_metrics, "conditions": {k: v.to_dict() for k, v in results.items()}}
    (out_path / "run_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_path / "detection_pairs.json").write_text(
        json.dumps(
            {"config": config, "counts": det_metrics, "pairs": snapshot.pair_records},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    summary = _write_summary(det_metrics, results, out_path, config, snapshot.pair_records)
    print(f"\n[phase D] results written to {out_path}/")

    # Full success: clear the checkpoint so a later fresh run never mistakes
    # this completed state for something to resume from.
    ckpt_mod.clear_checkpoint(ckpt_file)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run resolution comparison across conditions.")
    parser.add_argument("--backend", choices=["fake", "real"], default="fake")
    parser.add_argument("--all", action="store_true", help="Use all 20 seeds (default)")
    parser.add_argument("--limit", type=int, default=None, help="Use first N seeds")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Stage-1 cosine-similarity threshold. Cosine spans [-1, 1], so the default "
        "-1.0 sends every same-topic pair to the judge; 0.0 would silently drop "
        "negative-similarity pairs.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from an existing checkpoint in --out if present (default).",
    )
    parser.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        help="Ignore any existing checkpoint in --out and reprocess everything.",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    seeds = SEED_CONFLICTS
    if args.limit:
        seeds = seeds[: args.limit]

    print(f"== eval/run_comparison  backend={args.backend}  docs={len(seeds)} ==\n")
    report = run_comparison(
        seeds, backend=args.backend, threshold=args.threshold, out_dir=args.out, resume=args.resume
    )

    # Print per-condition accuracy
    print("\n== accuracy summary ==")
    for name, cr in report["conditions"].items():
        print(f"  {name:25s}  {cr['accuracy']:.1%}  (decisive={cr['decisive']}, correct={cr['correct']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
