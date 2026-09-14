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

Outputs ``results/run_comparison.json`` + ``results/summary.md`` and prints
the summary to stdout.
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
from memory.detector import ConflictDetector, Relationship
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

    @property
    def topics(self) -> list[str]:
        return list(self.seed_topic_map.keys())


def _snapshot_writes(writes, seeds: list[SeedConflict]) -> Snapshot:
    """Capture writes + per-item metadata into a serializable snapshot."""
    item_dicts = [w.item.to_dict() for w in writes]
    topic_map = {s.topic: s.doc_id for s in seeds}
    return Snapshot(seeds=seeds, item_dicts=item_dicts, pair_specs=[], seed_topic_map=topic_map)


def _detect_pairs(snapshot: Snapshot, detector: ConflictDetector) -> None:
    """Run two-stage detection and store pair specs (mutates snapshot)."""
    items = [MemoryItem.from_dict(d) for d in snapshot.item_dicts]
    pairs = detector.detect(items, relationships={Relationship.CONTRADICTION})
    for p in pairs:
        snapshot.pair_specs.append(
            {
                "item_a_id": p.item_a.id,
                "item_b_id": p.item_b.id,
                "topic": p.topic,
                "similarity": p.similarity,
                "rationale": p.rationale,
            }
        )


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def _detection_metrics(snapshot: Snapshot) -> dict:
    """Pair-level detection precision/recall/F1 against gold labels."""
    items_by_id = {d["id"]: MemoryItem.from_dict(d) for d in snapshot.item_dicts}
    tp = fp = fn = 0
    gold_negative_topics = set()
    for d in snapshot.seed_topic_map:
        seed = SEEDS_BY_ID.get(snapshot.seed_topic_map[d])
        if seed and seed.gold_excerpt_id == COEXIST:
            gold_negative_topics.add(d)

    flagged: set[frozenset] = set()
    for ps in snapshot.pair_specs:
        fid = frozenset([ps["item_a_id"], ps["item_b_id"]])
        flagged.add(fid)
        item_a = items_by_id[ps["item_a_id"]]
        item_b = items_by_id[ps["item_b_id"]]
        exc_a = item_a.metadata.get("excerpt_id", "")
        exc_b = item_b.metadata.get("excerpt_id", "")
        is_cross_excerpt = exc_a != exc_b
        is_gold_negative = ps["topic"] in gold_negative_topics
        if is_cross_excerpt and not is_gold_negative:
            tp += 1
        else:
            fp += 1

    # False negatives: cross-excerpt pairs in non-COEXIST topics not flagged
    # Build ground-truth cross-excerpt pairs
    items_by_topic: dict[str, list[MemoryItem]] = defaultdict(list)
    for it in items_by_id.values():
        items_by_topic[it.topic].append(it)
    for topic, titems in items_by_topic.items():
        if topic in gold_negative_topics:
            continue
        excerpt_groups: dict[str, list[MemoryItem]] = defaultdict(list)
        for it in titems:
            eid = it.metadata.get("excerpt_id", "")
            excerpt_groups[eid].append(it)
        for eid_a, items_a in excerpt_groups.items():
            for eid_b, items_b in excerpt_groups.items():
                if eid_a < eid_b:
                    for ia in items_a:
                        for ib in items_b:
                            fid = frozenset([ia.id, ib.id])
                            if fid not in flagged:
                                fn += 1

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
            if d.get("gold") is not None:
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


def _score_condition(
    name: str,
    resolver,
    snapshot: Snapshot,
    reconciler_llm=None,
    tmp_dir: str | None = None,
) -> ConditionResult:
    """Replay detection results into a fresh store, resolve, and score."""
    import tempfile
    import os
    db_dir = tmp_dir or tempfile.mkdtemp()
    db_path = os.path.join(db_dir, f"cond_{name}.db")
    store = SqliteMemoryStore(db_path)
    items = [MemoryItem.from_dict(d) for d in snapshot.item_dicts]
    for it in items:
        store.add(it)

    rec = ConditionResult(name=name)

    # Group pairs by topic -> conflict groups
    topic_pairs: dict[str, list[dict]] = defaultdict(list)
    for ps in snapshot.pair_specs:
        topic_pairs[ps["topic"]].append(ps)

    for topic, pairs in topic_pairs.items():
        all_items = store.list(topic=topic)
        live_items = [it for it in all_items if it.status != Status.SUPERSEDED]
        if len(live_items) < 2:
            continue
        conflict = Conflict(topic=topic, items=live_items)

        rec.total_conflicts += 1
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
            rec.coordination += 1
            from baselines.base import Resolution
            resolution = Resolution.coexist(
                topic=topic,
                strategy="coordination",
                item_ids=[it.id for it in live_items],
                rationale=llm_rec.rationale,
            )
            apply_resolution(store, resolution)
            confirmed_excerpts = all_excerpt_ids
            if gold == COEXIST:
                is_correct = True  # all excerpts expected
            else:
                is_correct = gold in confirmed_excerpts
            rec.decisive += 1
            decision["correct"] = is_correct
            decision["rationale"] = llm_rec.rationale
            rec.decisions.append(decision)
            if is_correct:
                rec.correct += 1
            else:
                rec.incorrect += 1
            continue

        # --- CREDIBILITY: delegate to resolver ---
        if name == "null":
            rec.contested += 1
            decision["correct"] = False
            decision["rationale"] = "null resolver: no resolution applied"
            rec.decisions.append(decision)
            continue

        resolution = resolver.resolve(conflict)
        apply_resolution(store, resolution)

        confirmed = [store.get(cid) for cid in resolution.confirmed_ids]
        confirmed = [c for c in confirmed if c is not None]
        confirmed_excerpts = {c.metadata.get("excerpt_id", "") for c in confirmed}

        if confirmed_excerpts:
            rec.decisive += 1
            if gold == COEXIST:
                is_correct = all_excerpt_ids.issubset(confirmed_excerpts)
            else:
                is_correct = confirmed_excerpts == {gold}
            decision["correct"] = is_correct
            decision["rationale"] = resolution.rationale
            rec.decisions.append(decision)
            if is_correct:
                rec.correct += 1
            else:
                rec.incorrect += 1
        else:
            rec.contested += 1
            decision["correct"] = False
            decision["rationale"] = resolution.rationale
            rec.decisions.append(decision)

        # Update reliability memory if resolver supports it
        if hasattr(resolver, "update_memory"):
            resolver.update_memory(
                live_items, resolution, correct=confirmed_excerpts and is_correct
            )

    return rec


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _write_summary(
    detection: dict,
    results: dict[str, ConditionResult],
    out_dir: Path,
    config: dict,
) -> str:
    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"**Backend:** {config['backend']}  ")
    lines.append(f"**Documents:** {config['n_docs']}  ")
    lines.append(f"**Agents:** {len(DEFAULT_ROSTER)}  ")
    lines.append(f"**Similarity threshold:** {config['threshold']}  ")
    lines.append("")

    lines.append("## Detection Metrics (pair-level)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in detection.items():
        lines.append(f"| {k} | {v} |")
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
    threshold: float = 0.0,
    out_dir: str = "results",
    llm_client=None,
    embedder_obj=None,
    reconcile_llm=None,
) -> dict:
    """Full comparison pipeline. Returns results dict."""
    from eval.fake_backend import FakeEmbedder, ScopedFakeLLM

    seeds = seeds or SEED_CONFLICTS

    # Build LLM / embedder for the specified backend
    if backend == "fake":
        llm_client = llm_client or ScopedFakeLLM()
        embedder_obj = embedder_obj or FakeEmbedder()
        reconcile_llm = reconcile_llm or ScopedFakeLLM()
    else:
        from common.llm import make_llm
        from memory.detector import SentenceEmbedder

        llm_client = llm_client or make_llm()
        embedder_obj = embedder_obj or SentenceEmbedder()
        reconcile_llm = reconcile_llm or llm_client

    # --- Phase A: run agents once, capture writes --------------------------
    import tempfile as _tmp
    _a_dir = _tmp.mkdtemp(prefix="eval_agent_")
    store_a = SqliteMemoryStore(os.path.join(_a_dir, "agents.db"))
    writes = orch_run(store_a, seeds=seeds, llm=llm_client)
    print(f"[phase A] ran {len(writes)} agent writes over {len(seeds)} seeds")

    # --- Phase B: detect contradictions once -------------------------------
    snapshot = _snapshot_writes(writes, seeds)
    detector = ConflictDetector(llm=llm_client, embedder=embedder_obj, similarity_threshold=threshold)
    _detect_pairs(snapshot, detector)
    det_metrics = _detection_metrics(snapshot)
    n_flagged = len(snapshot.pair_specs)
    n_topics = len(set(ps["topic"] for ps in snapshot.pair_specs))
    print(f"[phase B] detected {n_flagged} contradiction pairs across {n_topics} topics")
    print(f"          P={det_metrics['precision']:.3f}  R={det_metrics['recall']:.3f}  F1={det_metrics['f1']:.3f}")

    # --- Phase C: resolve under each condition -----------------------------
    all_resolvers = _build_resolvers()
    conditions = ["null", "last_write_wins", "majority_vote", "static_confidence"]
    # Auto-add reliability if available
    if "reliability_aware" in all_resolvers:
        conditions.append("reliability_aware")

    import tempfile as _tmp
    tmp_root = _tmp.mkdtemp(prefix="eval_")
    results: dict[str, ConditionResult] = {}
    for cond in conditions:
        t0 = time.perf_counter()
        if cond == "null":
            cr = _score_condition("null", None, snapshot, reconciler_llm=reconcile_llm, tmp_dir=tmp_root)
        else:
            resolver = all_resolvers[cond]
            cr = _score_condition(cond, resolver, snapshot, reconciler_llm=reconcile_llm, tmp_dir=tmp_root)
        elapsed = time.perf_counter() - t0
        results[cond] = cr
        print(
            f"[phase C] {cond:25s}  acc={cr.accuracy:.1%}  "
            f"correct={cr.correct}/{cr.decisive}  "
            f"contested={cr.contested}  ({elapsed:.2f}s)"
        )

    # --- Phase D: write results -------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = {
        "backend": backend,
        "n_docs": len(seeds),
        "threshold": threshold,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report = {"config": config, "detection": det_metrics, "conditions": {k: v.to_dict() for k, v in results.items()}}
    (out_path / "run_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = _write_summary(det_metrics, results, out_path, config)
    print(f"\n[phase D] results written to {out_path}/")

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
    parser.add_argument("--threshold", type=float, default=0.0, help="Stage-1 similarity threshold")
    args = parser.parse_args()

    seeds = SEED_CONFLICTS
    if args.limit:
        seeds = seeds[: args.limit]

    print(f"== eval/run_comparison  backend={args.backend}  docs={len(seeds)} ==\n")
    report = run_comparison(seeds, backend=args.backend, threshold=args.threshold, out_dir=args.out)

    # Print per-condition accuracy
    print("\n== accuracy summary ==")
    for name, cr in report["conditions"].items():
        print(f"  {name:25s}  {cr['accuracy']:.1%}  (decisive={cr['decisive']}, correct={cr['correct']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
