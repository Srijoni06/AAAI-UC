"""End-to-end demo: 5 summarizer agents over the seeded contradiction suite.

  1. spin up 5 summarizer agents (LLM backend from LLM_BACKEND: local | gemini).
     agent_A/C/D are a *correlated group* (shared model + near-identical prompt,
     same excerpt); agent_B and agent_E are independent.
  2. each agent reads one excerpt of each demo document and writes a one-sentence
     claim to shared memory, with real provenance (doc id, excerpt id, the
     verbatim slice it read).
  3. print every agent's write, grouped so the correlated group is obvious, and
     mark which agents landed on the gold excerpt.
  4. show the detected contradiction and "resolve" it with last-write-wins.

Run:  python demo.py            (first ~20 LLM calls, then served from cache)
      python demo.py --all      (run the full 20-document suite)

  Default backend is local (Ollama at localhost:11434, llama3.1:8b).
  Set LLM_BACKEND=gemini (+ GEMINI_API_KEY) for a verification run.

The default 4-document subset covers all four conflict types; the suite's
``COEXIST`` seed (doc-languages) and the rest are reached with ``--all``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.orchestrator import (
    DEFAULT_ROSTER,
    AgentWrite,
    agent_groups,
    correlated_groups,
    independent_agents,
    run,
)
from baselines.base import apply_resolution
from baselines.last_write_wins import LastWriteWins
from common.llm import make_llm, resolve_config
from domain.seed_conflicts import COEXIST, SEED_CONFLICTS, SEEDS_BY_ID, type_counts
from memory.store import MemoryStore, open_store

MEM_PATH = Path("demo_memory.db")

# One document per conflict type, kept short so the demo is cheap.
DEMO_DOC_IDS = [
    "doc-benchmark",   # factual    / subtle
    "doc-cost",        # magnitude  / obvious
    "doc-sota",        # staleness  / moderate
    "doc-attribution", # provenance / subtle
]


def _fresh(path: Path) -> None:
    """Remove the SQLite db and its WAL/SHM sidecars so each run starts clean."""
    base = str(path)
    for p in (base, base + "-wal", base + "-shm"):
        fp = Path(p)
        if fp.exists():
            fp.unlink()


def print_roster() -> None:
    print("== agent roster ==")
    for group, ids in correlated_groups().items():
        print(f"  correlated group {group!r} (shared model + near-identical prompt,")
        print(f"      reads the same excerpt): {', '.join(ids)}")
    print(f"  independent agents: {', '.join(independent_agents())}")
    print()


def print_writes_by_doc(writes: list[AgentWrite]) -> None:
    """Group the run's writes by document, then by agent group."""
    by_doc: dict[str, list[AgentWrite]] = {}
    for w in writes:
        by_doc.setdefault(w.seed.doc_id, []).append(w)

    groups = agent_groups(DEFAULT_ROSTER)
    corr = correlated_groups(DEFAULT_ROSTER)

    for doc_id, doc_writes in by_doc.items():
        seed = doc_writes[0].seed
        gold = (
            "COEXIST (both claims valid)"
            if seed.gold_excerpt_id == COEXIST
            else f"excerpt {seed.gold_excerpt_id!r}"
        )
        print(f"\n--- {doc_id}: {seed.title} ---")
        print(f"  Q: {seed.question}")
        print(
            f"  conflict type: {seed.conflict_type.value} | "
            f"difficulty: {seed.difficulty.value} | gold: {gold}"
        )

        w_by_agent = {w.agent_id: w for w in doc_writes}

        for gname, members in corr.items():
            first = w_by_agent[members[0]]
            print(
                f"\n  correlated group {gname!r}  "
                f"[all read excerpt {first.excerpt.excerpt_id!r} "
                f"- {first.excerpt.section}]"
            )
            for aid in members:
                w = w_by_agent[aid]
                print(f"    {aid} {_mark(w)}: {w.claim}")

        indeps = [g for g, ids in groups.items() if len(ids) == 1]
        print("\n  independent agents")
        for aid in indeps:
            w = w_by_agent[aid]
            print(
                f"    {aid} {_mark(w)}  [read excerpt {w.excerpt.excerpt_id!r} "
                f"- {w.excerpt.section}]: {w.claim}"
            )


def _mark(w: AgentWrite) -> str:
    if w.correct is None:
        return "(–)"
    return "(gold)" if w.correct else "(off-gold)"


def show_memory(store: MemoryStore) -> None:
    for it in store.list():
        print(f"  [{it.status.value:10}] {it.agent_id}  ({it.topic})")
        print(f"      {it.content}")
        src_id = it.metadata.get("source_id", "?")
        print(
            f"      source_type={it.source_type.value} origin={it.origin.value} "
            f"authority={it.authority.name} version={it.version} source_id={src_id}"
        )
        if it.evidence_span:
            span = it.evidence_span
            span = span if len(span) <= 90 else span[:87] + "..."
            print(f'      evidence="{span}"')


def main() -> int:
    run_all = "--all" in sys.argv[1:]

    res = resolve_config()
    print(res.banner())
    print()

    try:
        llm = make_llm()
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}")
        return 1
    assert llm.backend == res.backend

    seeds = (
        SEED_CONFLICTS
        if run_all
        else [SEEDS_BY_ID[d] for d in DEMO_DOC_IDS]
    )

    _fresh(MEM_PATH)
    store = open_store(MEM_PATH, backend="sqlite")

    print_roster()
    print(
        f"== running {len(DEFAULT_ROSTER)} agents on {len(seeds)} document(s) "
        f"(backend={llm.backend}, model={llm.agent_model}) ==\n"
        f"   full suite: {len(SEED_CONFLICTS)} docs, conflict types {type_counts()}"
    )

    try:
        writes = run(store, seeds=seeds, llm=llm)
    except RuntimeError as e:
        print(f"error: {e}")
        return 1

    print("\n== agent writes (grouped) ==")
    print_writes_by_doc(writes)

    conflicts = store.list_conflicts()
    print(f"\n\n== detected {len(conflicts)} conflict(s) via naive same-topic scan ==")
    for c in conflicts:
        seed = SEEDS_BY_ID.get(c.items[0].source_doc_id)
        gold = seed.gold_excerpt_id if seed else "?"
        print(f"\n  topic: {c.topic}  (gold excerpt: {gold})")
        for it in c.items:
            grp = it.metadata.get("agent_group", "?")
            exc = it.metadata.get("excerpt_id", "?")
            print(f"    - {it.agent_id:8} [{grp:11} | {exc:10}] {it.content}")

    resolver = LastWriteWins()
    print(f"\n== resolving every conflict with '{resolver.name}' ==")
    for c in conflicts:
        resolution = resolver.resolve(c)
        apply_resolution(store, resolution)
        winner = store.get(resolution.winner_id)
        print(f"\n  topic: {c.topic}")
        print(f"    winner:     {winner.agent_id} -> {winner.content}")
        print(f"    superseded: {resolution.superseded_ids}")
        print(f"    rationale:  {resolution.rationale}")

    print("\n== shared memory after resolution ==")
    show_memory(store)

    remaining = store.list_conflicts()
    print(f"\n== {len(remaining)} unresolved conflict(s) remain ==")
    print(f"(memory persisted to {MEM_PATH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
