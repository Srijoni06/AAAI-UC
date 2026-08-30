"""Milestone-1 end-to-end demo.

  1. spin up 2 summarizer agents (LLM backend from LLM_BACKEND: local | gemini)
  2. each reads a different excerpt of the same doc and writes a claim to
     shared memory
  3. show the memory contents and the detected contradiction
  4. run last-write-wins to "resolve" it and show the updated memory

Run:  python demo.py
  Default backend is local (Ollama at localhost:11434, llama3.1:8b).
  Set LLM_BACKEND=gemini (+ GEMINI_API_KEY) for a verification run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.orchestrator import DOCUMENTS, run
from baselines.base import apply_resolution
from baselines.last_write_wins import LastWriteWins
from common.llm import make_llm, resolve_config
from memory.store import MemoryStore, Status, open_store

MEM_PATH = Path("demo_memory.db")


def _fresh(path: Path) -> None:
    """Remove the SQLite db and its WAL/SHM sidecars so each run starts clean."""
    base = str(path)
    for p in (base, base + "-wal", base + "-shm"):
        fp = Path(p)
        if fp.exists():
            fp.unlink()


def show_memory(store: MemoryStore) -> None:
    for it in store.list():
        ts = f"{it.timestamp:.3f}"
        print(f"  [{it.status.value:10}] {it.agent_id}  ({it.topic})")
        print(f"      {it.content}")
        print(f"      src={it.source_doc_id}  ts={ts}")
        line = (
            f"      source_type={it.source_type.value}  authority={it.authority.name}  "
            f"origin={it.origin.value}  version={it.version}"
        )
        if it.evidence_span:
            line += f'  evidence="{it.evidence_span}"'
        print(line)


def main() -> int:
    # Print the resolved config BEFORE building any client, so a misread
    # LLM_BACKEND is visible even if client construction later fails.
    res = resolve_config()
    print(res.banner())
    print()

    try:
        llm = make_llm()
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}")
        return 1

    assert llm.backend == res.backend  # make_llm and resolve_config must agree

    _fresh(MEM_PATH)
    store = open_store(MEM_PATH, backend="sqlite")

    print(
        f"== running 2 agents on {len(DOCUMENTS)} documents "
        f"(backend={llm.backend}, model={llm.agent_model}) ==\n"
    )
    try:
        run(store, llm=llm)
    except RuntimeError as e:
        print(f"error: {e}")
        return 1

    print("== shared memory after agent writes ==")
    show_memory(store)

    conflicts = store.list_conflicts()
    print(f"\n== detected {len(conflicts)} conflict(s) ==")
    for c in conflicts:
        print(f"\n  topic: {c.topic}")
        for it in c.items:
            print(f"    - {it.agent_id}: {it.content}")

    resolver = LastWriteWins()
    print(f"\n== resolving with '{resolver.name}' ==")
    for c in conflicts:
        res = resolver.resolve(c)
        apply_resolution(store, res)
        winner = store.get(res.winner_id)
        print(f"\n  topic: {c.topic}")
        print(f"    winner:     {winner.agent_id} -> {winner.content}")
        print(f"    superseded: {res.superseded_ids}")
        print(f"    rationale:  {res.rationale}")

    print("\n== shared memory after resolution ==")
    show_memory(store)

    remaining = store.list_conflicts()
    print(f"\n== {len(remaining)} unresolved conflict(s) remain ==")
    print(f"\n(memory persisted to {MEM_PATH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
