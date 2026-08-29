"""Milestone-1 end-to-end demo.

  1. spin up 2 summarizer agents (real gemini-2.0-flash calls)
  2. each reads a different excerpt of the same doc and writes a claim to
     shared memory
  3. show the memory contents and the detected contradiction
  4. run last-write-wins to "resolve" it and show the updated memory

Run:  python demo.py         (requires GEMINI_API_KEY in the environment)
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.orchestrator import AGENT_MODEL, DOCUMENTS, make_client, run
from baselines.base import apply_resolution
from baselines.last_write_wins import LastWriteWins
from memory.store import MemoryStore, Status, open_store

MEM_PATH = Path("demo_memory.json")


def show_memory(store: MemoryStore) -> None:
    for it in store.list():
        ts = f"{it.timestamp:.3f}"
        print(f"  [{it.status.value:10}] {it.agent_id}  ({it.topic})")
        print(f"      {it.content}")
        print(f"      src={it.source_doc_id}  ts={ts}")


def main() -> int:
    try:
        client = make_client()
    except RuntimeError as e:
        print(f"error: {e}")
        return 1

    if MEM_PATH.exists():
        MEM_PATH.unlink()
    store = open_store(MEM_PATH)

    print(f"== running 2 agents on {len(DOCUMENTS)} documents ({AGENT_MODEL}) ==\n")
    run(store, client=client)

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
