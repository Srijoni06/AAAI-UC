"""Conflict candidate detection (NOT in milestone 1).

Planned: embed each memory item with sentence-transformers, cluster items by
cosine similarity to find claims that are "about the same thing", then ask the
LLM judge (gemini-2.5-pro) to verify whether a candidate pair is a genuine
contradiction or merely related. Replaces the naive same-topic scan in
``memory.store.MemoryStore.list_conflicts``.
"""

# TODO(milestone 2): implement embedding clustering + LLM-judge verification.
