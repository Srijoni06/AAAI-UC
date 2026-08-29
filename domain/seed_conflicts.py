"""Seeded contradiction suite (NOT in milestone 1).

Planned: 15-20 deliberate contradictions over a multi-agent literature-
summarization pipeline, across three types:
  - inter-agent factual   (agents disagree on a fact from overlapping excerpts)
  - temporal / staleness  (an old claim contradicted by a newer correction)
  - provenance            (claim attributed to the wrong source / author)
Each seed carries a gold label (which claim should win, or "coexist") so
``eval/run_comparison.py`` can score resolution accuracy.

Milestone 1 uses the 3 hand-built documents in ``agents/orchestrator.py``
instead.
"""

# TODO(milestone 2): generate the 15-20 seed suite with gold labels.
