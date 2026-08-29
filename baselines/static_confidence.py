"""Static-confidence resolver (NOT in milestone 1) - the primary comparison baseline.

Planned: fixed weight table over role / provenance / recency (LatticeMind-style
evidence-type weights, e.g. code-change=60, human-note=12, stale-observation=4).
Stateless: no memory of which agent has been reliable before. Our contribution
replaces this table with the online reliability signal in ``reliability/``.
"""

# TODO(milestone 2): implement fixed role/provenance/recency weighting.
