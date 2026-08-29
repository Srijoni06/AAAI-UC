"""Conflict reconciliation (NOT in milestone 1).

Planned: for a verified conflict, ask the LLM judge to classify it as
  - CREDIBILITY  -> one claim should defeat the other; hand to a resolver that
                    scores claims and marks losers SUPERSEDED / winner CONFIRMED.
  - COORDINATION -> multiple claims validly coexist; mark CONTESTED and keep all.
The resolver plugged in here is ``reliability.resolver`` (our contribution),
with ``baselines/*`` as comparison conditions.
"""

# TODO(milestone 3): implement CREDIBILITY vs COORDINATION classification.
