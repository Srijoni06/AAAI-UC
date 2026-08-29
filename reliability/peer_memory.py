"""Online per-agent reliability + pairwise correlation (NOT in milestone 1).

Planned:
  - competence[agent]      : running estimate of how often this agent's claims
                             survive reconciliation, updated from resolution
                             outcomes (post-decision correctness feedback).
  - correlation[(a, b)]     : whether two agents tend to be right/wrong together.
                             High correlation means their agreement is weak
                             corroboration (shared bias / correlated error), not
                             two independent confirmations.
Both are consumed by ``reliability.resolver`` as the credibility-weighting
function, replacing the static evidence-type table.
"""

# TODO(milestone 3): implement competence + correlation tracking.
