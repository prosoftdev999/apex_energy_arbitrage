"""Part E, Level 2 (compact distilled policy) -- SCOPE NOTE.

Depends on offline_expert.py, which was not run this phase (see that
file's docstring for the evidence-based reason: the "missing large
mechanism" premise was tested and found false this research phase).

The current best_candidate.py already IS a compact, hand-derived
(not learned/distilled) deterministic policy -- built from exact public
distributional parameters (Gauss-Hermite quadrature, Pareto-jump quantile
mixture, exact congestion-premium probability) rather than fitted from
offline rollouts. This was a deliberate choice validated across this
session: analytically-derived mechanisms with a clear causal justification
have reproduced positively across every seed tested, whereas none of this
session's experiments needed a learned/distilled model to reach that bar.

Revisit this file only if a future regret-cluster investigation (see
offline_expert.py) produces expert trajectories that a compact model
would meaningfully improve on beyond what analytic derivation already
captures.
"""

raise NotImplementedError(
    "See module docstring: depends on offline_expert.py, which was not run "
    "this phase for a documented, evidence-based reason."
)
