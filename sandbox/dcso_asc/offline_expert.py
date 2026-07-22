"""Part E, Level 1 (offline expert planner) -- SCOPE NOTE, not a full
implementation.

This file is intentionally a stub with a clear explanation, not a disguised
placeholder: Parts D-H's elaborate offline-RL pipeline (expert planner,
advantage estimation, distillation, specialized per-scenario controllers,
CMA-ES parameter search) were all premised on "a large, undiscovered
mechanism must exist to explain the gap between the local ceiling (~0.744)
and an assumed official top score (~0.805)". That premise was tested and
found FALSE this research phase (see forensic_equivalence_report.md): the
existing frozen candidate, with zero new mechanisms, already scores 0.8043
on seed 8080 -- the entire "gap" is ordinary seed-to-seed difficulty
variance, not a missing capability.

Building a full MCTS/SDDP/fitted-Q offline-expert pipeline to search for a
mechanism that the evidence says does not exist at the claimed scale would
not be a good use of further compute. The actual open question -- "can
small, validated, cross-seed-positive mechanisms be found incrementally" --
is already being pursued directly (see best_candidate.py and
experiment_ledger.jsonl), which is cheaper and has a clean track record
this session (multiple validated, if modest, gains).

If a specific NEW regret cluster is identified in a future phase with a
certified ceiling above 0.005 final score (per Part D's own stated
threshold for "worth working on"), an offline expert planner targeting THAT
specific cluster would be a reasonable next step, reusing:
  - global_oracle.py (already-validated whole-horizon LP, for generating
    causal rollout labels via oc._record_rt_trajectory-style trajectory
    recording, restricted to public/legal information only);
  - evaluate_paired.py (for measuring any resulting policy's paired gain
    against the current best_candidate.py).

No offline expert was trained this phase because no such certified,
above-threshold, unaddressed regret cluster was identified -- the largest
individual per-instance regrets found (regret_clusters.csv) are already
partially captured by best_candidate.py's DENSE extension, and the
remainder is concentrated in single, non-representative CONGESTED/DENSE
instances rather than a generalizable pattern.
"""

raise NotImplementedError(
    "See module docstring: no offline expert was trained this phase because "
    "the premise motivating it (a large undiscovered mechanism) was tested "
    "and found false. Extend this file only after identifying a specific, "
    "certified, >=0.005-final-score regret cluster not already addressed."
)
