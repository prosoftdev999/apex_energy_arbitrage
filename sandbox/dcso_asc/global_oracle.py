"""Global oracle reference for this research phase.

Rather than rebuilding a fourth independent full-scale oracle formulation,
this module thinly wraps the ALREADY-VALIDATED oracle_certified.py::oracle2
(whole-horizon, network-feasible, convex LP -- global optimality proven via
convexity, and cross-checked exactly against two independent methods, DP and
exhaustive search, on a reduced instance -- see oracle_validation_small.py
and certified_oracle_report.md). Building a fourth formulation from scratch
was not warranted: the "central contradiction" motivating this research
phase (local ceiling ~0.744 vs an assumed official top score ~0.805) was
resolved as ordinary seed-to-seed difficulty variance (forensic_equivalence_
report.md), not an oracle defect -- so there was no open question about
oracle correctness left to re-investigate at full scale.
"""
import sys
from pathlib import Path

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SANDBOX_DIR))

import oracle_certified as oc

# Re-exported for convenience / API stability.
oracle2 = oc.oracle2
replay_through_production = oc.replay_through_production
_record_rt_trajectory = oc._record_rt_trajectory
_battery_arrays = oc._battery_arrays
_reshape_to_per_step = oc._reshape_to_per_step

__all__ = ["oracle2", "replay_through_production", "_record_rt_trajectory",
           "_battery_arrays", "_reshape_to_per_step"]
