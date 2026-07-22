# Phase 1 — Full Oracle Diagnostic Report (seed 987654)

## Score formula proof (why profit-optimal = quality-optimal per instance)

`quality = clip((profit - baseline)/(baseline+1e-6), -10, 10)` is a clip of an
affine, strictly increasing function of `profit`. There is no cross-instance
coupling (each of the 100 instances is scored independently, then averaged).
Therefore **the profit-maximizing trajectory for a given instance is also
the quality-maximizing trajectory** for that instance — clip∘affine-increasing
is monotonic non-decreasing, so no alternative trajectory with lower profit
can produce higher clipped quality. `oracle2` (network-feasible, whole-horizon
profit-maximizing LP, already validated with zero bound/flow violations on
replay) is therefore exactly the score-optimal oracle per instance. `Final
Score = mean(quality_int)/1e7`, and since `quality_int` is a per-instance
integer rounding of the clipped quality, `Final Score ≈ mean(category means)/10`
to within rounding noise (verified: every effect measured this session is
≥1e-4, far above the 1e-6 rounding granularity).

## Oracle execution (all real, executed against the production LP + replay-verified)

| Scenario | Sample | Coverage | Per-instance oracle cost | Zero bound/flow violations |
|---|---|---|---:|---|
| BASELINE | 20/20 | full | 0.4s | yes |
| CONGESTED | 20/20 | full | 1.5s | yes |
| MULTIDAY | 5/20 | sample | 36s | yes |
| DENSE | 7/20 | sample | 130-172s | yes |
| CAPSTONE | 0/20 | **memory-infeasible** | attempted 43.9 GiB allocation, failed | N/A |

CAPSTONE's dense whole-horizon LP formulation (`oracle_certified.py::oracle2`)
requires a `(153600, 38400)` dense constraint matrix at B=100,T=192 — this
genuinely fails with `numpy._core._exceptions._ArrayMemoryError` on this
machine. This is a real computational limit of the CURRENT LP tooling
(dense matrix construction), not a shortcut. A sparse reformulation could
likely resolve it but was not attempted this phase given the analytically-
bounded impact (below).

## Category-level oracle results

| Scenario | n | Frozen mean | Oracle mean | Gain (mean) | Gain (min-max) |
|---|---:|---:|---:|---:|---:|
| BASELINE | 20 (full) | 1.7742 | 1.9809 | **+0.2067** | +0.076 to +0.671 |
| CONGESTED | 20 (full) | 5.2785 | 5.6978 | **+0.4193** | +0.000 to +1.542 |
| MULTIDAY | 5 (sample) | 10.0000 | 10.0000 | +0.0000 | all saturated |
| DENSE | 7 (sample) | 9.3927 | 9.8622 | +0.4695 | +0.000 to +1.884 |
| CAPSTONE | 0 | — | — | analytically bounded ≤ (10−9.973) | — |

**Important methodological correction confirmed**: the prior 48-instance
worst/median/high-weighted sample overstated CONGESTED's true gain
(claimed +0.954 vs the rigorous full-20 result of **+0.4193** — less than
half). This validates your objection to the biased-subset methodology.
Conversely, DENSE's gain was initially overstated by an unrepresentative
3-instance sample (+1.095) then corrected downward to +0.4695 with a
7-instance sample (4 of the added 4 instances were already perfectly
saturated) — still a genuine, non-trivial, previously-unrecognized finding.

## Ceiling calculation

**Scope A — BASELINE+CONGESTED only, MULTIDAY/DENSE/CAPSTONE frozen exactly
as originally instructed** (rigorous: full, unbiased 20-instance oracle for
both active categories):

```
ceiling = 0.7314848 + 0.2067*(20/100)/10 + 0.4193*(20/100)/10
        = 0.7314848 + 0.004134 + 0.008386
        = 0.7440048
gap to 0.75 = 0.0059952
```

**This proves, with the complete (not sampled) oracle for both active
categories, that Stage 1 is unreachable within the originally-instructed
scope.** This satisfies stopping condition B for that scope specifically.

**Scope B — all five scenarios, illegal theoretical maximum** (perfect
foresight; DENSE/MULTIDAY/CAPSTONE would need to be unfrozen):

```
+ DENSE (7-sample est.):        +0.009390
+ MULTIDAY (absolute max, 10.0 everywhere): +0.001520
+ CAPSTONE (absolute max, 10.0 everywhere): +0.000540
= 0.7554548
gap to 0.75 = −0.0054548  (theoretically ABOVE target)
```

This is the illegal, hindsight-only, MULTIDAY/CAPSTONE-generously-bounded
maximum — not causally achievable. But it identifies **DENSE as the
single most promising unexplored avenue**: it is the only category besides
BASELINE/CONGESTED with a *directly measured* (not just bounded) non-trivial
oracle gain, and unfreezing it is explicitly permitted by your own framing
("*initially* remain byte-identical").

## Decision

Given every causal mechanism built this session has captured roughly
5-10% of its corresponding oracle gap (not the full hindsight value — RT
price shocks are provably i.i.d. with a fresh reseed every step, so no
causal policy can capture the full illegal ceiling), a realistic causal
DENSE mechanism would likely add on the order of +0.0005-0.001 to final
score, not the full +0.0094 theoretical. Combined with BASELINE/CONGESTED's
already-captured advantage, this makes 0.75 a stretch even with DENSE
included — but it is the only remaining avenue with *any* directly-measured
headroom, versus zero for MULTIDAY (sampled saturated) and an unmeasurable
but tightly-bounded amount for CAPSTONE.

**Recommendation**: extend the already-validated exact Gaussian-Hermite +
Pareto-jump-quantile mixture mechanism (built and cross-seed-validated in
the prior research phase for BASELINE/CONGESTED) to DENSE, using DENSE's
own public constants (sigma=0.25, rho_jump=0.04, alpha=2.7 from
`scenarios.py`), rather than building the full generic SSO-FVI offline-RL
pipeline from scratch — the ceiling evidence does not justify that much
additional engineering investment when a cheaper, already-proven mechanism
class can be tested on this specific, evidence-backed opportunity first.
