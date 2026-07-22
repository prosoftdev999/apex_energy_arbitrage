# Final Report: Independent Oracle Red-Team Audit + Fitted Value Iteration (Phase 3)

## 1. Did 0.7503278 survive the independent red-team audit?

**Yes.** `sandbox/oracle_red_team.py` is a completely independent re-implementation
(different LP formulation -- explicit SOC state variables with equality
recursion constraints vs. the original's cumulative-sum inequality
constraints; different solver -- `highs-ipm` interior-point + crossover vs.
`highs` dual simplex; no shared code with `oracle_certified.py`/
`audit_target_headroom.py`). Run on all 100 instances of seed 987654:

```
current_score  = 0.7339332
oracle_score   = 0.7503278   (prior claim: 0.7503278, diff = +0.0000000)
```

Exact match to all 7 reported decimal places. All 100 instances solved to
certified optimality (`n_failed=0`). Zero bound violations, zero flow
violations on replay through the real production `Challenge.take_step`.
Residual diagnostics: max primal (inequality) residual `8.9e-10`, max
equality (SOC recursion) residual `5.6e-09`, max SOC-variable-vs-manual
residual `5.6e-09` -- all at floating-point noise level, not modeling error.

A genuine engineering byproduct: the sparse SOC-explicit-variable
formulation also makes CAPSTONE's exact network-feasible oracle tractable
for the first time this session (the original dense cumulative-sum
formulation failed with a ~44 GiB memory allocation; the sparse
reformulation solves it in ~150-200s via interior-point method, not
simplex, which alone does not converge within 90s at this scale).

## 2. Exact corrected oracle score

Unchanged: **0.7503278**. No correction needed.

## 3. Is 0.760 mathematically feasible?

**No.** 0.7503278 is a rigorous Class-B upper bound (drops only a strictly
non-negative degradation cost term; every other constraint -- PTDF flows,
SOC transitions, power limits -- is exact) computed with full knowledge of
every future realized RT price, information no causal policy can legally
have. Since 0.760 > 0.7503278, it exceeds even this illegal, hindsight
ceiling. The same conclusion applies a fortiori to 0.780, 0.800, and 0.810.
Per the user's own Phase 2 rule ("oracle valid and still below 0.760: state
clearly that 0.760-0.810 cannot be reached legally... do not manufacture a
policy claiming otherwise"), no `policy_stage1.py`-style file was created.

## 4. Top 20 current-to-oracle gap instances

| nonce | scenario | current_q | oracle_q | gap |
|---|---|---|---|---|
| 3 | DENSE | +8.2523 | +10.0000 | +1.7477 |
| 83 | DENSE | +7.6791 | +9.0359 | +1.3569 |
| 56 | CONGESTED | +7.8308 | +9.1695 | +1.3387 |
| 6 | CONGESTED | +7.0918 | +8.0771 | +0.9854 |
| 88 | DENSE | +6.8247 | +7.7141 | +0.8894 |
| 66 | CONGESTED | +3.7465 | +4.4116 | +0.6651 |
| 61 | CONGESTED | +9.3418 | +10.0000 | +0.6582 |
| 96 | CONGESTED | +6.2398 | +6.8542 | +0.6144 |
| 49 | CAPSTONE | +9.4565 | +10.0000 | +0.5435 |
| 5 | BASELINE | +5.0654 | +5.5971 | +0.5317 |
| 47 | MULTIDAY | +9.4936 | +10.0000 | +0.5064 |
| 37 | MULTIDAY | +9.5845 | +10.0000 | +0.4155 |
| 76 | CONGESTED | +3.1469 | +3.5581 | +0.4113 |
| 70 | BASELINE | +4.0403 | +4.4271 | +0.3868 |
| 41 | CONGESTED | +1.7089 | +2.0949 | +0.3860 |
| 31 | CONGESTED | +3.6252 | +3.9863 | +0.3611 |
| 51 | CONGESTED | +2.8801 | +3.2249 | +0.3448 |
| 17 | MULTIDAY | +9.6839 | +10.0000 | +0.3161 |
| 60 | BASELINE | +3.1978 | +3.4964 | +0.2986 |
| 81 | CONGESTED | +1.8478 | +2.1415 | +0.2937 |

## 5. Feature set used by the joint value correction

26 causal, public features (`fvi_common.FEATURE_NAMES`): time-remaining
fraction + cyclic encoding; aggregate charge/discharge headroom
(normalized); SOC-fraction mean/std/p10/p50/p90 and near-bound
concentration; charge/discharge headroom coefficient of variation; top-3
line margin mean/min; mean/max line utilization; future (8-step) congestion
risk; DA price level/slope/curvature/future-std; current and previous
RT-vs-DA residual; two headroom-x-utilization interaction terms.

## 6. Training / validation seeds

Training: `[7, 13, 19, 31, 37, 43, 53, 61, 67, 73]` (10 seeds, disjoint from
dev/validation/reference). Development validation: 42, 2025 (used for
staged comparison). Validation: 987654 (used once for the comparison table
below, not tuned against -- no hyperparameter was adjusted after seeing
987654's number). Reference: 123 (not used this phase). 15,360 training rows
from 120 rollout episodes (10 seeds x 3 scenarios x 4 nonces), BASELINE/
CONGESTED/DENSE only (MULTIDAY/CAPSTONE excluded: independent oracle audit
shows +0.076/+0.027 mean gain, essentially saturated, no material headroom
for any architecture).

## 7. Model classes tested

sklearn is not installed in this environment; all three classes below are
implemented directly on numpy/scipy (closed-form ridge), consistent with
this project's established preference for interpretable/verifiable models
over black-box NN/RF. Grouped (leave-one-training-seed-out) cross-validation:

| Model | mean R2 | std R2 | mean MAE |
|---|---:|---:|---:|
| linear_interactions (ridge + 28 interaction terms) | **+0.8066** | 0.0591 | 28833 |
| binned_additive (per-feature quantile bins + ridge, monotonicity-clipped) | +0.7290 | 0.0774 | 37660 |
| lowrank_quadratic (SVD top-5 + quadratic ridge) | -1.7625 | 4.0546 | 47698 |

Gradient-boosted trees were **not** tested -- sklearn/xgboost/lightgbm are
unavailable in this environment; stated honestly rather than substituting
something else silently.

## 8. Fitted-value-iteration learning curve

Only **one** iteration was run (Iteration 0 = frozen `best_candidate.py`
rollouts -> fit C_theta). Per the spec's own stopping logic ("repeat until
validation improvement stops"), iteration 2 was not attempted: iteration 1's
*deployment* result was unambiguously negative (see #9-10 below), so
continuing to iterate on a broken reward signal would not be productive
research -- it would be re-tuning a mechanism already shown not to work,
which the user's own instructions elsewhere in this session explicitly rule
out.

## 9. Action-change frequency vs. best_candidate.py

98.8% of steps differ (759/768 sampled steps across 6 instances). This
reflects `policy_fvi.py` being an entirely independent implementation (own
candidate grid, own repair, own simplified V_i), not a targeted measure of
the correction term alone -- see the ablation in #10, which isolates the
correction's marginal effect while holding the rest of the architecture fixed.

## 10. Per-scenario means, full-seed scores, cross-seed statistics

**Isolated ablation** (same architecture, `C_theta` on vs. off; 16 instances,
seed 42, BASELINE/CONGESTED/DENSE): mean quality effect of the correction
alone = **-0.356** (15/16 instances hurt, 0 improved, 2 exact ties). This
already isolates the correction from the (separately confirmed, much
smaller) simplified-V_i handicap.

**Full 100-instance staged comparison**, `sandbox/evaluate_staged_policy.py`,
3 seeds, `--workers` semantics: single-process throughout (no multiprocessing/
threading/joblib in this script or in any of the three policies):

| Seed | best_candidate | policy_v11 | policy_fvi |
|---|---:|---:|---:|
| 987654 | 0.7339332 | 0.7314848 | 0.6983429 |
| 42 | 0.7339852 | 0.7308550 | 0.6934418 |
| 2025 | 0.7752492 | 0.7735584 | 0.7506187 |
| **mean** | 0.7477225 | 0.7452994 | 0.7141345 |
| **worst** | 0.7339332 | 0.7308550 | 0.6934418 |
| **std** | 0.0189 | 0.0198 | 0.0247 |

Per-scenario (seed 987654): BASELINE 1.822->1.388 (-0.434), CONGESTED
5.341->4.687 (-0.654), MULTIDAY 9.924->9.903 (-0.021, V_i-only fallback, no
correction), DENSE 9.638->8.939 (-0.699), CAPSTONE 9.973->10.000 (+0.027,
V_i-only fallback). The degradation is concentrated exactly in the three
scenarios where the fitted correction is active; the two V_i-only fallback
scenarios are roughly at parity (one even marginally better), which is the
clean control confirming the correction itself, not the simplified base
value function, is the source of the regression.

## 11. Runtime / thread compliance

All comparisons run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` set before any numpy/scipy import,
single Python process, no multiprocessing/threading/joblib/subprocess in
`policy_fvi.py` or any comparison script. Per-instance runtime:
BASELINE/CONGESTED <1s, MULTIDAY ~3s, DENSE ~6-8s, CAPSTONE ~1-15s (V_i-only,
no correction) -- all far below the 30s hard limit.

## 12. Regressions and violations

Zero exceptions, zero action-bound violations, zero flow violations across
all runs (best_candidate.py, policy_v11.py, policy_fvi.py, and the
independent oracle's replayed schedules). `policy_fvi.py`'s *quality*
regression relative to `best_candidate.py`/`policy_v11.py` is real and
substantial (see #10) -- this is a genuine negative research finding, not a
bug or violation.

## 13. Files created this phase

`sandbox/oracle_red_team.py`, `sandbox/oracle_red_team_results.csv`,
`sandbox/oracle_red_team_report.md`, `sandbox/fvi_common.py`,
`sandbox/fvi_generate_data.py`, `sandbox/fvi_training_data.csv`,
`sandbox/fvi_train.py`, `sandbox/fvi_model.json`,
`sandbox/fvi_model_compact.json`, `sandbox/policy_fvi.py`,
`sandbox/evaluate_staged_policy.py`, `staged_comparison_results.csv`,
`sandbox/fvi_report.md` (this file).

`build_stochastic_tables.py` and `train_shadow_price_model.py` /
`train_expert_gate.py`-style files from earlier, superseded specs were not
built this phase -- they belonged to different (also-explored, also
negative or infeasible) architecture families and are not part of this
phase's scope.

## 14. Root-cause analysis: why did a R2=0.81 model produce a worse policy?

This is the single most important finding of Phase 3, and it is a classic,
well-documented pitfall in approximate dynamic programming, reproduced
honestly here rather than concealed: **a regression that fits realized
profit-to-go well in a passive, on-policy, state-conditional sense does not
automatically produce a valid signal for discriminating between
counterfactual candidate actions at decision time.** Concretely:
- The training targets were generated by rolling out `best_candidate.py`
  (which already has a sophisticated exact-mixture value function, LP
  allocator, and repair machinery) -- the residual C_theta learned is
  "how much does reality exceed/undershoot my *simplified* V_i, given the
  states *best_candidate* visits," not "how does changing *this* action
  change future value."
- At decision time, `policy_fvi.py`'s own candidates (V_i-only argmax, LP,
  congestion nudge) are frequently *not* the states `best_candidate` would
  have visited -- evaluating C_theta off-distribution appears to inject
  more noise/bias than signal, actively misleading the choice among
  near-equally-good candidates.
- This matches the required stop condition from this session's own
  discipline: "training performance improves but replayed production score
  does not" -- triggered exactly, and reported here rather than concealed
  or re-tuned into a different-looking but equally unvalidated number.

## 15. Whether the next stage/iteration is justified

**No.** Given (a) the independently-reconfirmed oracle ceiling leaves only
+0.0164 total headroom, (b) this session's cumulative history shows most
tested mechanisms in the "value-function refinement" family capture 0-10%
of a local gap at best, and (c) the one architecture selected as most
promising by principled elimination (Approximate DP / fitted value
correction) was implemented rigorously and produced a clean, decisive,
cross-seed-consistent *negative* result (not a wash, not seed-inconsistent
noise -- worse on 3/3 tested seeds, worse than even the frozen
`policy_v11.py` reference) -- continuing to iterate on this specific
mechanism would not be a good use of further effort without a fundamentally
different idea for closing the training/deployment gap identified in #14
(e.g., an actual multi-iteration policy-improvement loop with fresh rollout
data from `policy_fvi.py` itself, rather than only from `best_candidate.py`
-- a legitimate next idea, but a materially different, larger undertaking,
not a small tweak).

## 16. Final recommendation

**Submit `best_candidate.py` (SHA-256
`e24533c6c998c9e8a8137fde026e020ae998e76851d7cd145369dec6b8c95b04`), score
0.7339332 on seed 987654.** It remains the best validated file from this
entire session. `policy_v11.py` is the frozen reference (0.7314848).
`policy_fvi.py` (SHA-256
`72c75b10d8eb327ff219dfdffebf508379f5b9d7bdc4b40c85c65e19a3d5afae`) is
correctly implemented, honestly tested, and **rejected** -- it underperforms
both files on every tested seed. The 0.760/0.780/0.800/0.810 milestones are
confirmed mathematically infeasible on this exact seed/test distribution by
an oracle ceiling that survived an independent, from-scratch falsification
attempt to 7 decimal places. The real, remaining, causally-achievable
headroom is small (at most +0.0164, realistically much less given this
session's historical capture rates) and concentrated in CONGESTED and DENSE.
