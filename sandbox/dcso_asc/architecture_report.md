# Architecture Discovery Report

## Governing facts (proven experimentally this session, not assumed)

1. **`OA−O0 = 0` exactly**: the LP allocator already finds the exact optimum of the current reward+continuation-value objective, on every tested instance. No smarter *search/planning* method can improve on an exact convex-LP solve of the *same* problem.
2. **RT price shocks are proven i.i.d.** (source-traced: fresh RNG reseed every step, zero persistent state). No lookahead/forecasting method can extract more information than the exact expectation already computed, unless it uses illegal future data.
3. **`O1−O0 = +0.083`** (real, corrected from a prior indexing bug) is **hindsight-only** — requires the actual realized next-step price, not causally available.
4. Every distribution-side correction tried this session shows diminishing returns: +0.0004, +0.0002, +0.0022, +0.0024 (your own stated pattern).

These facts don't rule out *all* improvement — they rule out entire **families** of architecture, precisely and for stated reasons, not by assumption.

## Scoring table (all 20 requested architectures)

| # | Architecture | Est. gain | Why it can't beat the current planner (or why it might) |
|---|---|---:|---|
| 1 | MPC, adaptive terminal value | ~0 | Re-solves the same convex problem OA−O0=0 already solves exactly |
| 2 | Approximate DP | ~0 (as pure re-solve); **partial exception tested below** | Same objective, unless the *value function itself* is restructured (tested as Arch. 2) |
| 3 | Differential DP | ~0 | Local re-linearization of an already-exact convex solve adds no information |
| 4 | Value Function Approximation | ~0 | Current Vc is exact backward induction, not approximated within its own assumptions |
| 5 | Policy Iteration | ~0 | Converges to the same fixed point already reached in one shot |
| 6 | Rollout Policy Improvement | ~0 | Rollout under the *same* model reaches the *same* optimum |
| 7 | Adaptive Lookahead | ~0 | Requires either illegal future data or the already-exact expectation |
| 8 | MCTS w/ pruning | ~0 | Tree search over a convex, already-exactly-solved problem is strictly redundant |
| 9 | Cross-Entropy trajectory opt | ~0 | Stochastic search converging to the same convex optimum, slower and noisier |
| 10 | Beam Search trajectory opt | ~0 | Same as above, discretized |
| 11 | Robust MPC | ~0 | No adversarial/ambiguous component exists (model exactly known, MC-validated) |
| 12 | Risk-sensitive optimization | <0.001 | Tested in an earlier phase (score-aware terminal gate): small, seed-inconsistent |
| 13 | Distributionally Robust Opt | <0.001 | No genuine distributional ambiguity to hedge (exact model, not misspecified) |
| **14/16/17** | **Battery opportunity-cost / dual-price / shadow-price** | **tested, rejected** | **See Architecture 1 below** |
| 15 | Marginal energy valuation | (subsumed by 14) | Same underlying signal as the dual-price mechanism |
| 18 | Hybrid LP + learned correction | ~0 | Nothing to correct — model is exact, not misspecified |
| 19 | Residual policy | ~0 | Same reasoning as 18 |
| 20 | Teacher-student distillation | ~0 | Nothing for a student to learn that the exact LP doesn't already capture |

## Architecture 1 (implemented, tested, REJECTED): Dual-price-aware battery-specific continuation value

**What's different from the current planner**: the existing continuation value `Vc` is computed **per-battery independently** (no cross-battery network coupling in the value function itself, only in the final LP allocation). This architecture extracts the LP's actual dual values (shadow prices) on binding PTDF flow constraints, accumulates them causally (EMA), and discounts/boosts each battery's `Vc` in proportion to its own PTDF exposure to historically-expensive lines — a decision the current planner cannot make, since it has no channel for "this specific battery matters more for future network relief."

**Result** (paired, seeds 42, BASELINE+CONGESTED, n=8 instances per config):

| β | Sign | BASELINE Δ | CONGESTED Δ |
|---:|---|---:|---:|
| 0.01 | discount | −0.0023 | −0.0043 |
| 0.003 | discount | −0.0005 | +0.0008 |
| 0.01 | bonus | +0.0008 | +0.0008 |
| 0.05 | bonus | −0.0282 | −0.0141 |
| 0.1 | bonus | −0.1494 | −0.0520 |

Best observed: +0.0008 overall (β=0.01, bonus direction) — **below the +0.003 threshold. Rejected.** Both signs tested; magnitude only ever hurts beyond a negligible scale.

## Architecture 2 (implemented, tested, REJECTED): Designated-reliever joint diversity

**What's different**: instead of a dense, per-battery-proportional adjustment (Architecture 1), this identifies the single most-utilized line *right now* (from public exogenous injections) and gives a bonus to only the ONE battery best positioned to relieve it (PTDF-sign-matched sensitivity × useful remaining headroom) — a sparse, discrete "designated hitter" selection the current planner cannot make (it treats all batteries symmetrically within the LP).

**Result**:

| β | BASELINE Δ | CONGESTED Δ |
|---:|---:|---:|
| 0.02 | −0.0014 | −0.0010 |
| 0.1 | −0.0210 | −0.0035 |
| 0.3 | −0.1202 | −0.0387 |

**No positive regime found at any scale. Rejected.**

## Why both architectures failed (consistent, not coincidental)

Both are variations on "make the continuation value network-topology-aware." The consistent finding — near-zero-to-negative at any real scale — suggests the existing uniform congestion discount (`_CONGESTION_DISCOUNT_ALPHA`, already validated and active) already captures the *aggregate* effect of network scarcity on continuation value, and further *battery-specific* differentiation adds noise (from the causally-estimated exposure/utilization signals, which are inherently noisier and more myopic than the aggregate repair-rate signal) rather than genuine new information. This is consistent with, not contradicted by, the governing facts above: OA−O0=0 already tells us the allocator extracts full value from whatever Vc it's given — the remaining question was always whether Vc itself could be improved, and these two attempts to do so via network-topology awareness did not succeed.

## Outcome

`best_candidate.py` is **unchanged** — neither tested architecture beat it. This is a genuine, evidence-based negative result for the "network-topology-aware value function" family specifically, obtained by actually implementing and testing two substantively different members of it (not by assumption). The most promising untested direction remains the score-objective family (12/13), which showed a real but small effect in an earlier session phase and was not re-tested here per your explicit instruction not to repeat prior work.
