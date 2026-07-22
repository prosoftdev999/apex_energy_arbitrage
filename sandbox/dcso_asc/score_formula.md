# Exact score formula (verified from source, not memory)

Source: `energy_arbitrage/python/challenge.py`

```python
def compute_baseline(self):
    greedy_schedule, greedy_state = self._simulate(greedy_policy)
    greedy_total_profit = greedy_state.total_profit
    conservative_schedule, conservative_state = self._simulate(conservative_policy)
    conservative_total_profit = conservative_state.total_profit
    if greedy_total_profit > conservative_total_profit:
        return Solution(schedule=greedy_schedule), greedy_total_profit
    else:
        return Solution(schedule=conservative_schedule), conservative_total_profit

def evaluate_solution(self, solution):
    QUALITY_PRECISION = 1_000_000
    total_profit = self.evaluate_total_profit(solution)
    _, baseline_total_profit = self.compute_baseline()
    quality = (total_profit - baseline_total_profit) / (baseline_total_profit + 1e-6)
    quality = max(-10.0, min(quality, 10.0)) * QUALITY_PRECISION
    return round(quality)
```

## F(policy_profit, baseline_profit) exactly

```
b* = max(greedy_profit, conservative_profit)          [baseline, per instance]
q  = (W - b*) / (b* + 1e-6)                             [raw quality, W = policy final profit]
Q  = clip(q, -10, 10)                                    [clipped quality]
quality_int = round(Q * 1_000_000)                        [INTEGER rounding happens HERE, per instance]
```

Across the 100-instance harness (`test_policy_real.py::evaluate`):

```
raw_score   = mean(quality_int over 100 instances)        [mean of INTEGERS]
final_score = raw_score / 1e7
```

## Where F is linear / where it saturates / kinks

- **Linear region**: `q ∈ (-10, 10)` — `F` is exactly linear in `W` with slope `1/(b*+1e-6)`.
- **Saturation (flat)**: `q ≤ -10` or `q ≥ 10` — `dF/dW = 0` exactly. Any further profit change (up or down, within the clip) has **zero** marginal score effect.
- **Kinks**: exactly at `q = -10` and `q = +10` — `F` is continuous but non-differentiable there (a corner, not a jump).
- Rounding (`round(...)`) introduces additional **micro-kinks** every `1e-6/(1)` quality units, i.e., quantization noise at the `1e-6` scale — negligible relative to any real mechanism's effect size, verified: every mechanism this session moved quality by ≥1e-4, far above the rounding granularity, so rounding is not a live concern.

## Does expectation and clipping commute? NO — this is the central mathematical fact

```
E[F(W)] ≠ F(E[W])           whenever P(W crosses a kink) > 0
```

`F` is **concave** in `W` for `W < b*` region approach from below the -10 kink is irrelevant in practice (`W` is essentially always ≥ 0 economically); `F` is **linear then flat (globally concave, non-decreasing)** — i.e., `F(W)` is a concave, monotonically non-decreasing, piecewise-linear function of `W` (linear ramp, then flat at +10, mirrored at -10). By Jensen's inequality for a concave function: `E[F(W)] ≤ F(E[W])`. This means the RAW-DOLLAR-EV-MAXIMIZING policy (v_11/v15's actual objective) **overstates** the TRUE expected quality benefit of chasing extra EV precisely in the region approaching saturation — pushing W higher when already near/at the +10 kink adds strictly less expected quality than the same $ EV would add far from the kink. This was the concrete, empirically-tested finding of `policy_v14` two research phases ago: a real, but small (population-level) effect, because `P(W near a kink)` is empirically small (~2-4% of instances per the terminal-window audit).

## Does scenario averaging occur before or after clipping?

**After.** `quality_int` (the clipped, rounded, per-instance quantity) is computed per-instance first; the 100-instance mean is taken over these ALREADY-CLIPPED integers. This means an instance that is deeply negative (q=-14 raw) contributes only -10 to the average, not -14 — the harness is NOT simply averaging raw profit ratios, it is averaging **already-bounded** outcomes. This matters for risk formulation: variance-reduction on an ALREADY-BOUNDED-BELOW quantity has a smaller value proposition than variance-reduction on an unbounded quantity (the harness already protects against catastrophic single-instance drag).

## How baseline uncertainty affects marginal value

`b* = max(greedy_profit, conservative_profit)` is **fully deterministic given the realized RT price path** — it is not "uncertain" in a Bayesian sense once the episode starts (proven exactly, 0.0 error, via the shadow-baseline tracker built and validated two research phases ago). The only residual uncertainty in `b*` from a mid-episode viewpoint is the **not-yet-realized** portion of the RT price path for both `greedy` and `conservative`'s remaining scheduled actions (both fully deterministic action schedules, known in advance — only their $-value is uncertain, and only through the same, already-quantified, unavoidable RT randomness).
