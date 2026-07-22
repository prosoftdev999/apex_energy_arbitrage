"""Required pre-benchmark gate (per task spec): compare policy_v2 and policy_v4
actions on identical Baseline/Congested episodes. This is NOT a score benchmark
-- it exists only to prove the new controller produces materially different
decisions before any 20-/100-instance run is spent on it.
"""
import sys
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

import policy_v2
import policy_v4
from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

SCENARIOS = [(Scenario.BASELINE, [1, 2]), (Scenario.CONGESTED, [1, 2])]


def run_pair(scen, seed_byte):
    seed = bytes([seed_byte] * 32)
    ch2 = Challenge.generate_instance(seed, Track(s=scen))
    ch4 = Challenge.generate_instance(seed, Track(s=scen))

    import random
    rng2 = random.Random(); rng2.seed(ch2._hidden_seed)
    rng4 = random.Random(); rng4.seed(ch4._hidden_seed)
    state2 = ch2._initial_state(rng2)
    state4 = ch4._initial_state(rng4)

    T = ch2.num_steps
    diffs = []
    bound_violations = 0
    line_violations = 0
    errors = 0
    repairs4 = 0

    for t in range(T):
        try:
            a2 = np.array(policy_v2.policy(ch2.to_policy_view(), state2))
            a4 = np.array(policy_v4.policy(ch4.to_policy_view(), state4))
        except Exception as e:
            errors += 1
            print(f"  ERROR at step {t}: {e}")
            break

        for lo, hi in state2.action_bounds:
            pass
        for i, (lo, hi) in enumerate(state2.action_bounds):
            if not (lo - 1e-6 <= a2[i] <= hi + 1e-6):
                bound_violations += 1
        for i, (lo, hi) in enumerate(state4.action_bounds):
            if not (lo - 1e-6 <= a4[i] <= hi + 1e-6):
                bound_violations += 1

        inj2 = ch2.compute_total_injections(state2, a2.tolist())
        flows2 = ch2.network.compute_flows(inj2)
        inj4 = ch4.compute_total_injections(state4, a4.tolist())
        flows4 = ch4.network.compute_flows(inj4)
        try:
            ch2.network.verify_flows(flows2)
        except Exception:
            line_violations += 1
        try:
            ch4.network.verify_flows(flows4)
        except Exception:
            line_violations += 1

        diffs.append(np.abs(a2 - a4))

        next_seed = bytes([rng2.randint(0, 255) for _ in range(32)])
        state2 = ch2.take_step(state2, a2.tolist(), NextRTPricesGenerate(next_seed))
        state4 = ch4.take_step(state4, a4.tolist(), NextRTPricesGenerate(next_seed))
        repairs4 = policy_v4._BENCH_STATS["repair_count"]

    diffs = np.array(diffs)
    pct_battery_steps_differ = (diffs > 1e-6).mean() * 100
    pct_time_steps_any_differ = (diffs.max(axis=1) > 1e-6).mean() * 100
    mean_abs_diff = diffs.mean()
    max_abs_diff = diffs.max()

    return dict(
        pct_battery_steps_differ=pct_battery_steps_differ,
        pct_time_steps_any_differ=pct_time_steps_any_differ,
        mean_abs_diff=mean_abs_diff, max_abs_diff=max_abs_diff,
        bound_violations=bound_violations, line_violations=line_violations,
        errors=errors, repairs_v4=repairs4,
    )


def main():
    all_pass = True
    for scen, seeds in SCENARIOS:
        pct_list = []
        for sb in seeds:
            r = run_pair(scen, sb)
            print(f"{scen.name} seed_byte={sb}: pct_battery_steps_differ={r['pct_battery_steps_differ']:.1f}% "
                  f"pct_time_steps_any_differ={r['pct_time_steps_any_differ']:.1f}% "
                  f"mean_abs_diff={r['mean_abs_diff']:.3f} max_abs_diff={r['max_abs_diff']:.3f} "
                  f"bound_viol={r['bound_violations']} line_viol={r['line_violations']} "
                  f"errors={r['errors']} v4_repairs={r['repairs_v4']}")
            pct_list.append(r['pct_time_steps_any_differ'])
            if r['bound_violations'] or r['line_violations'] or r['errors']:
                all_pass = False
        mean_pct = float(np.mean(pct_list))
        gate = mean_pct > 20.0
        print(f"{scen.name}: mean pct_time_steps_any_differ={mean_pct:.1f}% -> GATE {'PASS' if gate else 'FAIL'} (>20% required)")
        if not gate:
            all_pass = False
    print()
    print("OVERALL GATE:", "PASS - proceed to 20-instance dev test" if all_pass else "FAIL - do not run benchmark")


if __name__ == "__main__":
    main()
