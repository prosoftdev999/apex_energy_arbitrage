"""Phase 3 + Phase 4 combined: root-cause classification of policy_v4's worst
instances, and MPC time-consistency check of its rolling-horizon plan (does
the target SOC / implied charge-discharge plan change wildly step to step?).
"""
import sys
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import policy_v4 as v4

MASTER_SEED = 123
BASELINE_NONCES = [45, 30, 25, 10, 5]     # 5 worst within our oracle sample (all negative quality)
CONGESTED_NONCES = [21, 36, 1, 46, 41]    # 5 worst within our oracle sample


def trace_instance(scen, nonce):
    seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scen))
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)

    v4._CACHE.clear()
    B = ch.num_batteries
    T = ch.num_steps
    target_hist = np.full((T, B), np.nan)
    action_hist = np.full((T, B), np.nan)
    desired_hist = np.full((T, B), np.nan)
    repaired_flags = np.zeros(T, dtype=bool)

    for t in range(T):
        key = id(view)
        entry = v4._CACHE.get(key)
        if entry is None or entry.get("sig") != v4._episode_signature(ch):
            ba = v4._battery_arrays(ch)
            V_all = v4._build_da_value_function(ch, ba)
            net = ch.network
            entry = {"ba": ba, "V_all": V_all, "sig": v4._episode_signature(ch),
                     "ptdf": np.asarray(net.ptdf, dtype=float), "limits": np.asarray(net.flow_limits, dtype=float),
                     "surprise_hist": [], "da_low_hist": []}
            v4._CACHE[key] = entry

        ba = entry["ba"]
        specialized = B <= v4._SPECIALIZED_MAX_B
        soc = np.array(state.socs, dtype=float)
        node = ba["node"]

        if specialized:
            H = v4._H_BASELINE if B <= 10 else v4._H_CONGESTED
            target = v4._target_soc_fracs(ch, ba, t, state, H, entry["surprise_hist"], entry)
            target = v4._apply_surprise_override(target, ch, state, node, entry["surprise_hist"])
            bounds = np.array(state.action_bounds, dtype=float)
            lb, ub = bounds[:, 0], bounds[:, 1]
            u_desired = np.clip(v4._desired_action_from_target(target, soc, ba), lb, ub)
            target_hist[t] = target
            desired_hist[t] = u_desired

        a = v4.policy(view, state)
        action_hist[t] = a
        repaired_flags[t] = v4._BENCH_STATS["repair_count"] > (repaired_flags[:t].sum() if t > 0 else 0)

        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))

    return dict(target=target_hist, action=action_hist, desired=desired_hist,
                final_profit=state.total_profit, T=T, B=B)


def analyze(scen_name, nonce, tr):
    T, B = tr["T"], tr["B"]
    target = tr["target"]
    action = tr["action"]
    desired = tr["desired"]

    # Phase 4: time-consistency of the rolling plan
    target_delta = np.abs(np.diff(target, axis=0))
    mean_step_change = np.nanmean(target_delta)
    frac_big_swing = np.nanmean(target_delta > 0.5)  # swings from near-0 to near-1 or similar

    # reversals: sign changes in the RETURNED action
    sign = np.sign(np.round(action, 6))
    reversals = 0
    for b in range(B):
        s = sign[:, b]
        s_nz = s[s != 0]
        if len(s_nz) > 1:
            reversals += int(np.sum(s_nz[1:] != s_nz[:-1]))

    # Phase 3: allocation removed value (desired vs returned differ)
    alloc_diff = np.abs(desired - action)
    frac_allocation_changed = np.mean(alloc_diff > 1e-6)
    mean_alloc_diff = np.mean(alloc_diff)

    print(f"{scen_name} nonce={nonce}: mean_target_step_change={mean_step_change:.3f} "
          f"frac_big_target_swings(>0.5)={frac_big_swing*100:.1f}% "
          f"action_reversals={reversals} allocation_changed={frac_allocation_changed*100:.1f}% "
          f"mean_alloc_diff={mean_alloc_diff:.3f} final_profit={tr['final_profit']:.1f}")

    return dict(scenario=scen_name, nonce=nonce, mean_target_step_change=mean_step_change,
                frac_big_target_swings=frac_big_swing, action_reversals=reversals,
                frac_allocation_changed=frac_allocation_changed, mean_alloc_diff=mean_alloc_diff)


def main():
    results = []
    for nonce in BASELINE_NONCES:
        tr = trace_instance(Scenario.BASELINE, nonce)
        results.append(analyze("BASELINE", nonce, tr))
    for nonce in CONGESTED_NONCES:
        tr = trace_instance(Scenario.CONGESTED, nonce)
        results.append(analyze("CONGESTED", nonce, tr))

    print("\n=== Phase 4 summary: is the rolling plan time-consistent? ===")
    for scen_name in ["BASELINE", "CONGESTED"]:
        rows = [r for r in results if r["scenario"] == scen_name]
        print(f"{scen_name}: mean target step-change={np.mean([r['mean_target_step_change'] for r in rows]):.3f} "
              f"(0=stable plan, 1=maximal flip-flop every step) | "
              f"mean big-swing frequency={np.mean([r['frac_big_target_swings'] for r in rows])*100:.1f}% of steps | "
              f"mean action reversals per instance={np.mean([r['action_reversals'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
