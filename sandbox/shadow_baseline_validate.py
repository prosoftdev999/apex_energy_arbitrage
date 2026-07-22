"""Section 1/2 validation: prove that greedy/conservative baselines can be
exactly shadow-tracked online using only causally-available information.

Key source facts established by reading challenge.py directly:
  - _simulate() reseeds `rng.seed(self._hidden_seed)` at the START of every
    call (used for greedy, conservative, AND the submitted policy's own run
    via grid_optimize/evaluate_total_profit) -- so the sequence of next_seed
    draws, and hence the realized RT-price/congestion trajectory, is
    IDENTICAL across all three "parallel worlds" for the same Challenge
    instance. Battery actions never consume rng draws.
  - greedy.policy() and conservative.policy() use ONLY: state.time_step,
    market.day_ahead_prices (public, full schedule), state.action_bounds
    (deterministic given the shadow battery's OWN soc), and
    challenge.exogenous_injections (public, full schedule) -- ZERO use of
    state.rt_prices in the ACTION-CHOICE logic itself (only compute_profit,
    for profit accounting, uses the already-observed rt_prices).
  - compute_total_injections/apply_action_to_soc/compute_action_bounds are
    exact production Battery/Challenge methods, called directly here (no
    reimplementation risk).

This script runs a real policy_v11 episode and, IN PARALLEL, maintains
shadow States for greedy and conservative using the exact production
policy functions, then compares the shadow-computed final profit against
the REAL challenge.compute_baseline() result for the same instance.
"""
import sys
from pathlib import Path
import random

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate, State
from competition.energy_arbitrage.python.scenarios import Scenario
from competition.energy_arbitrage.python.greedy import policy as greedy_policy
from competition.energy_arbitrage.python.conservative import policy as conservative_policy

import policy_v11 as v11

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def run_with_shadow_baselines(ch):
    v11._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)

    shadow_g = State(time_step=0, socs=list(state.socs), rt_prices=list(state.rt_prices),
                      exogenous_injections=list(state.exogenous_injections),
                      action_bounds=list(state.action_bounds), total_profit=0.0)
    shadow_c = State(time_step=0, socs=list(state.socs), rt_prices=list(state.rt_prices),
                      exogenous_injections=list(state.exogenous_injections),
                      action_bounds=list(state.action_bounds), total_profit=0.0)

    for t in range(ch.num_steps):
        a = v11.policy(view, state)

        ag = greedy_policy(view, shadow_g)
        ac = conservative_policy(view, shadow_c)
        profit_g = ch.compute_profit(shadow_g, ag)
        profit_c = ch.compute_profit(shadow_c, ac)
        shadow_g.total_profit += profit_g
        shadow_c.total_profit += profit_c

        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))

        # advance shadow SOC/bounds using the SAME (already-realized, shared)
        # rt_prices/exogenous_injections as the real trajectory -- confirmed
        # identical because price/congestion generation never depends on the
        # policy's chosen action, only on the shared next_seed sequence.
        next_socs_g = [ch.batteries[i].apply_action_to_soc(ag[i], shadow_g.socs[i]) for i in range(len(ag))]
        next_socs_c = [ch.batteries[i].apply_action_to_soc(ac[i], shadow_c.socs[i]) for i in range(len(ac))]
        next_bounds_g = [ch.batteries[i].compute_action_bounds(next_socs_g[i]) for i in range(len(next_socs_g))]
        next_bounds_c = [ch.batteries[i].compute_action_bounds(next_socs_c[i]) for i in range(len(next_socs_c))]

        if t + 1 < ch.num_steps:
            shadow_g = State(time_step=t + 1, socs=next_socs_g, rt_prices=list(state.rt_prices),
                              exogenous_injections=list(state.exogenous_injections),
                              action_bounds=next_bounds_g, total_profit=shadow_g.total_profit)
            shadow_c = State(time_step=t + 1, socs=next_socs_c, rt_prices=list(state.rt_prices),
                              exogenous_injections=list(state.exogenous_injections),
                              action_bounds=next_bounds_c, total_profit=shadow_c.total_profit)
        else:
            shadow_g.time_step = t + 1
            shadow_c.time_step = t + 1

    return state.total_profit, shadow_g.total_profit, shadow_c.total_profit


def main():
    scenarios = [("BASELINE", Scenario.BASELINE, [0, 5]), ("CONGESTED", Scenario.CONGESTED, [1, 6]),
                 ("MULTIDAY", Scenario.MULTIDAY, [2]), ("DENSE", Scenario.DENSE, [3]), ("CAPSTONE", Scenario.CAPSTONE, [4])]
    max_abs_err = 0.0
    for scen_name, scen, nonces in scenarios:
        for nonce in nonces:
            seed = seed_from_master_nonce(987654, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            _, real_baseline = ch.compute_baseline()
            v11_profit, shadow_g_profit, shadow_c_profit = run_with_shadow_baselines(ch)
            shadow_baseline = max(shadow_g_profit, shadow_c_profit)
            err = abs(shadow_baseline - real_baseline)
            max_abs_err = max(max_abs_err, err)
            rel_err = err / max(abs(real_baseline), 1.0)
            print(f"{scen_name:10s} nonce={nonce:3d} real_baseline={real_baseline:12.4f} "
                  f"shadow_baseline={shadow_baseline:12.4f} abs_err={err:.6f} rel_err={rel_err:.2e} "
                  f"(shadow_g={shadow_g_profit:.2f} shadow_c={shadow_c_profit:.2f}) v11_profit={v11_profit:.2f}")
    print(f"\nmax abs error across all tested instances: {max_abs_err:.6f}")


if __name__ == "__main__":
    main()
