"""Phase 3 step 1-2: generate fitted-value-iteration training data.

Iteration 0 base policy: frozen best_candidate.py (used strictly as a
black-box action-selector for rollouts, per the user's explicit spec --
"Iteration 0: frozen best_candidate policy" -- this is NOT the same as
policy_fvi.py reusing best_candidate's internal decision logic, which it
does not).

For each rollout episode: builds a fresh, independent per-battery separable
value function V_i(soc_i) (fvi_common.build_separable_value_function), runs
the full episode under best_candidate.policy, and for every step t records
  phi(s_t)                          -- causal, public feature vector
  y_t = profit_to_go(t) - sum_i V_i(soc_i,t)   -- fitted-value-iteration residual target

Scenarios: BASELINE, CONGESTED, DENSE only -- MULTIDAY/DENSE... (correction:
MULTIDAY and CAPSTONE are excluded; see below) -- per the independent oracle
audit, MULTIDAY (oracle gain +0.076) and CAPSTONE (+0.027) are within
0.03-0.08 of the absolute +10 ceiling already, essentially no headroom left
regardless of architecture, so no training effort is spent there. DENSE has
real headroom (+0.1997) and is included.

Training seeds are fixed, arbitrary, and disjoint from the dev seeds
(42, 2025, 314159, 271828, 161803), the untouched validation seed (987654),
and the reference seed (123).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import random
import best_candidate as base_policy
import fvi_common as fc

TRAINING_SEEDS = [7, 13, 19, 31, 37, 43, 53, 61, 67, 73]
DEV_SEEDS = {42, 2025, 314159, 271828, 161803}
VALIDATION_SEED = 987654
REFERENCE_SEED = 123
assert not (set(TRAINING_SEEDS) & DEV_SEEDS) and VALIDATION_SEED not in TRAINING_SEEDS \
    and REFERENCE_SEED not in TRAINING_SEEDS, "training seeds must be disjoint from reserved seeds"

TRAIN_SCENARIOS = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.DENSE]
NONCES_PER_SEED = 4
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def run_episode_and_extract(ch, scenario_b):
    ba = fc.battery_arrays_from_batteries(ch.batteries)
    node = ba["node"]
    da_all = np.asarray(ch.market.day_ahead_prices, dtype=float)
    exo_all = np.asarray(ch.exogenous_injections, dtype=float)
    ptdf = np.asarray(ch.network.ptdf, dtype=float)
    limits = np.asarray(ch.network.flow_limits, dtype=float)
    slack = ch.network.slack_bus
    T = ch.num_steps

    V_all = fc.build_separable_value_function(da_all, node, ba, scenario_b)

    base_policy._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)

    soc_hist = np.zeros((T, ba["B"]))
    profit_hist = np.zeros(T + 1)
    rt_mean_hist = np.zeros(T)
    da_mean_hist = da_all.mean(axis=1)

    for t in range(T):
        soc_hist[t] = state.socs
        rt_mean_hist[t] = float(np.mean(state.rt_prices))
        a = base_policy.policy(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
        profit_hist[t + 1] = state.total_profit

    rows = []
    for t in range(T):
        rt_prev = rt_mean_hist[t - 1] if t > 0 else None
        phi = fc.extract_features(t, T, soc_hist[t], ba, node, da_all, exo_all, ptdf, limits, slack,
                                   rt_now=rt_mean_hist[t], rt_prev=rt_prev, da_now=da_mean_hist[t])
        separable_baseline = fc.separable_value_at(V_all, t, soc_hist[t], ba)
        profit_to_go = profit_hist[T] - profit_hist[t]
        y = profit_to_go - separable_baseline
        rows.append((phi, y))
    return rows


def main():
    out_path = _SANDBOX_DIR / "fvi_training_data.csv"
    header = ["seed", "scenario", "nonce", "t"] + fc.FEATURE_NAMES + ["y"]
    n_rows = 0
    t_start = time.time()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seed in TRAINING_SEEDS:
            for scen in TRAIN_SCENARIOS:
                for nonce in range(NONCES_PER_SEED):
                    inst_seed = seed_from_master_nonce(seed, nonce)
                    ch = Challenge.generate_instance(inst_seed, Track(s=scen))
                    scenario_b = ch.num_batteries
                    rows = run_episode_and_extract(ch, scenario_b)
                    for t, (phi, y) in enumerate(rows):
                        w.writerow([seed, scen.name, nonce, t] + list(phi) + [y])
                        n_rows += 1
                print(f"seed={seed} {scen.name} done, n_rows={n_rows}, elapsed={time.time()-t_start:.0f}s", flush=True)
    print(f"\nsaved {out_path} ({n_rows} rows)")


if __name__ == "__main__":
    main()
