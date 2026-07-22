"""Paired, per-instance comparison of two policy modules across one or more
seeds, 100 instances each, single process (--workers 1 always -- this
script itself never spawns worker processes or threads).

Usage:
    python paired_evaluation.py <module_a> <module_b> --seeds S1,S2,...

Reports per-seed Final Score for both modules, the paired per-instance
quality delta (mean, worst regression, count improved/regressed), and a
95% confidence interval on the mean paired delta across all tested seeds
(seeds are the unit of replication -- this matches how the real competition
draws a fresh random seed each round).
"""
import argparse
import contextlib
import io
import random
import sys
from pathlib import Path

import numpy as np

_ID7_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ID7_DIR / "sandbox" / "_pkgroot"))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def run(mod, ch):
    mod._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    for t in range(ch.num_steps):
        a = mod.policy(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return state.total_profit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module_a")
    ap.add_argument("module_b")
    ap.add_argument("--seeds", required=True, help="comma-separated master seeds")
    ap.add_argument("--instances", type=int, default=100)
    args = ap.parse_args()

    mod_a = __import__(args.module_a)
    mod_b = __import__(args.module_b)
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed_mean = []
    all_deltas_by_seed = {}
    for master_seed in seeds:
        deltas = []
        for nonce in range(args.instances):
            scen = SCENARIO_ORDER[nonce % 5]
            seed = seed_from_master_nonce(master_seed, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            with contextlib.redirect_stdout(io.StringIO()):
                _, base = ch.compute_baseline()
            p_a = run(mod_a, ch)
            p_b = run(mod_b, ch)

            def q(p):
                return max(-10.0, min((p - base) / (base + 1e-6), 10.0))

            deltas.append(q(p_a) - q(p_b))
        deltas = np.array(deltas)
        all_deltas_by_seed[master_seed] = deltas
        print(f"seed={master_seed}: mean_delta_quality={deltas.mean():+.5f} "
              f"n_improved={int(np.sum(deltas > 1e-6))} n_regressed={int(np.sum(deltas < -1e-6))} "
              f"worst_regression={deltas.min():+.4f}", flush=True)
        per_seed_mean.append(deltas.mean())

    per_seed_mean = np.array(per_seed_mean)
    print(f"\n=== across {len(seeds)} seeds ===")
    print(f"mean paired quality delta: {per_seed_mean.mean():+.5f} "
          f"(Final Score delta = mean/10 = {per_seed_mean.mean()/10:+.7f})")
    if len(seeds) > 1:
        se = per_seed_mean.std(ddof=1) / np.sqrt(len(seeds))
        ci_lo, ci_hi = per_seed_mean.mean() - 1.96 * se, per_seed_mean.mean() + 1.96 * se
        print(f"95% CI on mean paired delta (seed as replication unit): [{ci_lo:+.5f}, {ci_hi:+.5f}] "
              f"(Final Score scale: [{ci_lo/10:+.7f}, {ci_hi/10:+.7f}])")


if __name__ == "__main__":
    main()
