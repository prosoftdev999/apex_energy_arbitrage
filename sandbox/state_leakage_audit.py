"""Section 1: state-leakage audit. Tests whether policy_v11's module-global
_CACHE (keyed by id(challenge)) can leak state between instances -- the
classic Python risk is that id() values get REUSED after garbage collection,
so a stale cache entry from a GC'd Challenge could in principle be picked up
by a new Challenge object that happens to land at the same memory address.
_episode_signature() is designed to guard against this (num_batteries,
num_nodes, num_lines, num_steps, battery-0 capacity, battery-0 node, and the
first DA price rounded to 6 decimals must all match), but this must be
proven empirically, not assumed.

Tests:
  1. Run instance X fresh (first thing in the process).
  2. Run instance X after 99 other instances have populated/evicted _CACHE.
  3. Repeat with the 99 preceding instances in reversed order.
  4. Repeat once more after an explicit _CACHE.clear().
  5. Compare actions and final profit bit-for-bit across all four runs.

Single process throughout, no multiprocessing.
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

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import policy_v11 as v11

SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def run_instance(seed_master, nonce):
    scen = SCENARIO_ORDER[nonce % 5]
    seed = seed_from_master_nonce(seed_master, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scen))
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    actions = []
    for _ in range(ch.num_steps):
        a = v11.policy(view, state)
        actions.append(list(a))
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return actions, state.total_profit


def main():
    TARGET_NONCE = 137  # arbitrary target instance to check for leakage
    MASTER = 987654

    # Run 1: target instance completely fresh (nothing else run first).
    actions_fresh, profit_fresh = run_instance(MASTER, TARGET_NONCE)

    # Run 2: target instance after 99 preceding instances (forward order).
    for n in range(0, 99):
        run_instance(MASTER, n)
    actions_after, profit_after = run_instance(MASTER, TARGET_NONCE)

    # Run 3: 99 preceding instances in REVERSED order.
    v11._CACHE.clear()
    for n in reversed(range(0, 99)):
        run_instance(MASTER, n)
    actions_rev, profit_rev = run_instance(MASTER, TARGET_NONCE)

    # Run 4: explicit cache clear immediately before the target instance.
    v11._CACHE.clear()
    actions_cleared, profit_cleared = run_instance(MASTER, TARGET_NONCE)

    def compare(name, a, p):
        max_diff = max((np.max(np.abs(np.array(x) - np.array(y))) for x, y in zip(actions_fresh, a)), default=0.0)
        print(f"{name}: max|action diff vs fresh|={max_diff:.3e}  "
              f"profit diff={abs(p - profit_fresh):.6e}  profit={p:.4f}")

    print(f"fresh: profit={profit_fresh:.4f}")
    compare("after 99 forward", actions_after, profit_after)
    compare("after 99 reversed (cache cleared first)", actions_rev, profit_rev)
    compare("cache cleared immediately before", actions_cleared, profit_cleared)


if __name__ == "__main__":
    main()
