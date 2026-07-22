"""Fast focused BASELINE-only scorer for isolated single-lever tests.
Runs only the 20 BASELINE dev-seed instances (nonce=0,5,...,95, seed=42)
against the real reference package and reports mean/median clipped quality.
"""
import importlib
import sys
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch

MASTER_SEED = 42
N = 20


def score(module_name):
    mod = importlib.import_module(module_name)
    quals = []
    max_step = 0.0
    for k in range(N):
        nonce = k * 5
        seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
        ch = Challenge.generate_instance(seed, Track(s=Scenario.BASELINE))
        ep = bch._run_episode(ch, mod.policy)
        max_step = max(max_step, ep["worst_step_time"])
        my_profit = ch.evaluate_total_profit(ep["solution"])
        _, baseline_profit = ch.compute_baseline()
        q = max(-10.0, min((my_profit - baseline_profit) / (baseline_profit + 1e-6), 10.0))
        quals.append(q)
    quals = np.array(quals)
    print(f"{module_name}: BASELINE mean={quals.mean():+.4f} median={np.median(quals):+.4f} "
          f"min={quals.min():+.4f} max={quals.max():+.4f} worst_step={max_step*1000:.1f}ms")
    return quals


if __name__ == "__main__":
    score(sys.argv[1])
