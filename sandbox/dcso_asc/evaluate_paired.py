"""Reusable paired-evaluation utility: runs two policies on IDENTICAL
challenge instances (common random numbers via the shared next_seed
sequence, matching the pattern used throughout this session) and reports
paired statistics -- mean diff, standard error, median, positive count,
worst regression. Single process.
"""
import sys
from pathlib import Path
import random

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE
SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def run_episode(mod, ch):
    mod._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    for _ in range(ch.num_steps):
        a = mod.policy(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return state.total_profit


def paired_eval(mod_a, mod_b, master_seed, nonces):
    """Returns per-instance (quality_a, quality_b, diff) plus summary stats."""
    import contextlib
    import io
    rows = []
    for nonce in nonces:
        scen = SCENARIO_ORDER[nonce % 5]
        seed = seed_from_master_nonce(master_seed, nonce)
        ch = Challenge.generate_instance(seed, Track(s=scen))
        with contextlib.redirect_stdout(io.StringIO()):
            _, base = ch.compute_baseline()
        pa = run_episode(mod_a, ch)
        pb = run_episode(mod_b, ch)
        qa = max(-10.0, min((pa - base) / (base + 1e-6), 10.0))
        qb = max(-10.0, min((pb - base) / (base + 1e-6), 10.0))
        rows.append(dict(nonce=nonce, scenario=scen.name, baseline=base, q_a=qa, q_b=qb, diff=qb - qa))

    diffs = np.array([r["diff"] for r in rows])
    summary = dict(
        n=len(diffs), mean_diff=float(np.mean(diffs)), se=float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0,
        median_diff=float(np.median(diffs)), n_positive=int(np.sum(diffs > 1e-6)),
        n_negative=int(np.sum(diffs < -1e-6)), worst_regression=float(np.min(diffs)),
    )
    return rows, summary


if __name__ == "__main__":
    import argparse
    import importlib
    ap = argparse.ArgumentParser()
    ap.add_argument("module_a")
    ap.add_argument("module_b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nonces", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    mod_a = importlib.import_module(args.module_a)
    mod_b = importlib.import_module(args.module_b)
    rows, summary = paired_eval(mod_a, mod_b, args.seed, args.nonces)
    for r in rows:
        print(f"  nonce={r['nonce']:3d} {r['scenario']:9s} q_a={r['q_a']:+.4f} q_b={r['q_b']:+.4f} diff={r['diff']:+.4f}")
    print(f"\nsummary: {summary}")
