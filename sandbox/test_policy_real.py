"""
Local test harness for an Energy Arbitrage `policy(challenge, state)` function,
running against the REAL production reference package found at
D:\\bittensor\\dev\\apex\\ID7\\energy_arbitrage\\python (Challenge, Battery, Market,
Network, greedy/conservative baselines, exact scoring formula) instead of a
hand-reconstructed approximation.

That package's modules import themselves as `competition.energy_arbitrage.python.*`
(matching the real repo layout `shared/competition/src/competition/energy_arbitrage/...`).
This directory contains a `_pkgroot/competition/` shim -- an empty `__init__.py`
plus a directory junction named `energy_arbitrage` pointing at the real
`ID7/energy_arbitrage` folder -- so the imports resolve without copying any
source. `_pkgroot` is added to sys.path below.

Usage:
    python test_policy_real.py <module_name> [--instances N] [--seed S]

Example:
    python test_policy_real.py policy_138231
    python test_policy_real.py policy
    python test_policy_real.py policy_new

This uses the exact scenario cycle, seed derivation (seed_from_master_nonce,
matching Rust's `master_seed ^ (nonce as u64).wrapping_mul(0xdeadbeefcafebabe)`),
Challenge generation, real greedy/conservative baselines (challenge.compute_baseline),
and the exact quality/quality_int/raw_score/final_score formulas used in
challenge.py / test_challenge.py. The only thing NOT reproduced is the live
per-round hidden seed itself (unknowable), so exact instance-by-instance
numbers will still differ from a specific live round, but the scoring
mechanics, baselines, and price/network model are now the real ones.
"""
import argparse
import importlib
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"

sys.path.insert(0, str(_PKGROOT))   # so `competition.energy_arbitrage.python.*` resolves
sys.path.insert(0, str(_ID7_DIR))   # so policy_new.py / policy.py / policy_138231.py import by name

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate, Solution
from competition.energy_arbitrage.python.scenarios import Scenario

SCENARIO_ORDER = [
    Scenario.BASELINE,
    Scenario.CONGESTED,
    Scenario.MULTIDAY,
    Scenario.DENSE,
    Scenario.CAPSTONE,
]

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE
PER_STEP_TIMEOUT_S = 30.0
TOTAL_TIMEOUT_S = 1200.0


def seed_from_master_nonce(master_seed: int, nonce: int) -> bytes:
    """Matches test_challenge.py / the real Rust seed derivation exactly."""
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def run_episode_timed(challenge: Challenge, policy_fn):
    """Re-implements Challenge._simulate with per-step timing instrumentation.
    Any bound/flow violation raises ValueError (via take_step), exactly like
    production -- there is no soft "no-op the bad step" fallback in the real
    reference code, so a violation fails the whole instance."""
    view = challenge.to_policy_view()
    rng = random.Random()
    rng.seed(challenge._hidden_seed)
    state = challenge._initial_state(rng)
    schedule = []
    worst_step_time = 0.0
    t0 = time.perf_counter()

    for _ in range(challenge.num_steps):
        st = time.perf_counter()
        action = policy_fn(view, state)
        dt = time.perf_counter() - st
        worst_step_time = max(worst_step_time, dt)
        if dt > PER_STEP_TIMEOUT_S:
            raise TimeoutError(f"step {state.time_step} took {dt:.2f}s > {PER_STEP_TIMEOUT_S}s limit")

        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = challenge.take_step(state, action, NextRTPricesGenerate(next_seed))
        schedule.append(action)

    total_time = time.perf_counter() - t0
    if total_time > TOTAL_TIMEOUT_S:
        raise TimeoutError(f"episode took {total_time:.2f}s > {TOTAL_TIMEOUT_S}s limit")

    return Solution(schedule=schedule), worst_step_time, total_time


def _evaluate_one(module_name, nonce, master_seed):
    """Runs in a worker process: fresh module import (its own module-level
    _CACHE, matching the isolation a real sandboxed subprocess would have),
    one instance end to end. Returns a plain dict (picklable) rather than
    the Scenario enum/Solution objects."""
    mod = importlib.import_module(module_name)
    policy_fn = mod.policy

    scenario = SCENARIO_ORDER[nonce % len(SCENARIO_ORDER)]
    seed = seed_from_master_nonce(master_seed, nonce)
    challenge = Challenge.generate_instance(seed, Track(s=scenario))

    try:
        solution, step_max, total_time = run_episode_timed(challenge, policy_fn)
        my_profit = challenge.evaluate_total_profit(solution)
        _, baseline_profit = challenge.compute_baseline()
        quality_f = (my_profit - baseline_profit) / (baseline_profit + 1e-6)
        quality_int = round(max(-10.0, min(quality_f, 10.0)) * 1_000_000)
        return dict(nonce=nonce, scenario=scenario.name, profit=my_profit, baseline=baseline_profit,
                    quality=quality_f, quality_int=quality_int, step_max=step_max, total_time=total_time, error=None)
    except Exception as e:
        return dict(nonce=nonce, scenario=scenario.name, profit=None, baseline=None,
                    quality=None, quality_int=-10_000_000, step_max=0.0, total_time=0.0, error=f"{type(e).__name__}: {e}")


def evaluate(module_name, num_instances=20, master_seed=42, label="policy", workers=None):
    """Runs instances in parallel across worker processes -- each instance
    is a fully independent episode (fresh Challenge, fresh policy-module
    import), so there is no shared state to coordinate, only results to
    collect. This does not change what is being tested, only how fast the
    100 (or however many) independent episodes are evaluated."""
    if workers is None:
        # Conservative default: leave headroom so the machine stays usable
        # for other work while a test runs, rather than pegging every core
        # (this is a dev workstation, not a dedicated batch server).
        workers = max(1, min((os.cpu_count() or 4) // 2, num_instances, 4))

    per_scenario = {s.name: [] for s in SCENARIO_ORDER}
    results = {}
    print(f"  [{label}] running {num_instances} instances across {workers} worker processes...", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_evaluate_one, module_name, nonce, master_seed): nonce for nonce in range(num_instances)}
        for fut in as_completed(futures):
            r = fut.result()
            results[r["nonce"]] = r
            if r["error"] is not None:
                print(f"  [{label}] {r['nonce']:3d} {r['scenario']:10s} FAILED: {r['error']}", flush=True)
            else:
                print(
                    f"  [{label}] {r['nonce']:3d} {r['scenario']:10s} profit={r['profit']:14.1f} "
                    f"baseline={r['baseline']:12.1f} quality={r['quality']:+7.3f} "
                    f"step_max={r['step_max']*1000:6.1f}ms total_time={r['total_time']:7.1f}s",
                    flush=True,
                )

    errors = 0
    worst_step_time = 0.0
    worst_total_time = 0.0
    all_quality_int = []
    for nonce in range(num_instances):
        r = results[nonce]
        if r["error"] is not None:
            errors += 1
        else:
            worst_step_time = max(worst_step_time, r["step_max"])
            worst_total_time = max(worst_total_time, r["total_time"])
        per_scenario[r["scenario"]].append(r["quality_int"])
        all_quality_int.append(r["quality_int"])

    raw_score = float(np.mean(all_quality_int)) if all_quality_int else 0.0
    final_score = raw_score / 1e7

    print(f"\n=== {label}: summary over {num_instances} instances (master_seed={master_seed}) ===")
    for s in SCENARIO_ORDER:
        vals = per_scenario[s.name]
        if vals:
            print(f"  {s.name:10s} n={len(vals):3d}  mean_quality={np.mean(vals)/1e6:+7.3f}  "
                  f"min={np.min(vals)/1e6:+7.3f}  max={np.max(vals)/1e6:+7.3f}")
    print(f"  Raw Score:   {raw_score:.1f}")
    print(f"  Final Score: {final_score:.7f}")
    print(f"  errors={errors}  worst_step_time={worst_step_time*1000:.1f}ms (limit {PER_STEP_TIMEOUT_S*1000:.0f}ms)")
    print(f"  worst_total_episode_time={worst_total_time:.1f}s (limit {TOTAL_TIMEOUT_S:.0f}s)")
    print("  This run uses the REAL production Challenge/Battery/Market/Network/baselines "
          "from energy_arbitrage/python. The only unknown vs. a live round is the hidden "
          "per-round master seed, so absolute numbers may still differ from a specific "
          "live submission, but the scoring mechanics and price/network model are exact.")

    return dict(per_scenario=per_scenario, all_quality_int=all_quality_int,
                errors=errors, raw_score=raw_score, final_score=final_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="e.g. policy_new, policy, or policy_138231 (looked up in the ID7 folder)")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="master seed (mirrors --seed in test_challenge.py)")
    ap.add_argument("--workers", type=int, default=None, help="parallel worker processes (default: CPU count, capped at 16)")
    args = ap.parse_args()

    # Sanity-import once in the main process so an ImportError/SyntaxError in
    # the module surfaces immediately with a clean traceback, rather than as
    # an opaque failure inside every worker process.
    importlib.import_module(args.module)

    t0 = time.perf_counter()
    evaluate(args.module, num_instances=args.instances, master_seed=args.seed, label=args.module, workers=args.workers)
    print(f"\n(wall time: {time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
