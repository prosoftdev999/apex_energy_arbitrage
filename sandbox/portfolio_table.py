"""Frozen per-instance policy portfolio table: v6, v8, v9, v10, v11 evaluated
on IDENTICAL instances (same challenge, same RT-price trajectory per instance
-- each policy gets its own freshly-seeded rng object so the price stream is
reproduced exactly regardless of which policy is stepping through it) across
14 master seeds (123, 987654, 42, 2025 dev seeds already used earlier this
session, plus 10 additional never-before-used seeds). This measures whether
policy diversity exists at all before any selector is built -- per explicit
instruction, no selector or policy_v13 work starts until this table and its
portfolio-upper-bound analysis (Section 2) are complete.

Writes incrementally to portfolio_results.csv (one row per (seed,nonce)) so a
long-running batch survives interruption; safe to re-run (skips completed
rows already in the CSV).
"""
import contextlib
import csv
import importlib
import io
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE
POLICY_NAMES = ["policy_v6", "policy_v8", "policy_v9", "policy_v10", "policy_v11"]

SEEDS = [123, 987654, 42, 2025,
         7, 13, 99, 777, 3141, 271828, 555555, 8080, 424242, 191919]
NUM_INSTANCES = 100
OUT_CSV = _SANDBOX_DIR / "portfolio_results.csv"

FIELDNAMES = (["seed", "nonce", "scenario", "baseline"]
              + [f"profit_{n}" for n in POLICY_NAMES]
              + [f"q_{n}" for n in POLICY_NAMES]
              + ["best", "gain_over_v11", "n_tied"])


def seed_from_master_nonce(master_seed: int, nonce: int) -> bytes:
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def _run_episode(mod, ch):
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


def _evaluate_instance(master_seed, nonce):
    scenario = SCENARIO_ORDER[nonce % len(SCENARIO_ORDER)]
    seed = seed_from_master_nonce(master_seed, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scenario))
    with contextlib.redirect_stdout(io.StringIO()):
        _, base_profit = ch.compute_baseline()

    profits = {}
    try:
        for name in POLICY_NAMES:
            mod = importlib.import_module(name)
            profits[name] = _run_episode(mod, ch)
    except Exception as e:
        return dict(seed=master_seed, nonce=nonce, scenario=scenario.name, baseline=base_profit,
                    error=f"{type(e).__name__}: {e}")

    qualities = {n: max(-10.0, min((profits[n] - base_profit) / (base_profit + 1e-6), 10.0)) for n in POLICY_NAMES}
    best_name = max(qualities, key=qualities.get)
    best_q = qualities[best_name]
    n_tied = sum(1 for q in qualities.values() if abs(q - best_q) <= 1e-5)
    row = dict(seed=master_seed, nonce=nonce, scenario=scenario.name, baseline=base_profit)
    row.update({f"profit_{n}": profits[n] for n in POLICY_NAMES})
    row.update({f"q_{n}": qualities[n] for n in POLICY_NAMES})
    row["best"] = best_name
    row["gain_over_v11"] = best_q - qualities["policy_v11"]
    row["n_tied"] = n_tied
    row["error"] = None
    return row


def _load_done():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((int(r["seed"]), int(r["nonce"])))
    return done


def main():
    done = _load_done()
    write_header = not OUT_CSV.exists()
    tasks = [(s, n) for s in SEEDS for n in range(NUM_INSTANCES) if (s, n) not in done]
    print(f"total tasks: {len(SEEDS)*NUM_INSTANCES}, already done: {len(done)}, remaining: {len(tasks)}", flush=True)

    workers = min(10, os.cpu_count() or 4)
    t0 = time.perf_counter()
    n_completed = 0
    n_errors = 0
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES + ["error"])
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_evaluate_instance, s, n): (s, n) for s, n in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                n_completed += 1
                if r.get("error"):
                    n_errors += 1
                    print(f"ERROR seed={r['seed']} nonce={r['nonce']} {r['scenario']}: {r['error']}", flush=True)
                writer.writerow(r)
                f.flush()
                if n_completed % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = n_completed / elapsed
                    eta = (len(tasks) - n_completed) / max(rate, 1e-9)
                    print(f"  {n_completed}/{len(tasks)} done  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s, errors={n_errors})", flush=True)

    print(f"\nDone. {n_completed} instances evaluated, {n_errors} errors, wall time {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
