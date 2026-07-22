"""Section 15/16/18 deliverable: paired comparison of policy_v15.py against
best_candidate.py (policy_v15's direct parent -- inherits its quadrature
Bellman machinery unchanged) or policy_v11.py, run single-process.

Usage (from repo root d:\\bittensor\\dev\\apex\\ID7):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        python sandbox/dcso_asc/compare_v11_v15.py \\
        --ref best_candidate --seeds 42,2025,314159 --scenarios BASELINE,CONGESTED,DENSE \\
        --nonces 16 --workers 1

Flags:
  --ref          reference module to diff against: best_candidate or policy_v11 (default best_candidate)
  --seeds        comma-separated dev seeds (NEVER include 987654 here -- that is the
                 validation seed, run exactly once at the end per Section 16)
  --scenarios    comma-separated scenario names (BASELINE/CONGESTED/MULTIDAY/DENSE/CAPSTONE)
  --nonces       number of nonces per (seed, scenario) pair, starting at 0
  --out          json output path (default: alongside this script)

Prints a per-scenario and per-seed mean_diff table plus the overall mean, and
applies the Section 18 acceptance-gate checks directly (beats ref on every
dev seed, avg gain >=+0.003, no dev seed <+0.001, no scenario regresses >0.03).
"""
import argparse
import collections
import contextlib
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

_ID7_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ID7_DIR / "sandbox" / "_pkgroot"))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import policy_v15 as v15


def _seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * 0xDEADBEEFCAFEBABE) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def _run(mod, ch):
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
    ap.add_argument("--ref", default="best_candidate", choices=["best_candidate", "policy_v11"])
    ap.add_argument("--seeds", default="42,2025,314159")
    ap.add_argument("--scenarios", default="BASELINE,CONGESTED,DENSE")
    ap.add_argument("--nonces", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.workers != 1:
        raise SystemExit("this harness is single-process only; --workers must be 1")

    ref_mod = __import__(args.ref)
    dev_seeds = [int(s) for s in args.seeds.split(",")]
    if 987654 in dev_seeds:
        raise SystemExit("seed 987654 is the validation seed -- never tune/compare against it here")
    scen_map = {s.name: s for s in Scenario}
    scenarios = [scen_map[s] for s in args.scenarios.split(",")]
    out_path = args.out or str(Path(__file__).resolve().parent / "compare_v11_v15_results.json")

    results = []
    t_start = time.time()
    for dev_seed in dev_seeds:
        for scen in scenarios:
            for nonce in range(args.nonces):
                seed = _seed_from_master_nonce(dev_seed, nonce)
                ch = Challenge.generate_instance(seed, Track(s=scen))
                with contextlib.redirect_stdout(io.StringIO()):
                    _, base = ch.compute_baseline()
                p_ref = _run(ref_mod, ch)
                p_new = _run(v15, ch)
                q_ref = max(-10.0, min((p_ref - base) / (base + 1e-6), 10.0))
                q_new = max(-10.0, min((p_new - base) / (base + 1e-6), 10.0))
                results.append(dict(dev_seed=dev_seed, scenario=scen.name, nonce=nonce,
                                     base=base, q_ref=q_ref, q_new=q_new, diff=q_new - q_ref))
            print(f"seed={dev_seed} {scen.name} done, elapsed={time.time() - t_start:.0f}s", flush=True)
            with open(out_path, "w") as f:
                json.dump(results, f)

    print(f"\n=== per scenario (ref={args.ref}) ===")
    by_scen = collections.defaultdict(list)
    for r in results:
        by_scen[r["scenario"]].append(r["diff"])
    worst_regression = 0.0
    for scen, diffs in by_scen.items():
        mn, mx = min(diffs), max(diffs)
        worst_regression = min(worst_regression, mn)
        print(f"{scen:10s} n={len(diffs):3d} mean_diff={np.mean(diffs):+.5f} "
              f"min={mn:+.5f} max={mx:+.5f} n_regressed(<-0.001)={sum(1 for d in diffs if d < -0.001)}")

    print("\n=== per dev seed ===")
    by_seed = collections.defaultdict(list)
    for r in results:
        by_seed[r["dev_seed"]].append(r["diff"])
    seed_means = {}
    for s, diffs in by_seed.items():
        seed_means[s] = float(np.mean(diffs))
        print(f"seed={s}: mean_diff={seed_means[s]:+.5f}")

    all_diffs = [r["diff"] for r in results]
    overall = float(np.mean(all_diffs))
    print(f"\nOVERALL n={len(all_diffs)} mean_diff={overall:+.5f}")

    print("\n=== Section 18 acceptance gate ===")
    beats_every_seed = all(v > 0 for v in seed_means.values())
    avg_gain_ok = overall >= 0.003
    no_seed_below_floor = all(v >= 0.001 for v in seed_means.values())
    no_regression_ok = worst_regression >= -0.03
    print(f"beats {args.ref} on every dev seed: {beats_every_seed} ({seed_means})")
    print(f"avg dev gain >= +0.003: {avg_gain_ok} (actual {overall:+.5f})")
    print(f"no dev seed gain < +0.001: {no_seed_below_floor}")
    print(f"no scenario regression > 0.03: {no_regression_ok} (worst {worst_regression:+.5f})")
    gate_pass = beats_every_seed and avg_gain_ok and no_seed_below_floor and no_regression_ok
    print(f"\nACCEPTANCE GATE: {'PASS' if gate_pass else 'FAIL'}")
    print(f"results saved to {out_path}")
    return gate_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
