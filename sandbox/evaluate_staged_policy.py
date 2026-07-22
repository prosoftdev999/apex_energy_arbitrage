"""Phase 5/6: staged comparison of best_candidate.py, policy_v11.py, and
policy_fvi.py across seed 987654 and independent seeds, 100 instances,
workers=1 (single process, this script itself does not spawn workers)."""
import contextlib
import csv
import io
import random
import sys
import time
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
sys.path.insert(0, str(_SANDBOX_DIR / "_pkgroot"))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import best_candidate as bc
import policy_v11 as v11
import policy_fvi as pf

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
    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [987654]
    n_instances = 100
    all_results = []
    for master_seed in seeds:
        t_start = time.time()
        rows = []
        for nonce in range(n_instances):
            scen = SCENARIO_ORDER[nonce % 5]
            seed = seed_from_master_nonce(master_seed, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            with contextlib.redirect_stdout(io.StringIO()):
                _, base = ch.compute_baseline()
            p_bc = run(bc, ch)
            p_v11 = run(v11, ch)
            p_fvi = run(pf, ch)

            def q(p):
                return max(-10.0, min((p - base) / (base + 1e-6), 10.0))

            row = dict(master_seed=master_seed, nonce=nonce, scenario=scen.name, baseline=base,
                       q_best_candidate=q(p_bc), q_policy_v11=q(p_v11), q_policy_fvi=q(p_fvi))
            rows.append(row)
        elapsed = time.time() - t_start
        qs_bc = np.array([r["q_best_candidate"] for r in rows])
        qs_v11 = np.array([r["q_policy_v11"] for r in rows])
        qs_fvi = np.array([r["q_policy_fvi"] for r in rows])
        print(f"seed={master_seed} (elapsed {elapsed:.0f}s): "
              f"best_candidate={qs_bc.mean()/10:.7f} policy_v11={qs_v11.mean()/10:.7f} "
              f"policy_fvi={qs_fvi.mean()/10:.7f}", flush=True)
        for scen in SCENARIO_ORDER:
            sub = [r for r in rows if r["scenario"] == scen.name]
            print(f"  {scen.name:10s} best_candidate={np.mean([r['q_best_candidate'] for r in sub]):8.4f} "
                  f"policy_v11={np.mean([r['q_policy_v11'] for r in sub]):8.4f} "
                  f"policy_fvi={np.mean([r['q_policy_fvi'] for r in sub]):8.4f}", flush=True)
        all_results.extend(rows)
        with open(_ID7_DIR / "staged_comparison_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            for r in all_results:
                w.writerow(r)
    print("saved staged_comparison_results.csv")


if __name__ == "__main__":
    main()
