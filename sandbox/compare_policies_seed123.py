"""Phase 1: same-seed, same-instance comparison of policy_v2/v3/v4.
Not a scored 100-instance benchmark -- a controlled diagnostic on a small,
fixed set of seed=123 instances (4 BASELINE, 4 CONGESTED, 2 MULTIDAY).
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

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch

MASTER_SEED = 123
SELECTIONS = [
    (Scenario.BASELINE, [0, 5, 10, 15]),
    (Scenario.CONGESTED, [1, 6, 11, 16]),
    (Scenario.MULTIDAY, [2, 7]),
]
POLICIES = ["policy_v2", "policy_v3", "policy_v4"]


def run_one(mod, scen, nonce):
    seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scen))
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)

    B = ch.num_batteries
    soc0 = np.array([b.soc_initial_mwh for b in ch.batteries])
    soc = soc0.copy()
    cap = np.array([b.capacity_mwh for b in ch.batteries])
    charged = 0.0
    discharged = 0.0
    reversals = 0
    prev_sign = np.zeros(B)
    import time
    t0 = time.perf_counter()

    for t in range(ch.num_steps):
        a = np.array(mod.policy(view, state))
        sign = np.sign(np.round(a, 6))
        reversals += int(np.sum((sign != 0) & (prev_sign != 0) & (sign != prev_sign)))
        prev_sign = np.where(sign != 0, sign, prev_sign)
        c = np.maximum(-a, 0.0); d = np.maximum(a, 0.0)
        charged += float(np.sum(c) * 0.25)
        discharged += float(np.sum(d) * 0.25)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a.tolist(), NextRTPricesGenerate(next_seed))
        for i, batt in enumerate(ch.batteries):
            soc[i] = batt.apply_action_to_soc(a[i], soc[i])

    runtime = time.perf_counter() - t0
    my_profit = state.total_profit
    stats = getattr(mod, "_BENCH_STATS", {})
    return dict(
        profit=my_profit, ending_soc_pct=float(np.mean(soc / cap * 100.0)),
        charged_mwh=charged, discharged_mwh=discharged, reversals=reversals,
        repair_count=stats.get("repair_count", 0), runtime=runtime,
    )


def main():
    mods = {name: importlib.import_module(name) for name in POLICIES}
    rows = []

    for scen, nonces in SELECTIONS:
        for nonce in nonces:
            seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
            ch_ref = Challenge.generate_instance(seed, Track(s=scen))
            _, baseline_profit = ch_ref.compute_baseline()

            results = {}
            for name, mod in mods.items():
                for m in mods.values():
                    m._CACHE.clear()
                r = run_one(mod, scen, nonce)
                results[name] = r

            profits = {name: results[name]["profit"] for name in POLICIES}
            qualities = {name: max(-10.0, min((p - baseline_profit) / (baseline_profit + 1e-6), 10.0))
                         for name, p in profits.items()}
            best = max(qualities, key=qualities.get)

            row = dict(nonce=nonce, scenario=scen.name, baseline_profit=baseline_profit)
            for name in POLICIES:
                row[f"{name}_profit"] = profits[name]
                row[f"{name}_quality"] = qualities[name]
                row[f"{name}_ending_soc_pct"] = results[name]["ending_soc_pct"]
                row[f"{name}_charged_mwh"] = results[name]["charged_mwh"]
                row[f"{name}_discharged_mwh"] = results[name]["discharged_mwh"]
                row[f"{name}_reversals"] = results[name]["reversals"]
                row[f"{name}_repair_count"] = results[name]["repair_count"]
                row[f"{name}_runtime"] = results[name]["runtime"]
            row["best_policy"] = best
            rows.append(row)

            print(f"nonce={nonce:3d} {scen.name:9s} baseline={baseline_profit:10.1f} | "
                  f"v2={profits['policy_v2']:10.1f} (q={qualities['policy_v2']:+.3f}) | "
                  f"v3={profits['policy_v3']:10.1f} (q={qualities['policy_v3']:+.3f}) | "
                  f"v4={profits['policy_v4']:10.1f} (q={qualities['policy_v4']:+.3f}) | best={best}")

    import csv
    with open("compare_policies_seed123_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n| Instance | Scenario | Baseline profit | v2 profit | v3 profit | v4 profit | Best policy |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['nonce']} | {r['scenario']} | {r['baseline_profit']:.1f} | "
              f"{r['policy_v2_profit']:.1f} | {r['policy_v3_profit']:.1f} | {r['policy_v4_profit']:.1f} | {r['best_policy']} |")


if __name__ == "__main__":
    main()
