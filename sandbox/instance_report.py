"""Instance-level score-gap report: all 100 seed=123 instances, v6 vs v8."""
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
import policy_v6
import policy_v8

MASTER_SEED = 123
SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]


def run_policy(mod, ch):
    mod._CACHE.clear()
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    B = ch.num_batteries
    cap = np.array([b.capacity_mwh for b in ch.batteries])
    charged = 0.0
    discharged = 0.0
    for t in range(ch.num_steps):
        a = mod.policy(view, state)
        arr = np.array(a)
        charged += float(np.sum(np.maximum(-arr, 0.0)) * 0.25)
        discharged += float(np.sum(np.maximum(arr, 0.0)) * 0.25)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    ending_soc_pct = None
    stats = getattr(mod, "_BENCH_STATS", {})
    return dict(profit=state.total_profit, charged=charged, discharged=discharged,
                repair_count=stats.get("repair_count", 0))


def main():
    rows = []
    for nonce in range(100):
        scen = SCENARIO_ORDER[nonce % 5]
        seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
        ch = Challenge.generate_instance(seed, Track(s=scen))
        _, baseline_profit = ch.compute_baseline()

        r6 = run_policy(policy_v6, ch)
        r8 = run_policy(policy_v8, ch)

        def q(p):
            return max(-10.0, min((p - baseline_profit) / (baseline_profit + 1e-6), 10.0))

        q6, q8 = q(r6["profit"]), q(r8["profit"])
        row = dict(
            nonce=nonce, scenario=scen.name, baseline_profit=baseline_profit,
            v6_profit=r6["profit"], v8_profit=r8["profit"],
            v6_quality=q6, v8_quality=q8,
            saturated=(q8 >= 9.9999),
            v8_charged_mwh=r8["charged"], v8_discharged_mwh=r8["discharged"],
            v8_repair_count=r8["repair_count"],
        )
        rows.append(row)
        print(f"nonce={nonce:3d} {scen.name:9s} baseline={baseline_profit:10.1f} "
              f"v6_q={q6:+8.3f} v8_q={q8:+8.3f} diff={q8-q6:+.4f} sat={row['saturated']}")

    import csv
    with open("instance_report_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== deficit summary by scenario ===")
    for scen in SCENARIO_ORDER:
        sub = [r for r in rows if r["scenario"] == scen.name]
        q8s = np.array([r["v8_quality"] for r in sub])
        deficits = 10.0 - q8s
        n_sat = int(np.sum(q8s >= 9.9999))
        print(f"{scen.name:9s}: n={len(sub)} n_saturated={n_sat} mean_q8={q8s.mean():+.3f} "
              f"total_deficit={deficits.sum():.2f} mean_deficit_unsaturated="
              f"{deficits[q8s<9.9999].mean() if n_sat<len(sub) else 0:.3f}")

    print("\n=== top 20 largest-deficit instances (all scenarios) ===")
    rows_sorted = sorted(rows, key=lambda r: -(10.0 - r["v8_quality"]))
    for r in rows_sorted[:20]:
        deficit = 10.0 - r["v8_quality"]
        print(f"  nonce={r['nonce']:3d} {r['scenario']:9s} baseline={r['baseline_profit']:10.1f} "
              f"v8_q={r['v8_quality']:+8.3f} deficit={deficit:7.3f}")


if __name__ == "__main__":
    main()
