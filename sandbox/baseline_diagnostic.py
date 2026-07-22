"""Root-cause diagnostic for BASELINE scenario underperformance (policy_v2).

Replays every BASELINE instance for a fixed dev seed with an instrumented copy
of policy_v2 (policy_v2_trace.py) that records the full per-battery-per-step
decision trace, plus runs the official greedy baseline on the identical
realized price path for comparison. Produces:
    baseline_trace.csv          -- full per-(instance,t,battery) trace
    baseline_diagnostic.md      -- aggregated findings
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
from competition.energy_arbitrage.python.greedy import policy as greedy_policy

import benchmark as bch

MASTER_SEED = 42
N_INSTANCES = 20  # BASELINE nonces 0,5,...,95


def run():
    mod = importlib.import_module("policy_v2_trace")
    all_rows = []
    ending_soc = []
    per_instance = []

    for k in range(N_INSTANCES):
        nonce = k * 5  # BASELINE is SCENARIO_ORDER[nonce % 5] == index 0
        seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
        ch = Challenge.generate_instance(seed, Track(s=Scenario.BASELINE))

        mod._TRACE.clear()
        ep = bch._run_episode(ch, mod.policy)
        my_profit = ch.evaluate_total_profit(ep["solution"])

        greedy_sched, greedy_state = ch._simulate(greedy_policy)
        greedy_profit = greedy_state.total_profit

        trace = list(mod._TRACE)
        for row in trace:
            row["nonce"] = nonce
            row["instance"] = k
            b = row["battery"]
            t = row["t"]
            row["greedy_action"] = float(greedy_sched[t][b])
        all_rows.extend(trace)

        cons_state = ch._simulate(
            importlib.import_module("competition.energy_arbitrage.python.conservative").policy
        )[1]
        baseline_profit = max(greedy_profit, cons_state.total_profit)
        quality = max(-10.0, min((my_profit - baseline_profit) / (baseline_profit + 1e-6), 10.0))

        ending_soc.append(ep["ending_mean_soc_pct"])
        per_instance.append(dict(
            nonce=nonce, my_profit=my_profit, greedy_profit=greedy_profit,
            baseline_profit=baseline_profit, quality=quality,
            ending_mean_soc_pct=ep["ending_mean_soc_pct"],
            ending_min_soc_pct=ep["ending_min_soc_pct"],
            ending_max_soc_pct=ep["ending_max_soc_pct"],
            repair_count=mod._BENCH_STATS["repair_count"],
            curtailed_mwh=mod._BENCH_STATS["curtailed_mwh"],
        ))
        print(f"  instance {k:2d} nonce={nonce:3d} quality={quality:+.3f} "
              f"ending_soc={ep['ending_mean_soc_pct']:.1f}% repairs={mod._BENCH_STATS['repair_count']}")

    return all_rows, per_instance


def analyze(all_rows, per_instance):
    import csv
    with open("baseline_trace.csv", "w", newline="") as f:
        fieldnames = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    n = len(all_rows)
    chosen = np.array([r["chosen_action"] for r in all_rows])
    unconstrained = np.array([r["unconstrained_action"] for r in all_rows])
    greedy = np.array([r["greedy_action"] for r in all_rows])
    repair_changed = np.array([r["repair_changed"] for r in all_rows])
    repair_engaged = np.array([r["repair_engaged"] for r in all_rows])
    value_chosen = np.array([r["value_chosen"] for r in all_rows])
    value_unconstrained = np.array([r["value_unconstrained"] for r in all_rows])
    marginal_value = np.array([r["marginal_value"] for r in all_rows])
    rt_price = np.array([r["rt_price"] for r in all_rows])
    da_price = np.array([r["da_price"] for r in all_rows])
    t_arr = np.array([r["t"] for r in all_rows])
    T_max = t_arr.max()
    soc_pct = np.array([r["soc_pct"] for r in all_rows])

    lines = []
    lines.append("# BASELINE root-cause diagnostic (policy_v2, dev seed=42, 20 instances)\n")

    # 1/4: congestion / repair impact
    frac_repair_engaged = repair_engaged.mean()
    frac_repair_changed = repair_changed.mean()
    removed_mag = np.abs(unconstrained - chosen)[repair_changed]
    total_desired_mag = np.abs(unconstrained).sum()
    total_removed_mag = np.abs(unconstrained - chosen).sum()
    lines.append("## 4. Network congestion / repair impact (BASELINE only)\n")
    lines.append(f"- repair codepath engaged on {frac_repair_engaged*100:.1f}% of battery-steps (episode-level infeasibility detected)")
    lines.append(f"- this battery's OWN action was actually changed on {frac_repair_changed*100:.1f}% of battery-steps")
    lines.append(f"- total |desired action| removed by repair: {total_removed_mag:.1f} MW-steps out of {total_desired_mag:.1f} MW-steps desired ({100*total_removed_mag/max(total_desired_mag,1e-9):.2f}%)")
    lines.append("")

    # 5: opportunity loss
    opp_loss = value_unconstrained - value_chosen
    lines.append("## 5. Opportunity loss (local best vs chosen feasible action)\n")
    lines.append(f"- mean opportunity loss per battery-step: {opp_loss.mean():.4f}")
    lines.append(f"- mean opportunity loss on repaired steps only: {opp_loss[repair_changed].mean() if repair_changed.any() else 0:.4f}")
    lines.append(f"- mean opportunity loss on NON-repaired steps: {opp_loss[~repair_changed].mean():.6f}  (should be ~0 if repair is the only source of loss)")
    lines.append(f"- total opportunity loss across all {n} battery-steps: {opp_loss.sum():.1f}")
    lines.append("")

    # 3: ending SOC / utilization
    lines.append("## 3. Battery utilization / ending SOC\n")
    ending_arr = np.array([r["ending_mean_soc_pct"] for r in per_instance])
    lines.append(f"- ending mean SOC%: mean={ending_arr.mean():.2f}  min={ending_arr.min():.2f}  max={ending_arr.max():.2f}  (SOC floor = 10%)")
    lines.append(f"- instances ending above 15% mean SOC: {(ending_arr > 15).sum()} / {len(ending_arr)}")
    lines.append("")

    # 6: DP marginal value behavior
    lines.append("## 6. DP marginal value of stored energy vs realized price\n")
    last8 = t_arr >= (T_max - 7)
    lines.append(f"- mean marginal_value(dVc/dSOC) over full horizon: {marginal_value.mean():.3f}  vs mean rt_price: {rt_price.mean():.3f}")
    lines.append(f"- mean marginal_value in FINAL 8 steps only: {marginal_value[last8].mean():.3f}  vs mean rt_price in final 8 steps: {rt_price[last8].mean():.3f}")
    holding_mask = (np.abs(chosen) < 1e-6) & (soc_pct > 15)
    lines.append(f"- battery-steps holding (action~0) while SOC>15%: {holding_mask.sum()} ({100*holding_mask.mean():.2f}% of all steps)")
    if holding_mask.sum() > 0:
        lines.append(f"  -> in these steps, mean marginal_value={marginal_value[holding_mask].mean():.3f} vs mean rt_price={rt_price[holding_mask].mean():.3f} (marginal_value > price would rationally justify holding)")
    lines.append("")

    # 7: candidate usage histogram
    lines.append("## 7. Candidate-action usage histogram (unconstrained/desired action)\n")
    lb_est = np.array([r["unconstrained_action"] for r in all_rows])
    # bucket by fraction of nonzero magnitude relative to the 3-level grid (1/3,2/3,1) or zero
    nz = unconstrained[np.abs(unconstrained) > 1e-6]
    zero_frac = (np.abs(unconstrained) <= 1e-6).mean()
    lines.append(f"- zero (hold): {zero_frac*100:.1f}%")
    lines.append(f"- nonzero actions: {(1-zero_frac)*100:.1f}% (magnitudes vary continuously with SOC-dependent bounds; grid is 7-point: full/2-3/1-3 charge, zero, 1-3/2-3/full discharge)")
    lines.append("")

    # 2: comparison vs greedy
    lines.append("## 2. Comparison against greedy baseline\n")
    sign_chosen = np.sign(np.round(chosen, 6))
    sign_greedy = np.sign(np.round(greedy, 6))
    agree = (sign_chosen == sign_greedy)
    lines.append(f"- action DIRECTION agrees with greedy on {agree.mean()*100:.1f}% of battery-steps")
    we_charge_greedy_not = ((sign_chosen < 0) & (sign_greedy >= 0)).sum()
    we_discharge_greedy_not = ((sign_chosen > 0) & (sign_greedy <= 0)).sum()
    greedy_charge_we_not = ((sign_greedy < 0) & (sign_chosen >= 0)).sum()
    greedy_discharge_we_not = ((sign_greedy > 0) & (sign_chosen <= 0)).sum()
    lines.append(f"- we charge / greedy doesn't: {we_charge_greedy_not} steps")
    lines.append(f"- we discharge / greedy doesn't: {we_discharge_greedy_not} steps")
    lines.append(f"- greedy charges / we don't: {greedy_charge_we_not} steps")
    lines.append(f"- greedy discharges / we don't: {greedy_discharge_we_not} steps")
    lines.append(f"- network-repair-removed-action steps: {repair_changed.sum()} ({repair_changed.mean()*100:.2f}%)")
    lines.append("")

    lines.append("## Per-instance summary\n")
    lines.append("| nonce | quality | my_profit | greedy_profit | baseline_profit | ending_mean_soc% | repairs |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in per_instance:
        lines.append(f"| {r['nonce']} | {r['quality']:+.3f} | {r['my_profit']:.1f} | {r['greedy_profit']:.1f} | {r['baseline_profit']:.1f} | {r['ending_mean_soc_pct']:.1f} | {r['repair_count']} |")

    with open("baseline_diagnostic.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    rows, per_inst = run()
    analyze(rows, per_inst)
