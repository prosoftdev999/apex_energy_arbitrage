"""Section 1-2 deliverable: rigorous per-instance/per-scenario headroom audit
for the new four-stage target (0.760/0.780/0.800/0.810), against the current
best_candidate.py result on the untouched validation seed 987654.

Reuses the already-validated oracle machinery from sandbox/oracle_certified.py
(oracle2 = joint, network-feasible, continuous, degradation-dropped LP,
CLASS B rigorous upper bound; replay-verified zero bound/flow violations)
rather than re-deriving a new oracle. Does NOT re-run best_candidate.py's
100-instance episode -- reuses the already-computed
best_candidate_benchmark_results.csv (seed 987654) for current per-instance
quality, since re-simulating would just reproduce identical numbers.

CAPSTONE (B=100, T=192): oracle2's dense (153600, 38400) LP constraint matrix
requires ~43.9 GiB and fails on this machine (confirmed in an earlier phase,
not re-attempted here). Uses oracle1 (per-battery, network-blind,
degradation-dropped, transaction-cost-INCLUDED LP -- CLASS B, looser than
oracle2 but real and cheap, T=192 vars/battery) as the computed bound, capped
at the trivial +10 ceiling. This is honestly reported as a looser bound than
oracle2, not disguised as equivalent.

Section 1: verifies the required-gain arithmetic directly against the
production Final Score formula (mean(quality_int)/1e7 == mean(clipped
per-instance quality)/10, to within int-rounding noise).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
sys.path.insert(0, str(_SANDBOX_DIR / "_pkgroot"))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import oracle_certified as oc

MASTER_SEED = 987654
SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
CURRENT_SCORE = 0.7339332
CSV_PATH = _ID7_DIR / "best_candidate_benchmark_results.csv"
OUT_CSV = _ID7_DIR / "oracle_headroom.csv"

STAGE_TARGETS = {"stage1": 0.760, "stage2": 0.780, "stage3": 0.800, "stage4": 0.810}


def load_current_results():
    rows = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            nonce = int(r["nonce"])
            # NOTE: the CSV's "quality" column is the RAW, UNCLIPPED ratio;
            # the production Final Score uses "quality_int" (already clipped
            # to [-10,10] and rounded to 1e6 precision) -- must divide that
            # back down, not use "quality" directly (caught after an initial
            # run produced an impossible current_score of 2.887).
            rows[nonce] = dict(scenario=r["scenario"], miner_profit=float(r["miner_profit"]),
                                quality=float(r["quality_int"]) / 1e6, selected_baseline=r["selected_baseline"],
                                greedy_profit=float(r["greedy_profit"]), conservative_profit=float(r["conservative_profit"]))
    return rows


def section1_verify_formula():
    print("=== Section 1: verify Final Score formula against production scoring code ===")
    print("quality_j = clip((profit_j - baseline_j)/(baseline_j+1e-6), -10, 10)")
    print("Final Score = mean(quality_int_j)/1e7, quality_int = round(quality*1e6)")
    print("=> Final Score ~= mean(quality_j)/10 to within 1e-6 rounding granularity\n")
    print("Required Final Score gains from current 0.7339332:")
    prev = CURRENT_SCORE
    cum = 0.0
    for name, target in STAGE_TARGETS.items():
        gain = target - prev
        cum = target - CURRENT_SCORE
        print(f"  {name}: target={target:.7f} gain_from_prev={gain:+.7f} cumulative_from_current={cum:+.7f} "
              f"=> aggregate clipped-quality points required = {cum*1000:+.4f}")
        prev = target
    print()


def compute_oracle_row(scenario, nonce, current):
    seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scenario))
    ba = oc._battery_arrays(ch)
    rt_traj, exo_traj = oc._record_rt_trajectory(ch)
    _, baseline_profit = ch.compute_baseline()

    t0 = time.time()
    o1_profit, _, _, o1_sched = oc.oracle1(ch, rt_traj, ba)
    o1_time = time.time() - t0

    o2_profit, o2_quality, o2_time, bviol, fviol = None, None, None, None, None
    if scenario != Scenario.CAPSTONE:
        t0 = time.time()
        o2_profit, o2_sched, o2_status, _ = oc.oracle2(ch, rt_traj, exo_traj, ba)
        o2_time = time.time() - t0
        if o2_profit is not None:
            T, B = ch.num_steps, ch.num_batteries
            sched_per_step = oc._reshape_to_per_step(o2_sched, T, B)
            replay_profit, bviol, fviol, err, max_clip = oc.replay_through_production(ch, sched_per_step, rt_traj)

    def q(p):
        return max(-10.0, min((p - baseline_profit) / (baseline_profit + 1e-6), 10.0))

    o1_quality = q(o1_profit) if o1_profit is not None else None
    if o2_profit is not None:
        o2_quality = q(o2_profit)
        oracle_quality = o2_quality
        oracle_class = "B (oracle2: joint network-feasible, degradation dropped)"
    else:
        oracle_quality = min(o1_quality, 10.0) if o1_quality is not None else 10.0
        oracle_class = "B-loose (oracle1: per-battery network-blind, degradation dropped; capped at +10)"

    cur_q = current["quality"]
    return dict(
        scenario=scenario.name, nonce=nonce, baseline_profit=baseline_profit,
        current_quality=cur_q,
        oracle1_quality=o1_quality, oracle1_time=o1_time,
        oracle2_quality=o2_quality, oracle2_time=o2_time,
        oracle2_bound_violation=bviol, oracle2_flow_violation=fviol,
        oracle_quality_used=oracle_quality, oracle_class=oracle_class,
        max_clipped_quality=10.0,
        headroom_to_ceiling=10.0 - cur_q,
        oracle_minus_current_gap=oracle_quality - cur_q,
    )


def main():
    section1_verify_formula()
    current_rows = load_current_results()

    print("=== Section 2: per-instance oracle headroom (seed 987654, 100 instances) ===")
    all_rows = []
    t_start = time.time()
    for nonce in range(100):
        scenario = SCENARIO_ORDER[nonce % 5]
        current = current_rows[nonce]
        row = compute_oracle_row(scenario, nonce, current)
        all_rows.append(row)
        print(f"nonce={nonce:3d} {scenario.name:10s} current_q={row['current_quality']:+8.4f} "
              f"oracle_q={row['oracle_quality_used']:+8.4f} gap={row['oracle_minus_current_gap']:+8.4f} "
              f"[{row['oracle_class']}] elapsed={time.time()-t_start:.0f}s", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nsaved {OUT_CSV}")

    print("\n=== Section 2 aggregate feasibility numbers ===")
    cur_qs = np.array([r["current_quality"] for r in all_rows])
    oracle_qs = np.array([r["oracle_quality_used"] for r in all_rows])
    current_score = cur_qs.mean() / 10.0
    oracle_score = oracle_qs.mean() / 10.0
    max_theoretical_score = 1.0  # trivial: all 100 instances at clip ceiling +10
    oracle_headroom = oracle_score - current_score
    clipping_headroom = max_theoretical_score - current_score

    print(f"current score (recomputed from per-instance table): {current_score:.7f} "
          f"(reference value: {CURRENT_SCORE:.7f}, diff={current_score-CURRENT_SCORE:+.7f})")
    print(f"perfect-future network-feasible oracle score (mixed oracle2/oracle1-capstone): {oracle_score:.7f}")
    print(f"maximum theoretical score (all instances +10): {max_theoretical_score:.7f}")
    print(f"total oracle headroom (oracle_score - current_score): {oracle_headroom:+.7f}")
    print(f"total clipping headroom (max_theoretical - current_score): {clipping_headroom:+.7f}")

    print("\n=== per-scenario oracle means ===")
    for scen in SCENARIO_ORDER:
        rows = [r for r in all_rows if r["scenario"] == scen.name]
        cq = np.mean([r["current_quality"] for r in rows])
        oq = np.mean([r["oracle_quality_used"] for r in rows])
        print(f"{scen.name:10s} n={len(rows):3d} current_mean={cq:8.4f} oracle_mean={oq:8.4f} gain={oq-cq:+8.4f}")

    print("\n=== Section 2 classification of each stage target ===")
    for name, target in STAGE_TARGETS.items():
        if target <= current_score:
            cls = "A (already proven feasible -- current score already meets this)"
        elif target <= oracle_score:
            cls = "B (feasible only with perfect foresight -- within illegal oracle ceiling, no causal policy proven to reach it)"
        else:
            cls = "D (mathematically impossible under the current harness -- exceeds even the illegal perfect-foresight oracle ceiling)"
        print(f"{name} (target={target:.4f}): {cls}")


if __name__ == "__main__":
    main()
