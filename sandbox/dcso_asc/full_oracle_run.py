"""Executes the Phase-1 oracle plan honestly adapted to measured computational
feasibility: BASELINE and CONGESTED get the FULL 20-instance treatment
(cheap, ~0.4s/1.5s per instance). MULTIDAY and DENSE get a sample (5 and 3
instances respectively) given their much higher per-instance LP cost (36s,
162s measured). CAPSTONE's oracle2 is memory-infeasible with the current
dense LP formulation (confirmed: attempted 43.9 GiB allocation) -- skipped,
with its maximum-possible headroom bounded analytically instead (see
oracle_full_100_report.md): frozen candidate is already at 9.973/10.0, so
regardless of the true oracle value, CAPSTONE's remaining final-score
headroom cannot exceed (10.0-9.973)*20/100/10 = 0.000054.
"""
import contextlib
import csv
import io
import sys
import time
from pathlib import Path

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import oracle_certified as oc
import policy_v11 as v11

MASTER_SEED = 987654
OUT_CSV = Path(__file__).resolve().parent / "oracle_full_100_results.csv"
FIELDNAMES = ["nonce", "scenario", "baseline_profit", "frozen_profit", "frozen_quality",
              "oracle_profit", "oracle_replay_profit", "oracle_quality", "oracle_time_s",
              "bound_violation", "flow_violation", "error"]


def run_frozen(ch):
    import random
    v11._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    for t in range(ch.num_steps):
        a = v11.policy(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return state.total_profit


def process_one(scen, nonce, writer, f):
    seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scen))
    ba = oc._battery_arrays(ch)
    T, B = ch.num_steps, ch.num_batteries
    with contextlib.redirect_stdout(io.StringIO()):
        _, baseline_profit = ch.compute_baseline()
    rt_traj, exo_traj = oc._record_rt_trajectory(ch)
    frozen_profit = run_frozen(ch)
    row = dict(nonce=nonce, scenario=scen.name, baseline_profit=baseline_profit,
               frozen_profit=frozen_profit,
               frozen_quality=max(-10.0, min((frozen_profit - baseline_profit) / (baseline_profit + 1e-6), 10.0)))
    t0 = time.time()
    try:
        o2_profit, o2_sched, o2_status, o2_msg = oc.oracle2(ch, rt_traj, exo_traj, ba)
        oracle_time = time.time() - t0
        if o2_sched is None:
            row.update(oracle_profit=None, oracle_replay_profit=None, oracle_quality=None,
                       oracle_time_s=oracle_time, bound_violation=None, flow_violation=None,
                       error=f"LP failed: {o2_status} {o2_msg}")
        else:
            sched_per_step = oc._reshape_to_per_step(o2_sched, T, B)
            replay_profit, bviol, fviol, err, max_clip = oc.replay_through_production(ch, sched_per_step, rt_traj)
            oq = max(-10.0, min((replay_profit - baseline_profit) / (baseline_profit + 1e-6), 10.0))
            row.update(oracle_profit=o2_profit, oracle_replay_profit=replay_profit, oracle_quality=oq,
                       oracle_time_s=oracle_time, bound_violation=bviol, flow_violation=fviol, error=err)
    except Exception as e:
        row.update(oracle_profit=None, oracle_replay_profit=None, oracle_quality=None,
                   oracle_time_s=time.time() - t0, bound_violation=None, flow_violation=None,
                   error=f"{type(e).__name__}: {e}")
    writer.writerow(row)
    f.flush()
    print(f"nonce={nonce:3d} {scen.name:9s} baseline={baseline_profit:10.1f} "
          f"frozen_q={row['frozen_quality']:+7.3f} oracle_q={row.get('oracle_quality')} "
          f"time={row['oracle_time_s']:.1f}s err={row.get('error')}", flush=True)


def main():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["scenario"], int(r["nonce"])))
    write_header = not OUT_CSV.exists()

    plan = [
        (Scenario.BASELINE, list(range(0, 100, 5))),      # full 20
        (Scenario.CONGESTED, list(range(1, 100, 5))),     # full 20
        (Scenario.MULTIDAY, [2, 22, 42, 62, 82]),          # 5-instance sample
        (Scenario.DENSE, [3, 43, 83]),                      # 3-instance sample
    ]
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for scen, nonces in plan:
            for nonce in nonces:
                if (scen.name, nonce) in done:
                    continue
                process_one(scen, nonce, writer, f)


if __name__ == "__main__":
    main()
