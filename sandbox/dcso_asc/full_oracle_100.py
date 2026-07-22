"""Phase 1: FULL 100-instance score-optimal oracle diagnostic, seed 987654,
all 5 scenario families -- not a subset. Reuses the already-validated
oracle2 (exact network-feasible whole-horizon LP, degradation-cost dropped
only, replay-verified against production with zero bound/flow violations)
and replay_through_production from oracle_certified.py.

Mathematical note on "score-optimal vs profit-optimal": for a SINGLE
instance, validator quality = clip((profit-baseline)/(baseline+eps), -10, 10)
is a monotonically non-decreasing function of profit (clip of an affine,
increasing function). There is no cross-instance coupling (each of the 100
instances is scored independently and averaged). Therefore the profit-
maximizing trajectory for a given instance is ALSO the quality-maximizing
trajectory for that instance -- maximizing raw profit cannot produce a worse
clipped-quality outcome than any other feasible trajectory, and the already-
validated oracle2 (a raw-profit-maximizing whole-horizon LP) is therefore
exactly the score-optimal oracle per instance. This is proven, not assumed:
clip(f(x)) is monotonic non-decreasing in x whenever f is affine-increasing
in x, for any clip range.

Single process throughout. Writes results incrementally so slower large-B
scenarios (MULTIDAY/DENSE/CAPSTONE, T=192) don't risk losing completed work.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import oracle_certified as oc
import policy_v11 as v11

SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
MASTER_SEED = 987654
OUT_CSV = Path(__file__).resolve().parent / "oracle_full_100_results.csv"

FIELDNAMES = ["nonce", "scenario", "baseline_profit", "frozen_profit", "frozen_quality",
              "oracle_profit", "oracle_replay_profit", "oracle_quality", "oracle_time_s",
              "bound_violation", "flow_violation", "replay_gap", "error"]


def _load_done():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add(int(r["nonce"]))
    return done


def run_frozen(ch):
    import random
    from competition.energy_arbitrage.python.challenge import NextRTPricesGenerate
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


def main():
    done = _load_done()
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for nonce in range(100):
            if nonce in done:
                continue
            scen = SCENARIO_ORDER[nonce % 5]
            seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            ba = oc._battery_arrays(ch)
            T, B = ch.num_steps, ch.num_batteries
            import io, contextlib
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
                               replay_gap=None, error=f"LP failed: {o2_status} {o2_msg}")
                else:
                    sched_per_step = oc._reshape_to_per_step(o2_sched, T, B)
                    replay_profit, bviol, fviol, err, max_clip = oc.replay_through_production(ch, sched_per_step, rt_traj)
                    oq = max(-10.0, min((replay_profit - baseline_profit) / (baseline_profit + 1e-6), 10.0))
                    row.update(oracle_profit=o2_profit, oracle_replay_profit=replay_profit, oracle_quality=oq,
                               oracle_time_s=oracle_time, bound_violation=bviol, flow_violation=fviol,
                               replay_gap=o2_profit - replay_profit if replay_profit is not None else None,
                               error=err)
            except Exception as e:
                row.update(oracle_profit=None, oracle_replay_profit=None, oracle_quality=None,
                           oracle_time_s=time.time() - t0, bound_violation=None, flow_violation=None,
                           replay_gap=None, error=f"{type(e).__name__}: {e}")

            writer.writerow(row)
            f.flush()
            print(f"nonce={nonce:3d} {scen.name:9s} baseline={baseline_profit:10.1f} "
                  f"frozen_q={row['frozen_quality']:+7.3f} oracle_q={row.get('oracle_quality')} "
                  f"time={row['oracle_time_s']:.1f}s err={row.get('error')}", flush=True)


if __name__ == "__main__":
    main()
