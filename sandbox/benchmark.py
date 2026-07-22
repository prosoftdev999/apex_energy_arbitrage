"""Diagnostic benchmark harness for energy_arbitrage policies.

Runs the REAL production reference package (same path-shim mechanism as
test_policy_real.py) across a fixed seed/nonce grid and records per-instance
diagnostics beyond the plain score: SOC trajectory, charged/discharged MWh,
and (if the policy module exposes an optional `_BENCH_STATS` dict that it
mutates during policy() calls) network-repair engagement counts. Policies
that don't define `_BENCH_STATS` simply get 0/blank for those columns --
this is an opt-in introspection hook, not a required interface.

Usage:
    python benchmark.py <module> [--instances N] [--seed S] [--workers W] [--out-prefix PATH]
"""
import argparse
import csv
import importlib
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

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate, Solution
from competition.energy_arbitrage.python.scenarios import Scenario

SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE
PER_STEP_TIMEOUT_S = 30.0
TOTAL_TIMEOUT_S = 1200.0


def seed_from_master_nonce(master_seed: int, nonce: int) -> bytes:
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def _run_episode(challenge: Challenge, policy_fn):
    view = challenge.to_policy_view()
    rng = random.Random()
    rng.seed(challenge._hidden_seed)
    state = challenge._initial_state(rng)
    schedule = []
    worst_step_time = 0.0
    t0 = time.perf_counter()

    B = challenge.num_batteries
    soc = np.array([b.soc_initial_mwh for b in challenge.batteries], dtype=float)
    soc_cap = np.array([b.capacity_mwh for b in challenge.batteries], dtype=float)
    charged_mwh = 0.0
    discharged_mwh = 0.0
    dt = 0.25
    soc_frac_trace = []

    for _ in range(challenge.num_steps):
        st = time.perf_counter()
        action = policy_fn(view, state)
        dtc = time.perf_counter() - st
        worst_step_time = max(worst_step_time, dtc)
        if dtc > PER_STEP_TIMEOUT_S:
            raise TimeoutError(f"step {state.time_step} took {dtc:.2f}s > {PER_STEP_TIMEOUT_S}s limit")

        a = np.asarray(action, dtype=float)
        c = np.maximum(-a, 0.0)
        d = np.maximum(a, 0.0)
        charged_mwh += float(np.sum(c) * dt)
        discharged_mwh += float(np.sum(d) * dt)

        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = challenge.take_step(state, action, NextRTPricesGenerate(next_seed))
        schedule.append(action)

        for i, batt in enumerate(challenge.batteries):
            soc[i] = batt.apply_action_to_soc(a[i], soc[i])
        soc_frac_trace.append(soc / np.maximum(soc_cap, 1e-9))

    total_time = time.perf_counter() - t0
    if total_time > TOTAL_TIMEOUT_S:
        raise TimeoutError(f"episode took {total_time:.2f}s > {TOTAL_TIMEOUT_S}s limit")

    soc_frac_trace = np.array(soc_frac_trace) if soc_frac_trace else np.zeros((0, B))
    final_frac = soc_frac_trace[-1] if len(soc_frac_trace) else soc / np.maximum(soc_cap, 1e-9)
    return dict(
        solution=Solution(schedule=schedule),
        worst_step_time=worst_step_time,
        total_time=total_time,
        charged_mwh=charged_mwh,
        discharged_mwh=discharged_mwh,
        ending_mean_soc_pct=float(np.mean(final_frac)) * 100.0,
        ending_min_soc_pct=float(np.min(final_frac)) * 100.0,
        ending_max_soc_pct=float(np.max(final_frac)) * 100.0,
    )


def _evaluate_one(module_name, nonce, master_seed):
    mod = importlib.import_module(module_name)
    policy_fn = mod.policy

    scenario = SCENARIO_ORDER[nonce % len(SCENARIO_ORDER)]
    seed = seed_from_master_nonce(master_seed, nonce)
    challenge = Challenge.generate_instance(seed, Track(s=scenario))

    row = dict(
        master_seed=master_seed, nonce=nonce, scenario=scenario.name,
        num_nodes=challenge.network.num_nodes, num_batteries=challenge.num_batteries,
        horizon=challenge.num_steps,
        miner_profit=None, greedy_profit=None, conservative_profit=None, selected_baseline=None,
        quality=None, quality_int=None, runtime=None, max_step_runtime=None,
        ending_mean_soc_pct=None, ending_min_soc_pct=None, ending_max_soc_pct=None,
        charged_mwh=None, discharged_mwh=None,
        repair_count=0, curtailed_mwh=0.0, zero_fallback_count=0,
        failure_reason="",
    )

    try:
        ep = _run_episode(challenge, policy_fn)
        my_profit = challenge.evaluate_total_profit(ep["solution"])
        greedy_sched, greedy_state = challenge._simulate(
            importlib.import_module("competition.energy_arbitrage.python.greedy").policy
        )
        cons_sched, cons_state = challenge._simulate(
            importlib.import_module("competition.energy_arbitrage.python.conservative").policy
        )
        greedy_profit = greedy_state.total_profit
        cons_profit = cons_state.total_profit
        baseline_profit = max(greedy_profit, cons_profit)
        selected = "greedy" if greedy_profit >= cons_profit else "conservative"

        quality_f = (my_profit - baseline_profit) / (baseline_profit + 1e-6)
        quality_int = round(max(-10.0, min(quality_f, 10.0)) * 1_000_000)

        stats = getattr(mod, "_BENCH_STATS", {})
        row.update(
            miner_profit=my_profit, greedy_profit=greedy_profit, conservative_profit=cons_profit,
            selected_baseline=selected, quality=quality_f, quality_int=quality_int,
            runtime=ep["total_time"], max_step_runtime=ep["worst_step_time"],
            ending_mean_soc_pct=ep["ending_mean_soc_pct"], ending_min_soc_pct=ep["ending_min_soc_pct"],
            ending_max_soc_pct=ep["ending_max_soc_pct"],
            charged_mwh=ep["charged_mwh"], discharged_mwh=ep["discharged_mwh"],
            repair_count=stats.get("repair_count", 0), curtailed_mwh=stats.get("curtailed_mwh", 0.0),
            zero_fallback_count=stats.get("zero_fallback_count", 0),
        )
    except Exception as e:
        row["failure_reason"] = f"{type(e).__name__}: {e}"
        row["quality_int"] = -10_000_000

    return row


def run_benchmark(module_name, num_instances=20, master_seed=42, workers=None, label=None):
    if workers is None:
        workers = max(1, min((os.cpu_count() or 4) // 2, num_instances, 4))
    label = label or module_name
    rows = {}
    print(f"[{label}] running {num_instances} instances (seed={master_seed}, workers={workers})...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_evaluate_one, module_name, nonce, master_seed): nonce for nonce in range(num_instances)}
        for fut in as_completed(futures):
            r = fut.result()
            rows[r["nonce"]] = r
            if r["failure_reason"]:
                print(f"  [{label}] {r['nonce']:3d} {r['scenario']:10s} FAILED: {r['failure_reason']}", flush=True)
            else:
                print(f"  [{label}] {r['nonce']:3d} {r['scenario']:10s} quality={r['quality']:+7.3f} "
                      f"runtime={r['runtime']:7.1f}s step_max={r['max_step_runtime']*1000:6.1f}ms", flush=True)
    return [rows[n] for n in range(num_instances)]


def write_csv(rows, path):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_md(rows, path, module_name, master_seed):
    by_scn = {s.name: [] for s in SCENARIO_ORDER}
    for r in rows:
        by_scn[r["scenario"]].append(r)

    lines = [f"# Benchmark summary: {module_name} (seed={master_seed}, n={len(rows)})\n"]
    lines.append("| Scenario | Mean | Median | P10 | Minimum | Maximum | Failures | Max Runtime (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    all_q = []
    for s in SCENARIO_ORDER:
        vals = [r["quality_int"] / 1e6 for r in by_scn[s.name] if r["quality_int"] is not None]
        fails = sum(1 for r in by_scn[s.name] if r["failure_reason"])
        rts = [r["runtime"] for r in by_scn[s.name] if r["runtime"] is not None]
        all_q.extend([r["quality_int"] for r in by_scn[s.name]])
        if vals:
            arr = np.array(vals)
            lines.append(
                f"| {s.name} | {arr.mean():+.3f} | {np.median(arr):+.3f} | {np.percentile(arr,10):+.3f} | "
                f"{arr.min():+.3f} | {arr.max():+.3f} | {fails} | {max(rts) if rts else 0:.1f} |"
            )
        else:
            lines.append(f"| {s.name} | -- | -- | -- | -- | -- | {fails} | -- |")

    raw_score = float(np.mean(all_q)) if all_q else 0.0
    final_score = raw_score / 1e7
    total_fails = sum(1 for r in rows if r["failure_reason"])
    max_rt = max((r["runtime"] for r in rows if r["runtime"] is not None), default=0.0)
    max_step = max((r["max_step_runtime"] for r in rows if r["max_step_runtime"] is not None), default=0.0)
    lines.append(f"\n**Final Score: {final_score:.7f}**  (raw={raw_score:.1f})\n")
    lines.append(f"errors={total_fails}  worst_step_time={max_step*1000:.1f}ms  worst_total_episode_time={max_rt:.1f}s\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return final_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-prefix", type=str, default=None)
    args = ap.parse_args()

    importlib.import_module(args.module)

    t0 = time.perf_counter()
    rows = run_benchmark(args.module, num_instances=args.instances, master_seed=args.seed, workers=args.workers)
    prefix = args.out_prefix or args.module
    csv_path = f"{prefix}_benchmark_results.csv"
    md_path = f"{prefix}_benchmark_summary.md"
    write_csv(rows, csv_path)
    final_score = write_summary_md(rows, md_path, args.module, args.seed)
    print(f"\nFinal Score: {final_score:.7f}")
    print(f"Wrote {csv_path}, {md_path}")
    print(f"(wall time: {time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
