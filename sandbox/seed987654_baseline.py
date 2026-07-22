"""Section 1: complete policy_v11 seed-987654 baseline, instance-level.

Extends the plain profit/quality report with cost/network diagnostics needed
for the rest of this research phase: terminal SOC, charged/discharged energy,
degradation and transaction cost, network-curtailment loss, line-utilization,
and the step of maximum single-step curtailment (a causal proxy for "missed
opportunity" -- the step where the network allocator gave up the most MWh of
otherwise-desired action).
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
_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.0
_BETA_DEG = 2.0


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def _run_detailed(mod_name, ch):
    mod = importlib.import_module(mod_name)
    mod._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    B = ch.num_batteries
    cap = np.array([b.capacity_mwh for b in ch.batteries], dtype=float)
    smin = np.array([b.soc_min_mwh for b in ch.batteries], dtype=float)
    smax = np.array([b.soc_max_mwh for b in ch.batteries], dtype=float)
    etac = np.array([b.efficiency_charge for b in ch.batteries], dtype=float)
    etad = np.array([b.efficiency_discharge for b in ch.batteries], dtype=float)
    node = np.array([b.node for b in ch.batteries], dtype=np.intp)
    # tracked independently -- state.socs is emptied by the production
    # framework after the terminal take_step, so "terminal SOC" must be
    # computed from the same affine battery dynamics used throughout this
    # project rather than read off the post-terminal state.
    soc = np.array([b.soc_initial_mwh for b in ch.batteries], dtype=float)
    net = ch.network
    ptdf = np.asarray(net.ptdf, dtype=float)
    limits = np.asarray(net.flow_limits, dtype=float)
    slack = net.slack_bus

    charged = discharged = deg_cost = tx_cost = 0.0
    max_util = 0.0
    n_active_constraint_steps = 0
    max_curtail, max_curtail_step = -1.0, -1
    curtailed_value = 0.0
    prev_curtailed = mod._BENCH_STATS.get("curtailed_mwh", 0.0)
    worst_step_time = 0.0
    t0 = time.perf_counter()

    for t in range(ch.num_steps):
        st = time.perf_counter()
        a = mod.policy(view, state)
        worst_step_time = max(worst_step_time, time.perf_counter() - st)
        arr = np.asarray(a, dtype=float)
        c = np.maximum(-arr, 0.0)
        d = np.maximum(arr, 0.0)
        charged += float(np.sum(c) * _DT)
        discharged += float(np.sum(d) * _DT)
        abs_a = np.abs(arr)
        deg_cost += float(np.sum(_KAPPA_DEG * (abs_a * _DT / np.maximum(cap, 1e-9)) ** _BETA_DEG))
        tx_cost += float(np.sum(_KAPPA_TX * abs_a * _DT))

        exo = np.asarray(state.exogenous_injections, dtype=float)
        flows = mod._flows_np(exo, arr, node, slack, ptdf)
        util = float(np.max(np.abs(flows) / np.maximum(limits, 1e-9)))
        max_util = max(max_util, util)
        if np.any(np.abs(flows) > 0.999 * limits):
            n_active_constraint_steps += 1

        cur_curtailed_mwh = mod._BENCH_STATS.get("curtailed_mwh", 0.0) - prev_curtailed
        if cur_curtailed_mwh > max_curtail:
            max_curtail, max_curtail_step = cur_curtailed_mwh, t
        if cur_curtailed_mwh > 0:
            price_mean = float(np.mean(np.asarray(state.rt_prices, dtype=float)[node]))
            curtailed_value += cur_curtailed_mwh * price_mean
        prev_curtailed = mod._BENCH_STATS.get("curtailed_mwh", 0.0)

        soc = np.clip(soc + etac * c * _DT - d * _DT / np.maximum(etad, 1e-9), smin, smax)

        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))

    total_time = time.perf_counter() - t0
    frac = (soc - smin) / np.maximum(smax - smin, 1e-9)
    return dict(
        profit=state.total_profit, charged_mwh=charged, discharged_mwh=discharged,
        degradation_cost=deg_cost, transaction_cost=tx_cost,
        terminal_soc_mean=float(np.mean(frac)), terminal_soc_min=float(np.min(frac)),
        terminal_soc_max=float(np.max(frac)),
        curtailed_value=curtailed_value, max_line_util=max_util,
        n_active_constraint_steps=n_active_constraint_steps,
        max_curtail_mwh=float(max(max_curtail, 0.0)), max_curtail_step=max_curtail_step,
        worst_step_time=worst_step_time, total_time=total_time,
    )


def _evaluate_one(master_seed, nonce, mod_name="policy_v11"):
    scenario = SCENARIO_ORDER[nonce % len(SCENARIO_ORDER)]
    seed = seed_from_master_nonce(master_seed, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scenario))
    with contextlib.redirect_stdout(io.StringIO()):
        _, base_profit = ch.compute_baseline()
    try:
        d = _run_detailed(mod_name, ch)
    except Exception as e:
        return dict(seed=master_seed, nonce=nonce, scenario=scenario.name, baseline=base_profit,
                    error=f"{type(e).__name__}: {e}")
    raw_q = (d["profit"] - base_profit) / (base_profit + 1e-6)
    clipped_q = max(-10.0, min(raw_q, 10.0))
    row = dict(seed=master_seed, nonce=nonce, scenario=scenario.name, baseline=base_profit,
               profit=d["profit"], raw_quality=raw_q, clipped_quality=clipped_q, error=None)
    row.update({k: v for k, v in d.items() if k != "profit"})
    return row


FIELDNAMES = ["seed", "nonce", "scenario", "baseline", "profit", "raw_quality", "clipped_quality",
              "charged_mwh", "discharged_mwh", "degradation_cost", "transaction_cost",
              "terminal_soc_mean", "terminal_soc_min", "terminal_soc_max",
              "curtailed_value", "max_line_util", "n_active_constraint_steps",
              "max_curtail_mwh", "max_curtail_step", "worst_step_time", "total_time", "error"]


def main(master_seed=987654, num_instances=100, workers=None, mod_name="policy_v11", out_name=None):
    workers = workers or min(10, os.cpu_count() or 4)
    out_csv = _SANDBOX_DIR / (out_name or f"{mod_name}_seed{master_seed}_detailed.csv")
    t0 = time.perf_counter()
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_evaluate_one, master_seed, n, mod_name): n for n in range(num_instances)}
        for fut in as_completed(futures):
            r = fut.result()
            rows[r["nonce"]] = r
            if r.get("error"):
                print(f"ERROR nonce={r['nonce']} {r['scenario']}: {r['error']}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for n in range(num_instances):
            w.writerow(rows[n])
    print(f"wrote {out_csv} ({num_instances} rows, {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=987654)
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--module", type=str, default="policy_v11")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(args.seed, args.instances, args.workers, args.module, args.out)
