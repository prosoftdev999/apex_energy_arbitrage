"""Lightweight, single-process trace of the 6 worst seed-987654 CONGESTED
instances (nonces 86, 11, 41, 26, 81, 16 -- clipped quality 1.19-2.19, all
with materially positive baseline profit). For each step records compact
scalars: desired (pre-repair grid-argmax) vs final action, the LOCAL
OBJECTIVE VALUE LOST to network enforcement (reward+Vc evaluated at desired
vs final action -- a clean $ measure, not a MWh*price proxy), line
utilization, and continuation marginal value. No multiprocessing.
"""
import csv
import random
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

import policy_v11 as v11

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE
WORST6 = [86, 11, 41, 26, 81, 16]


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


FIELDNAMES = ["nonce", "t", "rt_price_mean", "da_price_mean", "soc_frac_mean",
              "desired_sum_mw", "final_sum_mw", "desired_minus_final_mw",
              "value_lost", "repair_engaged", "min_line_headroom_frac",
              "costate_mean", "cum_profit"]


def trace_instance(master_seed, nonce, writer):
    seed = seed_from_master_nonce(master_seed, nonce)
    ch = Challenge.generate_instance(seed, Track(s=Scenario.CONGESTED))
    v11._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    B = ch.num_batteries
    node = np.array([b.node for b in ch.batteries], dtype=np.intp)
    smin = np.array([b.soc_min_mwh for b in ch.batteries], dtype=float)
    smax = np.array([b.soc_max_mwh for b in ch.batteries], dtype=float)
    net = ch.network
    ptdf = np.asarray(net.ptdf, dtype=float)
    limits = np.asarray(net.flow_limits, dtype=float)
    slack = net.slack_bus
    da_all = np.asarray(ch.market.day_ahead_prices, dtype=float)

    prev_repair_count = v11._BENCH_STATS.get("repair_count", 0)

    for t in range(ch.num_steps):
        soc0 = np.array(state.socs, dtype=float)
        bounds = np.array(state.action_bounds, dtype=float)
        lb, ub = bounds[:, 0], bounds[:, 1]
        rt_price = np.array(state.rt_prices, dtype=float)[node]
        da_price = da_all[t][node]

        final_action = v11.policy(view, state)
        final_arr = np.asarray(final_action, dtype=float)

        entry = v11._CACHE[id(view)]
        ba, V_all = entry["ba"], entry["V_all"]
        S = V_all.shape[2]
        V_next_raw = V_all[t + 1] if t + 1 < V_all.shape[0] else np.zeros((B, S))
        specialized = B <= v11._DISCOUNT_MAX_B
        if specialized and entry.get("steps_so_far", 0) >= 8:
            rate = entry.get("repairs_so_far", 0) / entry.get("steps_so_far", 1)
            V_next = V_next_raw * (1.0 - v11._CONGESTION_DISCOUNT_ALPHA * rate)
        else:
            V_next = V_next_raw

        u = v11._candidate_grid(lb, ub)
        abs_u = np.abs(u)
        reward = u * rt_price[:, None] * v11._DT - v11._KAPPA_TX * abs_u * v11._DT \
            - v11._KAPPA_DEG * (abs_u * v11._DT / ba["cap"][:, None]) ** v11._BETA_DEG
        new_soc = v11._apply_action_np(u, soc0[:, None], smin[:, None], smax[:, None],
                                        ba["etac"][:, None], ba["etad"][:, None], v11._DT)
        span = np.maximum(smax - smin, v11._EPS)[:, None]
        idx_frac = np.clip((new_soc - smin[:, None]) / span * (S - 1), 0.0, S - 1 - 1e-06)
        i0 = idx_frac.astype(np.intp)
        i1 = np.minimum(i0 + 1, S - 1)
        w = idx_frac - i0
        rows_ = np.arange(B)[:, None]
        Vc = V_next[rows_, i0] * (1.0 - w) + V_next[rows_, i1] * w
        total = reward + Vc
        idx = np.argmax(total, axis=1)
        desired_action = u[np.arange(B), idx]
        desired_value = float(np.sum(total[np.arange(B), idx]))

        # evaluate the FINAL action under the identical local objective
        final_value = float(v11._evaluate_total_value(
            final_arr, rt_price, ba["cap"], soc0, smin, smax, ba["etac"], ba["etad"], V_next, S))
        value_lost = desired_value - final_value  # >=0 in principle; small negative = LP found better than grid

        repair_engaged = int(v11._BENCH_STATS.get("repair_count", 0) > prev_repair_count)
        prev_repair_count = v11._BENCH_STATS.get("repair_count", 0)

        exo = np.asarray(state.exogenous_injections, dtype=float)
        flows = v11._flows_np(exo, final_arr, node, slack, ptdf)
        min_headroom_frac = float(np.min((limits - np.abs(flows)) / np.maximum(limits, 1e-9)))

        new_soc_final = v11._apply_action_np(final_arr, soc0, smin, smax, ba["etac"], ba["etad"], v11._DT)
        frac_final = np.clip((new_soc_final - smin) / span[:, 0] * (S - 1), 0.0, S - 1 - 1e-06)
        j0 = frac_final.astype(np.intp)
        j1 = np.minimum(j0 + 1, S - 1)
        dsoc_cell = span[:, 0] / (S - 1)
        costate = np.where(j1 != j0, (V_next[np.arange(B), j1] - V_next[np.arange(B), j0]) / np.maximum(dsoc_cell, 1e-9), 0.0)

        writer.writerow(dict(
            nonce=nonce, t=t, rt_price_mean=float(np.mean(rt_price)), da_price_mean=float(np.mean(da_price)),
            soc_frac_mean=float(np.mean((soc0 - smin) / np.maximum(smax - smin, 1e-9))),
            desired_sum_mw=float(np.sum(desired_action)), final_sum_mw=float(np.sum(final_arr)),
            desired_minus_final_mw=float(np.sum(desired_action) - np.sum(final_arr)),
            value_lost=value_lost, repair_engaged=repair_engaged,
            min_line_headroom_frac=min_headroom_frac, costate_mean=float(np.mean(costate)),
            cum_profit=state.total_profit,
        ))

        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, final_action, NextRTPricesGenerate(next_seed))


def main():
    out_csv = _SANDBOX_DIR / "congested_worst6_trace.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for nonce in WORST6:
            trace_instance(987654, nonce, writer)
            print(f"nonce={nonce} done", flush=True)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
